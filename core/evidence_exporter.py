# -*- coding: utf-8 -*-
"""
违章证据包导出 + 电子安全违章通知书生成

用法（任意已完成的输出目录）：
    from core.evidence_exporter import export_evidence_package, generate_rectification_notice
    export_evidence_package(output_dir, video_path=...)      # -> {"dir":..., "zip":..., "count":N}
    generate_rectification_notice(output_dir, video_path=...) # -> docx 路径

设计要点：
- 导出主队列候选事件；主队列不等同已确认，移出主队列的事件只留档、不导出素材
- 视频片段裁剪兼容中文输出路径：先写临时 ASCII 文件再 os.rename
- 安全违章通知书用 python-docx 生成（打包时需 hiddenimports 加 'docx'）
"""
import json
import hashlib
import io
import os
import tempfile
import zipfile
from datetime import datetime

import cv2

from core.annotation_renderer import (candidate_display_label, candidate_evidence_notes,
                                      candidate_original_title,
                                      candidate_review_diagnostics, detail_box_relocalization,
                                      detail_contextual_advice, detail_representative_reference,
                                      detail_review_disposition, detail_source_metadata)
from core.bbox_utils import as_number as _num, bbox_xywh
from core.config import get_config
from core.frame_lookup import normalize_timestamp
from core.helmet_evidence import helmet_evidence_note
from core.person_evidence import person_evidence_note
from core.area_localization import area_evidence_note
from core.atomic_io import atomic_write_json, atomic_write_text, create_unique_directory
from core.custom_scene import (
    events_in_scene, load_scenes, merge_similar_events, scene_span_text,
)
from core.review_policy import (enforce_incomplete_review_guard, source_video_review_guard,
                                split_review_queue)

# 通知书事件表格几何（厘米）：标签列宽 / 内容列宽 / 截图宽度
_LABEL_COL_CM = 4.2
_VALUE_COL_CM = 12.3
_EVENT_IMG_CM = 11

# 违章统计汇总表列宽（厘米）：等级 / 数量 / 占比
_SUMMARY_COL_CM = (5.5, 5.5, 5.5)
# 等级排序权重：重→轻。兼容模型输出与历史数据里"较大"与"较大隐患"两种写法；
# 未收录的等级统一排到末尾。
_SEVERITY_RANK = {
    "重大隐患": 0, "重大": 0,
    "较大隐患": 1, "较大": 1,
    "一般隐患": 2, "一般": 2,
    "轻微违章": 3, "轻微": 3,
}


def _incomplete_candidate(event):
    """缺失门禁或分析不完整均是待核，不依赖其他类别护栏恰好命中。"""
    return isinstance(event, dict) and (
        event.get("analysis_incomplete") is True or bool(event.get("gate_failure_guards")))


def _export_candidate_notes(event, report_incomplete=False):
    """仅改变导出展示，不覆盖原模型证据及门禁审计字段。"""
    notes = list(candidate_evidence_notes(event))
    if _incomplete_candidate(event):
        notes.append(("分析与门禁状态", "分析或门禁未完整执行，候选仅供人工核查，不能作为已确认违章"))
    elif report_incomplete:
        notes.append(("报告门禁状态", "报告存在门禁执行失败，须人工核查该事件受影响范围"))
    return notes


def _export_candidate_label(event, notes):
    label = candidate_display_label(event)
    if label:
        return label
    if not notes:
        return event.get("title", "") if isinstance(event, dict) else ""
    title = candidate_original_title(event)
    if not title.startswith("待核："):
        return "待核：" + title
    return title


def _export_event_view(event):
    """仅在导出副本上修复不完整候选与旧处置冲突，不改源报告。"""
    if not _incomplete_candidate(event):
        return event
    view = dict(event)
    previous = str(view.get("review_disposition") or "")
    if previous and previous != "low_priority_review":
        view.setdefault("export_prior_review_disposition", previous)
    if previous == "box_relocalize":
        view["needs_box_relocalize"] = True
    if view.get("needs_review") is False:
        view.setdefault("export_prior_needs_review", False)
    if view.get("ai_review_confirmed") is True:
        view.pop("ai_review_confirmed")
        view.setdefault("export_prior_ai_review_confirmed", True)
    view["analysis_incomplete"] = True
    view["export_incomplete_guard_applied"] = True
    enforce_incomplete_review_guard(view)
    return view


def _load_report(output_dir):
    """读入 report.json；不存在或内容损坏（截断/非法 JSON）时返回 None。

    原来直接 json.load 会把 JSONDecodeError 抛到 UI 线程之外，用户只看到
    "导出失败"而不知原因；截断的报告多半是历史崩溃残留，明确跳过更友好。
    """
    path = os.path.join(output_dir, "report.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        print(f"[evidence] report.json 无法解析（{type(e).__name__}: {e}），"
              f"请重新分析该视频后再导出")
        return None


def _load_report_context(output_dir, video_path, empty_msg):
    """载入报告并解析出 (report, video_info, vpath, events)。

    证据包与通知书的前置校验完全一致（报告缺失、无事件各自报错），抽出来避免两处漂移。
    返回的 ``events`` 是报告全量事件；「未确认」口径拆分由调用方用
    ``split_review_queue`` 完成，本函数不替调用方决定统计口径。
    """
    report = _load_report(output_dir)
    if report is None:
        raise ValueError("未找到 report.json，请先完成一次分析")
    video_info = report.get("video_info") or {}
    vpath = video_path or video_info.get("path") or ""
    events = [_export_event_view(e) for e in report.get("events") or []]
    if not events:
        raise ValueError(empty_msg)
    return report, video_info, vpath, events


def _video_fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _hash_stable_video(path):
    """流式读取文件并核对文件指纹；失败只返回 None，不记录异常正文。"""
    try:
        with open(path, 'rb') as stream:
            before = _video_fingerprint(os.fstat(stream.fileno()))
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            after = _video_fingerprint(os.fstat(stream.fileno()))
        current = _video_fingerprint(os.stat(path))
        if before != after or after != current:
            return None
        return digest, current
    except (OSError, TypeError, ValueError):
        return None


def _video_source_integrity(report, vpath):
    """区分旧报告未记录摘要、新报告来源无效、和当前视频字节核验成功。"""
    if 'source_video_provenance' not in report:
        return {'status': 'legacy_unverified', 'allow_clip': True,
                'note': '视频来源未核验，片段仅供参考，不能视为与报告同源的证据'}
    provenance = report.get('source_video_provenance')
    expected = provenance.get('sha256') if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or provenance.get('status') != 'digest_rechecked'
            or not isinstance(expected, str) or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected)):
        return {'status': 'source_unverified', 'allow_clip': False,
                'note': '视频来源未核验，不能生成事件片段'}
    actual = _hash_stable_video(vpath)
    if actual is None or actual[0] != expected:
        return {'status': 'digest_mismatch_or_unreadable', 'allow_clip': False,
                'note': '视频来源摘要不一致或无法读取，不能生成事件片段'}
    return {'status': 'digest_verified', 'allow_clip': True, 'sha256': expected, 'fingerprint': actual[1],
            'note': '当前视频摘要与报告记录一致；前后核验不能排除过程中短暂替换'}


def _video_source_unchanged(vpath, identity):
    """新报告在片段导出前后重算摘要并核对初始指纹。"""
    if identity.get('status') != 'digest_verified':
        return False
    actual = _hash_stable_video(vpath)
    return bool(actual and actual[0] == identity.get('sha256') and actual[1] == identity.get('fingerprint'))


def _stats_stale(report, kept, moved):
    """报告正文统计是否为旧版残留口径（需要提示用户重跑分析）。

    v1.13.38 及更早版本写入 disposition 后没有重建 scene_summary，正文合计
    仍是全量事件数；此时导出通知书/证据包会按主队列重算，正文与文书天然冲突。
    判定：有事件被移出主队列，且 scene_summary.violation_count 合计 != len(kept)。

    字段缺失或类型异常一律返回 False：宁可少提示，也不对正常报告误报。
    """
    if not isinstance(report, dict) or not moved:
        return False
    if "scene_summary" not in report:
        return False
    summary = report.get("scene_summary")
    if not isinstance(summary, list):
        return False
    total = 0
    for s in summary:
        if not isinstance(s, dict):
            return False
        v = s.get("violation_count")
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        total += v
    return total != len(kept)


def _fmt_ts(sec):
    """秒 → mm:ss；无效时间明确未知，不借用第 0 秒。"""
    value = normalize_timestamp(sec)
    if value is None:
        return "时间未知"
    if value >= 1_000_000_000:
        return f"约{value:.2e}秒"
    centiseconds = int(round(value * 100))
    minutes, remaining = divmod(centiseconds, 6000)
    seconds, fraction = divmod(remaining, 100)
    rendered = f"{minutes:02d}:{seconds:02d}" + (f".{fraction:02d}" if fraction else "")
    return ("约" if centiseconds / 100 != value else "") + rendered


def _fmt_duration(sec):
    value = normalize_timestamp(sec)
    if value is None:
        return "未知"
    rendered = f"{value:.2f}".rstrip("0").rstrip(".")
    return ("约" if float(rendered) != value else "") + rendered + "秒"


def _fmt_bbox(bbox):
    """bbox → 可读坐标串；None/缺失/非有限值一律按 0（不得让格式化崩掉整份通知书）"""
    if not bbox:
        return ""
    bx, by, bw, bh = bbox_xywh(bbox)
    return (f"x={bx:.0f} y={by:.0f} "
            f"w={bw:.0f} h={bh:.0f}（画面像素坐标）")


def _fmt_span(e):
    """只描述首末采样跨度；稀疏采样不能推断持续时间或动作连续。"""
    first, last = normalize_timestamp(e.get('first_seen')), normalize_timestamp(e.get('last_seen'))
    span = last - first if first is not None and last is not None and last >= first else None
    count = e.get('frame_count')
    count_text = f"共 {count} 个采样帧" if type(count) is int and count > 0 else "采样帧数未知"
    text = (f"{_fmt_ts(first)} — {_fmt_ts(last)}"
            f"（首末采样跨度 {_fmt_duration(span)}；{count_text}；不代表中间连续发生）")
    n_rep = int(_num(e.get("repeat_count"), 1))
    if n_rep > 1:
        text += f"，系统归并记录 {n_rep} 次（是否同一位置需核实）"
    return text


def _fmt_basis(e):
    """判定依据文本：『《规程》第 X 页：条款正文』，两者缺一按有则拼接。"""
    parts = (e.get("regulation") or "", e.get("regulation_text") or "")
    return "：".join(p for p in parts if p)


def _scene_label(scene):
    """场景标题文本：『名称（mm:ss — mm:ss）』"""
    name = ((scene or {}).get("name") or "").strip() or "自定义场景"
    span = scene_span_text(scene)
    return f"{name}（{span}）" if span else name


def _build_display_groups(confirmed, scenes):
    """把事件分配到自定义场景并做场景内相似合并 → [(场景标题, 条目列表)]。

    事件按帧序号交集归属**第一个**命中的场景，避免重复计数；未归入任何场景的
    事件落进最后一组（标题为空串）。没有自定义场景时返回 [(\"\", 原事件)]，
    导出行为与旧版完全一致。
    """
    if not scenes:
        return [("", list(confirmed or []))]
    used = set()
    groups = []
    for sc in scenes:
        member = [e for e in events_in_scene(confirmed, sc) if id(e) not in used]
        for e in member:
            used.add(id(e))
        if member:
            # 人员/安全帽线索各有独立原框和反证，同名合并会丢失后续事件证据。
            guarded_titles = {str(e.get("title") or "") for e in member if _export_candidate_notes(e)}
            if guarded_titles:
                protected = [e for e in member if str(e.get("title") or "") in guarded_titles]
                regular = [e for e in member if str(e.get("title") or "") not in guarded_titles]
                display = sorted(merge_similar_events(regular) + protected,
                                 key=lambda e: _num(e.get("first_seen")))
            else:
                display = merge_similar_events(member)
            groups.append((_scene_label(sc), display))
    rest = [e for e in confirmed if id(e) not in used]
    if rest:
        groups.append(("", rest))
    return groups


# 片段编码候选：优先体积小的 H.264，其次 MPEG-4（AVI 容器，老机器兼容好），
# 最后 MJPG（无编码器依赖，但约 5MB/s）。探测结果按进程缓存，避免每段重试。
_CLIP_CODEC_CANDIDATES = (
    ("h264", "avc1", ".mp4", "H.264"),
    ("mp4v", "mp4v", ".avi", "MPEG-4"),
    ("mjpg", "MJPG", ".avi", "MJPG"),
)
_CLIP_CODEC_CACHE = {}


def _codec_works(fourcc, ext):
    """探测编码器是否可用（写一帧到临时文件）"""
    import tempfile

    import numpy as np

    path = os.path.join(tempfile.gettempdir(), f"_codec_probe_{os.getpid()}{ext}")
    writer = None
    try:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), 10.0, (320, 240))
        if not writer.isOpened():
            return False
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        return True
    except Exception:
        return False
    finally:
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def pick_clip_codec(prefer="auto"):
    """选片段编码，返回 (fourcc, ext, label)。

    prefer: auto / h264 / mp4v / mjpg；强制项不可用时按 auto 顺序回退，
    保证任何环境都能导出（最差退回 MJPG）。
    """
    prefer = str(prefer or "auto").strip().lower()
    if prefer in _CLIP_CODEC_CACHE:
        return _CLIP_CODEC_CACHE[prefer]
    order = list(_CLIP_CODEC_CANDIDATES)
    if prefer != "auto":
        order.sort(key=lambda c: 0 if c[0] == prefer else 1)
    chosen = None
    for _key, fourcc, ext, label in order:
        if _codec_works(fourcc, ext):
            chosen = (fourcc, ext, label)
            break
    if chosen is None:      # 理论上 MJPG 永远可用；兜底避免 None 崩溃
        chosen = ("MJPG", ".avi", "MJPG")
    _CLIP_CODEC_CACHE[prefer] = chosen
    return chosen


def _clip_window(first, last, pad_sec, duration, max_sec=0.0, anchor=None):
    """算片段窗口 (start, end, truncated)。

    事件跨度超过 max_sec 时以锚点为中截取（锚点缺失取事件中点），
    并夹在 [0, duration] 内——防止长事件把整段视频都裁进证据包。
    """
    first, last = normalize_timestamp(first), normalize_timestamp(last)
    if first is None or last is None or last < first:
        raise ValueError("invalid_event_time")
    pad_sec = max(0.0, _num(pad_sec))
    duration = max(0.0, _num(duration))
    max_sec = max(0.0, _num(max_sec))
    if duration and (first > duration or last > duration):
        raise ValueError("event_time_outside_video")
    start = max(0.0, first - pad_sec)
    end = max(start + 1.0, last + pad_sec)
    if duration > 0:
        end = min(end, duration)
    if not max_sec or max_sec <= 0:
        return start, end, False
    span = end - start
    if span <= max_sec:
        return start, end, False
    center = normalize_timestamp(anchor)
    if center is None or not first <= center <= last:
        center = (first + last) / 2.0
    start = center - max_sec / 2.0
    if duration > 0:
        start = min(start, max(0.0, duration - max_sec))
    start = max(0.0, start)
    end = start + max_sec
    if duration > 0:
        end = min(end, duration)
    return start, end, True


def _clip_video(video_path, start_ts, end_ts, out_path, codec=None):
    """裁剪视频片段 [start_ts, end_ts]。

    codec: pick_clip_codec() 的返回值；缺省按 auto 选（H.264→MPEG-4→MJPG）。
    中文路径兼容：先写临时 ASCII 文件再 rename。
    独占临时文件，只有完整读出目标帧后才发布，绝不覆盖已有片段。"""
    if codec is None:
        codec = pick_clip_codec()
    fourcc, _ext, _label = codec
    tmp = None
    cap = None
    writer = None

    def _safe_close():
        """释放句柄（重复调用安全）——异常路径下不释放会累积句柄"""
        for obj in (writer, cap):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass

    try:
        start_ts, end_ts = normalize_timestamp(start_ts), normalize_timestamp(end_ts)
        if start_ts is None or end_ts is None or end_ts <= start_ts:
            return False
        fd, tmp = tempfile.mkstemp(prefix='_clip_tmp_', suffix=_ext, dir=os.path.dirname(out_path))
        os.close(fd)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        fps = _num(cap.get(cv2.CAP_PROP_FPS), None)
        w = _num(cap.get(cv2.CAP_PROP_FRAME_WIDTH), None)
        h = _num(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), None)
        total = _num(cap.get(cv2.CAP_PROP_FRAME_COUNT), None)
        if any(value is None or value <= 0 for value in (fps, w, h, total)):
            _safe_close()
            return False
        w, h, total = int(w), int(h), int(total)
        writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if not writer.isOpened():
            _safe_close()
            return False
        start_frame = max(0, int(start_ts * fps))
        end_frame = min(max(0, total - 1), int(end_ts * fps))
        if start_frame >= total or end_frame < start_frame:
            return False
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
            return False
        actual_frame = _num(cap.get(cv2.CAP_PROP_POS_FRAMES), None)
        if actual_frame is None or abs(actual_frame - start_frame) > 0.5:
            return False
        idx = start_frame
        while idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            idx += 1
        writer.release()
        writer = None
        cap.release()
        cap = None
        if idx > end_frame and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            if os.name == 'nt':
                os.rename(tmp, out_path)
            else:
                os.link(tmp, out_path)
                os.remove(tmp)
            tmp = None
            return True
        return False
    except Exception as ex:
        print(f"视频片段裁剪失败（{type(ex).__name__}）")
        return False
    finally:
        _safe_close()
        if tmp is not None and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _event_image_path(e, output_dir):
    """兼容旧路径 API；实际导出时还需一次性读取并校验图像字节。"""
    eid = int(_num(e.get('event_id'), 0))
    source = os.path.join(output_dir, 'events', f'event_{eid:02d}.jpg')
    data, _ = _event_image_bytes(e, output_dir)
    return source if data is not None else ''


def _event_image_bytes(e, output_dir):
    """新版图片必须匹配报告摘要；无状态旧图仅可作为未核验历史线索。"""
    eid = e.get('event_id')
    if isinstance(eid, str) and eid.isascii() and eid.isdecimal() and len(eid) <= 12:
        eid = int(eid)
    if isinstance(eid, bool) or not isinstance(eid, int) or eid < 0:
        return None, 'unavailable'
    recorded = 'event_image_status' in e
    if recorded and e.get('event_image_status') != 'synchronized':
        return None, 'unavailable'
    source = os.path.join(output_dir, 'events', f'event_{eid:02d}.jpg')
    try:
        root = os.path.realpath(os.path.join(output_dir, 'events'))
        output_root = os.path.realpath(output_dir)
        if (os.path.islink(source) or os.path.commonpath((output_root, root)) != output_root
                or os.path.commonpath((root, os.path.realpath(source))) != root):
            return None, 'unavailable'
        with open(source, 'rb') as stream:
            data = stream.read(100 * 1024 * 1024 + 1)
        if not data or len(data) > 100 * 1024 * 1024:
            return None, 'unavailable'
        if recorded:
            expected = e.get('event_image_sha256')
            if (not isinstance(expected, str) or len(expected) != 64
                    or any(c not in '0123456789abcdef' for c in expected)
                    or hashlib.sha256(data).hexdigest() != expected):
                return None, 'unverified_or_changed'
            return data, 'recorded_sha256_verified'
        return data, 'legacy_unverified'
    except (OSError, ValueError, TypeError):
        return None, 'unavailable'


def _copy_detail_card(e, pkg_dir, output_dir):
    """只导出本次报告明确标记 ready 且位于固定目录的说明卡。"""
    eid = e.get('event_id')
    if (e.get('detail_card_status') != 'ready' or isinstance(eid, bool)
            or not isinstance(eid, int) or eid < 0):
        return ''
    expected = f'events_detail/event_{eid:02d}.png'
    if e.get('detail_card_path') != expected:
        return ''
    source = os.path.join(output_dir, 'events_detail', f'event_{eid:02d}.png')
    output_root = os.path.realpath(output_dir)
    root = os.path.realpath(os.path.join(output_dir, 'events_detail'))
    try:
        safe = (os.path.isfile(source) and not os.path.islink(source)
                and os.path.commonpath((output_root, root)) == output_root
                and os.path.commonpath((root, os.path.realpath(source))) == root)
    except (OSError, ValueError):
        safe = False
    if not safe:
        return ''
    expected_hash = e.get('detail_card_sha256')
    if (not isinstance(expected_hash, str) or len(expected_hash) != 64
            or any(c not in '0123456789abcdef' for c in expected_hash)):
        return ''
    try:
        with open(source, 'rb') as stream:
            data = stream.read(100 * 1024 * 1024 + 1)
            if (len(data) > 100 * 1024 * 1024 or not data.startswith(b'\x89PNG\r\n\x1a\n')
                    or hashlib.sha256(data).hexdigest() != expected_hash):
                return ''
    except OSError:
        return ''
    name = f'event_{eid:02d}_detail.png'
    with open(os.path.join(pkg_dir, name), 'xb') as target:
        target.write(data)
    return name


def _copy_event_assets(e, pkg_dir, output_dir, vpath, duration, pad_sec,
                       clip_max_sec=0.0, codec=None):
    """复制/导出一个事件的素材：代表帧图片 + 视频片段。

    Returns:
        (img_name, clip_name, clip_ok, clip_info)
        clip_info: {"start", "end", "truncated", "codec"}；未导出片段时为 {}
    """
    e = e or {}
    eid = int(_num(e.get("event_id"), 0))
    # 代表帧可能根本没落盘（后置规则事件/导出时源图已被清理）。此前无条件
    # 写 event_XX.jpg，导致 metadata/清单声称有图、包内却没有该文件；这里
    # 只在实际复制成功时给出文件名，缺图留空串由调用方按「无代表帧」渲染。
    img_data, _ = _event_image_bytes(e, output_dir)
    img_name = ""
    if img_data is not None:
        img_name = f"event_{eid:02d}.jpg"
        with open(os.path.join(pkg_dir, img_name), 'xb') as stream:
            stream.write(img_data)
    codec = codec or pick_clip_codec()
    clip_name = f"event_{eid:02d}_clip{codec[1]}"

    clip_ok = False
    clip_info = {}
    if vpath and os.path.exists(vpath):
        try:
            start, end, truncated = _clip_window(
                e.get("first_seen"), e.get("last_seen"), pad_sec, duration,
                clip_max_sec, e.get("representative_timestamp"))
        except ValueError:
            clip_info = {"unavailable_reason": "event_time_unverified"}
        else:
            clip_ok = _clip_video(vpath, start, end, os.path.join(pkg_dir, clip_name), codec)
            if clip_ok:
                clip_info = {
                    "start": round(start, 1),
                    "end": round(end, 1),
                    "truncated": bool(truncated),
                    "codec": codec[2],
                }
    return img_name, clip_name, clip_ok, clip_info


def _event_meta_item(e, img_name, clip_name, clip_ok):
    """事件在 metadata.json 中的条目（字段即证据包对外契约，勿随意增删）

    event_id 统一转 int：报告可能缺该键（历史版本）或存成字符串，
    硬取键会让整份证据包导出在这里 KeyError。
    """
    e = e or {}
    item = {
        "event_id": int(_num(e.get("event_id"), 0)),
        "title": e.get("title", ""),
        "severity": e.get("severity", ""),
        "category": e.get("category", ""),
        "first_seen": e.get("first_seen"),
        "last_seen": e.get("last_seen"),
        "duration_sec": e.get("duration_sec"),
        "frame_count": e.get("frame_count"),
        "frame_indices": e.get("frame_indices", []),
        "bbox": e.get("bbox") or {},
        "bbox_pre_tighten": e.get("bbox_pre_tighten"),
        "representative_frame": e.get("representative_frame"),
        "representative_timestamp": e.get("representative_timestamp"),
        "stability": e.get("stability"),
        "description": e.get("description", ""),
        "regulation": e.get("regulation", ""),
        "regulation_text": e.get("regulation_text", ""),
        "image": img_name,
        "clip": clip_name if clip_ok else "",
        "clip_ok": clip_ok,
    }
    if 'event_image_status' in e:
        item['event_image_status'] = e.get('event_image_status')
        item['event_image_bbox'] = e.get('event_image_bbox')
        item['event_image_sha256'] = e.get('event_image_sha256')
        item['event_image_source_frame_sha256'] = e.get('event_image_source_frame_sha256')
    if img_name:
        item['event_image_integrity'] = ('recorded_sha256_verified'
            if e.get('event_image_status') == 'synchronized' else 'legacy_unverified')
    elif 'event_image_status' in e:
        item['event_image_integrity'] = 'not_exported_or_unverified'
    for key in ('evidence', 'target_location', 'detail_source', 'detail_notice', 'scene_name', 'scene_id',
                'detail_card_status', 'detail_card_meta', 'detail_card_source', 'detail_card_sha256',
                'rule_based', 'track_based', 'clip_based', 'detail_provenance_conflict',
                'analysis_incomplete', 'analysis_incomplete_blocked_disposition', 'gate_failure_guards',
                'gate_failure_prior_disposition', 'gate_failure_prior_ai_review_confirmed',
                'local_gate_frame_review_hold', 'local_gate_prior_disposition', 'local_gate_prior_ai_review_confirmed',
                'export_incomplete_guard_applied', 'export_prior_review_disposition',
                'export_prior_needs_review', 'export_prior_ai_review_confirmed',
                'helmet_evidence', 'helmet_evidence_requires_review',
                'helmet_evidence_blocked_disposition', 'person_evidence',
                'person_evidence_requires_review', 'person_evidence_blocked_disposition',
                'person_evidence_prior_ai_review_confirmed', 'lift_anchor_bbox_status',
                'area_localization', 'area_localization_requires_review',
                'area_localization_blocked_disposition',
                'source_video_review_hold', 'source_video_review_status', 'source_video_review_reason',
                'source_video_blocked_disposition', 'source_video_prior_needs_review',
                'source_video_prior_ai_review_confirmed',
                'event_image_prior_unverified', 'event_image_unverified_history',
                'frozen_assets_unverified_guard', 'golden_guard_prior_identity_claim',
                'height_gate_frame_unverified', 'height_gate_frame_status', 'height_gate_blocked_disposition',
                'height_gate_frame_index', 'height_gate_frame_timestamp', 'height_gate_frame_sha256',
                'height_gate_reference_bbox', 'height_gate_review_status', 'height_gate_prior_attempts',
                'height_gate_prior_ai_review_confirmed', 'height_gate_prior_needs_review',
                'strict_gate_frame_unverified', 'strict_gate_frame_status', 'strict_gate_blocked_disposition',
                'strict_gate_frame_index', 'strict_gate_frame_timestamp', 'strict_gate_frame_sha256', 'strict_gate_status',
                'strict_gate_prior_attempts', 'strict_gate_prior_ai_review_confirmed',
                'strict_gate_prior_needs_review', 'strict_gate_frame_verified_before_failure',
                'box_entity_frame_unverified', 'box_entity_frame_status', 'box_entity_prior_attempts',
                'box_entity_blocked_disposition', 'box_entity_prior_ai_review_confirmed',
                'daylight_frame_unverified', 'daylight_frame_status', 'daylight_frame_prior_disposition',
                'daylight_frame_prior_needs_review', 'daylight_frame_prior_confirmed',
                'daylight_golden_prior_disposition', 'daylight_golden_prior_needs_review',
                'daylight_golden_prior_ai_confirmed', 'ai_review_golden_exempt',
                'ai_review_frame_unverified', 'ai_review_frame_status', 'ai_review_frame_prior_disposition',
                'ai_review_frame_prior_needs_review', 'ai_review_frame_prior_confirmed',
                'needs_box_relocalize',
                'bbox_raw_1000', 'entity_reference_candidate_bbox_1000', 'entity_reference_conflict',
                'needs_review', 'review_disposition', 'suspected_false',
                'review_reason', 'ai_review_reason', 'ai_review_note'):
        if key in e:
            item[key] = e[key]
    return item


def _append_event_text(lines, item, report_incomplete=False):
    """把一个事件写进「清单.txt」的文本块。

    全部字段走 .get + 数值安全转换：报告可能来自旧版本或经 JSON 净化，
    缺字段/None 不应让整份证据包导出失败。
    """
    eid = int(_num(item.get("event_id"), 0))
    lines.append("-" * 56)
    evidence_notes = _export_candidate_notes(item, report_incomplete=report_incomplete)
    shown_title = _export_candidate_label(item, evidence_notes)
    lines.append(f"事件{eid:02d} [{item.get('severity') or '一般'}] {shown_title}")
    if evidence_notes:
        raw_title = candidate_original_title(item)
        lines.append(f"  原模型标题（待核）：{raw_title}")
        for label, note in evidence_notes:
            lines.append(f"  {label}：{note}")
        lines.append(f"  复核处置：{detail_review_disposition(item)['disposition_note']}")
        for _, label, value in candidate_review_diagnostics(item):
            lines.append(f"  {label}：{value}")
        box_note = detail_box_relocalization(item)["box_relocalization_note"]
        if box_note:
            lines.append(f"  框状态：{box_note}")
    if not evidence_notes and item.get("ai_review_note"):
        lines.append(f"  复核状态：{item['ai_review_note']}（不得作为已确认违章）")
    if item.get("scene"):
        lines.append(f"  所属场景：{item['scene']}")
    lines.append(f"  采样时间：{_fmt_span(item)}")
    lines.append("  " + detail_representative_reference(item)["text"])
    bbox = item.get("bbox") or {}
    if bbox:
        bx, by, bw, bh = bbox_xywh(bbox)
        lines.append(f"  位置坐标：x={bx:.0f} y={by:.0f} w={bw:.0f} h={bh:.0f}")
    source = detail_source_metadata(item)
    lines.append(f"  说明：{source['source_note']}")
    if item.get('analysis_incomplete') is True:
        lines.append('  分析状态：不完整，候选仅供人工核查')
    if item.get('event_image_integrity') == 'legacy_unverified':
        lines.append('  事件图：历史图未记录可核验摘要，仅供参考')
    elif item.get('event_image_integrity') == 'not_exported_or_unverified':
        lines.append('  事件图：未导出或摘要不匹配，不能作为已核验图片')
    for label, key in ((source['position_label'], 'target_location'),
                       (source['description_label'], 'description'),
                       (source['evidence_label'], 'evidence')):
        value = item.get(key)
        lines.append(f"  {label}：{value if isinstance(value, str) and value.strip() else '未提供'}")
    for label, value in detail_contextual_advice(item):
        lines.append(f"  {label}：{value}")
    for label, key in (("法规依据（待核）", "regulation"), ("法规条款（待核）", "regulation_text")):
        if item.get(key):
            lines.append(f"  {label}：{item[key]}")
    if item.get("clip_truncated"):
        win = item.get("clip_window") or []
        if len(win) == 2:
            lines.append(f"  片段范围：{_fmt_ts(win[0])} — {_fmt_ts(win[1])}"
                         "（事件跨度过长，已以锚点为中心截取）")
    if item.get("clip_unavailable_reason") == "event_time_unverified":
        lines.append("  片段说明：采样时间缺失、无效或超出视频范围，未导出片段。")
    if item.get("clip_source_status") == "legacy_unverified" and item.get("clip_ok"):
        lines.append("  片段来源：视频来源未核验，片段仅供参考。")
    elif item.get("clip_unavailable_reason") in ("video_source_unverified", "video_source_changed_during_export"):
        lines.append("  片段说明：视频来源未核验或导出期间发生变化，未提供片段。")
    # 无代表帧时不得渲染成「材料：、xxx」或空文件名；仍保留片段信息
    img = item.get("image", "") or ""
    clip = item.get("clip", "") or ""
    if item.get("assets_withheld") is True:
        material = "（未随包导出素材，请查看原报告及原输出目录）"
    elif img:
        material = img + (f"、{clip}" if item.get("clip_ok") else "（视频片段导出失败）")
    elif item.get("clip_ok") and clip:
        material = f"（无代表帧）、{clip}"
    elif clip:
        material = "（无代表帧；视频片段导出失败）"
    else:
        material = "（无代表帧）"
    lines.append(f"  材料：{material}")
    if item.get('detail_image'):
        lines.append(f"  事件详情卡（待核）：{item['detail_image']}")
    lines.append("")


# 已是压缩格式的素材不再 deflate：收益接近 0，耗时长，且大文件在部分
# Python 构建上会触发 zlib 崩溃（2026-10-19 实测 1.5GB AVI 段错误）
_ZIP_STORE_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".jpg", ".jpeg", ".png", ".webp")


def _zip_dir(pkg_dir):
    """把包目录压成 zip，返回 zip 路径。

    先写 .tmp 再原子替换：导出中途失败/断电时用户不会拿到打不开的半截 zip。
    媒体文件走 STORED、文本走 DEFLATED：既快又稳（见 _ZIP_STORE_EXTS 注释）。
    """
    zip_path = pkg_dir + ".zip"
    zip_tmp = zip_path + ".tmp"
    try:
        with zipfile.ZipFile(zip_tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for root, _dirs, files in os.walk(pkg_dir):
                for fn in files:
                    fp = os.path.join(root, fn)
                    rel = os.path.relpath(fp, os.path.dirname(pkg_dir))
                    comp = (zipfile.ZIP_STORED
                            if os.path.splitext(fn)[1].lower() in _ZIP_STORE_EXTS
                            else zipfile.ZIP_DEFLATED)
                    zf.write(fp, rel, compress_type=comp)
        os.replace(zip_tmp, zip_path)
    except Exception:
        try:
            if os.path.exists(zip_tmp):
                os.remove(zip_tmp)
        except OSError:
            pass
        raise
    return zip_path


def export_evidence_package(output_dir, video_path=None, pad_sec=3.0,
                            out_root=None, progress_cb=None):
    """导出主队列证据包（候选不等于已确认违章）。
    返回 {"dir": 目录, "zip": 压缩包, "count": 事件数}"""
    report, video_info, vpath, events = _load_report_context(
        output_dir, video_path, "没有违章事件，无需导出证据包")
    source_integrity = _video_source_integrity(report, vpath)
    # 未确认事件（AI 复核移出主队列）不计入违章数、不导出素材；它们仍保留在
    # report.json 与界面中可查，这里只影响证据包口径。
    confirmed, unconfirmed = split_review_queue(events)
    if not confirmed:
        raise ValueError("报告中的事件均已由 AI 复核移出主队列（未确认），无需导出证据包")

    total = len(confirmed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 每次导出独占目录，避免同秒重试将旧的、不再允许导出的图片混入新包。
    pkg_dir = create_unique_directory(out_root or output_dir, f"证据包_{ts}",
                                      reserved_suffixes=(".zip",))

    duration_record = normalize_timestamp(video_info.get("duration"))
    duration = duration_record if duration_record is not None else 0.0
    # 片段策略：上限秒数 + 编码（配置缺失/异常一律回退安全默认，不能让导出崩）
    try:
        cfg = get_config()
        clip_max_sec = float(cfg.get("evidence_clip_max_sec", 60.0) or 0)
        codec = pick_clip_codec(cfg.get("evidence_clip_codec", "auto"))
    except Exception:
        clip_max_sec, codec = 60.0, pick_clip_codec()
    # 自定义场景归属（事件标注用；证据包仍按原始事件逐条导出素材，不做合并）
    ev_scene = {}
    for sc in load_scenes(output_dir):
        label = _scene_label(sc)
        for e in events_in_scene(confirmed, sc):
            ev_scene.setdefault(int(_num((e or {}).get("event_id"), 0)), label)
    meta = {
        "导出时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "视频文件": video_info.get("filename", os.path.basename(vpath or "")),
        "视频时长秒": round(duration_record, 1) if duration_record is not None else None,
        "分析时间": report.get("analysis_time", ""),
        "违章事件数": total,
        "未确认不计入事件数": len(unconfirmed),
        "片段前后缓冲秒": pad_sec,
        "片段上限秒": clip_max_sec,
        "片段编码": codec[2],
        "事件": [],
    }
    if unconfirmed:
        meta["未确认事件ID"] = [int(_num(e.get("event_id"), 0)) for e in unconfirmed]
    helmet_candidates = [e for e in confirmed if helmet_evidence_note(e)]
    person_candidates = [e for e in confirmed if person_evidence_note(e)]
    area_candidates = [e for e in confirmed if area_evidence_note(e)]
    incomplete_candidates = [e for e in confirmed if _incomplete_candidate(e)]
    source_candidates = [e for e in confirmed if source_video_review_guard(e)]
    frozen_unknown_candidates = [e for e in confirmed if e.get("frozen_assets_unverified_guard") is True]
    frame_unknown_candidates = [e for e in confirmed if any(e.get(key) is True for key in (
        "height_gate_frame_unverified", "strict_gate_frame_unverified", "box_entity_frame_unverified",
        "daylight_frame_unverified", "ai_review_frame_unverified")) or
        e.get("height_gate_review_status") in ("uncertain", "support_unverified") or
        e.get("strict_gate_status") == "missing_prerequisite"]
    if helmet_candidates:
        meta["待核安全帽线索事件ID"] = [int(_num(e.get("event_id"), 0)) for e in helmet_candidates]
        meta["安全帽线索说明"] = "这些事件仅为待核查线索，不是已确认违章"
    if person_candidates:
        meta["待核人员身份线索事件ID"] = [int(_num(e.get("event_id"), 0)) for e in person_candidates]
        meta["人员身份线索说明"] = "这些事件的主体身份尚未确认，不是已确认违章"
    if area_candidates:
        meta["待核区域定位线索事件ID"] = [int(_num(e.get("event_id"), 0)) for e in area_candidates]
        meta["区域定位线索说明"] = "这些事件的区域位置尚未确认，背景候选不是违章框"
    if source_candidates:
        meta["来源级保守待核事件ID"] = [int(_num(e.get("event_id"), 0)) for e in source_candidates]
        meta["来源级保守待核说明"] = "来源级留队列不代表事件匹配金标或违章成立"
    if frozen_unknown_candidates:
        meta["冻结资产未核验事件ID"] = [int(_num(e.get("event_id"), 0)) for e in frozen_unknown_candidates]
        meta["冻结资产未核验说明"] = "资产无法核验时保守待核，不代表已匹配金标"
    if frame_unknown_candidates:
        meta["门禁证据未核验事件ID"] = [int(_num(e.get("event_id"), 0)) for e in frame_unknown_candidates]
        meta["门禁证据未核验说明"] = "原帧、框、检测器或门禁结论未能核验，不能当作确定结论"
    if incomplete_candidates:
        meta["分析或门禁不完整候选事件ID"] = [int(_num(e.get("event_id"), 0)) for e in incomplete_candidates]
        meta["分析或门禁不完整候选说明"] = "这些事件仅供人工核查，不是已确认违章"
    if report.get("source_video_candidate_completeness"):
        meta["来源完整性提示"] = "来源视频发生变化或来源未能核验，候选结果可能不完整，需重新分析并人工核查"
        meta["来源核验首个失败原因"] = report.get("source_video_first_failure_reason") or "unknown"
    if report.get("gate_failures"):
        meta["门禁执行不完整提示"] = "一个或多个门禁执行失败，相关候选仅供人工核查，结果不能视为完整验收"
        failures = report["gate_failures"] if isinstance(report["gate_failures"], list) else []
        meta["门禁失败阶段"] = [entry.get("gate") for entry in failures
                                if isinstance(entry, dict) and entry.get("gate") in
                                ("height", "strict", "box_entity", "helmet", "daylight")]
    has_candidates = bool(report.get("gate_failures")) or any(
        _export_candidate_notes(e) for e in confirmed)
    if has_candidates:
        meta["主队列候选数"] = total
        meta["违章事件数口径说明"] = "兼容字段，统计主队列候选；含未确认线索，不是已确认违章数"
    # 旧版报告（v1.13.38 及更早）正文统计未按主队列重建：本包按主队列重算，
    # 在 meta 里显式标注口径差异，避免与新文书冲突时无从解释。
    if _stats_stale(report, confirmed, unconfirmed):
        meta["统计口径提示"] = "报告正文统计为旧版口径，本证据包已按主队列重算"
        print("[evidence] 旧报告统计口径不一致（scene_summary 未按主队列重建），"
              "已在 metadata 标注并提示重新分析")
    if ev_scene:
        meta["自定义场景"] = sorted(set(ev_scene.values()))
    truncated_events = 0
    lines = [
        "=" * 56,
        "违章证据包清单",
        "=" * 56,
        f"视频文件：{meta['视频文件']}",
        f"分析时间：{meta['分析时间']}",
        f"违章事件：{total} 项\n",
    ]
    if has_candidates:
        lines[1] = "安全隐患核查线索清单（含未确认候选）"
        lines[-1] = f"主队列候选：{total} 项（含未确认线索，不能视作已确认违章）\n"
    if report.get("source_video_candidate_completeness"):
        lines.append("来源视频发生变化或来源未能核验，候选结果可能不完整，请重新分析并人工核查。")
        lines.append(f"来源核验首个失败原因：{report.get('source_video_first_failure_reason') or 'unknown'}\n")
    if report.get("gate_failures"):
        lines.append("一个或多个门禁执行失败，相关候选仅供人工核查，结果不能视为完整验收。")

    for i, e in enumerate(confirmed):
        if progress_cb:
            # event_id 用 .get：旧版 report.json 的事件可能缺该键（本模块其余处
            # 一律 .get 容错），硬取会在接了进度回调后中断整包导出
            progress_cb(int((i + 1) / total * 100),
                        f"导出事件{e.get('event_id', '?')}...")
        clip_video = vpath if source_integrity['allow_clip'] else ''
        img_name, clip_name, clip_ok, clip_info = _copy_event_assets(
            e, pkg_dir, output_dir, clip_video, duration, pad_sec, clip_max_sec, codec)
        item = _event_meta_item(e, img_name, clip_name, clip_ok)
        if not source_integrity['allow_clip']:
            item['clip_unavailable_reason'] = 'video_source_unverified'
        detail_name = _copy_detail_card(e, pkg_dir, output_dir)
        if detail_name:
            item['detail_image'] = detail_name
        item["scene"] = ev_scene.get(int(_num(e.get("event_id"), 0))) or e.get("scene") or ""
        # [AI-REVIEW] 已处置事件（框重定位/降级/低优先级）显式标注「未确认」，
        # 写进 metadata 与清单，避免证据包被误读成「确认违章」。
        try:
            from core.ai_review import review_disposition_note
            _ar_note = review_disposition_note(e)
        except Exception:
            _ar_note = ""
        if _ar_note:
            item["ai_review_note"] = _ar_note
        if clip_info.get("unavailable_reason"):
            item["clip_unavailable_reason"] = clip_info["unavailable_reason"]
        elif clip_info:
            item["clip_window"] = [clip_info["start"], clip_info["end"]]
            item["clip_truncated"] = clip_info["truncated"]
            if clip_info["truncated"]:
                truncated_events += 1
        meta["事件"].append(item)
    if source_integrity['status'] == 'digest_verified' and not _video_source_unchanged(vpath, source_integrity):
        # 审计期内源视频变化：仅移除本次独占证据包中新生成的片段；原视频和事件图不动。
        for item in meta['事件']:
            filename = item.get('clip') or ''
            if filename and item.get('clip_ok'):
                if os.path.basename(filename) != filename or not filename.startswith('event_'):
                    raise RuntimeError('unsafe_generated_clip_name')
                clip_path = os.path.join(pkg_dir, filename)
                if os.path.islink(clip_path):
                    raise RuntimeError('unsafe_generated_clip_path')
                try:
                    os.remove(clip_path)
                except OSError:
                    # 删除失败时不生成最终证据包，避免把不可信片段打进 zip。
                    raise RuntimeError('unverified_clip_cleanup_failed') from None
            item['clip'] = ''
            item['clip_ok'] = False
            item['clip_unavailable_reason'] = 'video_source_changed_during_export'
            item.pop('clip_window', None)
            item.pop('clip_truncated', None)
        source_integrity = {'status': 'source_changed_during_export', 'allow_clip': False,
                            'note': '视频来源在导出前后不一致，已停止提供本次片段'}
    meta['视频来源核验状态'] = source_integrity['status']
    meta['视频来源核验说明'] = source_integrity['note']
    lines.append('视频来源：' + source_integrity['note'])
    for item in meta['事件']:
        item['clip_source_status'] = source_integrity['status']
        _append_event_text(lines, item, report_incomplete=bool(report.get('gate_failures')))
    if unconfirmed:
        _unconfirmed_ids = "、".join(
            str(int(_num(e.get("event_id"), 0))) for e in unconfirmed)
        lines.append(f"未确认（不计入违章、未导出素材）：事件 {_unconfirmed_ids}")
        # 旧/异常报告可能仍有被移出主队列的人员或安全帽候选；素材不随包导出，
        # 但原模型描述、坐标和相互矛盾的复核线索须留档供人工核查。
        candidate_unconfirmed = [e for e in unconfirmed if _export_candidate_notes(e)]
        if candidate_unconfirmed:
            helmet_unconfirmed, person_unconfirmed, area_unconfirmed, incomplete_unconfirmed = [], [], [], []
            lines.append("以下线索不计入违章，不随包导出素材：")
            for e in candidate_unconfirmed:
                item = _event_meta_item(e, "", "", False)
                item["assets_withheld"] = True
                if helmet_evidence_note(e):
                    helmet_unconfirmed.append(item)
                if person_evidence_note(e):
                    person_unconfirmed.append(item)
                if area_evidence_note(e):
                    area_unconfirmed.append(item)
                if _incomplete_candidate(e):
                    incomplete_unconfirmed.append(item)
                _append_event_text(lines, item, report_incomplete=bool(report.get("gate_failures")))
            if helmet_unconfirmed:
                meta["未确认安全帽线索"] = helmet_unconfirmed
            if person_unconfirmed:
                meta["未确认人员身份线索"] = person_unconfirmed
            if area_unconfirmed:
                meta["未确认区域定位线索"] = area_unconfirmed
            if incomplete_unconfirmed:
                meta["未确认分析或门禁不完整线索"] = incomplete_unconfirmed
    # 原子写：导出中途失败/断电时不会留下半截 metadata/清单（与 zip 的
    # tmp+rename 策略一致；此前裸 open 写，中断会留下不完整的包目录）
    atomic_write_json(os.path.join(pkg_dir, "metadata.json"), meta)
    atomic_write_text(os.path.join(pkg_dir, "清单.txt"), "\n".join(lines))

    zip_path = _zip_dir(pkg_dir)

    if progress_cb:
        progress_cb(100, "证据包导出完成")
    try:
        size_mb = round(os.path.getsize(zip_path) / 1048576.0, 1)
    except OSError:
        size_mb = 0.0
    return {"dir": pkg_dir, "zip": zip_path, "count": total,
            "size_mb": size_mb, "codec": codec[2], "clip_max_sec": clip_max_sec,
            "truncated": truncated_events}


def _safe_name(name):
    for ch in '\\/:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip() or "视频违章分析"


def _build_scene_groups(report, events):
    """按报告合并后的场景分组，标签带场景序号与起止秒"""
    segs = report.get("scene_segments") or []
    if not segs:
        return []
    segs = sorted(segs, key=lambda x: float(x.get("start_time", 0)))
    buckets = {s.get("segment_id"): [] for s in segs}
    extra = []
    for e in events:
        ts = normalize_timestamp(e.get("representative_timestamp"))
        if ts is None:
            extra.append(e)
            continue
        sid = None
        for s in segs:
            if float(s.get("start_time", 0)) <= ts <= float(s.get("end_time", 0)):
                sid = s.get("segment_id")
                break
        if sid is None:
            near = min(segs, key=lambda x: min(abs(ts - float(x.get("start_time", 0))),
                                               abs(ts - float(x.get("end_time", 0)))))
            sid = near.get("segment_id")
        if sid in buckets:
            buckets[sid].append(e)
        else:
            extra.append(e)
    out = []
    order = 0
    for s in segs:
        items = buckets.get(s.get("segment_id")) or []
        if not items:
            continue
        order += 1
        name = (s.get("name") or "未命名场景").strip()
        label = f"第{order}场景：{name}（第{float(s.get('start_time', 0)):.0f}秒–第{float(s.get('end_time', 0)):.0f}秒）"
        out.append((label, sorted(items, key=lambda e: float(e.get("representative_timestamp") or 0))))
    if extra:
        out.append(("未归入场景", extra))
    return out


def generate_rectification_notice(output_dir, video_path=None, out_root=None):
    """生成安全违章通知书（.docx），返回文件路径"""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Pt

    report, video_info, vpath, events = _load_report_context(
        output_dir, video_path, "没有违章事件，无需生成安全违章通知书")
    source_integrity = _video_source_integrity(report, vpath)
    # 未确认事件（AI 复核移出主队列）只作线索留档，不计入统计/清单/场景分组。
    confirmed, unconfirmed = split_review_queue(events)
    if not confirmed:
        raise ValueError("报告中的事件均已由 AI 复核移出主队列（未确认），无需生成违章通知书")
    has_candidates = bool(report.get("gate_failures")) or any(_export_candidate_notes(e) for e in events)

    fname = video_info.get("filename") or os.path.basename(vpath or "") or os.path.basename(output_dir)
    doc = Document()
    # 默认中文字体
    style = doc.styles["Normal"]
    style.font.name = "宋体"
    style.font.size = Pt(11)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    heading = "安全隐患核查参考单（含未确认候选）" if has_candidates else "安 全 违 章 通 知 书"
    h = doc.add_heading(heading, level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in h.runs:
        run.font.size = Pt(20)
        run.font.name = "黑体"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")

    doc.add_paragraph("致：＿＿＿＿＿＿＿＿（施工单位）")
    doc.add_paragraph("工程名称：＿＿＿＿＿＿＿＿＿＿＿＿＿＿")
    _add_labeled_line(doc, "视频文件：", fname)
    _add_labeled_line(doc, "分析时间：", report.get("analysis_time", ""))
    _add_labeled_line(doc, "检查方式：", "无人机航拍视频 AI 自动分析")
    doc.add_paragraph("视频来源：" + source_integrity['note'])
    if has_candidates:
        doc.add_paragraph("本清单可能包含人员身份、头部状态、区域位置、来源级保守待核或门禁分析不完整等线索；这些线索不是已确认违章，"
                          "须现场人工核查，不能仅凭模型输出直接作为整改或处罚依据。")
    if report.get("source_video_candidate_completeness"):
        doc.add_paragraph("来源视频发生变化或来源未能核验，候选结果可能不完整，需重新分析并人工核查。"
                          "来源核验首个失败原因：" + str(report.get("source_video_first_failure_reason") or "unknown"))
    if report.get("gate_failures"):
        doc.add_paragraph("一个或多个门禁执行失败，相关候选仅供人工核查，结果不能视为完整验收。")

    # 自定义场景（人工圈选）：场景内相似违章合并为一条展示；无场景时行为同旧版
    groups = _build_scene_groups(report, confirmed) or _build_display_groups(confirmed, load_scenes(output_dir))
    display_items = [e for _label, items in groups for e in items]

    doc.add_heading("一、主队列候选统计" if has_candidates else "一、违章统计汇总", level=1)
    _add_summary_table(doc, display_items, candidates=has_candidates)

    list_label = "候选事件清单" if has_candidates else "违章事件清单"
    doc.add_heading("二、%s（按场景，共 %d 个场景 / %d 项）" %
                    (list_label, len(groups), len(display_items)), level=1)
    for label, items in groups:
        if label:
            doc.add_heading(label, level=2)
        elif len(groups) > 1:
            doc.add_heading("未归入场景", level=2)
        for e in items:
            _add_event_table(doc, e, output_dir, report_incomplete=bool(report.get("gate_failures")),
                             context_scene=label)

    docs = report.get("doc_checklist") or []
    if docs:
        doc.add_heading("三、资料核对清单（不参与图像判定，需现场/资料核对）", level=1)
        for i, it in enumerate(docs, 1):
            doc.add_paragraph(f"{i}. {it.get('item','')}　【{it.get('status','待核对')}】")
        doc.add_heading("四、核查与整改建议" if has_candidates else "四、整改要求", level=1)
    else:
        doc.add_heading("三、核查与整改建议" if has_candidates else "三、整改要求", level=1)
    if has_candidates:
        doc.add_paragraph("1. 先对候选线索进行现场人工核查；核查属实后按现场安全制度处置。")
        doc.add_paragraph("2. 如发现即时危险，应依据现场情况及时采取防护措施。")
        doc.add_paragraph("3. 整改完成后，请留存影像资料并由安全管理人员复核。")
    else:
        doc.add_paragraph("1. 请责任单位在收到本通知后立即组织整改；"
                          "涉及【重大】风险的应立即停工整改，消除隐患后方可复工。")
        doc.add_paragraph("2. 一般隐患原则上 3 个工作日内完成整改，并留存整改前后影像资料。")
        doc.add_paragraph("3. 整改完成后，请使用无人机拍摄同区域视频重新分析，"
                          "复核通过后予以销号。")

    doc.add_heading("五、说明" if docs else "四、说明", level=1)
    doc.add_paragraph("本通知书由系统根据分析结果自动生成，供安全管理人员参考，"
                      "最终判定以现场核查为准。")
    # 旧版报告正文统计未含最新 AI 复核处置：本通知书按主队列重算，必须显式
    # 告知差异，否则同一视频的旧 report.txt 与本通知书数字冲突且无解释。
    if _stats_stale(report, confirmed, unconfirmed):
        doc.add_paragraph(
            "注：本报告由旧版本生成，报告正文统计可能未包含最新 AI 复核处置；"
            "本通知书已按主队列重算，建议重新分析该视频以获得一致文本。")

    # [AI-REVIEW] 未确认事件（AI 复核移出主队列）：不重编号已有章节，附在落款前
    # 作线索留档。开关关闭 / 旧版报告无 review_disposition → unconfirmed 为空，
    # 本节不出现，输出与改前一致。
    if unconfirmed:
        doc.add_heading("附：AI 复核移出主队列事件（未确认，不计入违章）", level=1)
        doc.add_paragraph("以下事件未确认为违章，仅作线索留档，不计入本通知书统计。")
        for e in unconfirmed:
            e = e or {}
            if _export_candidate_notes(e):
                _add_event_table(doc, e, output_dir, include_image=False,
                                 report_incomplete=bool(report.get("gate_failures")))
                continue
            doc.add_paragraph(
                f"事件{e.get('event_id', '')}｜{str(e.get('title') or '')}"
                f"｜{str(e.get('review_disposition') or '')}"
                f"｜{str(e.get('ai_review_reason') or '')[:60]}")

    # 落款：安全巡视单位（盖章）+ 时间
    doc.add_paragraph("")
    doc.add_paragraph("")
    _add_right_line(doc, "安全巡视单位（盖章）：＿＿＿＿＿＿＿＿＿＿")
    _add_right_line(doc, "时间：＿＿＿＿年＿＿月＿＿日")

    file_label = "安全隐患核查参考单" if has_candidates else "安全违章通知书"
    stem = f"{file_label}_{_safe_name(os.path.splitext(fname)[0])}"
    destination = os.path.abspath(out_root or output_dir)
    # 先在目标目录完整写出独占临时文件，再无覆盖地发布；读者不能看到半个 DOCX。
    # Windows 的 os.rename 不覆盖现存目标；POSIX 使用原子 hard link 保证同一语义。
    import tempfile
    fd, temporary = tempfile.mkstemp(prefix=".notice_", suffix=".tmp", dir=destination)
    try:
        with os.fdopen(fd, "w+b") as stream:
            doc.save(stream)
            stream.flush()
            os.fsync(stream.fileno())
        for index in range(10000):
            suffix = "" if index == 0 else f"_{index:03d}"
            out_path = os.path.join(destination, f"{stem}{suffix}.docx")
            if os.path.lexists(out_path):
                continue
            try:
                if os.name == "nt":
                    os.rename(temporary, out_path)
                    temporary = None
                else:
                    os.link(temporary, out_path)
            except FileExistsError:
                continue
            except PermissionError:
                if os.path.lexists(out_path):
                    continue
                raise
            return out_path
        raise FileExistsError("unable to allocate a unique notice document")
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _add_labeled_line(doc, label, value):
    """加一行『标签：值』，标签加粗。"""
    p = doc.add_paragraph()
    p.add_run(label).bold = True
    p.add_run(str(value))
    return p


def _add_right_line(doc, text):
    """加一行右对齐文本（落款用）。"""
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.add_run(text)
    return p


def _severity_rank(sev):
    """等级排序权重：越严重越小；未收录的等级统一排到末尾。"""
    return _SEVERITY_RANK.get(sev, len(_SEVERITY_RANK))


def _add_summary_table(doc, confirmed, candidates=False):
    """违章统计汇总表：按违章等级统计项数与占比，表末给出合计。

    等级按「重大→较大→一般→轻微」排列，未收录的等级排在末尾并按名称稳定
    排序，保证同一份报告重复导出结果一致。
    """
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm

    counts = {}
    for e in confirmed:
        sev = str((e or {}).get("severity") or "").strip() or "一般隐患"
        counts[sev] = counts.get(sev, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (_severity_rank(kv[0]), kv[0]))
    total = len(confirmed)

    table = doc.add_table(rows=len(ordered) + 2, cols=3)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for row in table.rows:
        for cell, width in zip(row.cells, _SUMMARY_COL_CM):
            cell.width = Cm(width)

    for cell, text in zip(table.rows[0].cells, ("候选等级" if candidates else "违章等级", "数量（项）", "占比")):
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.paragraphs[0].add_run(text).bold = True
    for r, (sev, n) in enumerate(ordered, start=1):
        cells = table.rows[r].cells
        cells[0].paragraphs[0].add_run(sev)
        for cell, text in zip(cells[1:], (str(n), f"{n / total * 100:.1f}%")):
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cell.paragraphs[0].add_run(text)
    cells = table.rows[-1].cells
    cells[0].paragraphs[0].add_run("合计").bold = True
    for cell, text in zip(cells[1:], (str(total), "100.0%")):
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.paragraphs[0].add_run(text).bold = True
    return table


def _docx_safe_text(value):
    """保留普通原文，但不让模型输出中的非法 XML 字符中断整个通知书。"""
    raw = str(value).encode("utf-8", "replace").decode("utf-8")
    return "".join(c if c in "\t\r\n" or 0x20 <= ord(c) <= 0xD7FF
                   or 0xE000 <= ord(c) <= 0xFFFD or 0x10000 <= ord(c) <= 0x10FFFF
                   else " " for c in raw)


def _add_event_table(doc, e, output_dir, include_image=True, report_incomplete=False, context_scene=None):
    """把一个违章事件渲染成两列表格（左标签/右内容），样式与《安全违章通知书》样例一致。"""
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm

    e = e or {}
    eid = int(_num(e.get("event_id"), 0))
    evidence_notes = _export_candidate_notes(e, report_incomplete=report_incomplete)
    shown_title = _export_candidate_label(e, evidence_notes)
    rows = [
        ("候选事件序号" if evidence_notes else "违章事件序号", f"{eid:02d}"),
        ("候选等级" if evidence_notes else "违章等级", e.get("severity") or "一般隐患"),
        ("候选内容" if evidence_notes else "违章内容", shown_title),
    ]
    if evidence_notes:
        raw_title = candidate_original_title(e)
        rows.append(("原模型标题（待核）", raw_title))
        rows.extend(evidence_notes)
        rows.append(("复核处置（未确认）", detail_review_disposition(e)["disposition_note"]))
        rows.extend((label, value) for _, label, value in candidate_review_diagnostics(e))
        box_note = detail_box_relocalization(e)["box_relocalization_note"]
        if box_note:
            rows.append(("框状态", box_note))
    # 场景内相似合并后的条目：注明合并处数与各次出现时刻
    n_merged = int(_num(e.get("merged_count"), 1) or 1)
    if n_merged > 1:
        times = "、".join(e.get("merged_times") or [])
        rows.append(("合并说明",
                     f"该场景内同类违章共 {n_merged} 处（出现于 {times}），已合并为一条"))
    source = detail_source_metadata(e)
    context_event = dict(e)
    if not context_event.get('scene_name') and isinstance(context_scene, str) and context_scene.strip():
        context_event['scene'] = context_scene
    rows += [
        ("视频位置", _fmt_span(e)),
        ("代表帧记录（待核）", detail_representative_reference(e)["text"]),
        ("位置坐标信息", _fmt_bbox(e.get("bbox"))),
        (source['description_label'], e.get("description") or "未提供"),
        (source['position_label'], e.get("target_location") or "未提供"),
        (source['evidence_label'], e.get("evidence") or "未提供"),
        ("内容来源说明", source['source_note']),
        *detail_contextual_advice(context_event),
        ("法规依据（待核）", _fmt_basis(e)),
        ("候选证据视频" if evidence_notes else "违章视频", ""),
    ]
    # [AI-REVIEW] 已处置事件显式标注「未确认」；insert(-1) 保证「违章视频」
    # 仍是末行（后续 table.rows[-1] 靠它插图），不改变表格几何。
    try:
        from core.ai_review import review_disposition_note
        _ar_note = review_disposition_note(e)
    except Exception:
        _ar_note = ""
    if _ar_note:
        rows.insert(-1, ("复核状态（未确认）",
                         f"{_ar_note}；不得作为已确认违章"))
    if e.get('analysis_incomplete') is True:
        rows.insert(-1, ("分析状态", "不完整，候选仅供人工核查"))
    table = doc.add_table(rows=len(rows), cols=2)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for i, (label, value) in enumerate(rows):
        label_cell, value_cell = table.rows[i].cells
        label_cell.width = Cm(_LABEL_COL_CM)
        value_cell.width = Cm(_VALUE_COL_CM)
        label_cell.paragraphs[0].add_run(label).bold = True
        if value:
            value_cell.paragraphs[0].add_run(_docx_safe_text(value))

    img_data, image_integrity = _event_image_bytes(e, output_dir)
    if include_image and img_data is not None:
        cell = table.rows[-1].cells[1]
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.paragraphs[0].add_run().add_picture(io.BytesIO(img_data), width=Cm(_EVENT_IMG_CM))
        if image_integrity == 'legacy_unverified':
            cell.add_paragraph('历史图未记录可核验摘要，仅供参考。')
