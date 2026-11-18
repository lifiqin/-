# -*- coding: utf-8 -*-
"""原帧人工框修订侧录：不修改 report、系统框、证据图或冻结金标。"""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

import cv2
import numpy as np

from core.area_localization import AREA_TITLES, area_confirmation_guard, frame_pixel_sha256
from core.annotation_renderer import draw_label_with_bg
from core.event_image_sync import _find_exact_frame
from core.frame_lookup import normalize_timestamp
from core.helmet_evidence import _event_box


class AreaRevisionError(ValueError):
    """证据未能可靠对应；不得猜测帧、裁切或静默写入。"""


@dataclass(frozen=True)
class ManualAreaContext:
    output_dir: str
    report_sha256: str
    frame_path: str
    frame_relative_path: str
    frame_sha256: str
    event_id: int
    title: str
    frame_index: int
    timestamp: float
    width: int
    height: int
    original_bbox: dict | None
    context_search_window: dict | None
    context_visual_bbox: dict | None
    frame_pixel_source_status: str = 'unverified'
    frame_pixel_source_note: str = '原帧像素来源未经独立核验，帧号和时间仅依据报告元数据'
    reference_frame_pixels_sha256: str | None = None
    reported_original_frame_sha256: str | None = None
    frame_bytes_source_status: str = 'unverified_missing_sha256'


def _within(path, root):
    try:
        path = os.path.normcase(os.path.realpath(path))
        root = os.path.normcase(os.path.realpath(root))
        return os.path.commonpath((path, root)) == root
    except (OSError, TypeError, ValueError):
        return False


def _read_regular_file(path):
    path = os.fspath(path)
    if os.path.islink(path) or not os.path.isfile(path):
        raise AreaRevisionError('证据文件不存在或为链接')
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise AreaRevisionError('证据文件不可读') from exc


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _original_bbox(value):
    box = _event_box(value)
    if box is None:
        return None
    return {'x': box[0], 'y': box[1], 'w': box[2] - box[0], 'h': box[3] - box[1]}


def prepare_manual_area_revision(output_dir, event_id):
    """唯一精确匹配代表帧号和时间，返回只读上下文与原图像素。"""
    if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id < 0:
        raise AreaRevisionError('事件编号不合法')
    root = os.path.abspath(os.fspath(output_dir))
    if os.path.islink(root) or not os.path.isdir(root):
        raise AreaRevisionError('报告目录不存在或为链接')
    report_path = os.path.join(root, 'report.json')
    report_bytes = _read_regular_file(report_path)
    try:
        report = json.loads(report_bytes.decode('utf-8-sig'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AreaRevisionError('报告不是有效的 UTF-8 JSON') from exc
    if not isinstance(report, dict) or not isinstance(report.get('events'), list):
        raise AreaRevisionError('报告没有事件列表')
    events = [event for event in report['events']
              if isinstance(event, dict) and type(event.get('event_id')) is int
              and event['event_id'] == event_id]
    if len(events) != 1:
        raise AreaRevisionError('事件编号缺失或不唯一')
    event = events[0]
    title = event.get('title')
    if not isinstance(title, str) or title not in AREA_TITLES:
        raise AreaRevisionError('此事件不是支持人工定位的区域类别')
    index, timestamp = event.get('representative_frame'), normalize_timestamp(event.get('representative_timestamp'))
    if isinstance(index, bool) or not isinstance(index, int) or index < 0 or timestamp is None:
        raise AreaRevisionError('缺少精确的代表帧号或时间')
    results = report.get('frame_results')
    if not isinstance(results, list):
        raise AreaRevisionError('报告缺少帧映射')
    # 同一个原始帧号即使被配上不同时间，也不能任选其一冒充唯一来源。
    same_index = [item for item in results if isinstance(item, dict)
                  and item.get('frame_index') == index]
    if len(same_index) != 1 or type(same_index[0].get('frame_index')) is not int:
        raise AreaRevisionError('代表帧号不唯一或不合法')
    matched = _find_exact_frame({'representative_frame': index, 'representative_timestamp': timestamp},
                                same_index)
    if matched is None:
        raise AreaRevisionError('代表帧号和时间不能唯一匹配')
    if matched.get('annotation_frame_status') == 'unverified_frame_source':
        raise AreaRevisionError('代表帧来源已标记为无法核验')
    original_dir = os.path.join(root, 'original')
    if os.path.islink(original_dir) or not os.path.isdir(original_dir) or not _within(original_dir, root):
        raise AreaRevisionError('原帧目录不安全或不存在')
    source = matched.get('original_path')
    if isinstance(source, str) and source:
        source = source if os.path.isabs(source) else os.path.join(root, source)
    else:
        annotated = matched.get('annotated_path')
        if not isinstance(annotated, str) or not annotated:
            raise AreaRevisionError('代表帧没有原图映射')
        source = os.path.join(original_dir, os.path.basename(annotated))
    source = os.path.abspath(source)
    if not _within(source, original_dir):
        raise AreaRevisionError('代表帧不在本次报告的原图目录')
    data = _read_regular_file(source)
    frame_digest = _digest(data)
    reported_digest = None
    bytes_status = 'unverified_missing_sha256'
    if 'original_frame_sha256' in matched:
        expected_digest = matched['original_frame_sha256']
        if (not isinstance(expected_digest, str)
                or not re.fullmatch(r'[0-9a-fA-F]{64}', expected_digest)):
            raise AreaRevisionError('报告帧记录的原图文件摘要无效')
        if expected_digest.lower() != frame_digest:
            raise AreaRevisionError('原图文件摘要与报告帧记录不一致，不能在该图上纠框')
        reported_digest = expected_digest.lower()
        bytes_status = 'matched_reported_sha256'
    try:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise AreaRevisionError('代表帧无法解码') from exc
    if image is None or image.ndim != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise AreaRevisionError('代表帧为空或无法解码')
    height, width = image.shape[:2]
    if height * width > 100000000:
        raise AreaRevisionError('原图尺寸过大')
    search_box = visual_box = None
    pixel_status = 'unverified'
    pixel_note = '报告未记录原图文件摘要；原帧像素来源未经独立核验，请对照原视频'
    if reported_digest:
        pixel_status = 'matched_recorded_frame_bytes'
        pixel_note = '原图文件摘要与报告帧记录一致；仅是记录一致性，不代表独立验证，请对照原视频'
    reference_hash = None
    evidence = event.get('area_localization')
    if isinstance(evidence, dict) and evidence.get('evidence_scope') == 'representative_frame':
        source_time = normalize_timestamp(evidence.get('source_timestamp'))
        if (type(evidence.get('source_frame_index')) is not int
                or evidence['source_frame_index'] != index or source_time is None
                or not math.isclose(source_time, timestamp, abs_tol=1e-6, rel_tol=0)):
            raise AreaRevisionError('同帧区域证据的帧号或时间与代表帧不一致')
        reference_hash = evidence.get('reference_frame_pixels_sha256')
        if reference_hash is not None:
            if not isinstance(reference_hash, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', reference_hash):
                raise AreaRevisionError('同帧区域证据的像素摘要无效')
            if reference_hash.lower() != frame_pixel_sha256(image):
                raise AreaRevisionError('原图像素与同帧区域证据不一致，不能在该图上纠框')
            reference_hash = reference_hash.lower()
            pixel_status = 'matched_recorded_pixels'
            pixel_note = '与报告记录的同帧像素摘要一致；仅是记录一致性，不代表独立验证'
            if not reported_digest:
                pixel_note += '；报告未记录原图文件摘要，来源未经独立核验'
        if reference_hash and area_confirmation_guard(event):
            try:
                search_box = validate_manual_box(evidence.get('search_window'), int(width), int(height))
            except AreaRevisionError:
                pass
            visual = evidence.get('visual_context_candidate')
            if (isinstance(visual, dict) and visual.get('kind') == 'unverified_dark_region_context'
                    and visual.get('not_violation_box') is True
                    and visual.get('independent_verification') is False):
                try:
                    visual_box = validate_manual_box(visual.get('bbox'), int(width), int(height))
                except AreaRevisionError:
                    pass
    context = ManualAreaContext(
        output_dir=root, report_sha256=_digest(report_bytes), frame_path=source,
        frame_relative_path=Path(os.path.relpath(source, root)).as_posix(), frame_sha256=frame_digest,
        event_id=event_id, title=title, frame_index=index, timestamp=timestamp,
        width=int(width), height=int(height), original_bbox=_original_bbox(event.get('bbox')),
        context_search_window=search_box, context_visual_bbox=visual_box,
        frame_pixel_source_status=pixel_status, frame_pixel_source_note=pixel_note,
        reference_frame_pixels_sha256=reference_hash,
        reported_original_frame_sha256=reported_digest, frame_bytes_source_status=bytes_status)
    return context, image


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (OverflowError, TypeError, ValueError):
        return None


def validate_manual_box(box, width, height):
    """只接受原图像素内的有效矩形；这里没有修改降级几何护栏。"""
    if not isinstance(box, dict):
        raise AreaRevisionError('人工框不是有效的矩形')
    vals = [_finite(box.get(k)) for k in ('x', 'y', 'w', 'h')]
    if any(v is None for v in vals):
        raise AreaRevisionError('人工框坐标不合法')
    x, y, w, h = vals
    if (w < 1 or h < 1 or x < 0 or y < 0 or x + w > width or y + h > height
            or not all(math.isfinite(v) for v in (x + w, y + h))):
        raise AreaRevisionError('人工框超出原图或尺寸无效')
    return {'x': x, 'y': y, 'w': w, 'h': h}


def viewport_to_image(x, y, view_width, view_height, image_width, image_height, *, clamp=False):
    """Qt 等比居中画布坐标转原图，显式处理上下/左右留黑与 HiDPI 逻辑坐标。"""
    values = [_finite(v) for v in (x, y, view_width, view_height, image_width, image_height)]
    if any(v is None for v in values):
        return None
    x, y, vw, vh, iw, ih = values
    if min(vw, vh, iw, ih) <= 0:
        return None
    scale = min(vw / iw, vh / ih)
    ox, oy = (vw - iw * scale) / 2.0, (vh - ih * scale) / 2.0
    ix, iy = (x - ox) / scale, (y - oy) / scale
    if not clamp and not (0 <= ix <= iw and 0 <= iy <= ih):
        return None
    return max(0.0, min(iw, ix)), max(0.0, min(ih, iy))


def box_from_image_points(start, end, width, height):
    if not isinstance(start, (tuple, list)) or not isinstance(end, (tuple, list)) or len(start) != 2 or len(end) != 2:
        raise AreaRevisionError('人工框起止点不合法')
    vals = [_finite(v) for v in (*start, *end)]
    if any(v is None for v in vals):
        raise AreaRevisionError('人工框起止点不合法')
    x1, x2 = sorted((vals[0], vals[2]))
    y1, y2 = sorted((vals[1], vals[3]))
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise AreaRevisionError('人工框超出原图')
    left, top, right, bottom = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
    return validate_manual_box({'x': left, 'y': top, 'w': right - left, 'h': bottom - top}, width, height)


def _revision_dir(context, create):
    root = context.output_dir
    folder = os.path.join(root, 'manual_area_revisions')
    if os.path.islink(folder):
        raise AreaRevisionError('修订目录不能是链接')
    if create:
        os.makedirs(folder, exist_ok=True)
    if os.path.exists(folder) and (not os.path.isdir(folder) or not _within(folder, root)):
        raise AreaRevisionError('修订目录不安全')
    return folder


def save_manual_area_revision(context, box, note='', display_info=None):
    """每次新建旁路记录；不覆盖 report 或旧修订，不自动改变任何事件。"""
    if not isinstance(context, ManualAreaContext):
        raise AreaRevisionError('缺少可信的原帧上下文')
    fresh, image = prepare_manual_area_revision(context.output_dir, context.event_id)
    if (fresh.report_sha256 != context.report_sha256 or fresh.frame_sha256 != context.frame_sha256
            or fresh.frame_path != context.frame_path or fresh.frame_index != context.frame_index
            or fresh.timestamp != context.timestamp):
        raise AreaRevisionError('报告或原帧已变化，请重新加载后画框')
    box = validate_manual_box(box, fresh.width, fresh.height)
    if not isinstance(note, str) or len(note) > 1000 or '\x00' in note:
        raise AreaRevisionError('备注不合法或超过 1000 字')
    if display_info is not None:
        if (not isinstance(display_info, dict) or any(
                type(display_info.get(key)) is not bool for key in (
                    'original_model_box_shown', 'visual_context_shown', 'previous_manual_candidate_shown'))):
            raise AreaRevisionError('画框时的参考来源标记不合法')
        if display_info['visual_context_shown'] and not (fresh.context_search_window or fresh.context_visual_bbox):
            raise AreaRevisionError('没有同帧可核验的背景上下文')
    directory = _revision_dir(fresh, create=True)
    if os.path.islink(directory) or not _within(directory, fresh.output_dir):
        raise AreaRevisionError('修订目录不安全')
    unique = f'event_{fresh.event_id}_{uuid.uuid4().hex}'
    preview_path = os.path.join(directory, unique + '.png')
    preview = image.copy()
    left, top = int(math.floor(box['x'])), int(math.floor(box['y']))
    right = min(fresh.width - 1, int(math.ceil(box['x'] + box['w'])) - 1)
    bottom = min(fresh.height - 1, int(math.ceil(box['y'] + box['h'])) - 1)
    cv2.rectangle(preview, (left, top), (right, bottom), (0, 220, 0), 3)
    label_y = top - 30 if top >= 30 else min(max(0, fresh.height - 30), bottom + 3)
    label_x = min(max(0, left), max(0, fresh.width - 225))
    preview = draw_label_with_bg(preview, '人工候选框（未确认）', (label_x, label_y),
                                 font_size=18, text_color=(255, 255, 255), bg_color=(0, 100, 0))
    ok, encoded = cv2.imencode('.png', preview)
    if not ok:
        raise AreaRevisionError('无法生成独立人工候选预览')
    preview_bytes = encoded.tobytes()
    record = {
        'schema': 'manual_area_revision/v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'provenance': 'human_drawn_candidate', 'is_automated_prediction': False,
        'is_verified_ground_truth': False, 'updates_report_or_gold': False,
        'event_id': fresh.event_id, 'title': fresh.title, 'representative_frame': fresh.frame_index,
        'representative_timestamp': fresh.timestamp, 'report_sha256': fresh.report_sha256,
        'original_frame_relative_path': fresh.frame_relative_path, 'original_frame_sha256': fresh.frame_sha256,
        'image_width': fresh.width, 'image_height': fresh.height,
        'report_event_bbox': fresh.original_bbox,
        'original_model_bbox': fresh.original_bbox,
        'original_model_bbox_note': '兼容旧版字段；其值是原报告事件框，可能已经自动校准，并非模型原始框',
        'display_info_field_notes': {'original_model_box_shown': '兼容旧版字段，表示显示原报告框，可能已自动校准'},
        'frame_pixel_source_status': fresh.frame_pixel_source_status,
        'frame_pixel_source_note': fresh.frame_pixel_source_note,
        'reported_original_frame_sha256': fresh.reported_original_frame_sha256,
        'frame_bytes_source_status': fresh.frame_bytes_source_status,
        'source_independent_verification': False,
        'reference_frame_pixels_sha256': fresh.reference_frame_pixels_sha256,
        'manual_bbox': box, 'note': note,
        'display_info': dict(display_info) if display_info is not None else {'status': 'not_recorded'},
        'preview_kind': 'manual_candidate_visualization_not_ground_truth',
        'preview_relative_path': Path(os.path.relpath(preview_path, fresh.output_dir)).as_posix(),
        'preview_sha256': _digest(preview_bytes),
    }
    target = os.path.join(directory, unique + '.json')
    data = (json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    try:
        with open(preview_path, 'xb') as handle:
            handle.write(preview_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        with open(target, 'xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise AreaRevisionError('无法保存人工修订') from exc
    return target


def list_manual_area_revisions(context):
    """只列相同报告哈希、原图哈希和事件的记录；异常记录计数但不冒充有效。"""
    if not isinstance(context, ManualAreaContext):
        raise AreaRevisionError('缺少可信的原帧上下文')
    folder = _revision_dir(context, create=False)
    if not os.path.isdir(folder):
        return {'records': [], 'invalid_count': 0}
    entries, invalid = [], 0
    for path in sorted(Path(folder).glob(f'event_{context.event_id}_*.json')):
        try:
            if path.is_symlink() or path.stat().st_size > 65536:
                raise ValueError('unsafe file')
            row = json.loads(path.read_text(encoding='utf-8'))
            if (not isinstance(row, dict) or row.get('schema') != 'manual_area_revision/v1'
                    or row.get('report_sha256') != context.report_sha256
                    or row.get('original_frame_sha256') != context.frame_sha256
                    or row.get('event_id') != context.event_id
                    or row.get('representative_frame') != context.frame_index
                    or row.get('representative_timestamp') != context.timestamp):
                raise ValueError('mismatched revision')
            validate_manual_box(row.get('manual_bbox'), context.width, context.height)
            preview_relative = row.get('preview_relative_path')
            if (row.get('preview_kind') != 'manual_candidate_visualization_not_ground_truth'
                    or not isinstance(preview_relative, str) or not preview_relative):
                raise ValueError('missing preview')
            preview = os.path.join(context.output_dir, preview_relative)
            if not _within(preview, folder) or _digest(_read_regular_file(preview)) != row.get('preview_sha256'):
                raise ValueError('preview missing or mismatched')
            entries.append({'path': str(path), 'record': row})
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, AreaRevisionError):
            invalid += 1
    entries.sort(key=lambda entry: (str(entry['record'].get('created_at_utc') or ''), entry['path']))
    return {'records': entries, 'invalid_count': invalid}
