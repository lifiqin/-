# -*- coding: utf-8 -*-
"""自定义场景：人工把关键帧圈选为一个作业场景，报告导出时场景内相似违章合并

三件事：
1. 场景存取 —— custom_scenes.json 与 report.json 同目录（原子写）。独立成文件
   而不是塞进 report.json：重新分析会重写 report.json，人工圈选的场景不该被冲掉；
   导出链路（通知书/证据包）随时可读，程序重启后依然生效。
2. 事件归属 —— 事件的 frame_indices 与场景圈选帧有交集即归属该场景。
   按帧交集而不是存 event_id：事件随报告重算（缓存/校准）后 id 会变，
   帧序号与帧列表所见一致，归属关系始终成立。
3. 相似合并 —— 场景内标准名（title）相同的事件合并为一条场景级违章，
   汇总处数/帧数/时间跨度，等级取组内最重，代表帧与判定依据取最早出现的那处。

AI 命名（ai_name_scene）：取圈选帧等间隔最多 3 张图，走推理池让视觉模型起一个
简短场景名；任何失败抛异常，由调用方降级为「自定义场景N」兜底名。
"""
import os
import re

from core.bbox_utils import as_number as _num
from core.atomic_io import atomic_write_json

SCENES_FILENAME = "custom_scenes.json"

# 命名送图的限制（与场景判别同口径：命名不需要细节，小图请求明显更快）
NAME_MAX_IMAGE_SIDE = 768
NAME_MAX_IMAGES = 3

# 场景名上限：提示词要求 ≤12 字，放宽到 16 字兜底截断
NAME_MAX_CHARS = 16


# ==================== 场景存取 ====================

def _scenes_path(output_dir):
    return os.path.join(output_dir, SCENES_FILENAME)


def load_scenes(output_dir):
    """读入自定义场景列表；文件不存在/损坏返回 []（导出链路零负担）"""
    path = _scenes_path(output_dir)
    if not os.path.exists(path):
        return []
    import json
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scenes = (data or {}).get("scenes") or []
        return [s for s in scenes if isinstance(s, dict)]
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        print(f"[scene] {SCENES_FILENAME} 无法解析（{type(e).__name__}: {e}），按无场景处理")
        return []


def save_scenes(output_dir, scenes):
    atomic_write_json(_scenes_path(output_dir), {
        "version": 1,
        "scenes": list(scenes or []),
    })


def add_scene(output_dir, scene):
    """追加一个场景（scene_id 自动补齐唯一），返回更新后的列表"""
    scenes = load_scenes(output_dir)
    ids = {str(s.get("scene_id") or "") for s in scenes}
    n = 1
    while f"scene_{n:02d}" in ids:
        n += 1
    scene = dict(scene or {})
    scene.setdefault("scene_id", f"scene_{n:02d}")
    scenes.append(scene)
    save_scenes(output_dir, scenes)
    return scenes


def remove_scene(output_dir, scene_id):
    """删除指定场景，返回更新后的列表"""
    scenes = [s for s in load_scenes(output_dir)
              if str(s.get("scene_id") or "") != str(scene_id)]
    save_scenes(output_dir, scenes)
    return scenes


def fmt_ts(sec):
    """秒 → mm:ss；入参可能是 None/字符串/NaN（JSON 净化后可能为 null）"""
    sec = int(_num(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def scene_span_text(scene):
    """场景时间范围文本：mm:ss — mm:ss；缺参/非数值/NaN 一律返回空串"""
    span = (scene or {}).get("time_span") or []
    if len(span) != 2:
        return ""
    import math
    try:
        a, b = float(span[0]), float(span[1])
    except (TypeError, ValueError):
        return ""
    if not (math.isfinite(a) and math.isfinite(b)):
        return ""
    return f"{fmt_ts(a)} — {fmt_ts(b)}"


def fallback_scene_name(existing_count):
    """AI 命名失败时的兜底名"""
    return f"自定义场景{existing_count + 1}"


# ==================== 事件归属与相似合并 ====================

def events_in_scene(events, scene):
    """事件的帧序号与场景圈选帧有交集即归属该场景"""
    scene_frames = set()
    for f in (scene or {}).get("frame_indices") or []:
        try:
            scene_frames.add(int(_num(f)))
        except (TypeError, ValueError):
            continue
    if not scene_frames:
        return []
    out = []
    for e in events or []:
        ev_frames = set()
        for f in (e or {}).get("frame_indices") or []:
            try:
                ev_frames.add(int(_num(f)))
            except (TypeError, ValueError):
                continue
        if ev_frames & scene_frames:
            out.append(e)
    return out


def _heaviest_severity(sevs):
    """组内最重等级（_SEVERITY_RANK 越小越重；懒加载避免与 evidence_exporter 循环导入）"""
    from core.evidence_exporter import _SEVERITY_RANK
    best, best_rank = "", None
    for sev in sevs:
        sev = str(sev or "").strip() or "一般隐患"
        rank = _SEVERITY_RANK.get(sev, len(_SEVERITY_RANK))
        if best_rank is None or rank < best_rank:
            best, best_rank = sev, rank
    return best or "一般隐患"


def merge_similar_events(events):
    """场景内相似违章合并：标准名（title）相同 → 一条。

    合并语义（对齐 _add_event_table/_fmt_span 的字段契约，保证通知书直接可渲染）：
    - 首末时刻取组内最早/最晚，frame_count 求和、frame_indices 并集
    - 等级取组内最重
    - 代表帧/坐标/判定依据沿用组内首条（最早出现的那处）——事件图片按
      event_id 取图，merged 条目沿用首条 id 即可复用现成截图
    - merged_count/merged_times 记录合并明细，报告里展示「场景内同类 N 处」
    返回按首末时刻排序的合并条目列表（单条事件原样保留）。
    """
    groups, order = {}, []
    for e in events or []:
        e = e or {}
        t = str(e.get("title") or "")
        if not t:
            t = "未命名违章"
        if t not in groups:
            groups[t] = []
            order.append(t)
        groups[t].append(e)

    merged = []
    for t in order:
        grp = sorted(groups[t], key=lambda e: _num((e or {}).get("first_seen")))
        head = dict(grp[0] or {})
        if len(grp) == 1:
            merged.append(head)
            continue
        frames = set()
        for e in grp:
            for f in e.get("frame_indices") or []:
                try:
                    frames.add(int(_num(f)))
                except (TypeError, ValueError):
                    continue
        first = min(_num(e.get("first_seen")) for e in grp)
        last = max(_num(e.get("last_seen")) for e in grp)
        head["first_seen"] = first
        head["last_seen"] = last
        head["duration_sec"] = max(0.0, last - first)
        head["frame_count"] = sum(int(_num(e.get("frame_count"), 0)) for e in grp)
        head["frame_indices"] = sorted(frames)
        head["severity"] = _heaviest_severity(e.get("severity") for e in grp)
        head["merged_count"] = len(grp)
        head["merged_times"] = [fmt_ts(e.get("first_seen")) for e in grp]
        head["merged_event_ids"] = [int(_num(e.get("event_id"), 0)) for e in grp]
        desc = str(head.get("description") or "").strip()
        head["description"] = (
            f"该场景内共发现 {len(grp)} 处同类违章"
            f"（{fmt_ts(first)}—{fmt_ts(last)} 期间多次出现，已合并展示）。"
            + (f"典型情形：{desc}" if desc else "")
        )
        merged.append(head)

    merged.sort(key=lambda e: _num(e.get("first_seen")))
    return merged


# ==================== AI 场景命名 ====================

def clean_scene_name(text):
    """清洗模型返回的场景名：剥引号/前缀/换行，限长。无效返回空串"""
    text = str(text or "").strip()
    if not text:
        return ""
    text = text.splitlines()[0].strip()
    text = text.strip().strip("\"'“”‘’「」『』.。！!，,、 ").strip()
    text = re.sub(r"^(场景名|名称|场景|自定义场景)[:：]\s*", "", text).strip()
    text = re.sub(r"\s+", "", text)
    return text[:NAME_MAX_CHARS]


def ai_name_scene(ai_client, image_paths):
    """给一组帧画面取场景名；成功返回名称字符串，失败抛异常（调用方降级兜底）

    走 chat_completion（纯文本返回，内部已带推理池轮询 + 关思考 + 截断重试），
    命名是小任务：温度 0、额度 256，disable_thinking 下秒级返回。
    """
    import base64

    import cv2
    import numpy as np

    from core.prompts import SCENE_NAME_PROMPT

    paths = [p for p in (image_paths or []) if p and os.path.exists(p)]
    if not paths:
        raise ValueError("场景内没有可用的帧画面")
    if len(paths) > NAME_MAX_IMAGES:
        # 等间隔采样首/中/尾，覆盖场景全貌
        idx = sorted({round(i * (len(paths) - 1) / (NAME_MAX_IMAGES - 1))
                      for i in range(NAME_MAX_IMAGES)})
        paths = [paths[i] for i in idx]

    content = [{"type": "text", "text": SCENE_NAME_PROMPT}]
    for p in paths:
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        side = max(h, w)
        if side > NAME_MAX_IMAGE_SIDE:
            scale = NAME_MAX_IMAGE_SIDE / float(side)
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    if len(content) == 1:
        raise ValueError("帧画面全部读取失败")

    messages = [{"role": "user", "content": content}]
    resp_text = ai_client.chat_completion(messages, temperature=0.0, max_tokens=256)
    name = clean_scene_name(resp_text)
    if not name:
        raise ValueError("模型未返回有效场景名")
    return name
