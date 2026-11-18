# -*- coding: utf-8 -*-
"""VLM 两阶段放大 + 多采样共识：精修区域类事件的框（会调用 API）

依据：
- R-VLM（ACL2025）：粗框 → 按目标尺寸裁剪放大 → 二次预测 → 映射回原图，
  training-free 显著提升小目标 IoU；
- AVA-VLM（施工场景）：对远处作业人员/PPE 先全局判读再做高分辨率裁剪复审；
- ACS / ViGoR：多次采样候选框，按一致性选优（best-of-N）。

实测（29 条审计判错事件）：
  原框 8/29 通过但 0 条紧框；
  放大+双采样共识 7/19 通过、其中 5 条紧框（面积≤2×）。
即：通过率接近，但**框真正贴合目标**的比例从 0 提升到 5。

默认开启（zoom_refine_enabled=true，config 见 DEFAULT_CONFIG）；每次对区域类事件
做两次「不同外扩倍率 + temperature=0.0」的裁切复核；仅当结果有面积交集时
才可采用候选框。交集只是最低限度的一致性检查，不能证明目标或位置正确。
"""
import base64
import json
import math
import re

import cv2
import numpy as np

from core.bbox_utils import bbox_xywh
from core.frame_store import get_frame_pixels

AREA_KWS = ("孔洞", "材料", "护栏", "基坑", "临边", "洞口")
PROMPT = """这是一张施工现场照片（可能已放大）。请给出【{title}】所指目标的 bbox。
只输出 JSON：{{"x1": 0, "y1": 0, "x2": 0, "y2": 0}}
坐标用 0-1000 的归一化值（左上角 0,0，右下角 1000,1000）。
目标描述：{desc}
要求：框必须紧贴目标本体，不要框整片场地。"""


def _parse(t):
    m = re.search(r"\{.*\}", str(t or ""), re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except ValueError:
        return {}


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def _clamp(b, W, H):
    """返回画面内交集；完全出界或交集太小不造假框。"""
    try:
        if any(isinstance(v, bool) for v in b):
            return None
        x, y, w, h = (float(v) for v in b)
        W, H = float(W), float(H)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h, W, H)):
        return None
    if w <= 0 or h <= 0 or W <= 0 or H <= 0:
        return None
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    # 完全在画内时沿用原始宽高，避免浮点减法引入无意义的微小漂移。
    cw = w if x >= 0 and x + w <= W else x2 - x1
    ch = h if y >= 0 and y + h <= H else y2 - y1
    return x1, y1, cw, ch


def _consensus(boxes):
    if not boxes:
        return None
    if len(boxes) == 1:
        return boxes[0]
    best, score = None, -1.0
    for i, a in enumerate(boxes):
        s = sum(_iou(a, b) for j, b in enumerate(boxes) if j != i)
        if s > score:
            best, score = a, s
    # 两框得分必然相同，旧实现即使两框完全不相交也无条件选择第一框。
    # 没有任何面积交集不能称为一致；保留原框并交人工核查。
    return best if score > 0 else None


def refine_box(client, img, title, desc, old_box, samples=2, max_side=1024,
               return_status=False):
    """返回精修框或 None；可选返回 (结果, 状态)，不把框重叠当正确性认证。"""
    def finish(box, status):
        return (box, status) if return_status else box

    H, W = img.shape[:2]
    try:
        if any(isinstance(v, bool) for v in old_box):
            return finish(None, "invalid_original_box")
        x, y, w, h = (float(v) for v in old_box)
    except (TypeError, ValueError):
        return finish(None, "invalid_original_box")
    if (not all(math.isfinite(v) for v in (x, y, w, h))
            or w <= 0 or h <= 0 or x + w <= 0 or y + h <= 0 or x >= W or y >= H):
        return finish(None, "invalid_original_box")
    cx, cy = x + w / 2, y + h / 2

    # 阶段 1：双采样改用「两次不同外扩倍率」而不是同一张图跑两遍。
    # 原实现两次 chat_completion 的 messages 逐字节相同 → 第二次命中 L1 缓存，
    # 共识名存实亡；现在每次按 i 用不同 side 倍率重新裁切，两次输入真正不同，
    # 且 temperature 固定 0.0（满足确定性契约，不再用 0.6）。
    side_ratios = (2.5, 3.2)
    boxes = []
    requested_samples = max(1, samples)
    for i in range(requested_samples):
        side = max(w, h) * side_ratios[i % len(side_ratios)]
        x0 = int(max(0, cx - side / 2)); y0 = int(max(0, cy - side / 2))
        x1 = int(min(W, cx + side / 2)); y1 = int(min(H, cy + side / 2))
        if x1 - x0 < 64 or y1 - y0 < 64:
            continue
        crop = img[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        scale = max(1.0, max_side / float(max(ch, cw)))
        big = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                         interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".jpg", big)
        if not ok:
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        try:
            resp = client.chat_completion(
                [{"role": "user", "content": [
                    {"type": "text", "text": PROMPT.format(title=title, desc=desc)},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}],
                temperature=0.0, max_tokens=120)
        except Exception:
            continue
        info = _parse(resp)
        try:
            raw_vals = [info[k] for k in ("x1", "y1", "x2", "y2")]
            if any(isinstance(v, bool) for v in raw_vals):
                continue
            vals = [float(v) for v in raw_vals]
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in vals):
            continue
        vx1, vy1, vx2, vy2 = vals
        if max(vals) <= 1.5:
            vx1, vx2 = vx1 * big.shape[1], vx2 * big.shape[1]
            vy1, vy2 = vy1 * big.shape[0], vy2 * big.shape[0]
        else:
            vx1, vx2 = vx1 / 1000.0 * big.shape[1], vx2 / 1000.0 * big.shape[1]
            vy1, vy2 = vy1 / 1000.0 * big.shape[0], vy2 / 1000.0 * big.shape[0]
        if vx2 <= vx1 or vy2 <= vy1:
            continue
        # 模型只看过裁切图，超出裁切图的部分不能被映射为已观察的目标。
        vx1, vy1 = max(0.0, vx1), max(0.0, vy1)
        vx2, vy2 = min(float(big.shape[1]), vx2), min(float(big.shape[0]), vy2)
        if vx2 <= vx1 or vy2 <= vy1:
            continue
        # resize 的 int 取整可能让宽高真实缩放率不同；分别用实际尺寸逆映射。
        sx, sy = cw / float(big.shape[1]), ch / float(big.shape[0])
        bx = (vx1 * sx + x0, vy1 * sy + y0,
              (vx2 - vx1) * sx, (vy2 - vy1) * sy)
        clipped = _clamp(bx, W, H)
        if clipped is not None:
            boxes.append(clipped)
    if len(boxes) < requested_samples:
        return finish(None, "insufficient_votes")
    cb = _consensus(boxes)
    if cb is None:
        return finish(None, "inconsistent")
    # 合理性护栏：面积相对原框 0.15~6 倍，避免又放大框糊弄
    ratio = (cb[2] * cb[3]) / max(1.0, w * h)
    if not (0.15 <= ratio <= 6.0):
        return finish(None, "area_rejected")
    status = "applied_overlap" if requested_samples > 1 else "applied_single_unverified"
    return finish(cb, status)


def refine_events(events, frames_by_index, client, config=None, limit=None,
                  workers=8):
    """就地精修区域类事件框，返回处理数量（并发执行，避免串行拖慢分析）。"""
    from concurrent.futures import ThreadPoolExecutor

    config = config or {}
    n = 0
    stat = {"no_img": 0, "cand": 0}
    jobs = []
    for e in events:
        from core.area_localization import area_confirmation_guard
        from core.person_evidence import lift_occupancy_guard
        if lift_occupancy_guard(getattr(e, "rep_violation", None)):
            e.zoom_refine_status = "lift_relation_requires_review"
            continue
        if area_confirmation_guard(getattr(e, "rep_violation", None)):
            evidence = e.rep_violation.get("area_localization", {})
            status = str(evidence.get("status") or "")
            # 沿用历史人员冲突状态；其他定位不确定不得误写成人员冲突。
            if status.startswith("person_") or status == "other_sample_person_box_conflict":
                e.zoom_refine_status = "area_person_conflict_requires_review"
            else:
                e.zoom_refine_status = "area_location_requires_review"
            continue
        if limit is not None and n >= limit:
            break
        title = str(getattr(e, "rep_title", "") or "")
        if not any(k in title for k in AREA_KWS):
            continue
        stat["cand"] += 1
        try:
            obj = frames_by_index[e.anchor_frame]
            img = obj if isinstance(obj, np.ndarray) else get_frame_pixels(obj)
        except Exception:
            img = None
        if img is None:
            stat["no_img"] += 1
            e.zoom_refine_status = "unavailable_frame"
            continue
        source_bbox = getattr(e, "refined", None) or getattr(e, "fused", None) or {}
        if not isinstance(source_bbox, dict):
            e.zoom_refine_status = "invalid_original_box"
            continue
        # 兼容 xywh / 角点两种合法写法，但缺失、非法值不可默认为 0：
        # 位置未知不应该变成左上角的虚构原框。
        keys = (("x", "y", "w", "h") if any(k in source_bbox for k in ("x", "y", "w", "h"))
                else ("x1", "y1", "x2", "y2"))
        if any(isinstance(source_bbox.get(k), bool) for k in keys):
            e.zoom_refine_status = "invalid_original_box"
            continue
        old = bbox_xywh(source_bbox, default=float("nan"))
        jobs.append((e, img, title,
                     (getattr(e, "rep_violation", {}) or {}).get("description", "")[:140],
                     old, dict(source_bbox)))

    def work(job):
        e, img, title, desc, old, source_bbox = job
        new, status = refine_box(client, img, title, desc, old,
                                 samples=int(config.get("zoom_refine_samples", 2)),
                                 return_status=True)
        return e, new, status, source_bbox

    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            for e, new, status, source_bbox in ex.map(work, jobs):
                e.zoom_refine_status = status
                if new:
                    n += 1
                    e.pre_zoom_bbox = source_bbox
                    e.refined = {"x": new[0], "y": new[1], "w": new[2], "h": new[3]}
    print(f"[zoom] 候选{stat['cand']} 无帧{stat['no_img']} 精修成功{n}")
    return n
