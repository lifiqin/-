# -*- coding: utf-8 -*-
"""事件框本地守卫（零 API 开销）

背景：VLM 自报 bbox 在多目标场景会框错对象（框到旁边的人/车辆/地面），
框错等于报错位置。本模块用本地 YOLO 实体框做一致性检查：

- 人员类违章（安全帽/安全带/反光衣/高处/临边）：事件框必须与锚点帧的
  人员/头部/安全帽实体框有重叠（IoU ≥ iou_min），否则标记 box_suspect。
- 所有事件：框越界、面积占比异常（<0.02% 或 >60%）也标记。

只标记不删除——报告里显示"框待复核"，把判断权交给人，不静默改数据。
"""
from core.bbox_utils import bbox_xywh

PERSON_KWS = ("安全帽", "安全带", "反光衣", "个体防护", "高处作业",
              "临边防护", "坠落半径", "违规载人")


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def flag_suspect_boxes(events, frames_by_index, detector=None, iou_min=0.15,
                       frame_w=0, frame_h=0):
    """给事件打 box_suspect 标记，返回被标记数量。"""
    if not events:
        return 0
    n_flag = 0
    cache = {}

    for e in events:
        try:
            raw = getattr(e, "refined", None) or getattr(e, "fused", None) or {}
            box = bbox_xywh(raw)
            title = str(getattr(e, "rep_title", "") or "")
            flags = []

            # 1) 几何检查（不依赖 YOLO）
            x, y, w, h = box
            if w <= 1 or h <= 1:
                flags.append("框尺寸异常")
            if frame_w and frame_h:
                if x + w < 0 or y + h < 0 or x > frame_w or y > frame_h:
                    flags.append("框越界")
                ratio = (w * h) / float(frame_w * frame_h)
                if ratio > 0.6:
                    flags.append("框覆盖画面过大")
                elif ratio < 0.0002:
                    flags.append("框过小")

            # 2) 人员类：必须与 YOLO 人员/头部框重叠
            if detector is not None and any(k in title for k in PERSON_KWS):
                anchor = getattr(e, "anchor_frame", None)
                if anchor is not None and anchor not in cache:
                    px = None
                    try:
                        from core.frame_store import get_frame_pixels
                        import numpy as _np
                        try:
                            obj = frames_by_index[anchor]
                        except Exception:
                            obj = frames_by_index.get_pixels(anchor)
                        px = obj if isinstance(obj, _np.ndarray) else get_frame_pixels(obj)
                    except Exception:
                        px = None
                    if px is None:
                        cache[anchor] = []
                    else:
                        try:
                            # 优先用带小目标召回的实体清单（build_scene_prior），
                            # 单帧 detect 会漏掉远景小人，导致全员误标
                            if hasattr(detector, "build_scene_prior"):
                                ents = (detector.build_scene_prior(px) or {}).get("entities") or []
                                cache[anchor] = [
                                    {"label": e.get("type"), "bbox": (
                                        (e.get("bbox_px") or {}).get("x1", 0),
                                        (e.get("bbox_px") or {}).get("y1", 0),
                                        (e.get("bbox_px") or {}).get("x2", 0),
                                        (e.get("bbox_px") or {}).get("y2", 0))}
                                    for e in ents if e.get("type") in ("person", "head", "helmet")
                                ]
                                # 实体是角点，转成 xywh
                                cache[anchor] = [
                                    {"label": d["label"],
                                     "bbox": (d["bbox"][0], d["bbox"][1],
                                              d["bbox"][2] - d["bbox"][0],
                                              d["bbox"][3] - d["bbox"][1])}
                                    for d in cache[anchor]
                                ]
                            else:
                                cache[anchor] = [d for d in detector.detect(px)
                                                 if d.get("label") in ("person", "head", "helmet")]
                        except Exception:
                            cache[anchor] = []
                dets = cache.get(anchor) or []
                if dets:
                    best = max((_iou(box, d.get("bbox") or (0, 0, 0, 0)) for d in dets),
                               default=0.0)
                    if best < iou_min:
                        flags.append("框内无人员实体（可能框错对象）")

            if flags:
                e.box_suspect = True
                e.box_suspect_reason = "；".join(flags)
                n_flag += 1
            else:
                e.box_suspect = False
                e.box_suspect_reason = ""
        except Exception:
            continue
    return n_flag
