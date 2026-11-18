# -*- coding: utf-8 -*-
"""基于图像位置变化生成待核线索；不证明真实位移或进入危险半径。"""
import math
import cv2
import numpy as np

from core.event_image_sync import verified_exact_original
from core.frame_lookup import normalize_timestamp
from core.zones import dist_to_polygon

TITLE = "人员进入机械作业半径（轨迹）"


class _Track:
    __slots__ = ("tid", "box", "vel", "miss")

    def __init__(self, tid, box):
        self.tid = tid
        self.box = np.array(box, dtype=np.float32)
        self.vel = np.zeros(4, dtype=np.float32)
        self.miss = 0


def _valid_entity(ent, min_area, max_area=None):
    """v1.13.36 硬门控：过滤尺寸异常/低置信误检。

    6 条新视频实测里 `c46c8857` 260s 帧的 YOLO 结果：
      machine 125×134 / 182×105（红色设备、板房被当成机械）
      person  122×219 / 145×205 / 131×215（板房被当成人，占画面 5%）
    俯视航拍里真实人员一般只有 20×40~40×70 px，所以误检的特征是
    「太大」而不是「太小」。因此同时设面积下限与上限（仅对 person 严格）。
    conf 缺失时只按尺寸判断。
    """
    b = (ent or {}).get("bbox_px") or {}
    if not b:
        return False
    try:
        w = float(b["x2"]) - float(b["x1"])
        h = float(b["y2"]) - float(b["y1"])
        conf = float(ent.get("conf") or 0)
    except (KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(x) for x in (w, h, conf)) or w <= 0 or h <= 0:
        return False
    if min_area <= 0:          # 门控关闭：只要求框本身有效
        return True
    area = w * h
    if area < min_area or (max_area is not None and area > max_area):
        return False
    return conf == 0 or conf >= 0.30


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1e-6, ua)


def build_track_events(out_dir, frame_results, det, enter=0.55, far=0.9, rearm=1.2,
                       window=3, gap_merge=20, max_miss=6, entities_map=None,
                       zones=None, edge_near=0.03, edge_far=0.08, hard_gate=True, diagnostics=None):
    from core.entity_cache import get_for_row

    diag = diagnostics if isinstance(diagnostics, dict) else {}
    diag.update(frame_count=0, verified_frame_count=0, unverified_frame_count=0,
                unavailable_frame_count=0, invalid_frame_count=0, detector_failure_count=0,
                motion_confirmed=False, scope="sampled_frames_only", absence_is_not_safety=True)
    frame_results = frame_results if isinstance(frame_results, (list, tuple)) else []
    # 无时间的记录不能伪装成 0 秒；0 本身仍是合法时间。
    frames = []
    for row in frame_results:
        ts = normalize_timestamp(row.get("timestamp")) if isinstance(row, dict) else None
        if ts is not None:
            frames.append((ts, row))
        else:
            diag["invalid_frame_count"] += 1
    frames.sort(key=lambda item: item[0])
    def _verified_image(row, ts):
        event = {"representative_frame": row.get("frame_index"), "representative_timestamp": ts}
        _, image, status = verified_exact_original(event, frame_results, out_dir)
        return image if status == "ok" else None

    # 尺寸只取可核验原图；没有证据时使用兼容默认尺寸，并始终标为未核验。
    W, H = 960, 544
    for _ts, _r in frames:
        _im0 = _verified_image(_r, _ts)
        if _im0 is not None:
            H, W = _im0.shape[:2]
            break
    min_p_area = 0.0004 * W * H if hard_gate else 0.0   # 约 15×15（960×544 俯视人员下限）
    max_p_area = 0.012 * W * H if hard_gate else None   # 约 79×79，滤掉板房误检成的「巨人员」
    min_m_area = 0.006 * W * H if hard_gate else 0.0    # 约 56×56，滤掉小工具
    max_m_area = 0.30 * W * H if hard_gate else None    # 机械可较大，只挡整屏级误检
    tracks, nid = [], 1
    hist, armed, raw = [], True, []
    zhist, zarmed, zraw = {}, {}, []

    def _person_box(box):
        return {"x": float(box[0]), "y": float(box[1]),
                "w": float(box[2] - box[0]), "h": float(box[3] - box[1])}

    def _association(rows, *, machine=False):
        # IoU 只能说明关联未明显冲突；没有地面坐标、运动补偿或独立身份验证。
        times = [row[0] for row in rows]
        if any(b <= a for a, b in zip(times, times[1:])):
            return "ambiguous_time_order"
        if any(b - a > gap_merge for a, b in zip(times, times[1:])):
            return "ambiguous_time_gap"
        if len({row[4] for row in rows}) != 1:
            return "ambiguous_person_switch"
        if machine:
            def as_xyxy(b):
                return (b["x1"], b["y1"], b["x2"], b["y2"])
            if any(_iou(as_xyxy(a[3]), as_xyxy(b[3])) < 0.35 for a, b in zip(rows, rows[1:])):
                return "ambiguous_machine_switch"
        return "heuristic_iou_only"

    def _observation(rows, kind):
        return {"observation_times": [float(row[0]) for row in rows],
                "observation_ratios": [round(float(row[1]), 4) for row in rows],
                "person_track_ids": [int(row[4]) for row in rows],
                "association_state": _association(rows, machine=kind == "machine"),
                "camera_motion_compensated": False, "identity_independently_verified": False,
                "actual_danger_radius_verified": False, "motion_confirmed": False,
                "frame_source_verified": all(row[5] for row in rows),
                "time_basis": ("frame_results_exact_original_sha256" if all(row[5] for row in rows)
                               else "frame_results_timestamp_source_unverified")}
    for ts, r in frames:
        image = _verified_image(r, ts)
        if entities_map is not None:
            ents, cache_verified = get_for_row(entities_map, r)
            if ents is None:
                diag["unavailable_frame_count"] += 1
                continue
            frame_verified = bool(cache_verified and image is not None)
        else:
            if image is None:
                diag["unavailable_frame_count"] += 1
                continue
            try:
                prior = det.build_scene_prior(image)
            except Exception:
                diag["detector_failure_count"] += 1
                diag["unavailable_frame_count"] += 1
                continue
            # 检测过程中图片被替换不能把此次实体作为可信证据。
            if _verified_image(r, ts) is None:
                diag["unavailable_frame_count"] += 1
                continue
            if not isinstance(prior, dict) or not isinstance(prior.get("entities"), list):
                diag["detector_failure_count"] += 1
                diag["unavailable_frame_count"] += 1
                continue
            ents, frame_verified = prior["entities"], True
        if not isinstance(ents, list) or any(not isinstance(item, dict) for item in ents):
            diag["detector_failure_count"] += 1
            diag["unavailable_frame_count"] += 1
            continue
        diag["frame_count"] += 1
        diag["verified_frame_count" if frame_verified else "unverified_frame_count"] += 1
        # 硬门控：人/机械都必须有足够面积的框与可信 conf；统一 bucket_truck 类型名
        persons = [e["bbox_px"] for e in ents
                   if e.get("type") in ("person", "head")
                   and _valid_entity(e, min_p_area, max_p_area)]
        machines = [e["bbox_px"] for e in ents
                    if e.get("type") in ("machine", "bucket", "bucket_truck")
                    and _valid_entity(e, min_m_area, max_m_area)]
        boxes = [np.array([p["x1"], p["y1"], p["x2"], p["y2"]], dtype=np.float32) for p in persons]
        # 预测 + 匹配
        for t in tracks:
            t.box = t.box + t.vel
        matched = set()
        for b in boxes:
            best, bi = 0.35, None
            for i, t in enumerate(tracks):
                if i in matched:
                    continue
                v = _iou(b, t.box)
                if v > best:
                    best, bi = v, i
            if bi is None:
                tracks.append(_Track(nid, b)); nid += 1
                matched.add(len(tracks) - 1)
            else:
                t = tracks[bi]
                t.vel = 0.6 * t.vel + 0.4 * (b - t.box)
                t.box = b
                t.miss = 0
                matched.add(bi)
        for i, t in enumerate(tracks):
            if i not in matched:
                t.miss += 1
        tracks = [t for t in tracks if t.miss <= max_miss]
        # 最近人机距离
        best = None
        for t in tracks:
            if t.miss > 0 or not machines:
                continue
            b = t.box
            pcx, pcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            for mb in machines:
                mcx, mcy = (mb["x1"] + mb["x2"]) / 2, (mb["y1"] + mb["y2"]) / 2
                h = max(1, mb["y2"] - mb["y1"])
                rr = (((pcx - mcx) ** 2 + (pcy - mcy) ** 2) ** 0.5) / h
                if best is None or rr < best[0]:
                    best = (rr, b.copy(), dict(mb), t.tid)
        # 临边距离不依赖机械：每个区域取本帧最近人员，身份切换必须标记不确定。
        if zones:
            for zi, z in enumerate(zones):
                if not isinstance(z, dict):
                    continue
                poly = z.get("polygon") or []
                try:
                    if not isinstance(poly, (list, tuple)) or len(poly) > 4096:
                        continue
                    polygon_norm = [(float(x), float(y)) for x, y in poly]
                    polygon = np.asarray([(x * W, y * H) for x, y in polygon_norm], dtype=np.float32)
                    if len(polygon) < 3 or not np.isfinite(polygon).all():
                        continue
                except (TypeError, ValueError, OverflowError):
                    continue
                choices = []
                for t in tracks:
                    if t.miss:
                        continue
                    pbox = t.box.copy()
                    center = (float((pbox[0] + pbox[2]) / 2), float((pbox[1] + pbox[3]) / 2))
                    try:
                        d = dist_to_polygon(center, polygon_norm, W, H)
                        inside = cv2.pointPolygonTest(polygon, center, False) >= 0
                    except (TypeError, ValueError, OverflowError, cv2.error):
                        continue
                    if d is not None and math.isfinite(d):
                        choices.append((0.0 if inside else d, t.tid, pbox, inside))
                if not choices:
                    continue
                d, person_id, pbox, inside = min(choices, key=lambda item: (item[0], item[1]))
                # 落在人工标注多边形内作为区域线索；不等于真实坠落危险或违章。
                rr = 0.0 if inside else d / max(1, H)
                history = zhist.setdefault(zi, [])
                history.append((ts, rr, pbox, None, person_id, frame_verified))
                if rr > edge_far:
                    zarmed[zi] = True
                if zarmed.get(zi, True) and len(history) >= window:
                    observations = history[-window:]
                    prev = [h[1] for h in observations[:-1]]
                    if prev and max(prev) > edge_far and rr < edge_near:
                        zraw.append({"ts": ts, "zone": z.get("name", "临边"), "zone_index": zi,
                                     "inside_user_polygon": bool(inside),
                                     "from": float(max(prev)), "to": float(rr),
                                     "bbox": _person_box(pbox), **_observation(observations, "edge")})
                        zarmed[zi] = False
        if best is None:
            continue
        ratio, pbox, mbox, person_id = best
        hist.append((ts, ratio, pbox, mbox, person_id, frame_verified))
        if ratio > rearm:
            armed = True
        if armed and len(hist) >= window:
            prev = [h[1] for h in hist[-window:-1]]
            if prev and max(prev) > far and ratio < enter:
                observations = hist[-window:]
                raw.append({"ts": ts, "from": float(max(prev)), "to": float(ratio),
                            "bbox": {"x": float(min(pbox[0], mbox["x1"])),
                                     "y": float(min(pbox[1], mbox["y1"])),
                                     "w": float(max(pbox[2], mbox["x2"]) - min(pbox[0], mbox["x1"])),
                                     "h": float(max(pbox[3], mbox["y2"]) - min(pbox[1], mbox["y1"]))},
                            **_observation(observations, "machine")})
                armed = False
    def _merge(lst):
        out = []
        for e in lst:
            if out and e["ts"] - out[-1]["ts"] <= gap_merge:
                if e["to"] < out[-1]["to"]:
                    out[-1] = e
            else:
                out.append(e)
        return out

    merged = []
    for zi in sorted({e["zone_index"] for e in zraw}):
        merged.extend(_merge([dict(e, kind="edge") for e in zraw if e["zone_index"] == zi]))
    merged.extend(_merge([dict(e, kind="machine") for e in raw]))
    merged.sort(key=lambda x: x["ts"])
    incomplete = (diag["invalid_frame_count"] or diag["unavailable_frame_count"]
                  or diag["unverified_frame_count"])
    diag["status"] = "no_usable_frames" if not diag["frame_count"] else (
        "partial_or_unverified" if incomplete else "sampled_frames_checked")
    diag["candidate_count"] = len(merged)
    return merged
