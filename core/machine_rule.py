# -*- coding: utf-8 -*-
"""机械作业半径规则：用现成 person/machine 检测器算几何关系，不依赖 VLM 框"""
from core.frame_lookup import find_frame_by_timestamp, normalize_timestamp

TITLE = "机械作业无监护/安全距离不足"


def _frame_path(out_dir, ts):
    return find_frame_by_timestamp(out_dir, ts)


def _center(b):
    return ((b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2)


def _inside(pt, b, pad=0.0):
    w, h = b["x2"] - b["x1"], b["y2"] - b["y1"]
    return (b["x1"] - pad * w) <= pt[0] <= (b["x2"] + pad * w) and (b["y1"] - pad * h) <= pt[1] <= (b["y2"] + pad * h)


def detect_pairs(im, det, close_ratio=0.5):
    """返回 [(person_bbox, machine_bbox, dist_ratio)]，已排除驾驶室内人员。"""
    ents = det.build_scene_prior(im).get("entities", [])
    persons = [e for e in ents if e.get("type") in ("person", "head")]
    machines = [e for e in ents if e.get("type") in ("machine", "bucket")]
    pairs = []
    for p in persons:
        pb = p.get("bbox_px") or {}
        if not pb:
            continue
        pc = _center(pb)
        for m in machines:
            mb = m.get("bbox_px") or {}
            if not mb:
                continue
            if _inside(pc, mb):          # 驾驶室内操作员，不算
                continue
            if not _inside(pc, mb, pad=0.25):   # 人必须贴着机械，远处不算
                continue
            if (mb["y2"] - mb["y1"]) < 60 or (mb["x2"] - mb["x1"]) < 60:
                continue
            mc = _center(mb)
            h = max(1.0, mb["y2"] - mb["y1"])
            ratio = (((pc[0] - mc[0]) ** 2 + (pc[1] - mc[1]) ** 2) ** 0.5) / h
            if ratio <= close_ratio:
                pairs.append((pb, mb, ratio))
    return pairs


def _pairs_from_ents(ents, close_ratio=0.5):
    persons = [e for e in ents if e.get("type") in ("person", "head")]
    machines = [e for e in ents if e.get("type") in ("machine", "bucket")]
    pairs = []
    for p in persons:
        pb = p.get("bbox_px") or {}
        if not pb:
            continue
        pc = _center(pb)
        for m in machines:
            mb = m.get("bbox_px") or {}
            if not mb:
                continue
            if _inside(pc, mb):
                continue
            if not _inside(pc, mb, pad=0.25):
                continue
            if (mb["y2"] - mb["y1"]) < 60 or (mb["x2"] - mb["x1"]) < 60:
                continue
            mc = _center(mb)
            h = max(1.0, mb["y2"] - mb["y1"])
            ratio = (((pc[0] - mc[0]) ** 2 + (pc[1] - mc[1]) ** 2) ** 0.5) / h
            if ratio <= close_ratio:
                pairs.append((pb, mb, ratio))
    return pairs


def build_events(out_dir, frame_results, det, close_ratio=0.5, min_hits=2, gap=20,
                 entities_map=None, diagnostics=None):
    """仅生成稀疏采样线索；不能证明持续动作、目标身份或真实危险距离。"""
    from core.entity_cache import build_entity_cache, get_for_row

    diag = diagnostics if isinstance(diagnostics, dict) else {}
    diag.update(frame_count=0, unavailable_frame_count=0, unverified_frame_count=0,
                scope="sampled_frames_only", absence_is_not_safety=True)
    frame_results = frame_results if isinstance(frame_results, (list, tuple)) else []
    if entities_map is None:
        entities_map = build_entity_cache(out_dir, frame_results, det)
    if hasattr(entities_map, "audit"):
        diag["entity_cache"] = dict(entities_map.audit)
    hits = []
    for r in frame_results:
        ts = normalize_timestamp(r.get("timestamp")) if isinstance(r, dict) else None
        if ts is None:
            diag["unavailable_frame_count"] += 1
            continue
        ents, verified = get_for_row(entities_map, r)
        if not isinstance(ents, list) or any(not isinstance(item, dict) for item in ents):
            diag["unavailable_frame_count"] += 1
            continue
        diag["frame_count"] += 1
        diag["unverified_frame_count"] += int(not verified)
        try:
            pairs = _pairs_from_ents(ents, close_ratio)
        except (KeyError, TypeError, ValueError, OverflowError):
            diag["unavailable_frame_count"] += 1
            continue
        for pb, mb, ratio in pairs:
            x1 = min(pb["x1"], mb["x1"]); y1 = min(pb["y1"], mb["y1"])
            x2 = max(pb["x2"], mb["x2"]); y2 = max(pb["y2"], mb["y2"])
            hits.append({"ts": ts, "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                         "ratio": ratio, "frame_index": r.get("frame_index"), "verified": verified})
    hits.sort(key=lambda h: h["ts"])
    groups, cur = [], []
    for h in hits:
        if not cur or h["ts"] - cur[-1]["ts"] <= gap:
            cur.append(h)
        else:
            groups.append(cur); cur = [h]
    if cur:
        groups.append(cur)
    events = []
    for g in groups:
        # 多个主体出现在同一帧不是多帧，且不同帧的主体不一定相同。
        observation_times = sorted({h["ts"] for h in g})
        if len(observation_times) < min_hits:
            continue
        best = min(g, key=lambda x: x["ratio"])
        events.append({
            "title": TITLE,
            "first_seen": g[0]["ts"], "last_seen": g[-1]["ts"],
            "representative_timestamp": best["ts"],
            "representative_frame": best["frame_index"],
            "bbox": best["bbox"],
            "hits": len(observation_times), "observation_times": observation_times,
            "dist_ratio": round(best["ratio"], 2),
            "rule_based": True, "needs_review": True,
            "frame_source_verified": all(h["verified"] for h in g),
            "identity_independently_verified": False,
            "actual_danger_radius_verified": False, "motion_confirmed": False,
        })
    diag["candidate_count"] = len(events)
    diag["status"] = ("no_usable_frames" if not diag["frame_count"] else
                      "partial_or_unverified" if diag["unavailable_frame_count"] or diag["unverified_frame_count"]
                      else "sampled_frames_checked")
    return events
