# -*- coding: utf-8 -*-
"""场景合并：自动（同名相邻）/ 手动（选中多个）→ scene_overrides.json"""
import json
import os


def _path(output_dir):
    return os.path.join(output_dir, "scene_overrides.json")


def load_overrides(output_dir):
    try:
        return json.load(open(_path(output_dir), encoding="utf-8")).get("groups", [])
    except Exception:
        return []


def save_overrides(output_dir, groups):
    json.dump({"groups": groups}, open(_path(output_dir), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def auto_groups(scene_segments, max_gap=30):
    """同名且间隔≤max_gap 的相邻段自动并组。"""
    groups, cur = [], None
    for s in sorted(scene_segments, key=lambda x: x.get("start_time", 0)):
        sid = s.get("segment_id")
        if cur is None:
            cur = {"ids": [sid], "name": s.get("name") or "", "start": s.get("start_time", 0), "end": s.get("end_time", 0)}
            continue
        gap = float(s.get("start_time", 0)) - float(cur["end"])
        if (s.get("name") or "") == cur["name"] and gap <= max_gap:
            cur["ids"].append(sid)
            cur["end"] = s.get("end_time", cur["end"])
        else:
            if len(cur["ids"]) > 1:
                groups.append({"ids": cur["ids"], "name": cur["name"], "source": "auto"})
            cur = {"ids": [sid], "name": s.get("name") or "", "start": s.get("start_time", 0), "end": s.get("end_time", 0)}
    if cur and len(cur["ids"]) > 1:
        groups.append({"ids": cur["ids"], "name": cur["name"], "source": "auto"})
    return groups


def apply_overrides(scene_segments, groups):
    """把分组应用到场景段：返回合并后的场景列表（保留原 id 集合与时间范围）。"""
    if not groups:
        return [dict(s, seg_ids=[s.get("segment_id")]) for s in scene_segments]
    merged_id = {}
    for g in groups:
        for sid in g.get("ids", []):
            merged_id[sid] = g
    out, seen = [], set()
    for s in sorted(scene_segments, key=lambda x: x.get("start_time", 0)):
        sid = s.get("segment_id")
        g = merged_id.get(sid)
        if not g:
            out.append(dict(s, seg_ids=[sid]))
            continue
        key = id(g)
        if key in seen:
            continue
        seen.add(key)
        members = [x for x in scene_segments if x.get("segment_id") in g["ids"]]
        out.append({
            "segment_id": g["ids"][0],
            "seg_ids": list(g["ids"]),
            "name": g.get("name") or (members[0].get("name") if members else ""),
            "start_time": min(float(m.get("start_time", 0)) for m in members),
            "end_time": max(float(m.get("end_time", 0)) for m in members),
            "merged": True,
        })
    return out
