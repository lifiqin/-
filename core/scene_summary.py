# -*- coding: utf-8 -*-
"""按场景汇总：同一场景内的违章合并成一条场景报告，不再逐帧重复"""
from collections import defaultdict

from core.frame_lookup import normalize_timestamp


def _valid_timestamps(events):
    values = [normalize_timestamp(e.get("representative_timestamp")) for e in events]
    return [value for value in values if value is not None]


def _scene_sort_key(item):
    values = _valid_timestamps(item[1])
    return (0, min(values)) if values else (1, 0)


def _rounded_timestamp(value):
    timestamp = normalize_timestamp(value)
    return round(timestamp, 1) if timestamp is not None else None


def _format_timestamp(value):
    timestamp = normalize_timestamp(value)
    return "时间未知" if timestamp is None else f"{timestamp:.0f}s"


def build_scene_summary(events, scene_segments, output_dir=None):
    """返回 [{scene_id,name,start_time,end_time,violations:[...]}]

    若 output_dir 存在 scene_overrides.json，则先按人工/自动合并结果重排场景。
    """
    if output_dir:
        try:
            from core.scene_merge import load_overrides, apply_overrides
            scene_segments = apply_overrides(scene_segments, load_overrides(output_dir))
        except Exception:
            pass
    by_scene = defaultdict(list)
    sid_map = {}
    valid_segments = [s for s in (scene_segments or [])
                      if isinstance(s, dict)
                      and normalize_timestamp(s.get("start_time")) is not None
                      and normalize_timestamp(s.get("end_time")) is not None
                      and normalize_timestamp(s.get("start_time")) <= normalize_timestamp(s.get("end_time"))]
    for e in events:
        ts = normalize_timestamp(e.get("representative_timestamp"))
        sid = -1
        if ts is not None:
            for s in valid_segments:
                if normalize_timestamp(s.get("start_time")) <= ts <= normalize_timestamp(s.get("end_time")):
                    sid = s.get("segment_id", -1)
                    sid_map[sid] = s.get("seg_ids") or [sid]
                    break
            else:
                try:
                    from core.event_scene_dedupe import _scene_of
                    sid = _scene_of(ts, valid_segments)[0]
                    for s in valid_segments:
                        if s.get("segment_id") == sid:
                            sid_map[sid] = s.get("seg_ids") or [sid]
                            break
                except Exception:
                    pass
        by_scene[sid].append(e)

    seg_map = {s.get("segment_id", -1): s for s in valid_segments}
    out = []
    for sid, evs in sorted(by_scene.items(), key=_scene_sort_key):
        scene_times = _valid_timestamps(evs)
        seg = seg_map.get(sid, {}) if scene_times else {}
        seg = dict(seg, seg_ids=sid_map.get(sid, seg.get("seg_ids") or [sid]))
        groups = defaultdict(list)
        for e in evs:
            groups[e.get("title", "")].append(e)
        items = []
        for title, lst in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            lst_sorted = sorted(lst, key=lambda e: -(normalize_timestamp(e.get("duration_sec")) or 0))
            best = lst_sorted[0]
            item_times = _valid_timestamps(lst)
            items.append({
                "title": title,
                "count": len(lst),
                "first_ts": _rounded_timestamp(min(item_times)) if item_times else None,
                "representative_frame": best.get("representative_frame"),
                "representative_timestamp": normalize_timestamp(best.get("representative_timestamp")),
                "bbox": best.get("bbox"),
                "severity": best.get("severity"),
            })
        first_ts = min(scene_times) if scene_times else None
        last_ts = max(scene_times) if scene_times else None
        start_time = normalize_timestamp(seg.get("start_time"))
        end_time = normalize_timestamp(seg.get("end_time"))
        scene = {
            "scene_id": sid,
            "seg_ids": seg.get("seg_ids") or [sid],
            "name": seg.get("name") or "未分段",
            "start_time": _rounded_timestamp(start_time if start_time is not None else first_ts),
            "end_time": _rounded_timestamp(end_time if end_time is not None else last_ts),
            "violation_types": len(items),
            "violation_count": len(evs),
            "violations": items,
        }
        unknown_count = len(evs) - len(scene_times)
        if unknown_count:
            scene["unknown_time_count"] = unknown_count
        out.append(scene)
    return out


def render_scene_text(scene_summary, limit=40):
    lines = ["", "=" * 60, "按场景汇总（同一场景合并，不逐帧重复）", "=" * 60]
    for s in scene_summary[:limit]:
        unknown = f"（{s['unknown_time_count']} 条时间未知）" if s.get("unknown_time_count") else ""
        lines.append(f"\n【场景 {s['scene_id']}｜{s['name']}】{_format_timestamp(s['start_time'])} - {_format_timestamp(s['end_time'])}，"
                     f"{s['violation_types']} 类 / {s['violation_count']} 处{unknown}")
        for v in s["violations"]:
            lines.append(f"  · {v['title']} ×{v['count']}（首次 {_format_timestamp(v['first_ts'])}）")
    return "\n".join(lines)
