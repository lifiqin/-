# -*- coding: utf-8 -*-
"""区域性问题：按（场景+标题）合并，并改成场景级标注（不画整片大框）"""
from collections import defaultdict

ZONE_KEYWORDS = (
    "裸土", "渣土", "扬尘", "建筑垃圾", "废料", "场地积水", "围挡",
    "钢筋骨架", "材料堆放", "材料堆码", "安全网破损", "裸土/渣土",
)


def is_zone(title):
    t = str(title or "")
    return any(k in t for k in ZONE_KEYWORDS)


def collapse_zone_events(events, scene_segments):
    """events: report["events"] 的 dict 列表。区域性问题按 scene+title 合并。"""
    def scene_of(ts):
        for s in scene_segments or []:
            if s.get("start_time", 0) <= ts <= s.get("end_time", 0):
                return s.get("segment_id", -1)
        return -1

    groups = defaultdict(list)
    order = []
    for e in events:
        if not is_zone(e.get("title")):
            order.append(("single", e))
            continue
        key = (scene_of(float(e.get("representative_timestamp") or 0)), e.get("title"))
        if key not in groups:
            order.append(("group", key))
        groups[key].append(e)

    out = []
    for kind, key in order:
        if kind == "single":
            out.append(key)
            continue
        lst = groups[key]
        if not lst:
            continue
        rep = dict(lst[0])
        rep["repeat_count"] = len(lst)
        rep["merged_event_ids"] = [x.get("event_id") for x in lst]
        rep["scene_level"] = True
        rep["scene_id"] = key[0]
        rep["bbox"] = None            # 场地级问题不画大框
        rep["box_note"] = "区域性问题：按场景标注，不提供单点框"
        rep["first_seen"] = min(x.get("first_seen", 0) for x in lst)
        rep["last_seen"] = max(x.get("last_seen", 0) for x in lst)
        out.append(rep)
    return out
