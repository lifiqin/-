# -*- coding: utf-8 -*-
"""安全帽反证只改变处置，不删除已产生的事件。"""
import os

from core.ai_review import is_golden_exempt
from core.geometry_guard import geometry_guard_hit
from core.review_policy import source_video_review_guard


def apply_helmet_rebut_gate(events, output_dir, config):
    """复用旧开关，几何小目标及冻结金标保留在主复核队列。"""
    stats = {"checked": 0, "demoted": 0, "guard_kept": 0}
    if not config.get("helmet_rebut_drop_enabled", True):
        return stats
    tag = os.path.basename(os.path.normpath(output_dir or ""))
    for event in events or []:
        violation = dict(getattr(event, "rep_violation", None) or {})
        if "安全帽" not in getattr(event, "rep_title", "") or not violation.get("helmet_rebut"):
            continue
        stats["checked"] += 1
        view = {"event_id": getattr(event, "display_id", None)}
        protected = (geometry_guard_hit(event) or is_golden_exempt(view, config, tag)
                     or source_video_review_guard(violation))
        violation.update({"needs_review": protected,
                          "review_disposition": "low_priority_review" if protected else "rejected",
                          "review_reason": "安全帽视觉反证；" + ("命中护栏，保留人工复核" if protected else "保留事件待人工核查")})
        violation["helmet_rebut_guard_kept" if protected else "helmet_rebut_demote"] = True
        event.rep_violation = violation
        stats["guard_kept" if protected else "demoted"] += 1
    return stats
