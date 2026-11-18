# -*- coding: utf-8 -*-
"""降级前的统一小目标护栏；无效或缺失坐标按不可判处理。"""
import math

MIN_AREA_PX2 = 5000
MIN_LONG_SIDE_PX = 40


def event_bbox(event):
    """取得报告字典或 Event 的像素 xywh 框，优先采用最终报告使用的框。"""
    if isinstance(event, dict):
        return event.get("bbox")
    box = getattr(event, "refined", None)
    if box:
        return box
    samples = getattr(event, "samples", None) or []
    return samples[0].get("bbox") if samples and isinstance(samples[0], dict) else None


def _box_requires_guard(box):
    if not isinstance(box, dict):
        return True
    try:
        width = float(box.get("w", 0))
        height = float(box.get("h", 0))
    except (TypeError, ValueError, OverflowError):
        return True
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        return True
    return width * height < MIN_AREA_PX2 or max(width, height) < MIN_LONG_SIDE_PX


def geometry_guard_hit(event):
    """小目标、无效框或历史小框锁定时阻止降级。"""
    # 精修可能扩大 bbox；不能把生成的新像素当成原始分辨率的提高。
    # 历史字段不存在或为 None 时不另加限制；存在但无效时保守保护。
    if isinstance(event, dict):
        if bool(event.get("dedupe_geometry_guard")):
            return True
        history = (event.get("bbox_pre_zoom"), event.get("bbox_pre_tighten"))
    else:
        if bool((getattr(event, "rep_violation", None) or {}).get("dedupe_geometry_guard")):
            return True
        history = (getattr(event, "pre_zoom_bbox", None),
                   getattr(event, "pre_tighten_bbox", None))
    return (_box_requires_guard(event_bbox(event))
            or any(_box_requires_guard(box) for box in history if box is not None))


def enforce_review_guard(event, golden=False, *, golden_identity_matched=None):
    """纠正护栏绕过；返回值表示处置是否修正，身份不可核验时另写审计状态。"""
    if not isinstance(event, dict):
        return False
    if golden and golden_identity_matched is False:
        # 先纠正无法核验的旧身份声称，即使事件本来就在待复核队列。
        if event.pop("ai_review_golden_exempt", None) is True:
            event.setdefault("golden_guard_prior_identity_claim", True)
        event["frozen_assets_unverified_guard"] = True
    disposition = str(event.get("review_disposition") or "")
    small = geometry_guard_hit(event)
    moved = disposition in ("rejected", "box_relocalize", "dropped", "low_priority_review")
    lost_required_review = (small and bool(event.get("dedupe_geometry_review_required"))
                            and not event.get("needs_review"))
    if (not (golden and (not event.get("needs_review") or moved))
            and not (small and moved) and not lost_required_review):
        return False
    event["needs_review"] = True
    event["review_disposition"] = "low_priority_review"
    if small:
        event["ai_review_geometry_guard"] = True
    if golden and golden_identity_matched is True:
        event["ai_review_golden_exempt"] = True
    return True
