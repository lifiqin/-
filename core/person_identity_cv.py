"""人员候选的第二本地检测器比对（研究实验件，未接入生产，默认不运行）。

本模块只接收已经本地运行完成的检测结果，不加载模型、不访问网络、不做负向定性。
即使权重哈希不同，也不能断言不同检测器的训练数据与误差统计独立。
所有输出仅供离线人工审查，不得参与实时自动判断、删事件或移动审核队列。
"""
import math
import re


_SHA256 = re.compile(r"^[a-fA-F0-9]{64}$")


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (OverflowError, TypeError, ValueError):
        return None


def _xywh(box, width, height):
    if not isinstance(box, dict):
        return None
    values = [_finite(box.get(key)) for key in ("x", "y", "w", "h")]
    if any(value is None for value in values):
        return None
    x, y, w, h = values
    if (x < 0 or y < 0 or w <= 0 or h <= 0 or not math.isfinite(x + w)
            or not math.isfinite(y + h) or x + w > width or y + h > height):
        return None
    return (x, y, x + w, y + h)


def _ratios(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    if area_a <= 0 or area_b <= 0:
        return (0.0, 0.0, 0.0)
    union = area_a + area_b - intersection
    return (intersection / union if union else 0.0,
            intersection / area_a, intersection / area_b)


def _valid_sha(value):
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def compare_person_observation(candidate_bbox, *, frame_sha256, frame_size,
                               primary_model_sha256, secondary, min_confidence=0.25):
    """验证同帧、不同权重的空间交叉观察；只能作复核线索。

    secondary 必须形如 ``{status:'ok', frame_sha256, model_sha256,
    full_frame_covered:True, detections:[{class:'person', confidence, bbox:{x,y,w,h}}]}``。
    bbox 全部是原图像素坐标；若含分片检测，调用方必须先正确映射回原图。
    missing/no match 都不是“不是人”，matched 也不是“确认是人”。
    """
    out = {"observation": "unavailable", "reason": "invalid_input",
           "independence": "different_weights_do_not_prove_independent_errors",
           "needs_review": True, "automatic_disposition_permitted": False}

    def unavailable(reason):
        out["reason"] = reason
        return out

    if not isinstance(frame_size, dict):
        return unavailable("invalid_frame_size")
    width, height = _finite(frame_size.get("width")), _finite(frame_size.get("height"))
    threshold = _finite(min_confidence)
    if (not width or not height or width <= 0 or height <= 0 or width > 2 ** 31 - 1
            or height > 2 ** 31 - 1 or not width.is_integer() or not height.is_integer()
            or threshold is None or not 0 <= threshold <= 1):
        return unavailable("invalid_frame_size_or_threshold")
    target = _xywh(candidate_bbox, width, height)
    if not target:
        return unavailable("invalid_candidate_bbox")
    if not _valid_sha(frame_sha256) or not _valid_sha(primary_model_sha256):
        return unavailable("missing_provenance")
    if not isinstance(secondary, dict) or secondary.get("status") != "ok":
        return unavailable("secondary_not_available")
    other_frame = secondary.get("frame_sha256")
    other_model = secondary.get("model_sha256")
    if not _valid_sha(other_frame) or other_frame.lower() != frame_sha256.lower():
        return unavailable("frame_mismatch")
    if not _valid_sha(other_model):
        return unavailable("missing_secondary_model_provenance")
    if other_model.lower() == primary_model_sha256.lower():
        return unavailable("same_model_artifact")
    if secondary.get("full_frame_covered") is not True:
        return unavailable("incomplete_frame_coverage")
    detections = secondary.get("detections")
    if not isinstance(detections, list):
        return unavailable("invalid_detection_list")
    matched = []
    for item in detections:
        if not isinstance(item, dict):
            return unavailable("invalid_detection")
        if not isinstance(item.get("class"), str) or not item.get("class"):
            return unavailable("invalid_detection_class")
        if item.get("class") != "person":
            continue
        score = _finite(item.get("confidence"))
        box = _xywh(item.get("bbox"), width, height)
        if score is None or not 0 <= score <= 1 or box is None:
            return unavailable("invalid_person_detection")
        if score < threshold:
            continue
        iou, target_covered, detection_covered = _ratios(target, box)
        # 要求双向覆盖，避免一整个机械框只因覆盖小目标而被当作同一人。
        if iou >= 0.25 and target_covered >= 0.35 and detection_covered >= 0.35:
            matched.append((score, box, iou))
    matched.sort(key=lambda item: (-item[0], item[1]))
    # 同一对象在全图与多个切片中可能被重复检出；高度重合才视为同簇。
    groups = []
    for item in matched:
        if not any(_ratios(item[1], group[1])[0] >= 0.5 for group in groups):
            groups.append(item)
    out["secondary_model_sha256"] = other_model.lower()
    out["frame_sha256"] = frame_sha256.lower()
    out["candidate_bbox"] = dict(candidate_bbox)
    out["matched_groups"] = len(groups)
    if not groups:
        out.update(observation="no_match", reason="secondary_missed_or_different_object")
    elif len(groups) > 1:
        out.update(observation="ambiguous", reason="multiple_distinct_overlapping_person_candidates")
    else:
        out.update(observation="matched", reason="spatially_consistent_detector_candidates",
                   secondary_confidence=groups[0][0], iou=groups[0][2])
    return out


def attach_person_observation(event, observation):
    """离线回放工具：只给已留在主复核队列的候选加审计数据，不改处置或框。"""
    if (not isinstance(event, dict) or not isinstance(observation, dict)
            or observation.get("observation") not in ("unavailable", "no_match", "matched", "ambiguous")
            or observation.get("needs_review") is not True
            or observation.get("automatic_disposition_permitted") is not False):
        raise ValueError("event and a safe comparison observation are required")
    if (event.get("needs_review") is not True
            or event.get("review_disposition") in ("rejected", "box_relocalize", "dropped")):
        raise ValueError("existing review guard must retain the event first")
    reference = observation.get("candidate_bbox")
    if reference is not None and event.get("bbox") != reference:
        raise ValueError("observation bbox does not match the event bbox")
    event["person_cv_observation"] = dict(observation)
