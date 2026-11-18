"""Read coarse color cues in native helmet crops; never confirm safety or identity."""

import math
from numbers import Real

import cv2
import numpy as np

from core.helmet_evidence import _upper_candidate

_MIN_SIDE = 16
_MIN_HELMET_CONF = 0.45
_MIN_PERSON_CONF = 0.35
_MIN_DOMINANCE = 0.78
_MIN_QUADRANT = 0.65
_MAX_SAMPLE_SIDE = 160


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        v = float(value)
    except (OverflowError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _box_from(row, width, height):
    values = row.get("bbox")
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    bounds = tuple(_number(v) for v in values)
    if any(v is None for v in bounds):
        return None
    x1, y1, x2, y2 = bounds
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return None
    return (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))


def _box_payload(box):
    return dict(zip(("x1", "y1", "x2", "y2"), box)) if box is not None else None


def _size(box):
    return box[2] - box[0], box[3] - box[1]


def _local_head_box(child, parent):
    w, h = _size(child)
    pw, ph = _size(parent)
    # For a positive association, require overlap with the proposed person
    # rather than accepting an unattached object merely within the head margin.
    overlap_w = max(0, min(child[2], parent[2]) - max(child[0], parent[0]))
    overlap_h = max(0, min(child[3], parent[3]) - max(child[1], parent[1]))
    return (w <= 0.80 * pw and h <= 0.55 * ph and overlap_w >= 0.65 * w
            and overlap_h >= 0.30 * h and _upper_candidate(child, parent))


def _clipped(box, width, height):
    return box[0] <= 1 or box[1] <= 1 or box[2] >= width - 1 or box[3] >= height - 1


def _score_color(image, box):
    x1, y1, x2, y2 = box
    w, h = _size(box)
    # Use an inset patch rather than the whole object box. Pixel decimation is
    # intentionally bounded and never interpolates invented color pixels.
    ix, iy = max(1, w // 6), max(1, h // 6)
    step = max(1, math.ceil(max(w, h) / _MAX_SAMPLE_SIDE))
    crop = image[y1 + iy:y2 - iy:step, x1 + ix:x2 - ix:step]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = (hsv[:, :, i] for i in range(3))
    chromatic = (sat >= 125) & (val >= 110)
    palette = (
        ("red", chromatic & ((hue <= 6) | (hue >= 174))),
        ("orange", chromatic & (hue >= 10) & (hue <= 19)),
        ("yellow", chromatic & (hue >= 24) & (hue <= 38)),
        ("blue", chromatic & (hue >= 96) & (hue <= 130)),
        # Bright neutral, not merely unsaturated. Extensive clipping may be
        # glare/overexposure, so it never votes for white.
        ("white", (sat <= 24) & (val >= 226) & (val <= 250)),
    )
    for color, mask in palette:
        if np.mean(mask) < _MIN_DOMINANCE:
            continue
        cy, cx = (n // 2 for n in mask.shape)
        quadrants = (mask[:cy, :cx], mask[:cy, cx:], mask[cy:, :cx], mask[cy:, cx:])
        if all(np.mean(q) >= _MIN_QUADRANT for q in quadrants):
            return color
    return "unknown"


def observe_helmet_colors(frame_bgr, detections):
    """Return optional color observations for detector-proposed helmet boxes.

    Inputs must be a single unannotated native uint8 BGR frame and that same
    frame's ``YoloDetector.detect_tiled`` rows (xyxy boxes). No image is modified.
    The caller must bind the two inputs to the same frame; dimensions alone
    cannot authenticate provenance. A visually plausible color is neither
    wearer identity nor proof that anyone is wearing a helmet.
    """
    if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
            or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3
            or not isinstance(detections, (list, tuple))):
        return []
    height, width = frame_bgr.shape[:2]
    by_label = {key: [] for key in ("person", "helmet", "head")}
    for row in detections:
        if not isinstance(row, dict):
            continue
        label = row.get("label")
        if not isinstance(label, str) or label not in by_label:
            continue
        box = _box_from(row, width, height)
        if box is not None:
            conf = _number(row.get("conf"))
            if conf is not None and not 0 <= conf <= 1:
                conf = None
            by_label[label].append((box, conf))

    observations = []
    for helmet, conf in by_label["helmet"]:
        w, h = _size(helmet)
        possible_people = [(person, score) for person, score in by_label["person"]
                           if _local_head_box(helmet, person)]
        person, person_conf = possible_people[0] if len(possible_people) == 1 else (None, None)
        color = "unknown"
        if _clipped(helmet, width, height):
            reason = "helmet_at_frame_edge"
        elif min(w, h) < _MIN_SIDE or not 0.48 <= w / h <= 2.08:
            reason = "helmet_size_or_shape_unreliable"
        elif conf is None or conf < _MIN_HELMET_CONF:
            reason = "helmet_detector_confidence_unreliable"
        elif person is None:
            reason = "multiple_person_candidates" if possible_people else "no_linked_person_candidate"
        elif _clipped(person, width, height):
            reason = "person_at_frame_edge"
        elif person_conf is None or person_conf < _MIN_PERSON_CONF:
            reason = "person_detector_confidence_unreliable"
        elif sum(_local_head_box(box, person) for box, _ in by_label["helmet"]) != 1:
            reason = "multiple_helmet_candidates"
        elif any(_local_head_box(box, person) for box, _ in by_label["head"]):
            reason = "conflicting_head_candidate"
        else:
            color = _score_color(frame_bgr, helmet)
            reason = "no_clear_dominant_color" if color == "unknown" else "dominant_color_in_native_crop"
        observations.append({
            "helmet_bbox_px": _box_payload(helmet),
            "person_bbox_px": _box_payload(person),
            "color": color,
            "status": "unknown" if color == "unknown" else "visual_candidate",
            "reason": reason,
            "needs_review": True,
        })
    return observations
