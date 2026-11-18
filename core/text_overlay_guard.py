"""Opt-in text-overlap observation. This never decides a violation or creates a box."""

import hashlib
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from core.event_image_sync import verified_exact_original
from core.review_policy import hold_text_overlay_check


MODEL_SHA256 = 'd2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9'
MAX_MODEL_BYTES = 8 * 1024 * 1024
MAX_INPUT_PIXELS = 8_000_000
MIN_LONG_EDGE = 80
MIN_ASPECT = 5
OVERLAP_THRESHOLD = 0.5


class TextOverlayDetector:
    """Approximate local PP-OCRv4 DB postprocess; not official OCR or recognition."""

    def __init__(self, model_path):
        path = Path(model_path)
        if not path.is_file() or path.stat().st_size > MAX_MODEL_BYTES:
            raise ValueError('model_unavailable')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != MODEL_SHA256:
            raise ValueError('model_digest_mismatch')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self._session = ort.InferenceSession(raw, sess_options=options, providers=['CPUExecutionProvider'])

    def mask(self, image):
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('invalid_image')
        h, w = image.shape[:2]
        if h < 1 or w < 1:
            raise ValueError('invalid_image')
        ratio = max(1.0, 736.0 / min(h, w))
        rw = max(32, round(int(w * ratio) / 32) * 32)
        rh = max(32, round(int(h * ratio) / 32) * 32)
        if rw * rh > MAX_INPUT_PIXELS:
            raise ValueError('input_too_large')
        resized = cv2.resize(image, (rw, rh))
        tensor = ((resized.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        output = self._session.run(None, {self._session.get_inputs()[0].name: tensor})[0]
        if (output.ndim != 4 or output.shape[:2] != (1, 1)
                or output.shape[2:] != (rh, rw) or not np.isfinite(output).all()):
            raise ValueError('invalid_model_output')
        prediction = output[0, 0]
        if float(prediction.min()) < -0.0001 or float(prediction.max()) > 1.0001:
            raise ValueError('invalid_model_probability')
        binary = cv2.dilate((prediction > 0.3).astype(np.uint8), np.ones((2, 2), dtype=np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1000:
            raise ValueError('candidate_limit_exceeded')
        selected = []
        for contour in contours:
            center, sides, angle = cv2.minAreaRect(contour)
            a, b = sides
            if min(a, b) < 3 or a + b <= 0:
                continue
            points = cv2.boxPoints((center, sides, angle))
            x, y, bw, bh = cv2.boundingRect(points.astype(np.float32))
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(rw, x + bw), min(rh, y + bh)
            if x1 <= x0 or y1 <= y0:
                continue
            local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.fillPoly(local, [(points - [x0, y0]).astype(np.int32)], 1)
            if cv2.mean(prediction[y0:y1, x0:x1], local)[0] < 0.5:
                continue
            pad = (a * b * 1.6) / (2.0 * (a + b))
            expanded = cv2.boxPoints((center, (a + 2 * pad, b + 2 * pad), angle))
            expanded[:, 0] = np.clip(np.round(expanded[:, 0] * w / rw), 0, w - 1)
            expanded[:, 1] = np.clip(np.round(expanded[:, 1] * h / rh), 0, h - 1)
            short, long = sorted(cv2.minAreaRect(expanded.astype(np.float32))[1])
            if short >= 4 and long >= MIN_LONG_EDGE and long / short >= MIN_ASPECT:
                selected.append(expanded.astype(np.int32))
        mask = np.zeros((h, w), dtype=np.uint8)
        if selected:
            cv2.fillPoly(mask, selected, 1)
        return mask


def _box_region(box, shape):
    if not isinstance(box, dict):
        return None
    try:
        values = [box[k] for k in ('x', 'y', 'w', 'h')]
        if any(isinstance(v, bool) for v in values):
            return None
        x, y, width, height = (float(v) for v in values)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    h, w = shape
    if (not all(math.isfinite(v) for v in (x, y, width, height)) or width <= 0 or height <= 0
            or x < 0 or y < 0 or x + width > w or y + height > h):
        return None
    return math.floor(x), math.floor(y), math.ceil(x + width), math.ceil(y + height)


def inspect_text_overlap(events, frame_results, output_dir, detector):
    """Mark suspicious or uncheckable candidates; preserve event count and original boxes.

    ``detector.mask`` receives an already verified exact source frame and returns a
    binary mask of the same size. No-overlap is only an observation, not a verdict.
    """
    stats = Counter()
    cache = {}
    for event in events or []:
        if not isinstance(event, dict):
            continue
        if event.get('scene_level') and not event.get('bbox'):
            hold_text_overlay_check(event, 'scene_without_box')
            stats['scene_without_box'] += 1
            continue
        if detector is None:
            hold_text_overlay_check(event, 'detector_unavailable')
            stats['unavailable'] += 1
            continue
        frame, image, status = verified_exact_original(event, frame_results, output_dir)
        if status != 'ok' or image is None:
            hold_text_overlay_check(event, 'frame_unverified')
            stats['frame_unverified'] += 1
            continue
        box = _box_region(event.get('bbox'), image.shape[:2])
        if box is None:
            hold_text_overlay_check(event, 'box_unverified')
            stats['box_unverified'] += 1
            continue
        digest = frame['original_frame_sha256']
        if digest not in cache:
            try:
                mask = detector.mask(image)
                if (not isinstance(mask, np.ndarray) or mask.shape != image.shape[:2]
                        or not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all()):
                    raise ValueError('invalid_mask')
                cache[digest] = mask
            except Exception:
                cache[digest] = None
        mask = cache[digest]
        if mask is None:
            hold_text_overlay_check(event, 'detector_failed')
            stats['detector_failed'] += 1
            continue
        x1, y1, x2, y2 = box
        overlap = float(mask[y1:y2, x1:x2].mean())
        event['text_overlay_overlap'] = round(overlap, 6)
        event['text_overlay_frame_sha256'] = digest
        if overlap >= OVERLAP_THRESHOLD:
            hold_text_overlay_check(event, 'suspected_text_overlap')
            stats['suspected'] += 1
        else:
            event['text_overlay_check_status'] = 'no_overlap_observed'
            stats['no_overlap_observed'] += 1
    return dict(stats)


def run_configured_guard(report, frame_results, output_dir, config):
    """Disabled by default. An unavailable enabled detector fails closed."""
    if config.get('text_overlay_guard_enabled', False) is not True:
        return
    try:
        detector = TextOverlayDetector(config.get('text_overlay_model_path') or '')
    except Exception:
        detector = None
    report['text_overlay_guard_stats'] = inspect_text_overlap(
        report.get('events') or [], frame_results, output_dir, detector)
