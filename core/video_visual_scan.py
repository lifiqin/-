"""Local representative-video scan; intentionally isolated from violation reports."""

import hashlib
import math
from pathlib import Path

import cv2
import numpy as np


MAX_SCAN_SECONDS = 30
MAX_SAMPLE_HZ = 5
_MAX_PIXELS = 4096 * 2304


class VisualScanError(RuntimeError):
    """An unavailable source, decoder, or detector, not an empty finding."""


def describe_frame(frame, frame_index, fps):
    height, width = frame.shape[:2]
    return {
        "frame_index": frame_index,
        "timestamp": frame_index / fps,
        "width": width,
        "height": height,
        "bgr8_sha256": hashlib.sha256(np.ascontiguousarray(frame)).hexdigest(),
    }


def _validate_number(value, label, minimum, maximum):
    if (type(value) not in (int, float) or not minimum <= value <= maximum):
        raise ValueError(f"无效的{label}（允许范围 {minimum}～{maximum}）")


def _capture(path):
    if not path.is_file():
        raise VisualScanError("源视频文件不存在")
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or not 0.5 <= fps <= 240 or not 1 <= total <= 10**9 or int(total) != total:
            raise VisualScanError("无法打开视频或无法核对帧率和帧数")
        return cap, fps, int(total)
    except BaseException:
        cap.release()
        raise


def _verify_next(cap, expected):
    actual = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if not math.isfinite(actual) or abs(actual - expected) > 0.25:
        raise VisualScanError("解码器无法核对原视频帧位置")


def _seek(cap, index, total):
    if not 0 <= index < total or not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
        raise VisualScanError("请求的时刻不在可读取的视频范围内")
    _verify_next(cap, index)


def _read(cap, index):
    ok, frame = cap.read()
    _verify_next(cap, index + 1)
    if (not ok or frame is None or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3
            or not 0 < frame.shape[0] * frame.shape[1] <= _MAX_PIXELS):
        raise VisualScanError("该原视频帧不可解码或分辨率超过扫描上限")
    return frame


def _file_stamp(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev


def read_reference_frame(video, start=0.0):
    """Return the exact comparison image and binding metadata for zone authoring."""
    _validate_number(start, "起始秒数", 0, 24 * 3600)
    path = Path(video)
    stamp = _file_stamp(path)
    cap, fps, total = _capture(path)
    try:
        index = round(start * fps)
        _seek(cap, index, total)
        image = _read(cap, index)
        if stamp != _file_stamp(path):
            raise VisualScanError("读取期间源视频发生变化")
        return image, {"source_name": path.name, **describe_frame(image, index, fps)}
    finally:
        cap.release()


def _export_temporal_evidence(cap, fps, total, frames, events, reference, sink, limit):
    """Re-read selected frames by index and pixel hash, with a fixed memory bound.

    Give representative frames priority so early context cannot exhaust the
    budget before a later candidate appears. Never use a similar/nearby frame.
    """
    by_index = {row["frame_index"]: row for row in frames}
    ordered = dict.fromkeys([
        *(event["representative_frame"] for event in events),
        *(event_frame for event in events for event_frame in event["observation_frames"]),
        *([reference["frame_index"]] if reference is not None else []),
        *(row["frame_index"] for row in frames
          if any(item["status"] == "visual_candidate" for item in row["helmet_colors"])),
        *(row["frame_index"] for row in frames if row.get("person_crosschecks")),
        *by_index,
    ])
    for index in list(ordered)[:limit]:
        current = by_index[index]
        _seek(cap, index, total)
        image = _read(cap, index)
        meta = describe_frame(image, index, fps)
        if any(meta[key] != current[key] for key in ("bgr8_sha256", "width", "height", "timestamp")):
            raise VisualScanError("采样帧二次解码与原观测像素不一致；未生成可用报告")
        current["evidence"] = sink(meta, image)
    for event in events:
        row = by_index[event["representative_frame"]]
        event["source_frame_evidence"] = row["evidence"]
        event["evidence_frame_index"] = row["frame_index"] if row["evidence"] is not None else None
    return by_index[reference["frame_index"]]["evidence"] if reference is not None else None


class FixedViewGuard:
    """Reject visible drift/low support against ONE reference, never re-anchor.

    Checks distributed background-like features in the image plane. It does
    not recover motion, identify foreground objects, or prove a stationary UAV.
    """

    _WIDTH = 480
    _MARGIN = 12
    _CELL_MIN = 5
    _WINDOW = (21, 21)
    _PYRAMID_LEVEL = 1  # Search only local drift; coarse tracking spreads foreground changes.

    @classmethod
    def _gray(cls, bgr):
        h, w = bgr.shape[:2]
        dims = (cls._WIDTH, max(1, round(h * cls._WIDTH / w)))
        if min(h, w) < 64 or not 100 <= dims[1] <= 1000:
            return None
        return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), dims, interpolation=cv2.INTER_AREA)

    def __init__(self, reference_bgr, zones=()):
        self.shape = reference_bgr.shape
        self._ref = self._gray(reference_bgr)
        self._points = None
        # All cells touched by a zone or its immediate neighborhood are
        # mandatory. A temporarily covered, unrelated cell must not blind a
        # well-supported local zone; we still require three distributed cells.
        self._required = set(range(4)) if not zones else set()
        for zone in zones:
            xs, ys = zip(*zone["polygon"])
            for cell in range(4):
                left, top = (cell % 2) / 2, (cell // 2) / 2
                if (min(xs) <= left + 0.52 and max(xs) >= left - 0.02
                        and min(ys) <= top + 0.52 and max(ys) >= top - 0.02):
                    self._required.add(cell)
        if self._ref is None:
            return
        h, w = self._ref.shape
        points, cells = [], []
        for cell in range(4):
            left, top = (cell % 2) * (w // 2), (cell // 2) * (h // 2)
            x1, y1 = left + self._MARGIN, top + self._MARGIN
            x2, y2 = left + w // 2 - self._MARGIN, top + h // 2 - self._MARGIN
            tile = self._ref[y1:y2, x1:x2]
            corners = cv2.goodFeaturesToTrack(tile, maxCorners=50, qualityLevel=0.02, minDistance=8)
            if corners is None or len(corners) < self._CELL_MIN:
                continue
            points.extend(corners + np.array([x1, y1], dtype=np.float32))
            cells.extend([cell] * len(corners))
        textured = set(cells)
        if len(textured) < 3 or not self._required <= textured:
            return
        self._points = np.asarray(points, dtype=np.float32)
        self._cells = np.asarray(cells)

    def assess(self, current_bgr):
        result = {"usable": False, "reason": "insufficient_reference_texture", "support": [],
                  "supported_cells": [], "required_cells": sorted(self._required)}
        if self._points is None:
            return result
        result["reason"] = "image_dimensions_changed"
        if current_bgr.shape != self.shape:
            return result
        current = self._gray(current_bgr)
        result["reason"] = "insufficient_motion_support"
        if current is None:
            return result
        try:
            moved, ok, error = cv2.calcOpticalFlowPyrLK(
                self._ref, current, self._points, None,
                winSize=self._WINDOW, maxLevel=self._PYRAMID_LEVEL)
            if moved is None or ok is None or error is None or not np.isfinite(moved).all():
                return result
            back, reverse_ok, _ = cv2.calcOpticalFlowPyrLK(
                current, self._ref, moved, None,
                winSize=self._WINDOW, maxLevel=self._PYRAMID_LEVEL)
            if back is None or reverse_ok is None:
                return result
        except cv2.error:
            return result
        reference_points = self._points.reshape(-1, 2)
        moved = moved.reshape(-1, 2)
        displacement = moved - reference_points
        consistency = np.linalg.norm(back.reshape(-1, 2) - reference_points, axis=1)
        good = (ok.ravel() != 0) & (reverse_ok.ravel() != 0) & (consistency <= 0.6) & (error.ravel() <= 14)
        supports = [int(np.count_nonzero(good & (self._cells == i))) for i in range(4)]
        result["support"] = supports
        supported = [i for i, count in enumerate(supports)
                     if count >= max(self._CELL_MIN, math.ceil(np.count_nonzero(self._cells == i) * 0.45))]
        result["supported_cells"] = supported
        if len(supported) < 3 or not self._required <= set(supported):
            return result
        cell_shift = max(np.linalg.norm(np.median(displacement[good & (self._cells == i)], axis=0))
                         for i in supported)
        p80 = np.percentile(np.linalg.norm(displacement[good & np.isin(self._cells, supported)], axis=1), 80)
        result.update(max_cell_shift_480px=round(float(cell_shift), 3),
                      p80_drift_480px=round(float(p80), 3))
        if cell_shift > 1.0 or p80 > 1.6:
            result["reason"] = "reference_view_drift"
            return result
        result.update(usable=True, reason="consistent_with_reference_view")
        return result


def _people(detections):
    return [
        {"type": "person", "conf": row.get("conf"),
         "bbox_px": dict(zip(("x1", "y1", "x2", "y2"), row["bbox"]))}
        for row in detections if isinstance(row, dict) and row.get("label") == "person"
        and isinstance(row.get("bbox"), (tuple, list)) and len(row["bbox"]) == 4
    ]


def _person_crosscheck_boxes(people, width, height):
    """Use only unique original in-frame geometry; never invent/clamp a box."""
    boxes = {}
    invalid = 0
    for person in people:
        raw = person["bbox_px"]
        coords = tuple(raw[key] for key in ("x1", "y1", "x2", "y2"))
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in coords):
            invalid += 1
            continue
        x, y, x2, y2 = coords
        if not (0 <= x < x2 <= width and 0 <= y < y2 <= height):
            invalid += 1
            continue
        boxes.setdefault(coords, dict(x=x, y=y, w=x2 - x, h=y2 - y))
    return [boxes[key] for key in sorted(boxes)], invalid


def _people_in_supported_view(people, view, width, height):
    def eligible(person):
        b = person["bbox_px"]
        box = [b[key] for key in ("x1", "y1", "x2", "y2")]
        if any(type(coord) not in (float, int) or not math.isfinite(coord) for coord in box):
            return False
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        cell = int(cx >= width / 2) + 2 * int(cy >= height / 2)
        return cell in view["supported_cells"]

    return [person for person in people if eligible(person)] if view["usable"] else []


def _local_detector():
    from core.yolo_detector import DEFAULT_MODEL_NAME, YoloDetector

    model_path = Path(__file__).resolve().parents[1] / "models" / DEFAULT_MODEL_NAME
    detector = YoloDetector(model_path=str(model_path), offline_cpu=True)
    if not detector.available:
        raise VisualScanError("未能加载本地安全帽／人员模型")
    with model_path.open("rb") as model:
        return detector, hashlib.file_digest(model, "sha256").hexdigest()


def scan_video(video, *, zones=(), reference=None, start=0.0, seconds=10.0, sample_hz=None,
               helmet_colors=False, evidence_sink=None, zone_association="iou",
               secondary_person_model=None, temporal_evidence=False, temporal_frame_limit=24):
    """Analyze local frames without loading credentials, APIs or violation state.

    Zone coordinates are accepted only with an exact reference binding.
    The optional evidence sink stores unmarked frames for review.
    Temporal evidence is re-decoded after detection, with representative priority
    and exact pixel verification, so early samples cannot consume the whole budget.
    """
    from core.helmet_color import observe_helmet_colors
    from core.zone_presence import ZoneAssociationError, ZonePresence

    _validate_number(start, "起始秒数", 0, 24 * 3600)
    _validate_number(seconds, "扫描时长", 0.1, MAX_SCAN_SECONDS)
    if type(temporal_evidence) is not bool:
        raise ValueError("temporal_evidence 必须为布尔值")
    if temporal_evidence and (type(temporal_frame_limit) is not int or not 0 <= temporal_frame_limit <= 40):
        raise ValueError("时序原帧保存上限必须为 0～40")
    if zone_association not in ("iou", "bytetrack"):
        raise ValueError("区域关联方式必须是 iou 或 bytetrack")
    if zone_association == "bytetrack" and not zones:
        raise ValueError("ByteTrack 试用必须提供已画区域")
    if sample_hz is None:
        sample_hz = 5.0 if zone_association == "bytetrack" else 3.0
    _validate_number(sample_hz, "抽样频率", 1, MAX_SAMPLE_HZ)
    if not zones and not helmet_colors and secondary_person_model is None:
        raise ValueError("请选择区域人员检查、安全帽颜色观察或显式人员二检")
    if zones and not isinstance(reference, dict):
        raise ValueError("区域检查必须绑定手工标注所用的原视频参考帧")
    region_options = {"record_observation_bboxes": True} if temporal_evidence else {}
    region = ZonePresence(zones, **region_options) if zones else None
    path = Path(video)
    stamp = _file_stamp(path)
    cap, fps, total = _capture(path)
    crosschecker = None
    try:
        first = round(start * fps)
        requested_stop = first + max(1, math.ceil(seconds * fps))
        stop = min(total, requested_stop)
        step = max(1, math.ceil(fps / sample_hz))
        _seek(cap, first, total)
        anchor = _read(cap, first)
        ref_frame = {"source_name": path.name, **describe_frame(anchor, first, fps)}
        if region and any(reference.get(k) != ref_frame[k]
                          for k in ("source_name", "frame_index", "width", "height", "bgr8_sha256")):
            raise ValueError("区域文件不匹配当前视频和起始帧，请重新生成区域参考页")
        guard = FixedViewGuard(anchor, zones) if region else None
        if region and step / fps > 1.25:
            raise ValueError("实际抽样间隔过长，无法形成连续区域观察")
        associator = None
        if zone_association == "bytetrack":
            from core.zone_bytetrack import ByteTrackAssociator

            try:
                associator = ByteTrackAssociator(fps / step)
            except ZoneAssociationError as exc:
                raise VisualScanError(str(exc)) from exc
            region = ZonePresence(zones, observed_associator=associator, **region_options)
        detector, model_sha = _local_detector()
        if secondary_person_model is not None:
            from core.person_crosscheck import PersonCrosschecker

            try:
                crosschecker = PersonCrosschecker(secondary_person_model, model_sha)
            except Exception as exc:
                raise VisualScanError("人员二检模型／依赖加载失败；请检查已有本地 .pt 权重，未自动下载或回退") from exc
        frames, events, failed = [], [], 0
        crosscheck_failed = 0
        previous_digest = None
        reference_evidence = evidence_sink(ref_frame, anchor) if region and evidence_sink and not temporal_evidence else None
        for index in range(first, stop):
            if index == first:
                frame = anchor
            elif (index - first) % step == 0:
                frame = _read(cap, index)
            else:
                if not cap.grab():
                    raise VisualScanError("扫描窗口内视频不能完整解码")
                continue
            meta = describe_frame(frame, index, fps)
            view = guard.assess(frame) if guard else None
            if view and view["usable"] and meta["bgr8_sha256"] == previous_digest:
                # Repeated decoded images are not additional temporal evidence.
                view.update(usable=False, reason="duplicate_image_content")
            previous_digest = meta["bgr8_sha256"]
            detections = []
            attempted = helmet_colors or bool(view and view["usable"]) or (crosschecker is not None and not region)
            detection_status = "not_run_unstable_view"
            if attempted:
                try:
                    detections = detector.detect_tiled(frame)
                    if not isinstance(detections, list):
                        raise TypeError("invalid local detector response")
                    detection_status = "ok"
                except Exception:
                    detection_status = "detector_failed"
                    failed += 1
            people = _people(detections)
            crosschecks = []
            crosscheck_status = "not_run_primary_unavailable"
            invalid_boxes = 0
            if crosschecker is not None and detection_status == "ok":
                crosscheck_status = "not_run_no_primary_candidates"
                boxes, invalid_boxes = _person_crosscheck_boxes(people, meta["width"], meta["height"])
                if boxes:
                    try:
                        crosschecks = crosschecker.observe(frame, boxes)
                        if (not isinstance(crosschecks, list) or len(crosschecks) != len(boxes)
                                or any(not isinstance(row, dict) or row.get("candidate_bbox") != box
                                       or row.get("frame_sha256") != meta["bgr8_sha256"]
                                       or row.get("observation") not in ("matched", "no_match", "ambiguous", "unavailable")
                                       or row.get("needs_review") is not True
                                       or row.get("automatic_disposition_permitted") is not False
                                       for box, row in zip(boxes, crosschecks, strict=True))):
                            raise ValueError("invalid crosscheck observation")
                    except Exception:
                        crosschecks = [{"candidate_bbox": dict(box), "frame_sha256": meta["bgr8_sha256"],
                                        "observation": "unavailable", "reason": "secondary_observation_failed",
                                        "needs_review": True, "automatic_disposition_permitted": False}
                                       for box in boxes]
                    crosscheck_status = ("unavailable" if any(row["observation"] == "unavailable" for row in crosschecks)
                                         else "ok")
                if invalid_boxes:
                    crosscheck_status = "unavailable"
                crosscheck_failed += crosscheck_status == "unavailable"
            if region:
                stable_people = _people_in_supported_view(people, view, meta["width"], meta["height"])
                candidates = region.update(meta["timestamp"], index, (meta["width"], meta["height"]), stable_people,
                                           frame_usable=view["usable"] and detection_status == "ok")
            else:
                candidates = []
            colors = observe_helmet_colors(frame, detections) if helmet_colors and detection_status == "ok" else []
            loc = None
            if not temporal_evidence and (candidates or any(o["status"] == "visual_candidate" for o in colors)
                                          or crosschecks):
                if index == first and reference_evidence is not None:
                    loc = reference_evidence
                elif evidence_sink:
                    loc = evidence_sink(meta, frame)
            for event in candidates:
                if event["representative_frame"] != index:
                    raise VisualScanError("候选帧与当前证据帧不一致，停止导出")
                if crosschecker is not None:
                    matching = [row for row in crosschecks if row["candidate_bbox"] == event["bbox"]]
                    event["person_crosscheck"] = (dict(matching[0]) if len(matching) == 1 else {
                        "candidate_bbox": dict(event["bbox"]), "frame_sha256": meta["bgr8_sha256"],
                        "observation": "unavailable", "reason": "primary_bbox_not_unique",
                        "needs_review": True, "automatic_disposition_permitted": False})
                events.append({**event, "source_frame_evidence": loc,
                               "evidence_frame_index": index if loc is not None else None})
            zone_frame = region.diagnostics["last_frame"] if region else None
            frames.append({**meta, "view": view, "detection_status": detection_status, "zone_frame": zone_frame,
                           "person_detection_count": len(people), "helmet_colors": colors, "evidence": loc})
            if crosschecker is not None:
                frames[-1].update(person_crosschecks=crosschecks, person_crosscheck_status=crosscheck_status,
                                  person_crosscheck_invalid_bbox_count=invalid_boxes)
        if temporal_evidence and evidence_sink:
            reference_evidence = _export_temporal_evidence(
                cap, fps, total, frames, events, ref_frame if region else None, evidence_sink, temporal_frame_limit)
        if stamp != _file_stamp(path):
            raise VisualScanError("扫描期间源视频发生变化；本次结果不可用")
        usable = sum(bool(f["zone_frame"] and f["zone_frame"]["status"] == "ok") for f in frames)
        result = {
            "schema_version": 1, "mode": "offline_visual_candidates", "api_calls": 0,
            "source_name": path.name, "time_basis": "nominal_frame_index_over_fps_not_capture_time",
            "source_fps": fps, "sample_step_frames": step, "sample_hz": fps / step,
            "reference": ref_frame, "reference_evidence": reference_evidence,
            "window": {"start_frame": first, "stop_frame_exclusive": stop,
                       "truncated_by_video_end": stop < requested_stop},
            "detector_model_sha256": model_sha, "source_file_stat_rechecked": True,
            "zones": list(zones), "zone_usable_sample_count": usable if region else None,
            "zone_diagnostics": region.diagnostics if region else None,
            "helmet_colors_requested": bool(helmet_colors), "detector_failure_count": failed,
            "status": ("partial" if failed or crosscheck_failed or stop < requested_stop
                       or (region and (usable < len(frames) or usable < 3))
                       else "sampled_only"),
            "absence_is_not_safety": True, "frames": frames, "zone_candidates": events,
        }
        if associator is not None:
            result["zone_association"] = associator.report_metadata
        if crosschecker is not None:
            result["person_crosscheck_metadata"] = crosschecker.report_metadata
            result["person_crosscheck_failure_count"] = crosscheck_failed
        if temporal_evidence:
            result["temporal_evidence_requested"] = True
        return result
    finally:
        if crosschecker is not None:
            crosschecker.close()
        cap.release()
