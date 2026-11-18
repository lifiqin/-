"""用户画定图像区域中的疑似人员进入/出现线索。无网络、无相机标定。"""

import copy
import math
from dataclasses import dataclass, field
from numbers import Real

import cv2
import numpy as np

_SCOPE = "user_polygon_image_center_sampled_frames"
_EPS = 1e-10
_MAX_ENTITIES = 256


class ZoneAssociationError(RuntimeError):
    """A requested association step is unavailable; it is not an empty detection."""


def _finite(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _orient(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _on_segment(a, b, p):
    return all(min(a[i], b[i]) - _EPS <= p[i] <= max(a[i], b[i]) + _EPS for i in (0, 1))


def _intersects(a, b, c, d):
    ac, ad, ca, cb = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    if ((ac > _EPS and ad < -_EPS or ad > _EPS and ac < -_EPS)
            and (ca > _EPS and cb < -_EPS or cb > _EPS and ca < -_EPS)):
        return True
    return any(abs(v) <= _EPS and _on_segment(x, y, p) for v, x, y, p in (
        (ac, a, b, c), (ad, a, b, d), (ca, c, d, a), (cb, c, d, b)))


def _load_zones(zones):
    if not isinstance(zones, (tuple, list)) or not 1 <= len(zones) <= 16:
        raise ValueError("必须提供 1～16 个手绘区域")
    parsed = []
    for i, zone in enumerate(zones, 1):
        name = zone.get("name") if isinstance(zone, dict) else None
        polygon = zone.get("polygon") if isinstance(zone, dict) else None
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80 or not name.isprintable():
            raise ValueError(f"区域 {i} 名称必须为 1～80 个可显示字符")
        if not isinstance(polygon, (list, tuple)) or not 3 <= len(polygon) <= 64:
            raise ValueError(f"区域 {i} 必须有 3～64 个多边形顶点")
        points = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"区域 {i} 顶点必须为 [x, y]")
            xy = tuple(_finite(v) for v in point)
            if any(v is None or not 0 <= v <= 1 for v in xy):
                raise ValueError(f"区域 {i} 顶点必须为 0～1 的有限数值")
            points.append(xy)
        n = len(points)
        edges = list(zip(points, points[1:] + points[:1]))
        if len(set(points)) != n:
            raise ValueError(f"区域 {i} 有重复顶点")
        area2 = sum(a[0] * b[1] - a[1] * b[0] for a, b in edges)
        if abs(area2) < 2e-6:
            raise ValueError(f"区域 {i} 多边形面积不足或退化")
        for j, (a, b) in enumerate(edges):
            c = points[(j + 2) % n]
            if abs(_orient(a, b, c)) <= _EPS and (_on_segment(a, b, c) or _on_segment(b, c, a)):
                raise ValueError(f"区域 {i} 多边形边回折")
            for k in range(j + 2, n):
                if (j != 0 or k != n - 1) and _intersects(a, b, *edges[k]):
                    raise ValueError(f"区域 {i} 多边形自交")
        parsed.append((name.strip(), tuple(points)))
    return tuple(parsed)


def _iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    aa, ab = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (aa + ab - overlap)


def _person_box(entity, width, height, min_confidence):
    if entity.get("type") != "person":
        return None
    conf = _finite(entity.get("conf"))
    box = entity.get("bbox_px")
    if conf is None or not min_confidence <= conf <= 1 or not isinstance(box, dict):
        return None
    xyxy = tuple(_finite(box.get(key)) for key in ("x1", "y1", "x2", "y2"))
    if any(v is None for v in xyxy):
        return None
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    if (not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)
            or min(w, h) < 5 or w * h > 0.20 * width * height):
        return None
    return xyxy


@dataclass
class _ZoneState:
    outside: list = field(default_factory=list)
    inside: list = field(default_factory=list)
    # 只允许短暂经过边界不确定带，不把很久以前的区外证据接到区内。
    outside_anchor: tuple = ()
    emitted: bool = False


@dataclass
class _Track:
    tid: int
    bbox: tuple
    zones: list
    observed_id: int | None = None


class ZonePresence:
    """有界、确定性的逐帧状态机。一个实例只接一段顺序图像。

    适配器必须把不可信的相机位姿/镜头变化/解码标成 frame_usable=False。
    只看图像中心，不确认姓名、工作职责、物理边界、越栏动作或违章。
    """

    def __init__(self, zones, *, min_confidence=0.30, min_inside_frames=3, min_outside_frames=2,
                 max_gap_seconds=1.25, boundary_margin=0.0125, observed_associator=None,
                 record_observation_bboxes=False):
        self._zones = _load_zones(zones)
        if type(record_observation_bboxes) is not bool:
            raise ValueError("record_observation_bboxes 必须是布尔值")
        self._record_observation_bboxes = record_observation_bboxes
        conf, gap, margin = map(_finite, (min_confidence, max_gap_seconds, boundary_margin))
        if conf is None or not 0 < conf <= 1:
            raise ValueError("min_confidence 必须在 (0, 1]")
        if gap is None or not 0 < gap <= 5:
            raise ValueError("max_gap_seconds 必须在 (0, 5]")
        if margin is None or not 0 < margin <= 0.2:
            raise ValueError("boundary_margin 必须在 (0, 0.2]")
        for n in (min_inside_frames, min_outside_frames):
            if type(n) is not int or not 2 <= n <= 12:
                raise ValueError("区域连续帧要求必须为 2～12 的整数")
        self._min_conf = conf
        self._min_in = min_inside_frames
        self._min_out = min_outside_frames
        self._gap = gap
        self._margin = margin
        self._size = None
        self._contours = ()
        self._timestamp = None
        self._frame_index = None
        self._tracks = []
        self._next_id = 1
        self._observed_associator = observed_associator
        self._association_state = (observed_associator.association_state if observed_associator is not None
                                   else "unique_adjacent_image_iou_only")
        self._diag = dict(scope=_SCOPE, zone_count=len(self._zones), frame_count=0, usable_frame_count=0,
                          unusable_frame_count=0, no_valid_person_frame_count=0,
                          accepted_person_box_count=0, rejected_person_box_count=0,
                          ambiguous_association_count=0, boundary_observation_count=0,
                          entry_candidate_count=0, presence_candidate_count=0, resets={},
                          last_frame=None, absence_is_not_safety=True)
        if observed_associator is not None:
            self._diag["unassigned_observed_box_count"] = 0

    @property
    def diagnostics(self):
        """返回独立、JSON 可序列化的逐帧最新状态和累计诊断。"""
        return copy.deepcopy(self._diag)

    def _break(self, reason):
        self._tracks = []
        if self._observed_associator is not None:
            self._observed_associator.reset()
        reasons = self._diag["resets"]
        reasons[reason] = reasons.get(reason, 0) + 1

    def _reject(self, reason, timestamp, frame_index):
        self._break(reason)
        self._diag["unusable_frame_count"] += 1
        self._diag["last_frame"] = dict(status=reason, timestamp=timestamp, frame_index=frame_index,
                                        accepted_person_count=0, candidate_count=0)
        return []

    def _associate(self, boxes):
        """仅把互为唯一的可信 IoU 重叠当同一图像目标。丢帧或歧义不续链。"""
        previous = self._tracks
        eligible = []
        links = [[] for _ in previous]
        for j, box in enumerate(boxes):
            options = [i for i, track in enumerate(previous) if _iou(track.bbox, box) >= 0.30]
            eligible.append(options)
            for i in options:
                links[i].append(j)
        tracks = []
        for box, options in zip(boxes, eligible):
            if len(options) == 1 and len(links[options[0]]) == 1:
                track = previous[options[0]]
                track.bbox = box
            else:
                if options:
                    self._diag["ambiguous_association_count"] += 1
                track = _Track(self._next_id, box, [_ZoneState() for _ in self._zones])
                self._next_id += 1
            tracks.append(track)
        self._tracks = tracks

    def _associate_observed(self, boxes, timestamp):
        """IDs may link current raw boxes to the immediately preceding observation only."""
        assignments = self._observed_associator.associate(boxes, timestamp)
        if (not isinstance(assignments, dict) or any(box not in boxes or type(tid) is not int or tid < 0
                                                  for box, tid in assignments.items())
                or len(set(assignments.values())) != len(assignments)):
            raise ZoneAssociationError("跟踪器未返回一对一的本帧观测框")
        previous = {track.observed_id: track for track in self._tracks}
        tracks = []
        for box in sorted(boxes):
            observed_id = assignments.get(box)
            if observed_id is None:
                continue
            track = previous.get(observed_id)
            if track is None:
                track = _Track(self._next_id, box, [_ZoneState() for _ in self._zones], observed_id)
                self._next_id += 1
            else:
                track.bbox = box
            tracks.append(track)
        self._tracks = tracks
        self._diag["unassigned_observed_box_count"] += len(boxes) - len(assignments)

    def _candidate(self, track, zi, state):
        outside = state.outside_anchor
        current = state.inside[-1]
        evidence = (*outside, *state.inside)
        x1, y1, x2, y2 = track.bbox
        name = self._zones[zi][0]
        entry = bool(outside)
        kind = "zone_entry" if entry else "zone_presence"
        experimental = self._observed_associator is not None
        self._diag["entry_candidate_count" if entry else "presence_candidate_count"] += 1
        description = ((("ByteTrack 在相邻可用采样中关联了多个疑似人框，" if experimental
                         else "与上一采样框可唯一重叠关联的疑似人员，")
                        + f"框中心先在手绘区域「{name}」外、后在其内。")
                       if entry else f"多个相邻有效采样帧在手绘区域「{name}」内观察到疑似人员框中心，未见其从区外进入。")
        # Both association methods can track a false positive such as a pile.
        title = "待核：疑似人员进入划定区域" if entry else "待核：区域内疑似人员框"
        event = dict(kind=kind, zone_index=zi, zone_name=name,
                    title=title,
                    description=description + "仅为二维图像线索，不确认身份、实际入侵或跨越围栏。",
                    review_reason="请回看连续原视频，核对人员、区域位置与进出过程；不得仅凭图像框或未检出认定现场情况。",
                    first_seen=evidence[0][0], last_seen=current[0], representative_timestamp=current[0],
                    representative_frame=current[1], bbox=dict(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
                    observation_times=[item[0] for item in evidence],
                    observation_frames=[item[1] for item in evidence],
                    observation_regions=["outside"] * len(outside) + ["inside"] * len(state.inside),
                    person_track_id=track.tid, anchor="person_box_image_center", scope=_SCOPE,
                    association_state=self._association_state, needs_review=True,
                    motion_confirmed=False, intrusion_confirmed=False, camera_motion_compensated=False,
                    identity_independently_verified=False, physical_boundary_verified=False,
                    absence_is_not_safety=True)
        if experimental or self._record_observation_bboxes:
            event["observation_bboxes"] = [
                dict(x=a, y=b, w=c - a, h=d - b) for _, _, (a, b, c, d) in evidence
            ]
        return event

    def _observe_zone(self, track, zi, ts, index, margin):
        state = track.zones[zi]
        x1, y1, x2, y2 = track.bbox
        point = ((x1 + x2) / 2, (y1 + y2) / 2)
        d = cv2.pointPolygonTest(self._contours[zi], point, True)
        # 除全图尺度的缓冲区外，保留一小段随框尺寸变化的定位余量。
        threshold = max(margin, 0.20 * min(x2 - x1, y2 - y1))
        sample = (ts, index, track.bbox)
        if d < -threshold:
            state.inside.clear()
            state.outside.append(sample)
            del state.outside[:-self._min_out]
            if len(state.outside) == self._min_out:
                state.outside_anchor = tuple(state.outside)
                state.emitted = False
            return None
        state.outside.clear()
        if abs(d) <= threshold:
            self._diag["boundary_observation_count"] += 1
            state.inside.clear()
            if state.outside_anchor and ts - state.outside_anchor[-1][0] > 2 * self._gap:
                state.outside_anchor = ()
            return None
        if state.emitted:
            return None
        if (not state.inside and state.outside_anchor
                and ts - state.outside_anchor[-1][0] > 2 * self._gap):
            state.outside_anchor = ()
        state.inside.append(sample)
        if len(state.inside) < self._min_in:
            return None
        state.emitted = True
        return self._candidate(track, zi, state)

    def update(self, timestamp, frame_index, frame_size, entities, *, frame_usable=True):
        """消费一次采样；`frame_size=(宽, 高)`；返回本帧新增待核线索。"""
        self._diag["frame_count"] += 1
        ts = _finite(timestamp)
        idx = frame_index if type(frame_index) is int and frame_index >= 0 else None
        if ts is None or ts < 0 or idx is None:
            return self._reject("invalid_frame_identity", ts, idx)
        if self._timestamp is not None and (ts <= self._timestamp or idx <= self._frame_index):
            return self._reject("non_monotonic_frame", ts, idx)
        if self._timestamp is not None and ts - self._timestamp > self._gap:
            self._break("time_gap")
        self._timestamp, self._frame_index = ts, idx
        if frame_usable is not True:
            return self._reject("frame_unusable", ts, idx)
        if (not isinstance(frame_size, (list, tuple)) or len(frame_size) != 2
                or any(type(x) is not int or not 1 <= x <= 32768 for x in frame_size)):
            return self._reject("invalid_frame_size", ts, idx)
        width, height = frame_size
        if self._size != (width, height):
            changed = self._size is not None
            self._size = (width, height)
            self._contours = tuple(np.asarray([(x * width, y * height) for x, y in points], dtype=np.float32)
                                   for _, points in self._zones)
            if changed:
                return self._reject("frame_size_changed", ts, idx)
        if (not isinstance(entities, (list, tuple)) or len(entities) > _MAX_ENTITIES
                or any(not isinstance(entity, dict) for entity in entities)):
            return self._reject("invalid_entities", ts, idx)
        boxes = {}
        for entity in entities:
            if entity.get("type") != "person":
                continue
            box = _person_box(entity, width, height, self._min_conf)
            if box is None:
                self._diag["rejected_person_box_count"] += 1
            else:
                boxes[box] = max(boxes.get(box, 0.0), _finite(entity.get("conf")))
        # Input iteration order must not affect tracker IDs or duplicate scores.
        boxes = dict(sorted(boxes.items()))
        if self._observed_associator is None:
            self._associate(sorted(boxes))
        else:
            try:
                self._associate_observed(boxes, ts)
            except ZoneAssociationError:
                return self._reject("association_unavailable", ts, idx)
        self._diag["usable_frame_count"] += 1
        self._diag["accepted_person_box_count"] += len(boxes)
        if not boxes:
            self._diag["no_valid_person_frame_count"] += 1
        margin = self._margin * min(width, height)
        output = []
        for zi in range(len(self._zones)):
            for track in self._tracks:
                candidate = self._observe_zone(track, zi, ts, idx, margin)
                if candidate is not None:
                    output.append(candidate)
        self._diag["last_frame"] = dict(status="ok", timestamp=ts, frame_index=idx,
                                        accepted_person_count=len(boxes), candidate_count=len(output))
        return output
