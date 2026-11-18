"""Optional ByteTrack backend for offline, image-plane zone observations."""

from importlib.metadata import PackageNotFoundError, version

import numpy as np

from core.zone_presence import ZoneAssociationError


PINNED_VERSIONS = {"trackers": "2.7.0", "supervision": "0.31.0"}


def _load_backend():
    try:
        installed = {name: version(name) for name in PINNED_VERSIONS}
    except PackageNotFoundError as exc:
        raise ZoneAssociationError("ByteTrack 试用依赖未安装；请在隔离环境按 requirements-bytetrack.txt 安装") from exc
    if installed != PINNED_VERSIONS:
        raise ZoneAssociationError("ByteTrack 试用依赖版本与已测版本不一致；请检查 requirements-bytetrack.txt")
    try:
        from supervision import Detections
        from trackers import ByteTrackTracker
    except (ImportError, OSError) as exc:
        raise ZoneAssociationError("无法加载 ByteTrack 试用依赖；未回退为其他跟踪算法") from exc
    return ByteTrackTracker, Detections


class ByteTrackAssociator:
    """Return only IDs assigned to unchanged detector boxes in the current image."""

    association_state = "bytetrack_adjacent_observations_only"

    def __init__(self, sample_rate):
        self._tracker_type, self._detections_type = _load_backend()
        self._rate = sample_rate
        self._tracker = None

    @property
    def report_metadata(self):
        return {"method": "roboflow_bytetrack", "versions": dict(PINNED_VERSIONS), "experimental": True,
                "confirmation": "observation_only_identity_unverified",
                "uses_camera_compensation": False, "uses_reidentification": False}

    def reset(self):
        self._tracker = None

    def associate(self, boxes, timestamp):
        # ZonePresence already validated and deduplicated the pixel boxes.
        raw_boxes = list(boxes)
        xyxy = np.asarray(raw_boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(list(boxes.values()), dtype=np.float32)
        indexes = np.arange(len(raw_boxes), dtype=np.int64)
        try:
            data = self._detections_type(xyxy=xyxy.copy(), confidence=scores, data={"source_index": indexes.copy()})
            if self._tracker is None:
                self._tracker = self._tracker_type(frame_rate=self._rate)
            result = self._tracker.update(data, timestamp=timestamp)
            count = len(result)
            if not count:
                return {}
            ids = np.asarray(result.tracker_id)
            selected = np.asarray(result.data["source_index"])
            if (ids.shape != (count,) or selected.shape != (count,) or ids.dtype.kind not in "iu"
                    or selected.dtype.kind not in "iu" or np.any(ids < -1) or np.any(selected < 0)
                    or np.any(selected >= len(raw_boxes)) or len(np.unique(selected)) != count
                    or not np.array_equal(result.xyxy, xyxy[selected])):
                raise ZoneAssociationError("ByteTrack 返回的观测框、来源索引或轨迹 ID 不一致")
            positive = ids[ids >= 0]
            if len(np.unique(positive)) != len(positive):
                raise ZoneAssociationError("ByteTrack 同帧出现冲突轨迹，不能续接")
            return {raw_boxes[int(i)]: int(tid) for i, tid in zip(selected, ids, strict=True) if tid >= 0}
        except ZoneAssociationError:
            raise
        except Exception as exc:
            raise ZoneAssociationError("ByteTrack 本帧关联失败，已停止沿用上一帧轨迹") from exc
