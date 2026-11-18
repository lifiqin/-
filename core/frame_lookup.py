# -*- coding: utf-8 -*-
"""按真实时间查找原始帧：有界、确定性、失败即不提供证据。"""
import math
import os
import re

MAX_FRAME_TIME_DELTA_SEC = 3.0
_FRAME_NAME = re.compile(r"^frame_\d+_(\d+)s\.(?:jpg|jpeg|png)$", re.I)


def normalize_timestamp(value):
    """仅接受有限、非负的时间；缺失值不能默认为视频第 0 秒。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        timestamp = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return timestamp if math.isfinite(timestamp) and timestamp >= 0 else None


def find_frame_by_timestamp(output_dir, timestamp):
    """最近帧距目标不超过 3 秒才可用；距离相同时取较早秒及稳定文件名。"""
    timestamp = normalize_timestamp(timestamp)
    if timestamp is None or not output_dir:
        return None
    try:
        folder = os.path.join(os.fspath(output_dir), "original")
        with os.scandir(folder) as entries:
            candidates = []
            for entry in entries:
                match = _FRAME_NAME.fullmatch(entry.name)
                if match is None or not entry.is_file():
                    continue
                second = int(match.group(1))
                delta = abs(second - timestamp)
                if delta <= MAX_FRAME_TIME_DELTA_SEC:
                    candidates.append((delta, second, entry.name.casefold(), entry.name, entry.path))
    except (OSError, TypeError, ValueError):
        return None
    return min(candidates)[-1] if candidates else None
