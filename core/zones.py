# -*- coding: utf-8 -*-
"""区域多边形：基坑临边等，坐标归一化 0~1"""
import json
import os

import numpy as np


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def for_video(zones, video_path):
    if not zones:
        return []
    name = os.path.basename(video_path)
    for k, v in zones.items():
        if k == name or name.startswith(k):
            return v
    return []


def _pt_seg_dist(p, a, b):
    p, a, b = np.array(p, float), np.array(a, float), np.array(b, float)
    ab = b - a
    t = np.dot(p - a, ab) / max(1e-9, np.dot(ab, ab))
    t = max(0.0, min(1.0, t))
    return float(np.linalg.norm(p - (a + t * ab)))


def dist_to_polygon(pt, poly_norm, w, h):
    """pt 为像素坐标，poly_norm 归一化多边形；返回像素距离"""
    poly = [(x * w, y * h) for x, y in poly_norm]
    if len(poly) < 3:
        return None
    d = min(_pt_seg_dist(pt, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))
    return d
