# -*- coding: utf-8 -*-
"""孔洞双标准：VLM 判定有无 -> 小模型在 VLM 区域内精修框"""
import os

import cv2
import numpy as np
import onnxruntime as ort

_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
_MODELS = [os.path.join(_DIR, "hole_det.onnx"), os.path.join(_DIR, "hole_det_v2.onnx")]
_sessions = []


def _session(i=0):
    global _sessions
    if not _sessions:
        _sessions = []
        for m in _MODELS:
            if os.path.exists(m):
                _sessions.append(ort.InferenceSession(m, providers=["DmlExecutionProvider", "CPUExecutionProvider"]))
    return _sessions[i] if i < len(_sessions) else None


def _iou(a, b):
    x1, y1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    x2, y2 = min(a["x"] + a["w"], b["x"] + b["w"]), min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / max(1e-6, ua)


def _inside(pt, b, pad=0.2):
    w, h = b["w"], b["h"]
    return (b["x"] - pad * w) <= pt[0] <= (b["x"] + b["w"] + pad * w) and \
           (b["y"] - pad * h) <= pt[1] <= (b["y"] + b["h"] + pad * h)


def detect(im, conf=0.12, idx=0):
    sess = _session(idx)
    if sess is None:
        return []
    H, W = im.shape[:2]
    s = 640
    r = min(s / W, s / H)
    nw, nh = int(W * r), int(H * r)
    lb = np.full((s, s, 3), 114, np.uint8)
    dx, dy = (s - nw) // 2, (s - nh) // 2
    lb[dy:dy + nh, dx:dx + nw] = cv2.resize(im, (nw, nh))
    x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0].T
    res = []
    for row in out:
        c = float(row[4])
        if c < conf:
            continue
        cx, cy, w, h = row[:4]
        res.append({"x": (cx - w / 2 - dx) / r, "y": (cy - h / 2 - dy) / r,
                    "w": w / r, "h": h / r, "score": c})
    return res


def detect_tiled(im, conf=0.10, idx=0):
    """2x2 重叠分块检测；完整覆盖图像，不据此直接新增孔洞事件。"""
    H, W = im.shape[:2]
    if not H or not W:
        return []
    # 每块覆盖至少 55%，从两端开始；旧版 50% 的块从 45% 处开始，
    # 会漏掉右侧/底部 5%。ceil 和 clamp 保证奇数与极小图像仍覆盖。
    tile_w = min(W, (11 * W + 19) // 20)
    tile_h = min(H, (11 * H + 19) // 20)
    out = []
    for y0 in (0, H - tile_h):
        for x0 in (0, W - tile_w):
            crop = im[y0:y0 + tile_h, x0:x0 + tile_w]
            for d in detect(crop, conf, idx):
                d["x"] += x0
                d["y"] += y0
                out.append(d)
    return out


def refine_hole_box(im, vlm_box, conf=0.12, iou_min=0.15):
    """返回 (box, score) 或 (None, reason)。只在 VLM 框附近选，不新增孔洞。"""
    if im is None or not vlm_box:
        return None, "no_image_or_box"
    b = {"x": vlm_box["x"], "y": vlm_box["y"], "w": vlm_box["w"], "h": vlm_box["h"]}
    for idx in range(len(_MODELS)):
        cands = []
        pool = detect(im, conf, idx) + detect_tiled(im, max(0.08, conf - 0.02), idx)
        for d in pool:
            dc = (d["x"] + d["w"] / 2, d["y"] + d["h"] / 2)
            if _iou(d, b) >= iou_min or _inside(dc, b):
                cands.append(d)
        if not cands:
            continue
        best = max(cands, key=lambda d: d["score"])
        if _iou(best, b) < 0.10:
            continue
        return {"x": float(best["x"]), "y": float(best["y"]),
                "w": float(best["w"]), "h": float(best["h"])}, float(best["score"])
    return None, "detector_miss"
