# -*- coding: utf-8 -*-
"""孔洞/洞口几何校验：框内必须有“凹陷/暗区”特征，否则视为疑似误报"""
import cv2
import numpy as np


def _inside(box, shape):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def looks_like_hole(im, bbox):
    """返回 (是否像孔洞, 原因)。面积、纵横比、暗像素比例三重约束。"""
    if im is None or not bbox:
        return False, "无图或无框"
    x, y, w, h = bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 0), bbox.get("h", 0)
    H, W = im.shape[:2]
    area = (w * h) / float(W * H)
    if area < 0.0006:
        return False, "框过小，非孔洞"
    if area > 0.25:
        return False, "框过大，非孔洞"
    ratio = w / max(1.0, h)
    if ratio < 0.2 or ratio > 5:
        return False, "纵横比异常"
    r = _inside((x, y, x + w, y + h), im.shape)
    if not r:
        return False, "框出界"
    x1, y1, x2, y2 = r
    crop = im[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark_ratio = float((gray < 85).mean())
    std = float(gray.std())
    # 孔洞：有相当比例暗区，但不能整片均匀（均匀多为黑布/物体）
    if dark_ratio < 0.18:
        return False, "暗区不足，非孔洞"
    if std < 12:
        return False, "纹理均匀，非孔洞"
    return True, f"dark={dark_ratio:.2f},std={std:.1f}"


# ---------------------------------------------------------------------------
# 吊装容器（吊桶/料斗）反证（v1.13.44，默认关闭）
#
# 背景：用户 2026-11-10 实测「明明是吊一口锅，被识别为孔洞」。俯视下吊桶与
# 圆口桩孔高度同构：都是「近圆轮廓 + 较暗内部」。
# 13 条真实孔洞里有 2 条就是圆形桩孔（7da29f11 e15/e20），所以**绝不能按圆形
# 排除** —— 只能靠结构证据区分：
#   S1 吊索穿过：一条近竖直、细、连续的线，且延长线对准框心（强信号）
#   S2 金属加强圈：桶沿的环状高光/径向梯度（中信号）
#   S3 液面反光：泥浆液面的成片低饱和高亮带（中信号）
# 决策：S1 且（S2 或 S3）才判为容器；只命中一个中信号时不动判定。
#
# 已知限制（未做）：
#   ① 钻杆排除只做了几何（线宽>6px / 线长>3×框高），没有接检测器实体做
#      「线端点连机械」的判断；真实桩孔旁若钻杆极细长，仍可能被误判为吊索。
#   ② 无真实吊桶样本，阈值只能靠合成图与几何推演标定，**未经真实验收**。
#      因此开关 hole_container_rebut_enabled 默认 False。
# ---------------------------------------------------------------------------

_LIFT_MIN_SIDE_PX = 24          # 框长边小于此值不做容器判定（信息不足）
_LIFT_SQUARE_RATIO = 0.60       # minAreaRect 两边比 ≥0.60 视为近圆/近方
_LIFT_SOBEL_ANGLE_TOL = 12.0    # 吊索与竖直方向允许偏差（度）
_LIFT_ROPE_WIDTH_MAX = 6        # 吊索线宽上限（像素）
_LIFT_ROPE_WIDTH_MIN = 1
_LIFT_ROPE_MIN_LEN_RATIO = 0.60  # 吊索可见长度至少 0.6×框高
_LIFT_ROPE_LONG_RATIO = 3.0     # 线长 > 3×框高 → 判为钻杆/塔材，不计 S1
_LIFT_ROPE_CENTER_TOL = 0.35    # 延长线到框心距离 ≤ 0.35×min(w,h)


def _largest_dark_rect(crop):
    """返回框内最大暗区的 minAreaRect 边长比；无有效暗区返回 0。"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.05 * crop.shape[0] * crop.shape[1]:
        return 0.0
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    if w <= 0 or h <= 0:
        return 0.0
    return float(min(w, h) / max(w, h))


def _find_rope(im, box):
    """S1：在框上方扩展区找近竖直细线，返回 (是否命中, 说明)。

    线宽用「同一行连续像素数」的中位数估计；线长与框高比较，过长的判为钻杆。
    """
    H, W = im.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    top = max(0, int(y1 - 1.2 * bh))
    bot = min(H, int(y1 + 0.6 * bh))
    left = max(0, int(x1 - 0.35 * bw))
    right = min(W, int(x2 + 0.35 * bw))
    if bot - top < 10 or right - left < 6:
        return False, "ROI 过小"

    roi = im[top:bot, left:right]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    vert = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    lines = cv2.HoughLinesP(vert, 1, np.pi / 180.0, threshold=20,
                            minLineLength=int(_LIFT_ROPE_MIN_LEN_RATIO * bh),
                            maxLineGap=8)
    if lines is None:
        return False, "无线段"

    cx = (x1 + x2) / 2.0 - left
    for seg in np.asarray(lines).reshape(-1, 4):
        lx1, ly1, lx2, ly2 = [float(v) for v in seg]
        dx, dy = lx2 - lx1, ly2 - ly1
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-3:
            continue
        # 与竖直方向夹角：|dx| / length
        if abs(dx) / length > _LIFT_SOBEL_ANGLE_TOL / 90.0:
            continue
        if length > _LIFT_ROPE_LONG_RATIO * bh:
            continue                       # 贯穿全画面 → 钻杆/塔材，不计 S1
        # 线宽：沿 y 取该线段附近一行的横向连续像素数
        mx = int(round((lx1 + lx2) / 2.0))
        my = int(round((ly1 + ly2) / 2.0))
        row = edges[min(edges.shape[0] - 1, max(0, my))]
        if not (0 <= mx < row.size):
            continue
        wpx = 1
        k = mx - 1
        while k >= 0 and row[k]:
            wpx += 1
            k -= 1
        k = mx + 1
        while k < row.size and row[k]:
            wpx += 1
            k += 1
        if not (_LIFT_ROPE_WIDTH_MIN <= wpx <= _LIFT_ROPE_WIDTH_MAX):
            continue
        # 延长线到框心的水平距离（线近似竖直，取线中点 x 作代表）
        if abs(mx - cx) > _LIFT_ROPE_CENTER_TOL * min(bw, bh):
            continue
        return True, "吊索竖直穿过框心（宽 %dpx，长 %.0fpx）" % (wpx, length)
    return False, "无符合的竖直线"


def _ring_score(crop):
    """S2：极坐标展开后，环带径向梯度是否显著且方向朝内。"""
    h, w = crop.shape[:2]
    if min(h, w) < 16:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    size = int(min(h, w) // 2)
    polar = cv2.warpPolar(gray, (size, 180), (w / 2.0, h / 2.0), size,
                          cv2.WARP_POLAR_LINEAR)
    # 沿角度方向求均值 → 径向亮度曲线
    prof = polar.mean(axis=0)
    grad = np.abs(np.gradient(prof))
    if grad.size == 0 or grad.max() <= 1e-6:
        return 0.0
    med = float(np.median(grad)) + 1e-6
    peak = float(grad.max())
    # 峰必须落在半径 60%~100% 的环带（桶沿），且显著高于中位数
    peak_idx = int(np.argmax(grad))
    if not (0.55 * size <= peak_idx <= size):
        return 0.0
    return peak / med


def _liquid_band(crop):
    """S3：框上部是否存在成片低饱和高亮带（泥浆液面反光）。"""
    h, w = crop.shape[:2]
    if h < 12 or w < 12:
        return False
    band = crop[int(0.10 * h):int(0.60 * h), :]
    if band.size == 0:
        return False
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = ((s <= 60) & (v >= 150)).astype(np.uint8)
    ratio = float(mask.mean())
    if not (0.03 <= ratio <= 0.75):
        return False
    # 要求成片：连通域水平跨度 ≥ 0.4×宽
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_WIDTH]) >= 0.4 * w and int(stats[i, cv2.CC_STAT_AREA]) >= 40:
            return True
    return False


def looks_like_lifting_container(im, bbox):
    """判断框内是否为吊装容器（吊桶/料斗）而非孔洞。

    Args:
        im: BGR 帧；``None`` 直接返回 False。
        bbox: ``{"x","y","w","h"}``（与 ``looks_like_hole`` 同格式）。

    Returns:
        ``(bool, str)``：True 表示「像吊装容器」，调用方应把事件降级为
        needs_review，**不要物理删除**。
    """
    if im is None or not bbox:
        return False, "无图或无框"
    try:
        x, y, w, h = (float(bbox.get(k, 0)) for k in ("x", "y", "w", "h"))
    except Exception:
        return False, "框字段异常"
    if max(w, h) < _LIFT_MIN_SIDE_PX:
        return False, "框过小"
    r = _inside((x, y, x + w, y + h), im.shape)
    if not r:
        return False, "框出界"
    x1, y1, x2, y2 = r
    crop = im[y1:y2, x1:x2]
    if crop.size == 0:
        return False, "空裁切"

    box_ratio = min(w, h) / max(w, h)
    if box_ratio < _LIFT_SQUARE_RATIO:
        return False, "框为长条（比 %.2f），按沟槽处理" % box_ratio

    rope, rope_why = _find_rope(im, (x1, y1, x2, y2))
    if not rope:
        return False, "无吊索：%s" % rope_why

    ring = _ring_score(crop)
    liquid = _liquid_band(crop)
    if ring >= 1.6 or liquid:
        detail = []
        if ring >= 1.6:
            detail.append("桶沿环带梯度 %.1f×" % ring)
        if liquid:
            detail.append("液面反光带")
        return True, "吊装容器：%s + %s" % (rope_why, "、".join(detail))
    return False, "有吊索但无桶沿/液面证据（环带 %.1f×）" % ring
