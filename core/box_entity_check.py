# -*- coding: utf-8 -*-
"""框-实体一致性检查：标题要求的实体不在框内 → 标 needs_review（不删除）

依据 100 条人工复核：4 类高频误报是“框在机械/路面/材料上，标题却是人”。
"""
import os

import cv2
import numpy as np

from core.frame_lookup import find_frame_by_timestamp, normalize_timestamp

PERSON_TITLES = ("安全帽", "个体防护", "高处")
MACHINE_TITLES = ("机械作业", "吊装", "场内人车")

# ---- 配网带电作业豁免（v1.13.42）----
# 缺陷（用户 2026-11-11 复核，e76a9eb47f7f e17 ts=705）：标题「临近带电体作业
# 安全距离不足」，框内其实是**绝缘斗臂车的绝缘斗与斗臂总成**（黄色绝缘斗 +
# 橙色内衬 + 下垂电缆 + 绝缘操作杆），无裸导线、无人员/工具贴近导电体、无放电
# 痕迹；该视频为规范配网带电作业（绝缘服 + 导线绝缘毯 + 专职监护人 + 警戒带），
# GT 判定 0 违章。根因：带电作业本身就要求贴近带电体，「距离/贴近」当判据在
# 原理上会把整类规范作业判成违章，调阈值解决不了。故加本地护栏：同帧检测到
# 斗臂车（实体表里只有 ``bucket_truck``，没有裸 ``bucket``）且与事件框重叠
# ≥50% → 只降级为待复核，不删除。
_LIVE_LINE_TITLE_KEY = "临近带电体"
_LIVE_LINE_ENTITY_TYPES = ("bucket_truck",)
# 判据 A：事件框基本落在斗臂车上（重叠 ≥50%）→ 直接判定
_LIVE_LINE_MIN_OVERLAP = 0.5
# 判据 B：框中心落在斗臂车框外扩 25% 的范围内 → 视为「同一作业单元」。
# 为什么需要 B：带电作业的 14 号事件常把框打在线缆/斗臂上而不是车身上，
# 实测 e76a9eb47f7f 重跑后的 e17（ts=655，框打在线缆上）与斗臂车框重叠仅
# 0.23，只用 A 会漏；而外扩是有界的，不会把远处的 14 号事件也吞进来
# （8aecce68 ts=790 那条室内事件**根本没有 bucket_truck 实体**，B 也不会命中）。
_LIVE_LINE_CENTER_EXPAND = 0.25
_LIVE_LINE_REASON = "配网带电作业形态（斗臂车/绝缘斗），贴近带电体为正常作业"


def _cfg(config, key, default=None):
    """兼容 ``dict`` 与 ``core.config.Config``（都有 .get），None 回退默认。"""
    if config is None:
        return default
    try:
        value = config.get(key, default)
    except Exception:
        return default
    return default if value is None else value


def _ent_box(ent):
    b = (ent or {}).get("bbox_px") or {}
    try:
        x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
    except Exception:
        return None
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _iou(a, b):
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1e-6, ua)


def _expand(box, ratio, shape):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, y1 = x1 - w * ratio, y1 - h * ratio
    x2, y2 = x2 + w * ratio, y2 + h * ratio
    W, H = shape
    return (max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2))


def _load_image(output_dir, ts):
    fp = _frame_by_ts(output_dir, ts)
    return _read_frame(fp)


def _read_frame(fp):
    if not fp:
        return None
    try:
        return cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


def _frame_by_ts(output_dir, ts):
    return find_frame_by_timestamp(output_dir, ts)


def _overlap(box, ent):
    x1, y1, x2, y2 = box
    b = ent.get("bbox_px") or {}
    ix = max(0, min(x2, b.get("x2", 0)) - max(x1, b.get("x1", 0)))
    iy = max(0, min(y2, b.get("y2", 0)) - max(y1, b.get("y1", 0)))
    return ix * iy / ((x2 - x1) * (y2 - y1) + 1e-6)


# 框内未命中时的外扩倍数（同中心，总边长 ×1.6），专治 bbox 偏移。
# 必须在 helmet_color_rebut 定义前赋值：它被用作默认参数，模块加载时即求值。
_HELMET_EXPAND_SCALE = 1.6

# v1.13.43：外扩窗口的彩色帽点固定下限。
# 依据 12 条视频全批复核（QA 逐条 4x 放大判读）：外扩窗口里 2.8% / 8.3%
# 是噪声（把绿防尘网当帽点、真·没戴帽被判已戴），31%/43%/52% 才是真帽壳。
# 25% 能干净切开。HSV 白色分支已整体移除（白帽改由检测器 helmet 兜底）。
_HELMET_EXPAND_MIN_COLORED = 0.25

# v1.13.43：框内窗口的彩色帽点下限。原自适应阈值低至 1.2%~5%，
# 深藏青工装（6.1%）就能命中，把真·没戴帽判成已戴。10% 起判。
_HELMET_INSIDE_MIN_COLORED = 0.10



def helmet_color_rebut(im, box, min_ratio=None, expand_scale=_HELMET_EXPAND_SCALE):
    """安全帽视觉反证：框内上半部出现彩色帽点 → 与「未戴安全帽」矛盾。

    航拍俯视下安全帽是头顶的高饱和彩色圆点（蓝/黄/橙/红）或白色亮圆点。
    现有 helmet 反校验依赖专用检测器命中，俯视小目标常检不到，这里用 HSV
    颜色做零成本兜底。

    判定窗口（v1.13.38 起）：
    1. 框内（上半 65%），阈值随框面积自适应（见 :func:`_adaptive_min_ratio`）；
    2. 框内未命中 → 同中心外扩 ``expand_scale`` 倍再检一次，专治 bbox 偏移。

    命中还必须存在**紧致帽点**（面积/填充率/长宽比，见
    :func:`_has_compact_helmet_dot`）：大面积色块（躯干/围挡/草地）不算。

    Args:
        im: BGR 帧。
        box: ``(x1, y1, x2, y2)`` 像素框。
        min_ratio: 彩色占比阈值；``None`` 时按窗口面积自适应（下限 1.2%）。
        expand_scale: 外扩倍数；``None``/≤1 表示不做外扩。

    Returns:
        ``(是否反证成立, 说明)``。蓝/黄/橙/红用高饱和色占比，白帽因易与
        水泥/纸板混淆，额外要求框本身较小且白色占比更高。
    """
    if im is None:
        return False, ""
    try:
        shape = im.shape[:2]
    except Exception:
        return False, ""
    hit, why = _window_hit(im, box, min_ratio,
                           colored_min_ratio=_HELMET_INSIDE_MIN_COLORED)
    if hit:
        return True, why
    if expand_scale and expand_scale > 1.0:
        H, W = shape
        ebox = _scale_box(box, expand_scale, (W, H))
        try:
            same = tuple(float(v) for v in box) == ebox
        except (TypeError, ValueError):
            same = False
        if not same:
            hit, why2 = _window_hit(im, ebox, min_ratio, colored_min_ratio=_HELMET_EXPAND_MIN_COLORED)
            if hit:
                return True, "外扩 %.1fx：%s" % (expand_scale, why2)
    return False, ""


# ---------------------------------------------------------------------------
# 安全帽反证加强（v1.13.38，用户 2026-11-10 反馈「很多人带了安全帽你说没带」）
#
# 实测根因（用户原图 393x336，两名蓝帽工人 + 黄绿反光背心）：
#   ① 框一大/人一远，帽点占框面积比就掉到 3% 以下 → 固定 min_ratio 漏判；
#   ② 框偏移（框没框住人）时，框内根本没有帽点，且检测器也检不到 person，
#      ``mark_suspect`` 里 `if ok and "安全帽" in title` 让整段反证被跳过 →
#      「框里连人都没有」却照样报「人没戴安全帽」。
# 因此这里做四件事：形状约束（拒绝大面积色块）+ 阈值随框面积自适应 +
# 外扩窗口兜底 + 整帧帽点兜底。框内反证由 helmet_gate 处置；无论窗口
# 如何都不能物理删除事件，小目标和冻结金标保留 needs_review。
# ---------------------------------------------------------------------------

# 自适应阈值：参照框面积占比（占画面 2% 时仍用 base=3%），框越大阈值越低
_HELMET_RATIO_REF_AREA_FRAC = 0.02
# 下限保护：无论框多大都不低于 1.2%（不无脑降到 0，避免整片天空/地面被判成帽点）
_HELMET_MIN_RATIO_FLOOR = 0.012
# 上限保护：极小框里帽子本应占很大比例，阈值最多抬到 5%，避免过严漏判
_HELMET_MIN_RATIO_CEIL = 0.05
# 整帧兜底的帽点下限
_HELMET_FRAME_MIN_DOTS = 2
# 整帧/窗口帽点检测的紧致度门槛：饱和度高、块形紧致、面积有上限
_HELMET_DOT_MIN_AREA = 24
# 帽点面积上限（占整帧）。2.5% 而非 2%：200x200 单测里 r=16 的圆点是
# 2.01% 整帧，旧 2% 会把「近景大帽点」判死；2.5% 在 960x544 上仍是
# 114x114 的巨块，不会把躯干色块当帽点。
_HELMET_DOT_MAX_FRAC = 0.025
# 单个连通域占窗口面积上限：≥55% 视为「大面积色块」，不构成帽点证据。
# 实测：合成 fixture 躯干占框 75.7%、c46c e3 蓝衣框单块 94% → 拒；
# 而 8aecce e24 真人框最大块仅 31% → 仍可命中（HEAD 正向能力保留）。
_HELMET_DOT_MAX_WINDOW_FRAC = 0.55
_HELMET_DOT_SAT_MIN = 110
_HELMET_DOT_VAL_MIN = 90
_HELMET_DOT_FILL_MIN = 0.45
_HELMET_DOT_ASPECT = (0.4, 2.6)
# ---- 彩色帽点占比闸门（v1.13.41，默认关闭）----
# 标定依据（QA 2026-11-11 实测 v1.mp4）：真·戴蓝帽的框内蓝色占比 42.3%，
# 「戴蓝帽但没穿反光背心」的 e83 为 17.7%，而同视频 5 条真·没戴帽全是 0.0%。
# 阈值取 12%（蓝色）/ 20%（黄橙红合计），对 0% 与 17.7% 两档都留了余量。
# **白色故意不纳入闸门**：白帽与水泥/砂石同色，本批实测「真·没戴帽」的事件框内
# 白色占比高达 49%~66%（v1 e1/e20/e23/e91），白色在本数据上不可分；白帽仍由
# helmet_color_rebut 的 small_box 分支兜底，本闸门只统计不判定。


def _adaptive_min_ratio(box_area, frame_area, base=0.03):
    """按框面积自适应彩色帽点占比阈值。

    框越大，帽子在框内占比越小（俯拍人员只占框的一小块），阈值按
    ``base * sqrt(ref_frac / box_frac)`` 缩放，并夹在
    ``[_HELMET_MIN_RATIO_FLOOR, _HELMET_MIN_RATIO_CEIL]`` 之间。

    Args:
        box_area: 检测窗口像素面积。
        frame_area: 整帧像素面积。
        base: 基准阈值（框面积占画面 2% 时的取值）。

    Returns:
        float 阈值（永远 > 0）。
    """
    try:
        box_area = float(box_area)
        frame_area = float(frame_area)
    except (TypeError, ValueError):
        return base
    if box_area <= 0 or frame_area <= 0:
        return base
    frac = box_area / frame_area
    ratio = base * ((_HELMET_RATIO_REF_AREA_FRAC / max(frac, 1e-6)) ** 0.5)
    return float(min(_HELMET_MIN_RATIO_CEIL, max(_HELMET_MIN_RATIO_FLOOR, ratio)))


def _scale_box(box, scale, shape):
    """同中心缩放框（``scale=1.6`` 表示总宽高各 ×1.6），并裁剪到画面内。"""
    W, H = shape
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * float(scale), (y2 - y1) * float(scale)
    return (max(0.0, cx - w / 2.0), max(0.0, cy - h / 2.0),
            min(float(W), cx + w / 2.0), min(float(H), cy + h / 2.0))


def _has_compact_helmet_dot(mask, window_area, max_area):
    """掩膜里是否存在**紧致帽点**（复用 ``_HELMET_DOT_*`` 形状门槛）。

    只有「面积合理 + 填充率高 + 长宽比接近圆」的连通域才算帽点。大面积
    色块（躯干/围挡/草地/立管整段）要么超过 ``max_area``，要么单块占窗口
    ≥``_HELMET_DOT_MAX_WINDOW_FRAC``，都会被排除。

    Args:
        mask: 布尔掩膜（窗口上半部）。
        window_area: 窗口像素面积。
        max_area: 单个连通域的绝对面积上限（整帧 ``_HELMET_DOT_MAX_FRAC``）。

    Returns:
        bool：至少一个连通域通过形状门槛。
    """
    binary = np.ascontiguousarray(mask.astype(np.uint8))
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    for idx in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[idx])
        if area < _HELMET_DOT_MIN_AREA or w <= 0 or h <= 0:
            continue
        if area > max_area or area >= _HELMET_DOT_MAX_WINDOW_FRAC * window_area:
            continue
        if area / float(w * h) < _HELMET_DOT_FILL_MIN:
            continue
        if not (_HELMET_DOT_ASPECT[0] <= h / float(w) <= _HELMET_DOT_ASPECT[1]):
            continue
        return True
    return False


def _window_hit(im, box, min_ratio=None, colored_min_ratio=None):
    """单个窗口的帽点判定（返回 ``(hit, 说明)``）。

    命中必须同时满足：彩色占比 ≥ 阈值，且掩膜里至少有一个
    **紧致帽点**（见 :func:`_has_compact_helmet_dot`）。只判彩色占比会把
    躯干、围挡、草地等大面积色块误当帽点——这正是 v1.13.38 首版的缺陷。

    v1.13.43：**HSV 白色分支已整体移除**。QA 全批 12 条视频逐条 4x 复核：
    白色衣裤、白墙、水泥地在框内/外扩窗口都会大面积命中（实测头顶区白占比
    43%~51%），无法与白帽壳区分，把真·没戴帽判成「疑似已佩戴」——这是漏报。
    白帽改由检测器的 ``helmet`` 实体兜底（见 ``mark_suspect`` 反向校验 1）；
    宁可多留误报，也不把没戴帽的人判成已戴。

    Args:
        im: BGR 帧（``None`` 直接不命中）。
        box: ``(x1, y1, x2, y2)`` 像素框。
        min_ratio: 彩色占比阈值；``None`` 时按窗口面积自适应。
        colored_min_ratio: 彩色占比的**固定下限**；给定时与自适应阈值取大者。

    Returns:
        ``(bool, str)``；命中时说明里带实际占比与所用阈值，便于日志归因。
    """
    if im is None:
        return False, ""
    try:
        H, W = im.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return False, ""
        # 帽在头顶：只看窗口上半 65%
        yb = min(y2, y1 + max(1, int((y2 - y1) * 0.65)))
        crop = im[y1:yb, x1:x2]
        if crop.size == 0:
            return False, ""
        total = float(crop.shape[0] * crop.shape[1])
        dot_max_area = _HELMET_DOT_MAX_FRAC * W * H
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        colored = (((hue >= 95) & (hue <= 135)) | ((hue >= 8) & (hue <= 40))
                   | (hue <= 8) | (hue >= 165)) & (sat >= 70) & (val >= 70)
        cr = float(np.count_nonzero(colored)) / total
        thr = (_adaptive_min_ratio(crop.shape[0] * crop.shape[1], W * H)
               if min_ratio is None else float(min_ratio))
        thr = max(thr, _HELMET_MIN_RATIO_FLOOR)
        if colored_min_ratio is not None:
            thr = max(thr, float(colored_min_ratio))
        if cr >= thr and _has_compact_helmet_dot(colored, total, dot_max_area):
            return True, "检出彩色帽点（%.1f%% ≥ 阈值 %.1f%%）" % (cr * 100, thr * 100)
    except Exception:
        return False, ""
    return False, ""


def helmet_dots_in_frame(im):
    """整帧紧凑彩色帽点计数（用于「框根本没框到人」的兜底反证）。

    与 :func:`helmet_color_rebut` 的整窗占比口径不同：这里数的是**紧致小色块**
    （高饱和 + 块内填充率 ≥0.45 + 长宽比 0.4~2.6 + 面积不超过画面 2.5%），
    目的是排除大块棕色裸土/红色围挡这类「占满一片」的色块——实测纯灰图、
    棕土噪声图计数均为 0，而用户原图（两名蓝帽工人）计数 ≥2。

    Args:
        im: BGR 帧。

    Returns:
        int 帽点个数（画面不可用时 0）。
    """
    if im is None:
        return 0
    try:
        if not hasattr(im, "shape") or im.size == 0:
            return 0
        H, W = im.shape[:2]
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        mask = (((hue >= 95) & (hue <= 135)) | ((hue >= 20) & (hue <= 45))
                | (hue <= 20)) & (sat >= _HELMET_DOT_SAT_MIN) & (val >= _HELMET_DOT_VAL_MIN)
        binary = np.ascontiguousarray(mask.astype(np.uint8))
        n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
        max_area = _HELMET_DOT_MAX_FRAC * W * H
        count = 0
        for idx in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[idx])
            if area < _HELMET_DOT_MIN_AREA or area > max_area or w <= 0 or h <= 0:
                continue
            if area / float(w * h) < _HELMET_DOT_FILL_MIN:
                continue
            if not (_HELMET_DOT_ASPECT[0] <= h / float(w) <= _HELMET_DOT_ASPECT[1]):
                continue
            count += 1
        return count
    except Exception:
        return 0


def helmet_colored_ratio(im, box):
    """统计框内（以及框上部 50%）的彩色/白色帽点占比（**只作补充信号**）。

    背景（QA 2026-11-11 实测，用户反馈「很多人带了安全帽你说没带」）：``v1.mp4``
    的 e75「未戴安全帽」框内**高饱和蓝色像素占比 42.3%**（蓝帽清晰可见、占人形
    约 1/5），而同视频 5 条**真·没戴帽**的事件该比例全为 **0.0%** —— 这个信号
    与「戴没戴」高度可分，可作为"看得见帽"的廉价补充判据。

    色域口径：蓝色 ``H∈[95,130] S≥90 V≥70``；黄/橙 ``H∈[8,40]``；红 ``H≤8 或
    H≥165``；白 ``S≤45 且 V≥160``（沿用 :func:`helmet_color_rebut` 的色域范围，
    阈值另设、互不影响）。**白色只统计不判定**：白帽与水泥/砂石同色，实测无法
    区分（v1.13.43 起 HSV 白色分支已移除，白帽交给检测器 helmet 实体）。

    Args:
        im: BGR 帧（``None`` → 全 0 统计）。
        box: ``(x1, y1, x2, y2)`` 像素框。

    Returns:
        dict：``blue`` / ``colored`` / ``white``（框内占比，0~1）、
        ``blue_top``（框上部 50% 的蓝色占比）、``area``（框内像素数）。
    """
    stats = {"blue": 0.0, "colored": 0.0, "white": 0.0,
             "blue_top": 0.0, "area": 0}
    if im is None:
        return stats
    try:
        H, W = im.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return stats
        crop = im[y1:y2, x1:x2]
        if crop.size == 0:
            return stats
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        blue = (hue >= 95) & (hue <= 130) & (sat >= 90) & (val >= 70)
        yellow = (hue >= 8) & (hue <= 40) & (sat >= 90) & (val >= 70)
        red = ((hue <= 8) | (hue >= 165)) & (sat >= 90) & (val >= 70)
        white = (sat <= 45) & (val >= 160)
        total = float(crop.shape[0] * crop.shape[1])
        stats["area"] = int(total)
        stats["blue"] = float(np.count_nonzero(blue)) / total
        stats["colored"] = float(np.count_nonzero(blue | yellow | red)) / total
        stats["white"] = float(np.count_nonzero(white)) / total
        top = crop[:max(1, int(crop.shape[0] * 0.5))]
        if top.size:
            top_blue = cv2.split(cv2.cvtColor(top, cv2.COLOR_BGR2HSV))
            _h, _s, _v = top_blue
            mask = (_h >= 95) & (_h <= 130) & (_s >= 90) & (_v >= 70)
            stats["blue_top"] = float(np.count_nonzero(mask)) / float(top.shape[0] * top.shape[1])
    except Exception:
        return {"blue": 0.0, "colored": 0.0, "white": 0.0, "blue_top": 0.0, "area": 0}
    return stats


def helmet_rebut_detail(im, box, allow_frame_fallback=False, expand_scale=_HELMET_EXPAND_SCALE):
    """安全帽视觉反证（框内 → 外扩 → 整帧兜底），返回命中窗口与说明。

    Args:
        im: BGR 帧。
        box: 事件框 ``(x1, y1, x2, y2)``。
        allow_frame_fallback: 是否允许整帧兜底。调用方只在「框内无 person 实体」
            时传 True——那种形态就是「框根本没框到人，却报人没戴帽」。
        expand_scale: 外扩倍数（同中心，总宽高 ×expand_scale）。

    Returns:
        ``(hit, 说明, 窗口)``；窗口取值 ``"框内" | "外扩" | "整帧兜底" | ""``。
    """
    if im is None:
        return False, "", ""
    try:
        H, W = im.shape[:2]
    except Exception:
        return False, "", ""
    hit, why = _window_hit(im, box, None,
                           colored_min_ratio=_HELMET_INSIDE_MIN_COLORED)
    if hit:
        return True, why, "框内"
    if expand_scale and expand_scale > 1.0:
        ebox = _scale_box(box, expand_scale, (W, H))
        hit, why2 = _window_hit(im, ebox, None, colored_min_ratio=_HELMET_EXPAND_MIN_COLORED)
        if hit:
            return True, "外扩 %.1fx：%s" % (expand_scale, why2), "外扩"
    if allow_frame_fallback:
        dots = helmet_dots_in_frame(im)
        if dots >= _HELMET_FRAME_MIN_DOTS:
            return (True, "整帧检出 %d 个紧凑彩色帽点" % dots, "整帧兜底")
    return False, "", ""


def _live_line_view(event):
    """事件 → 判定用的最小视图（``title`` / ``event_id`` / 自述文本）。"""
    if isinstance(event, dict):
        return {
            "event_id": event.get("event_id"),
            "title": str(event.get("title") or ""),
            "description": str(event.get("description") or ""),
            "review_reason": str(event.get("review_reason") or ""),
        }
    rep = getattr(event, "rep_violation", None) or {}
    return {
        "event_id": getattr(event, "display_id", None),
        "title": str(getattr(event, "rep_title", "") or ""),
        "description": str(rep.get("description") or ""),
        "review_reason": str(rep.get("review_reason") or ""),
    }


def _apply_event_fields(event, payload):
    """把字段写回事件（字典直接 update，``Event`` 写 ``rep_violation``）。"""
    if isinstance(event, dict):
        event.update(payload)
        return
    rep = dict(getattr(event, "rep_violation", None) or {})
    rep.update(payload)
    event.rep_violation = rep


def _live_line_entity_hit(box, entity, threshold=_LIVE_LINE_MIN_OVERLAP):
    """斗臂车实体是否与事件框构成「同一带电作业单元」。

    两条有界判据（任一成立即可）：
    A. 事件框与斗臂车框重叠 ≥ ``threshold``（框基本打在车/斗上）；
    B. 事件框**中心**落在斗臂车框外扩 ``_LIVE_LINE_CENTER_EXPAND`` 的范围内
       （框打在线缆/斗臂上，与斗臂车贴着）。

    Args:
        box: 事件框 ``(x1, y1, x2, y2)``。
        entity: 检测实体字典。
        threshold: 判据 A 的重叠比例。

    Returns:
        bool。
    """
    if str((entity or {}).get("type") or "") not in _LIVE_LINE_ENTITY_TYPES:
        return False
    ent_box = _ent_box(entity)
    if not ent_box:
        return False
    if _overlap(box, entity) >= threshold:
        return True
    try:
        cx = (float(box[0]) + float(box[2])) / 2.0
        cy = (float(box[1]) + float(box[3])) / 2.0
    except (TypeError, ValueError):
        return False
    ex = (ent_box[2] - ent_box[0]) * _LIVE_LINE_CENTER_EXPAND
    ey = (ent_box[3] - ent_box[1]) * _LIVE_LINE_CENTER_EXPAND
    return (ent_box[0] - ex) <= cx <= (ent_box[2] + ex) and \
        (ent_box[1] - ey) <= cy <= (ent_box[3] + ey)


def _representative_frame_and_box(event, output_dir, frame_results):
    """唯一锚帧与原图 SHA 均通过，且最终框与该帧样本同源才提供证据。

    返回已经解码和校验过的像素而非可变路径，避免校验后再读另一幅图。
    当前精修链未记录可核验的锚帧摘要绑定；不同框不能仅凭距离或精修标记
    假定来自同帧。在缺少这种来源证明时保留候选待核，不删除也不改框。
    """
    from types import SimpleNamespace

    from core.height_gate import _representative_original_box
    from core.strict_gate import _verified_tracked_anchor

    target = event
    if isinstance(event, dict):
        target = SimpleNamespace(anchor_frame=event.get("representative_frame"),
                                 anchor_ts=event.get("representative_timestamp"),
                                 refined=event.get("bbox"), samples=event.get("samples"))
    samples = getattr(target, "samples", None)
    if not isinstance(samples, (list, tuple)):
        return None, "invalid_anchor_samples"
    try:
        sample, row, image, status = _verified_tracked_anchor(target, output_dir, frame_results)
    except (AttributeError, TypeError, ValueError, OverflowError, MemoryError):
        return None, "invalid_anchor_or_frame_results"
    if status != "ok":
        return None, status
    box, status = _representative_original_box(target, sample, image)
    if status != "ok":
        return None, status
    return (normalize_timestamp(target.anchor_ts), row["original_frame_sha256"], image, box), "ok"


def _event_payload(event):
    return event if isinstance(event, dict) else dict(getattr(event, "rep_violation", None) or {})


def _set_event_payload(event, value):
    if not isinstance(event, dict):
        event.rep_violation = value


def _hold_box_entity_review(event, note, *, frame_status=None):
    """不完整证据只能保守留队；旧处置与旧确认标志转入审计。"""
    from core.review_policy import enforce_incomplete_review_guard

    value = _event_payload(event)
    previous = str(value.get("review_disposition") or "")
    if previous in ("rejected", "box_relocalize", "dropped", "low_priority_confirm"):
        value.setdefault("box_entity_blocked_disposition", previous)
    if previous == "box_relocalize" or (frame_status and "bbox" in frame_status):
        value["needs_box_relocalize"] = True
    if value.get("ai_review_confirmed") is True:
        value.pop("ai_review_confirmed")
        value.setdefault("box_entity_prior_ai_review_confirmed", True)
    value["needs_review"] = True
    value["review_disposition"] = "low_priority_review"
    previous_reason = str(value.get("review_reason") or "")
    if note not in previous_reason:
        value["review_reason"] = (previous_reason + "；" if previous_reason else "") + note
    if frame_status:
        value["box_entity_frame_unverified"] = True
        value["box_entity_frame_status"] = frame_status
        value["analysis_incomplete"] = True
        enforce_incomplete_review_guard(value)
    _set_event_payload(event, value)


def _archive_previous_box_entity_evidence(event, kind):
    """每次重跑先撤销旧帽子/带电反证，旧结论只作为历史审计。"""
    value = _event_payload(event)
    fields = (("helmet_rebut", "helmet_rebut_window", "helmet_rebut_reason",
               "helmet_rebut_review_only", "helmet_rebut_demote", "helmet_rebut_guard_kept")
              if kind == "helmet" else ("live_line_work_demote",))
    old = {key: value.pop(key) for key in fields if key in value}
    if not old:
        return
    had_rebuttal = any(old.get(key) is True for key in
                       ("helmet_rebut", "helmet_rebut_review_only", "helmet_rebut_demote",
                        "helmet_rebut_guard_kept", "live_line_work_demote"))
    if had_rebuttal and value.get("review_reason"):
        old["review_reason"] = value.pop("review_reason")
    if "helmet_rebut" in old:
        value["helmet_rebut"] = False
    history = value.get("box_entity_prior_attempts")
    if not isinstance(history, list):
        history = [] if history is None else [{"unrecognized_prior_audit": history}]
    history.append({"scope": kind, "previous_evidence": old})
    value["box_entity_prior_attempts"] = history
    _set_event_payload(event, value)
    if had_rebuttal:
        _hold_box_entity_review(event, "此前框实体反证已转入历史，本次仍需人工核查")


def apply_live_line_work_exempt(events, output_dir, config=None, det=None,
                                threshold=_LIVE_LINE_MIN_OVERLAP, frame_results=None):
    """配网带电作业豁免：框落在斗臂车上 → 「临近带电体」只降级、不删除。

    判定链（全部本地、确定性、零 API）：
    1. 标题含「临近带电体」（其余标题一律不看）；
    2. 冻结金标「成立」命中 → 不动（金标最高优先，约束 2）；
    3. 代表帧实体里存在 ``bucket_truck``（本项目实体表**没有**裸 ``bucket``
       类型，故只用 ``bucket_truck``），且与事件框构成同一作业单元
       （重叠 ≥ ``threshold``，或框中心落在斗臂车框外扩 25% 内，见
       :func:`_live_line_entity_hit`）；
    4. 命中 → ``needs_review=True`` + ``review_reason`` 写明形态 +
       ``live_line_work_demote=True``。**不动其它判据、不写任何删除标记**。

    开关 ``live_line_work_exempt_enabled`` 默认 **True**（这是已确证的误报：
    e76a9eb47f7f e17），关闭时不写任何字段、事件 JSON 逐字节不变。

    Args:
        events: ``Event`` 列表（原地修改 ``rep_violation``）；也接受事件字典。
        output_dir: 输出目录（含 ``original/`` 帧图）。
        config: 配置对象/字典（``live_line_work_exempt_enabled``、
            ``ai_review_golden_exempt_path``）。
        det: 可选检测器（单测注入替身）；缺省走 ``get_detector()``。
        threshold: 事件框与斗臂车框的最小重叠比例（默认 0.5，防泛化）。

    Returns:
        dict：``enabled`` / ``checked`` / ``demoted`` / ``golden_kept`` / ``no_frame``。
    """
    stats = {"enabled": False, "checked": 0, "demoted": 0,
             "golden_kept": 0, "no_frame": 0}
    if not bool(_cfg(config, "live_line_work_exempt_enabled", True)):
        return stats
    from core.ai_review import is_golden_exempt
    if det is None:
        from core.yolo_detector import get_detector
        det = get_detector()
    stats["enabled"] = True
    tag = os.path.basename(os.path.normpath(output_dir or "")) or ""
    cache = {}
    for event in events or []:
        view = _live_line_view(event)
        if _LIVE_LINE_TITLE_KEY not in view["title"]:
            continue
        stats["checked"] += 1
        _archive_previous_box_entity_evidence(event, "live_line")
        try:
            protected = is_golden_exempt(view, config, tag)
        except Exception:
            _hold_box_entity_review(event, "冻结保护无法核验，未执行带电作业实体反证",
                                    frame_status="golden_guard_unavailable")
            stats["no_frame"] += 1
            continue
        if protected:
            _hold_box_entity_review(event, "冻结身份键或冻结资产保护，保留人工复核")
            stats["golden_kept"] += 1
            continue
        representative, status = _representative_frame_and_box(event, output_dir, frame_results)
        if not representative:
            stats["no_frame"] += 1
            _hold_box_entity_review(event, "代表帧或最终框未核验，未执行带电作业实体反证",
                                    frame_status=status)
            continue
        _ts, digest, im, box = representative
        if digest not in cache:
            try:
                prior = det.build_scene_prior(im)
            except Exception:
                stats["no_frame"] += 1
                _hold_box_entity_review(event, "实体检测失败，未执行带电作业实体反证",
                                        frame_status="detector_failed")
                continue
            entities = prior.get("entities") if isinstance(prior, dict) else None
            cache[digest] = entities if isinstance(entities, list) and all(
                isinstance(item, dict) for item in entities) else None
        checked_again, status = _representative_frame_and_box(event, output_dir, frame_results)
        if not checked_again or checked_again[1] != digest:
            stats["no_frame"] += 1
            _hold_box_entity_review(event, "实体检测期间代表帧或最终框发生变化，未执行带电作业实体反证",
                                    frame_status=status if not checked_again else "frame_changed")
            continue
        _ts, _digest, im, box = checked_again
        ents = cache[digest]
        if ents is None:
            stats["no_frame"] += 1
            _hold_box_entity_review(event, "实体检测未完成，未执行带电作业实体反证",
                                    frame_status="detector_output_invalid")
            continue
        hit = any(_live_line_entity_hit(box, x, threshold) for x in ents)
        if not hit:
            continue
        _apply_event_fields(event, {
            "needs_review": True,
            "review_reason": _LIVE_LINE_REASON,
            "live_line_work_demote": True,
        })
        stats["demoted"] += 1
        print("[live-line] e%s %s 降级：框内检出斗臂车/绝缘斗" % (
            view.get("event_id"), view.get("title")), flush=True)
    return stats


def mark_suspect(events, output_dir, threshold=0.03, config=None, frame_results=None):
    """原地给可疑事件加 needs_review；返回统计。

    Args:
        events: ``Event`` 列表（原地修改 ``rep_violation``）。
        output_dir: 本次输出目录（读取 ``original/`` 帧图）。
        threshold: 实体与框的重叠阈值（默认 0.03，保持既有语义）。
        config: 可选配置对象/字典；用于读
            ``helmet_color_ratio_gate_enabled``（默认 False）与
            ``live_line_work_exempt_enabled``（默认 True）。缺省时彩色帽点
            判据不生效（旧行为），带电作业豁免按默认开关执行。
    """
    # 依赖初始化失败必须交给调用方的独立门禁故障护栏，不能伪装成零候选。
    from core.yolo_detector import get_detector
    det = get_detector()
    # 配网带电作业豁免（v1.13.42）：独立前置一遍，只认「临近带电体」标题。
    _live_line = apply_live_line_work_exempt(events, output_dir, config, det=det,
                                            frame_results=frame_results)
    cache, checked, flagged, no_frame = {}, 0, 0, 0
    for e in events:
        title = getattr(e, "rep_title", "")
        if not (any(k in title for k in PERSON_TITLES) or any(k in title for k in MACHINE_TITLES)):
            continue
        _archive_previous_box_entity_evidence(e, "helmet")
        representative, status = _representative_frame_and_box(e, output_dir, frame_results)
        if not representative:
            v = _event_payload(e)
            v["helmet_rebut"] = False
            _set_event_payload(e, v)
            _hold_box_entity_review(e, "代表帧或最终框未核验，框实体核验未执行", frame_status=status)
            no_frame += 1
            flagged += 1
            continue
        ts, digest, im, box = representative
        if digest not in cache:
            try:
                prior = det.build_scene_prior(im)
            except Exception:
                v = _event_payload(e)
                v["helmet_rebut"] = False
                _set_event_payload(e, v)
                _hold_box_entity_review(e, "实体检测失败，框实体核验未执行", frame_status="detector_failed")
                no_frame += 1
                flagged += 1
                continue
            entities = prior.get("entities") if isinstance(prior, dict) else None
            cache[digest] = entities if isinstance(entities, list) and all(
                isinstance(item, dict) for item in entities) else None
        checked_again, status = _representative_frame_and_box(e, output_dir, frame_results)
        if not checked_again or checked_again[1] != digest:
            v = _event_payload(e)
            v["helmet_rebut"] = False
            _set_event_payload(e, v)
            _hold_box_entity_review(e, "实体检测期间代表帧或最终框发生变化，框实体核验未执行",
                                    frame_status=status if not checked_again else "frame_changed")
            no_frame += 1
            flagged += 1
            continue
        ts, _digest, im, box = checked_again
        ents = cache[digest]
        if ents is None:
            v = _event_payload(e)
            v["helmet_rebut"] = False
            _set_event_payload(e, v)
            _hold_box_entity_review(e, "实体检测结果无效，框实体核验未执行",
                                    frame_status="detector_output_invalid")
            no_frame += 1
            flagged += 1
            continue
        checked += 1
        why = ""
        if any(k in title for k in PERSON_TITLES):
            ok = any(x.get("type") == "person" and _overlap(box, x) > threshold for x in ents)
        else:
            ok = any(x.get("type") in ("machine", "bucket_truck") and _overlap(box, x) > threshold for x in ents)
        # 机械/吊装类：框内或邻近必须有机械实体，否则疑似误报
        if ok and title in MACHINE_TITLES:
            mx1, my1, mx2, my2 = _expand(box, 0.15, (1000, 1000))
            # 硬条件：框内必须同时有人与机械（IoU 或中心落入框内），否则疑似误报
            def _in_box(ent, b, extra=0.1):
                eb = _ent_box(ent)
                if not eb:
                    return False
                ex = (b[2] - b[0]) * extra
                ey = (b[3] - b[1]) * extra
                cx = (eb[0] + eb[2]) / 2
                cy = (eb[1] + eb[3]) / 2
                return (b[0] - ex) <= cx <= (b[2] + ex) and (b[1] - ey) <= cy <= (b[3] + ey)

            has_m = any(e.get("type") in ("machine", "bucket") and _in_box(e, box) for e in ents)
            has_p = any(e.get("type") in ("person", "head") and _in_box(e, box) for e in ents)
            if not (has_m and has_p):
                ok, why = False, "框内未同时包含人与机械"

        # 孔洞/洞口类：框内必须有凹陷暗区特征
        if ok and ("孔洞" in title or "洞口" in title):
            try:
                from core.hole_guard import looks_like_hole
                if im is None:
                    im = _load_image(output_dir, ts)
                good, why2 = looks_like_hole(im, box)
                # v1.13.44：吊装容器反证（默认关闭）。俯视下吊桶/料斗与圆口桩孔
                # 同为「近圆 + 暗内部」，只能靠结构证据区分（吊索/桶沿/液面）。
                # 只降级不删除 —— 孔洞是真违章高发类，误删代价高。
                if good and bool(_cfg(config, "hole_container_rebut_enabled", False)):
                    from core.hole_guard import looks_like_lifting_container
                    is_c, why3 = looks_like_lifting_container(im, box)
                    if is_c:
                        good, why2 = False, why3
                if good:
                    x1, y1, x2, y2 = box
                    if (y1 + y2) / 2 < 350:
                        good, why2 = False, "位置偏高，非地面孔洞"
                    for ent in ents:
                        if ent.get("type") in ("person", "machine", "bucket") and _iou(box, _ent_box(ent)) > 0.25:
                            good, why2 = False, "框压人/机械，非孔洞"
                            break
                if not good:
                    ok, why = False, why2
            except Exception:
                pass

        # v1.13.38：安全帽类反证**不再以 ok 为前提**，但要区分「删除」与
        # 「待复核」：旧实现写的是 `if ok and "安全帽" in title`——当框内
        # 没有 person 实体（典型的「框偏移，框里根本没人」）时，整段反证
        # 被跳过，事件照样以「未戴安全帽」报出（用户 2026-11-10 实测）。
        # 拆成两段：
        #   ① 检测器 helmet 实体反证：只在 ok（有人）时才有意义，
        #      维持 HEAD 语义——只标待复核，不写 helmet_rebut 物理删除；
        #   ② HSV 视觉反证：无论 ok 都跑，但只有「框内 + 形状合格 + 有人」
        #      才允许走 helmet_rebut 删除；外扩/整帧/无人窗口一律降级为
        #      needs_review，避免把立管、草地、躯干色块当帽点误删事件。
        if "安全帽" in title:
            has_person = any(x.get("type") == "person" and _overlap(box, x) > threshold
                             for x in ents)
            det_helmet_hit = False
            # 反向校验 1：框内上半部检出 helmet 实体 → 大概率已经戴了
            x1, y1, x2, y2 = box
            if ok:
                for ent in ents:
                    if ent.get("type") != "helmet":
                        continue
                    b = ent.get("bbox_px") or {}
                    cy = (b.get("y1", 0) + b.get("y2", 0)) / 2
                    cx = (b.get("x1", 0) + b.get("x2", 0)) / 2
                    if x1 <= cx <= x2 and y1 <= cy <= y1 + (y2 - y1) * 0.35:
                        det_helmet_hit = True
                        ok = False
                        why = "框内检出安全帽，疑似已佩戴"
                        break
            # 反向校验 2（v1.13.36）：检测器在航拍俯视小目标上常检不到 helmet，
            # 6 条新视频判读里「人都戴蓝帽仍报未戴、框落在地面」反复出现。
            # 用 HSV 彩色帽点兜底：框内 → 外扩 1.6× → 整帧（无 person 时）。
            # v1.13.43 起白色不再参与 HSV 判定（见 _window_hit 文档）。
            if im is None:
                im = _load_image(output_dir, ts)
            hit, why2, window = helmet_rebut_detail(
                im, box, allow_frame_fallback=not has_person)
            if hit:
                ok = False
                if not det_helmet_hit:
                    why = "视觉反证[%s]：%s，疑似已佩戴" % (window, why2)
                v = getattr(e, "rep_violation", None) or {}
                v["helmet_rebut_window"] = window
                v["helmet_rebut_reason"] = why2
                if window == "框内" and has_person and not det_helmet_hit:
                    # 唯一允许物理删除的路径：框内 + 形状合格 + 有人，
                    # 且检测器没有给出 helmet 实体（检测器命中维持 HEAD 语义）。
                    v["helmet_rebut"] = True
                else:
                    # 外扩/整帧/框内无人/检测器已命中：保留主队列待复核
                    v["helmet_rebut_review_only"] = True
                e.rep_violation = v
            # v1.13.43：v1.13.41 的「彩色帽点占比闸门」已废弃。
            # 它原先写在 `if ok:` 里，而 `ok` 在此之前已被 helmet_rebut_detail
            # 清掉；框内阈值仅 4.9%、又有三级窗口，新闸门 12% 被严格支配，
            # 全批 ratio_demoted 恒为 0（QA 开关闸各跑三条视频，事件集合逐字段
            # 一致已证实）。改为只记录诊断占比，不再参与判定。
            if im is not None:
                try:
                    stats = helmet_colored_ratio(im, box)
                    v = dict(getattr(e, "rep_violation", None) or {})
                    v["helmet_color_ratio_stats"] = {
                        "blue": round(float(stats.get("blue", 0.0)), 4),
                        "colored": round(float(stats.get("colored", 0.0)), 4),
                        "white": round(float(stats.get("white", 0.0)), 4),
                        "blue_top": round(float(stats.get("blue_top", 0.0)), 4),
                        "area": int(stats.get("area", 0)),
                    }
                    e.rep_violation = v
                except Exception:
                    pass
        if not ok:
            v = getattr(e, "rep_violation", None) or {}
            v["needs_review"] = True
            v["suspected_false"] = True
            v["review_reason"] = why or "框内未见标题要求的实体"
            e.rep_violation = v
            flagged += 1
    return {"checked": checked, "flagged": flagged, "no_frame": no_frame,
            "live_line_demoted": _live_line.get("demoted", 0)}
