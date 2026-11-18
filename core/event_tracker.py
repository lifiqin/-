# -*- coding: utf-8 -*-
"""
违章事件追踪器
把逐帧独立检出的散框合并为跨帧"违章事件"，全部本地计算，零API开销：
- 事件合并：同类违章 + 位置重叠(IoU) + 时间窗
- 事件化：相邻帧同类、位置重叠的违章合并为事件，稳定度校准坐标
- 坐标校准：中位数融合剔除离群框；跨帧模板匹配投票修正代表帧位置；
  逐帧模板跟踪使前后帧框一致，并补上模型漏检的帧
"""
from bisect import bisect_left, bisect_right
from collections import Counter

import cv2
import numpy as np

# 几何/文本工具统一在 bbox_utils 实现（此前 IoU/中位框/标题相似度三处各写一份）
from core.bbox_utils import iou_xywh as _iou
from core.bbox_utils import median_box as _median_box
from core.bbox_utils import titles_similarity as _titles_similar
from core.bbox_utils import as_number as _num

CATEGORY_KEYWORDS = [
    ("安全帽", ["安全帽"]),
    ("高处作业防护", ["安全带", "高处", "高空抛物", "交叉作业", "登高", "攀爬", "坠落", "梯子", "爬梯", "脚手架", "支撑架", "吊篮",
                  "临时拉线", "防倒杆", "倒杆塔", "缆风绳", "抱杆", "跨越架", "封网", "跨越施工"]),
    # 个体防护置于临边防护之前：两者都含"防护"，靠顺序 + 最长匹配消歧（见 L7）
    ("个体防护", ["个体防护", "防护用品", "反光衣", "防护服"]),
    # 基坑沟槽置于临边防护之前：标题同时含"基坑"与"栏杆/护栏"时（如"基坑未设置
    # 安全防护栏杆"），坑字语境优先——按命名纠偏约定，这类是基坑类而非临边类违章
    ("基坑沟槽", ["基坑", "沟槽", "开挖", "坍塌", "堆土", "堆料", "护梯", "坑内", "坑内积水",
                  "支护", "边坡", "放坡", "超挖", "管涌", "流土", "毗邻", "变形超限"]),
    ("拆除作业", ["拆除", "拆塔", "撤线"]),
    ("临边防护", ["临边", "护栏", "围栏", "栏杆", "临边防护", "楼面临边", "探头板", "脚手板"]),
    ("孔洞防护", ["孔洞", "洞口", "盖板", "竖井", "斜井", "井口未封闭", "井口敞开"]),
    ("临时用电", ["用电", "电缆", "电线", "配电", "触电", "带电", "外电", "架空线路", "拖地", "泡水"]),
    ("机械起重", ["机械", "起重", "吊装", "吊臂", "吊物下方", "专人指挥", "挖掘机", "吊车", "臂架", "钻机", "叉车", "切割机", "混行", "人车", "人机",
                  "载人", "索道", "支腿", "垫板", "牵引", "受力绳索", "内角侧", "吊钩", "吊索具", "塔机", "群塔", "构件缺陷", "锈蚀", "报废"]),
    # 人员资质置于机械起重之后：与"机械"无重叠，顺序无影响
    ("人员资质", ["资质", "特种作业", "证件", "无证", "作业票", "交底", "培训", "监护", "审批", "专人"]),
    ("工程质量", ["偷工减料", "不合格材料", "不按图", "关键部位", "按合格签字"]),
    ("材料堆放", ["杂物", "堆放", "乱堆", "废料"]),
    ("有限空间", ["有限空间", "井内", "井下作业", "封闭空间"]),
    ("文明施工", ["裸土", "渣土", "扬尘", "建筑垃圾", "废料", "积水", "排水", "围挡", "堆码超高", "安全网"]),
    ("警示警戒", ["警示", "标志", "标识", "警戒线", "警戒区", "警示牌", "交通警示", "穿越围栏", "跨越围栏"]),
    ("动火消防", ["动火", "灭火", "消防", "吸烟", "明火", "气瓶", "氧气", "乙炔"]),
]

VOTE_SCORE = 0.45
TRACK_SCORE = 0.50
GAP_FILL_SCORE = 0.55

# ---------- 事件匹配 / 邻近回退参数 ----------
MATCH_RECENT_SAMPLES = 8                # IoU 匹配时回看的最近样本数（配合事件中位框，见 bug 清单 M4）
SMALL_OBJ_MAX_DIM = 200.0               # "小目标"判定：框最长边超过该值时不启用邻近回退
NEIGHBOR_DIST_FACTOR = 1.5              # 邻近回退最大中心距 = max(两框最长边) × 该系数
NEIGHBOR_DIST_SHORT_SIDE_RATIO = 0.14   # 未知图像尺寸时按短边比例估算（约等效 1080p 的 150px，见 L1）
NEIGHBOR_MAX_DIST_DEFAULT_PX = 150.0    # 图像尺寸未知时的兜底阈值


def category_of(title):
    """按关键词把违章标题归到类别。

    最长匹配关键词优先：标题命中多个类别时，取"匹配到最长关键词"的类别（越长越具体），
    长度相同则按 CATEGORY_KEYWORDS 定义顺序取先者。
    原实现"第一个命中即返回"会让含共有词（如"防护"）的标题误分类（bug 清单 L7）。
    """
    t = title or ""
    best_cat = None
    best_len = 0
    for cat, kws in CATEGORY_KEYWORDS:
        for kw in kws:
            if kw in t and len(kw) > best_len:
                best_len = len(kw)
                best_cat = cat
    return best_cat


def same_event_title(a, b):
    """不同具体标题不能仅因大类、位置相同而合并。

    逐帧标题通常已在 VLM 出口归一。未知/空标题不作模糊猜配，避免把
    建筑垃圾的描述嫁接到积水事件；不同写法可能多留事件，不能因此删线索。
    """
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    a, b = a.strip(), b.strip()
    return bool(a and b and a == b)


def _crop(gray, box):
    """按 bbox 从灰度图裁模板。

    键值一律走 _num 容错：samples 里的 bbox 可能是 dict 但缺键/含 None
    （add() 只挡了"非 dict"），硬取 box["x"] 会 KeyError 中断整个 finalize。
    缺失分量按 0 兜底后，裁剪区域必然退化（<8px）→ 返回 None 跳过该帧。
    """
    h, w = gray.shape[:2]
    x1 = int(max(0, _num(box.get("x"))))
    y1 = int(max(0, _num(box.get("y"))))
    x2 = int(min(w, _num(box.get("x")) + _num(box.get("w"))))
    y2 = int(min(h, _num(box.get("y")) + _num(box.get("h"))))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return gray[y1:y2, x1:x2]


def _search_region(gray, box, margin):
    """模板搜索区域（box 应为完整的 fused 中位框；仍做容错防外层数据异常）"""
    h, w = gray.shape[:2]
    bx, by = _num(box.get("x")), _num(box.get("y"))
    bw, bh = _num(box.get("w")), _num(box.get("h"))
    x1 = int(max(0, bx - margin))
    y1 = int(max(0, by - margin))
    x2 = int(min(w, bx + bw + margin))
    y2 = int(min(h, by + bh + margin))
    return x1, y1, x2, y2


def _match(search_roi, tpl):
    if search_roi is None or tpl is None:
        return 0.0, None
    if tpl.std() < 5:
        return 0.0, None
    if (search_roi.shape[0] < tpl.shape[0]
            or search_roi.shape[1] < tpl.shape[1]):
        return 0.0, None
    res = cv2.matchTemplate(search_roi, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return float(max_val), max_loc


def _sharpness(gray, box):
    crop = _crop(gray, box)
    if crop is None:
        return 0.0
    return cv2.Laplacian(crop, cv2.CV_64F).var()


def tighten_bbox_local(frame_bgr, box, max_shrink=0.45, max_center_shift=0.25):
    """本地框收缩（零API开销）：GrabCut在框内分离目标本体，返回紧致包围盒（原帧像素坐标）。
    修正大模型框整体偏大、含大量背景的问题。保守策略：只收缩不放大、
    单边收缩不超max_shrink、中心偏移不超边长的max_center_shift倍，任一校验不过返回None。"""
    h, w = frame_bgr.shape[:2]
    try:
        x, y = int(box["x"]), int(box["y"])
        bw, bh = int(box["w"]), int(box["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if bw < 32 or bh < 32:
        return None
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + bw), min(h, y + bh)
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None

    pad = 4
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)
    roi = frame_bgr[ry1:ry2, rx1:rx2]
    rh, rw = roi.shape[:2]

    # GrabCut初始化：框边缘窄条=确定背景，中央50%=可能前景、中心30%=确定前景。
    # 原「中心50%=确定前景」在框稍有偏移时会把背景污染进前景GMM，
    # 误分割导致中心漂移并传播到该事件所有帧
    mask = np.full((rh, rw), cv2.GC_PR_BGD, dtype=np.uint8)
    bx = max(3, int(rw * 0.15))
    by = max(3, int(rh * 0.15))
    mask[:by, :] = cv2.GC_BGD
    mask[rh - by:, :] = cv2.GC_BGD
    mask[:, :bx] = cv2.GC_BGD
    mask[:, rw - bx:] = cv2.GC_BGD
    mask[int(rh * 0.25):int(rh * 0.75) + 1, int(rw * 0.25):int(rw * 0.75) + 1] = cv2.GC_PR_FGD
    mask[int(rh * 0.35):int(rh * 0.65) + 1, int(rw * 0.35):int(rw * 0.65) + 1] = cv2.GC_FGD

    try:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(roi, mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k5)

    # 前景占比校验：太少=目标没分出来，太多=等于原框没有收缩价值
    ratio = float(fg.sum()) / float(rw * rh)
    if ratio < 0.08 or ratio > 0.92:
        return None

    ys, xs = np.where(fg > 0)
    # 映射回原帧坐标，并裁到原框范围内（只收缩不放大）
    nx1 = min(max(rx1 + int(xs.min()), x1), x2 - 12)
    ny1 = min(max(ry1 + int(ys.min()), y1), y2 - 12)
    nx2 = max(min(rx1 + int(xs.max()) + 1, x2), nx1 + 12)
    ny2 = max(min(ry1 + int(ys.max()) + 1, y2), ny1 + 12)
    nw, nh = nx2 - nx1, ny2 - ny1

    # 收缩幅度限制：单边最多收max_shrink（防止把目标本体切掉）
    if nw < bw * (1 - max_shrink) or nh < bh * (1 - max_shrink):
        return None
    # 中心偏移限制：收缩后框心不应大幅移动（防分割到旁边目标）
    ocx, ocy = x + bw / 2.0, y + bh / 2.0
    ncx, ncy = nx1 + nw / 2.0, ny1 + nh / 2.0
    if abs(ncx - ocx) > bw * max_center_shift or abs(ncy - ocy) > bh * max_center_shift:
        return None
    if nw < 12 or nh < 12:
        return None

    # 边缘证据校验：收缩后的框四边应压在图像边缘（梯度）上，至少不比原框差。
    # GrabCut误分割到平坦区域（如把坑内阴影或附近地面当目标）时新框边缘
    # 梯度会明显低于原框——此时拒绝收缩，保留原框（宁大勿偏）
    try:
        t = 3
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        def _edge_mean(bx1, by1, bx2, by2):
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(rw, bx2), min(rh, by2)
            if bx2 - bx1 < 4 or by2 - by1 < 4:
                return 0.0
            strips = [
                mag[by1:min(by1 + t, by2), bx1:bx2],
                mag[max(by2 - t, by1):by2, bx1:bx2],
                mag[by1:by2, bx1:min(bx1 + t, bx2)],
                mag[by1:by2, max(bx2 - t, bx1):bx2],
            ]
            vals = [float(s.mean()) for s in strips if s.size]
            return sum(vals) / len(vals) if vals else 0.0

        old_edges = _edge_mean(x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1)
        new_edges = _edge_mean(nx1 - rx1, ny1 - ry1, nx2 - rx1, ny2 - ry1)
        if new_edges < old_edges * 0.9:
            return None
    except cv2.error:
        return None

    return {"x": int(nx1), "y": int(ny1), "w": int(nw), "h": int(nh)}


class ViolationEvent:
    def __init__(self, event_id, frame_index, timestamp, violation):
        self.id = event_id
        self.display_id = event_id
        self.category = category_of(violation.get("title", ""))
        self.samples = []
        # 重复出现计数：同一位置同类违章间隔较久再次出现时 +1（见 _merge_repeat_events）
        self.repeat_count = 1
        self.repeat_ts = [timestamp]   # 每一次出现的起始时间
        self.add(frame_index, timestamp, violation)

    def add(self, frame_index, timestamp, violation):
        bbox = violation.get("bbox")
        if not isinstance(bbox, dict):
            bbox = {"x": 0, "y": 0, "w": 0, "h": 0}
        self.samples.append({
            "frame_index": frame_index,
            "timestamp": timestamp,
            "bbox": dict(bbox),
            "violation": violation,
        })

    @property
    def last_ts(self):
        return self.samples[-1]["timestamp"]

    @property
    def first_ts(self):
        return self.samples[0]["timestamp"]

    def match_score(self, violation, timestamp, iou_th, max_gap,
                    neighbor_max_dist=NEIGHBOR_MAX_DIST_DEFAULT_PX):
        if timestamp - self.last_ts > max_gap:
            return -1.0
        bbox = violation.get("bbox")
        if not isinstance(bbox, dict):
            return -1.0
        # 数值一律走 as_number：w/h 为 None 时 `None <= 0` 会 TypeError（调用方
        # update() 无兜底，会中断整个事件追踪）；NaN 的比较恒为 False 会穿透检查
        bw, bh = _num(bbox.get("w")), _num(bbox.get("h"))
        if bw <= 0 or bh <= 0:
            return -1.0
        if not same_event_title(self.rep_title, violation.get("title")):
            return -1.0
        cat = category_of(violation.get("title", ""))
        if self.category is not None and cat is not None and cat != self.category:
            return -1.0
        if self.category is None or cat is None:
            if _titles_similar(self.rep_title, violation.get("title", "")) < 0.35:
                return -1.0
        # 匹配基准：事件中位框（代表整体位置，对无人机缓慢漂移鲁棒）+ 最近若干帧样本。
        # 原实现只比较最后 4 帧，长事件目标漂移后会漏匹配、同一目标被拆成多个事件（bug 清单 M4）
        candidates = self.samples[-MATCH_RECENT_SAMPLES:]
        if self.samples:
            candidates = list(candidates) + [{"bbox": _median_box([s["bbox"] for s in self.samples])}]
        best = 0.0
        for s in candidates:
            best = max(best, _iou(bbox, s["bbox"]))
        if best >= iou_th:
            return best
        # 小目标邻近回退：模型框帧间跳动大时，同类别+时间紧邻+中心距离在阈值内
        # 也视为同一目标（仅限人员/设备尺度的小框，大场景框仍靠IoU）
        bx, by = _num(bbox.get("x")), _num(bbox.get("y"))
        if timestamp - self.last_ts <= 30.0:
            for s in self.samples[-2:]:
                b2 = s["bbox"]
                if not isinstance(b2, dict):
                    continue
                b2x, b2y = _num(b2.get("x")), _num(b2.get("y"))
                b2w, b2h = _num(b2.get("w")), _num(b2.get("h"))
                d1 = max(bw, bh)
                d2 = max(b2w, b2h)
                if d1 > SMALL_OBJ_MAX_DIM or d2 > SMALL_OBJ_MAX_DIM:
                    continue
                cx1, cy1 = bx + bw / 2.0, by + bh / 2.0
                cx2, cy2 = b2x + b2w / 2.0, b2y + b2h / 2.0
                dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                if dist <= min(max(d1, d2) * NEIGHBOR_DIST_FACTOR, neighbor_max_dist):
                    return iou_th * 0.9  # 略低于IoU阈值，表示匹配但置信度稍低
        return -1.0

    @property
    def rep_title(self):
        titles = [s["violation"].get("title", "") for s in self.samples]
        titles = [t for t in titles if t]
        if not titles:
            return ""
        return Counter(titles).most_common(1)[0][0]

    def finalize(self, gray_fn, img_w, img_h, min_frames, report_mode="all"):
        self.min_frames = max(1, int(min_frames or 1))
        # 同帧多个样本去重（保留最后一个）
        per_frame = {}
        for s in self.samples:
            per_frame[s["frame_index"]] = s
        self.samples = sorted(per_frame.values(), key=lambda s: s["timestamp"])
        self.frame_count = len(self.samples)
        # 上报口径由 event_report_mode 决定（此前该参数恒为 True、min_frames 形同虚设）：
        #   "all"            —— 任何检出的违章都直接作为确认事件上报（默认，保持既有行为）
        #   "confirmed_only" —— 画面数达到 min_frames 才确认为事件，不足的标记为待复核
        #                       （不进入成果/证据包，用于抑制单帧误报）
        self.confirmed = (report_mode != "confirmed_only"
                          or self.frame_count >= max(1, int(self.min_frames or 1)))

        boxes = [s["bbox"] for s in self.samples]
        self.fused = _median_box(boxes)

        # 锚帧：目标最清晰的帧
        anchor_s = None
        anchor_score = -1.0
        for s in self.samples:
            g = gray_fn(s["frame_index"])
            sc = _sharpness(g, s["bbox"]) if g is not None else 0.0
            if sc > anchor_score:
                anchor_score = sc
                anchor_s = s
        if anchor_s is None:
            anchor_s = self.samples[0]
        self.anchor_frame = anchor_s["frame_index"]
        self.anchor_ts = anchor_s["timestamp"]
        self.rep_violation = dict(anchor_s["violation"])
        self.rep_violation["title"] = self.rep_title

        if not self.confirmed:
            self.refined = dict(self.fused)
            self.stability = None
            self.per_frame = {
                self.anchor_frame: {
                    "bbox": dict(self.fused), "score": 1.0, "detected": True,
                }
            }
            return

        self.refined, self.stability = self._vote_on_anchor(gray_fn)
        # 投票分散（稳定度低）说明模板匹配不可靠：放弃校准，回退中位数框
        self.tracking_reliable = self.stability is not None and self.stability <= 0.35
        if not self.tracking_reliable:
            self.refined = dict(self.fused)
        self.per_frame = self._track_frames(gray_fn, img_w, img_h)

    def _vote_on_anchor(self, gray_fn):
        """各帧以自己的框截图为模板，到锚帧中投票目标位置，取中位数"""
        anchor_gray = gray_fn(self.anchor_frame)
        if anchor_gray is None:
            return dict(self.fused), None

        w, h = self.fused["w"], self.fused["h"]
        margin = max(0.8 * max(w, h), 40)
        sx1, sy1, sx2, sy2 = _search_region(anchor_gray, self.fused, margin)
        search_roi = anchor_gray[sy1:sy2, sx1:sx2]

        centers = []
        n = len(self.samples)
        step = max(1, n // 12)
        picked = self.samples[::step]
        if picked[-1] != self.samples[-1]:
            picked.append(self.samples[-1])

        for s in picked:
            g = gray_fn(s["frame_index"])
            if g is None:
                continue
            tpl = _crop(g, s["bbox"])
            if tpl is None:
                continue
            score, loc = _match(search_roi, tpl)
            if score >= VOTE_SCORE and loc is not None:
                cx = sx1 + loc[0] + tpl.shape[1] / 2.0
                cy = sy1 + loc[1] + tpl.shape[0] / 2.0
                centers.append((cx, cy))

        if len(centers) < 2:
            return dict(self.fused), None

        arr = np.array(centers)
        med = np.median(arr, axis=0)
        # 离群剔除：个别帧的模板误匹配会把投票点甩到远处，先按 MAD 剔除再取中位数，
        # 避免锚帧框被个别错误投票拉偏（框漂移来源之一）
        dists = np.linalg.norm(arr - med, axis=1)
        mad = float(np.median(dists))
        limit = max(3.0 * mad, 0.5 * float(np.hypot(w, h)))
        inliers = arr[dists <= limit]
        if len(inliers) >= 2:
            arr = inliers
            med = np.median(arr, axis=0)
            dists = np.linalg.norm(arr - med, axis=1)
        stability = float(np.median(dists) / max(np.hypot(w, h), 1e-6))

        refined = {
            "x": med[0] - w / 2.0,
            "y": med[1] - h / 2.0,
            "w": w,
            "h": h,
        }
        return refined, stability

    def _track_frames(self, gray_fn, img_w, img_h):
        """锚帧校准框为模板，逐帧定位目标；检测结果帧必有框，漏检帧仅高分时补框。
        跟踪不可靠的事件只用中位数框且不补漏。
        注意：无人机在移动，同一目标在各帧的像素位置不同——锚帧框/中位数框
        不能跨帧套用，回退时优先用该帧自己检测到的bbox（它是在该帧上检测的，
        天然对准该帧里的目标本体）。"""
        own_boxes = {s["frame_index"]: dict(s["bbox"]) for s in self.samples}
        if not self.tracking_reliable:
            return {
                s["frame_index"]: {
                    "bbox": dict(own_boxes.get(s["frame_index"], self.fused)),
                    "score": 0.0, "detected": True,
                }
                for s in self.samples
            }
        anchor_gray = gray_fn(self.anchor_frame)
        if anchor_gray is None:
            return {}
        tpl = _crop(anchor_gray, self.refined)
        if tpl is None:
            return {}

        w, h = self.refined["w"], self.refined["h"]
        # 搜索范围：目标尺寸的0.75倍（向四周扩展搜索）
        margin = max(0.75 * max(w, h), 50)
        sample_frames = {s["frame_index"] for s in self.samples}
        first_fi = self.samples[0]["frame_index"]
        last_fi = self.samples[-1]["frame_index"]

        result = {}
        for fi, g in gray_fn.items_in_range(first_fi, last_fi):
            detected = fi in sample_frames
            if fi == self.anchor_frame:
                result[fi] = {"bbox": dict(self.refined), "score": 1.0,
                              "detected": detected}
                continue
            if g is None:
                continue
            sx1, sy1, sx2, sy2 = _search_region(g, self.refined, margin)
            score, loc = _match(g[sy1:sy2, sx1:sx2], tpl)
            if score >= TRACK_SCORE and loc is not None:
                box = {
                    "x": sx1 + loc[0],
                    "y": sy1 + loc[1],
                    "w": tpl.shape[1],
                    "h": tpl.shape[0],
                }
                result[fi] = {"bbox": box, "score": score, "detected": detected}
            elif detected:
                # 该帧模板匹配失败但VLM检测到了目标：用该帧自己的检测框，
                # 不套用锚帧框（无人机移动后锚帧坐标在该帧上可能已偏离目标）
                own = own_boxes.get(fi)
                result[fi] = {"bbox": dict(own) if own else dict(self.refined),
                              "score": score, "detected": detected}
            # 漏检帧且匹配低分：不补框
        return result

    def build_violation(self, bbox):
        """把事件转成一条"违规"记录（用于报告/标注图/整改通知书）。

        bbox 容错：缺键或非数值时回退到事件中位框/0 —— 一条不完整的框不应该
        让整个事件阶段崩溃（同类问题见 bbox_utils.median_box 的加固）。
        """
        src = bbox if isinstance(bbox, dict) else {}
        fallback = self.fused if isinstance(getattr(self, "fused", None), dict) else {}

        def _num(key):
            for holder in (src, fallback):
                try:
                    return float(holder.get(key))
                except (TypeError, ValueError):
                    continue
            return 0.0

        v = dict(self.rep_violation)
        v["bbox"] = {"x": _num("x"), "y": _num("y"), "w": _num("w"), "h": _num("h")}
        v["event_id"] = self.display_id
        v["event_confirmed"] = self.confirmed
        v["event_frames"] = self.frame_count
        title = self.rep_title or "违章"
        v["title"] = f"{title}（事件{self.display_id}）"
        return v

    def stability_label(self):
        if self.stability is None:
            return "—"
        if self.stability <= 0.15:
            return "高"
        if self.stability <= 0.35:
            return "中"
        return "低"


class _GrayCache:
    """帧号->灰度图缓存，避免重复转换"""

    def __init__(self, frames):
        self.frames = frames
        self.cache = {}
        # 排序键只算一次，避免每次遍历都全量 sorted（见 L9）
        self._sorted_keys = sorted(frames.keys())

    def __call__(self, frame_index):
        if frame_index not in self.cache:
            img = self.frames.get(frame_index)
            if img is None:
                self.cache[frame_index] = None
            else:
                self.cache[frame_index] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.cache[frame_index]

    def items_in_range(self, start_fi, end_fi):
        """只迭代 [start_fi, end_fi] 区间内的帧。

        在已排序帧号上二分定位，不再为靠后事件重复扫描此前所有帧。
        只对区间内帧按原顺序取灰度，保留按需解码及缓存语义。
        """
        start = bisect_left(self._sorted_keys, start_fi)
        stop = bisect_right(self._sorted_keys, end_fi, lo=start)
        for index in range(start, stop):
            fi = self._sorted_keys[index]
            yield fi, self(fi)


class EventTracker:
    def __init__(self, match_iou=0.25, max_gap=90.0, min_frames=2, img_size=None,
                 report_mode="all", merge_repeats=True, repeat_iou=0.35):
        self.match_iou = match_iou
        self.max_gap = max_gap
        self.min_frames = max(1, int(min_frames or 1))
        # 上报口径："all"=全部上报（默认）；"confirmed_only"=达到 min_frames 才确认
        self.report_mode = report_mode
        # 同类重复违章自动合并（同一位置间隔较久再次出现 → 并入原事件记次数）
        self.merge_repeats = bool(merge_repeats)
        self.repeat_iou = float(repeat_iou)
        # 图像短边（用于把邻近回退阈值从绝对像素改为按画面比例，见 L1）
        self.img_short_side = 0
        if img_size:
            try:
                w, h = int(img_size[0] or 0), int(img_size[1] or 0)
                self.img_short_side = min(w, h)
            except (TypeError, ValueError, IndexError):
                self.img_short_side = 0
        self.events = []
        self.frame_map = {}
        self._next_id = 1

    def _neighbor_max_dist(self):
        """邻近回退的最大中心距：优先按图像短边比例，未知尺寸时退回默认像素阈值（见 L1）"""
        if self.img_short_side:
            return self.img_short_side * NEIGHBOR_DIST_SHORT_SIDE_RATIO
        return NEIGHBOR_MAX_DIST_DEFAULT_PX

    def update(self, frame_index, timestamp, violations):
        ids = [None] * len(violations)
        claimed = set()
        for i, v in enumerate(violations):
            best_e = None
            # 初始阈值设为 match_iou * 0.9，让小目标邻近回退也能命中
            # （邻近回退返回 iou_th * 0.9，正常IoU匹配返回 >= iou_th）
            best_score = self.match_iou * 0.9
            for e in self.events:
                if e.id in claimed:
                    continue
                if timestamp - e.last_ts > self.max_gap:
                    continue
                try:
                    score = e.match_score(v, timestamp, self.match_iou, self.max_gap,
                                          self._neighbor_max_dist())
                except Exception as ex:
                    # 单条违章匹配失败不能中断整个追踪（update 无外层兜底，
                    # 一个坏 bbox 会把整场分析带崩）——视作不匹配，落到新事件
                    print(f"[tracker] 违章匹配异常（视作新事件）: {type(ex).__name__}: {ex}")
                    score = -1.0
                if score >= best_score:
                    best_e = e
                    best_score = score
            if best_e is None:
                e = ViolationEvent(self._next_id, frame_index, timestamp, v)
                self._next_id += 1
                self.events.append(e)
                claimed.add(e.id)
                ids[i] = e.id
            else:
                best_e.add(frame_index, timestamp, v)
                claimed.add(best_e.id)
                ids[i] = best_e.id
            # 把 event_id 写入 violation dict，后续校准用 _event_id 查找，
            # 不依赖列表索引对齐（脆弱的隐式不变量）
            v["_event_id"] = ids[i]
        self.frame_map[frame_index] = ids

    def finalize(self, frames, img_w=0, img_h=0):
        """frames: {frame_index: BGR图}，返回按首次出现排序的事件列表"""
        self._merge_split_events()
        if self.merge_repeats:
            self._merge_repeat_events()
        gray_fn = _GrayCache(frames)
        for e in self.events:
            e.finalize(gray_fn, img_w, img_h, self.min_frames, self.report_mode)
        self.events.sort(key=lambda e: e.first_ts)
        for i, e in enumerate(self.events, 1):
            e.display_id = i
        return self.events

    def _merge_events(self, max_rounds, can_merge, on_merge=None):
        """多轮合并同类事件：每轮扫描所有事件对，命中 can_merge 即把后者并入前者。

        重复执行至无合并或达 max_rounds（合并可能让原本不相邻的事件变为可合并）。
        on_merge(a, b) 在吸收前做额外累计（如重复合并的次数/出现时刻）。
        """
        merged = True
        rounds = 0
        while merged and rounds < max_rounds:
            merged = False
            rounds += 1
            self.events.sort(key=lambda e: e.first_ts)
            for i in range(len(self.events)):
                a = self.events[i]
                if a is None:
                    continue
                for j in range(i + 1, len(self.events)):
                    b = self.events[j]
                    if b is None:
                        continue
                    if not can_merge(a, b):
                        continue
                    if on_merge:
                        on_merge(a, b)
                    self._absorb_event(a, b)
                    self.events[j] = None
                    merged = True
            self.events = [e for e in self.events if e is not None]

    def _merge_split_events(self):
        """合并被快速移动拆开的同类事件（位置接近+时间衔接，且不同帧）

        重要约束：同一帧出现的同类事件不合并 —— 它们是不同目标（如同一画面里
        有3个未戴安全帽的人），合并会导致多目标被当成一个事件，最终画面上
        只剩1个框但报告里列了多个违规。
        """
        self._merge_events(3, self._can_merge_split)

    # ---- 事件合并的公共判定/动作（_merge_split_events 与 _merge_repeat_events 共用） ----

    @staticmethod
    def _events_share_frames(a, b):
        """两事件是否有共享帧（同帧同类 = 画面里的不同目标，绝不合并）"""
        a_frames = {s["frame_index"] for s in a.samples}
        b_frames = {s["frame_index"] for s in b.samples}
        return bool(a_frames & b_frames)

    @staticmethod
    def _median_iou(a, b):
        """两事件中位框的 IoU"""
        fa = _median_box([s["bbox"] for s in a.samples])
        fb = _median_box([s["bbox"] for s in b.samples])
        return _iou(fa, fb)

    def _absorb_event(self, a, b):
        """把事件 b 并入事件 a：样本按时间重排、_event_id 与 frame_map 同步改指 a"""
        a.samples.extend(b.samples)
        a.samples.sort(key=lambda s: s["timestamp"])
        # 同步 violation dict 的 _event_id 与 frame_map，保持引用一致
        for s in b.samples:
            s["violation"]["_event_id"] = a.id
        for fi, ids in self.frame_map.items():
            for k in range(len(ids)):
                if ids[k] == b.id:
                    ids[k] = a.id

    def _can_merge_split(self, a, b):
        """拆分合并判定：同类 + 时间衔接（≤2×max_gap）+ 不同帧 + 位置接近"""
        if not same_event_title(a.rep_title, b.rep_title):
            return False
        if a.category is None or b.category is None or a.category != b.category:
            return False
        if b.first_ts - a.last_ts > self.max_gap * 2:
            return False
        # 关键修复：两事件有重叠帧（同一帧里各有样本）说明是不同目标，不合并
        if self._events_share_frames(a, b):
            return False
        return self._median_iou(a, b) >= 0.35

    def _can_merge_repeat(self, a, b):
        """重复合并判定：同类（或标题相似）+ 不同帧 + 位置重合（repeat_iou）"""
        if not same_event_title(a.rep_title, b.rep_title):
            return False
        # 类别一致才考虑（都无类别时退回标题相似度）
        if a.category is not None and b.category is not None:
            if a.category != b.category:
                return False
            if _titles_similar(a.rep_title, b.rep_title) < 0.45:
                return False
        else:
            if _titles_similar(a.rep_title, b.rep_title) < 0.5:
                return False
        # 同帧重叠 = 画面里的不同目标，不合并
        if self._events_share_frames(a, b):
            return False
        return self._median_iou(a, b) >= self.repeat_iou

    def _merge_repeat_events(self):
        """同类重复违章合并：同一位置、同类违章，间隔超过常规时间窗后再次出现
        （如同一工点上午、下午各出现一次未戴安全帽），不再各成一条事件，
        而是并入首次事件并累计"重复出现次数"，清单里展示为一条。

        判定条件（全部满足才合并）：
        - 类别相同（或都无类别时标题相似度足够）
        - 中位框 IoU >= repeat_iou（位置重合，视为同一地点/同一目标）
        - 两事件没有共享任何帧（同帧同类 = 画面里的不同目标，绝不合并）

        局限说明：不同人先后出现在同一位置且违章相同的情况会被并入同一条，
        但清单会列出每次出现的时刻，人工复核时可分辨。
        """
        def _accumulate(a, b):
            # 次数与各次出现时刻累计，样本/映射由 _absorb_event 处理
            a.repeat_count += b.repeat_count
            a.repeat_ts.extend(b.repeat_ts)

        self._merge_events(4, self._can_merge_repeat, _accumulate)
