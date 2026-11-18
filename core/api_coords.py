# -*- coding: utf-8 -*-
"""坐标解析 / 网格换算 / YOLO 校准 / 网格自洽（从 api_client.py 拆出）"""

import math
import cv2

from core.bbox_utils import bbox_corners
from core.helmet_evidence import _event_box, _overlap_smaller, is_helmet_candidate
from core.person_evidence import is_person_identity_candidate, mark_lift_occupancy_candidate
from core.violation_vocab import is_harness_candidate, is_lift_occupancy_candidate


NORM_SCALE = 1000.0                 # 归一化坐标基准


NORM_MAX = 1000.5                   # 归一化上界（含浮点容差）


MIN_VALID_BOX = 1.0                 # 宽或高小于该值的框判为无效


CENTER_FMT_MIN_OVERSHOOT = 200.0    # 判定"模型实际给的是中心点格式"所需的最小越界幅度


YOLO_CALIB_MIN_IOU = 0.18           # YOLO 校准匹配的最低 IoU（防误配把框拉到别的目标上）
                                   # 原 0.10 过松：无人机俯视下车辆/驾驶室/建材易被误检成
                                   # person，且 VLM 框本身粗糙时轻微重叠就会把人员框拽到错误目标
YOLO_CALIB_MAX_SHIFT_RATIO = 0.35  # YOLO 校准允许的最大中心位移（相对 VLM 框对角线）
                                   # 原 0.5 可允许近半个框的位移，人员站在车旁时极易整体滑到车上
YOLO_CALIB_MIN_CONF = 0.35         # YOLO 实体置信度低于该值不参与校准（低置信多为误检，贴了更难看）
YOLO_STRONG_CALIB_MIN_CONF = 0.55  # 唯一人员强校准的置信度门槛（高于常规校准：强校准
                                   # 无需 VLM 框佐证就直接贴框，门槛必须更严）。
                                   # helmet 锚点也用此门槛不放宽：0.4x 置信段的 helmet
                                   # 误检率不低（实测斗臂车帧把绝缘遮蔽罩黄色软毯误检为
                                   # helmet conf=0.485，放宽门槛会把框拽到遮蔽罩上）


def _scale_to_orig(x, y, bw, bh, sx, sy):
    """按 sx/sy 缩放框并取整（发送图坐标 → 原图像素坐标）。"""
    return {"x": round(x * sx), "y": round(y * sy),
            "w": round(bw * sx), "h": round(bh * sy)}


def _make_bbox_getter(bbox):
    """构造 bbox 取值器：非法或非有限值（NaN/Infinity）一律返回 None。

    模型的 JSON 里可能出现 NaN/Infinity（Python 的 json 解析默认接受这两个字面量）：
    它们能"穿透"所有大小比较（NaN 参与比较恒为 False、inf 恒大于阈值），一路混进
    report.json 使其成为非法 JSON，或在 int() 换算时抛 OverflowError。入口直接拒掉。
    """
    def _f(*keys):
        for k in keys:
            if k in bbox:
                if isinstance(bbox[k], bool):
                    return None
                try:
                    val = float(bbox[k])
                except (TypeError, ValueError):
                    return None
                return val if math.isfinite(val) else None
        return None
    return _f


def _center_box_intersection(cx, cy, w, h):
    """中心框与左、上边界求交；不能平移右、下边界或把出界框移回画面。

    上边界以外的坐标无歧义；右、下边界可能是发送图像素而非 0-1000，
    此处不擅自按 1000 裁剪，留给坐标尺度识别和最终画面交集处理。
    """
    left, top = cx - w / 2, cy - h / 2
    x1, y1 = max(0.0, left), max(0.0, top)
    x2, y2 = cx + w / 2, cy + h / 2
    if x2 <= x1 or y2 <= y1:
        return None
    return {"x": x1, "y": y1,
            "w": x2 - x1 if left < 0 else w,
            "h": y2 - y1 if top < 0 else h}


def _bbox_from_center(get, w, h):
    """中心点格式 cx,cy,w,h → 坐标左上角及尺寸。"""
    cx = get("cx", "center_x", "x_center", "centerX")
    cy = get("cy", "center_y", "y_center", "centerY")
    if cx is None or cy is None or w is None or h is None or w <= 0 or h <= 0:
        return None
    return _center_box_intersection(cx, cy, w, h)


def _bbox_from_corners(get):
    """角点格式 x1,y1,x2,y2（模型标准输出格式）→ 归一化 x,y,w,h"""
    x1 = get("x1", "x_1", "left", "xmin", "x_min")
    y1 = get("y1", "y_1", "top", "ymin", "y_min")
    x2 = get("x2", "x_2", "right", "xmax", "x_max")
    y2 = get("y2", "y_2", "bottom", "ymax", "y_max")
    if None in (x1, y1, x2, y2):
        return None
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    # x1/y1 轻微越界统一 clamp 到 0，不再因 x1<0 整个丢弃（bug 清单 M1：与 y1 保持一致）
    x1, y1 = max(0.0, x1), max(0.0, y1)
    if x2 - x1 < MIN_VALID_BOX or y2 - y1 < MIN_VALID_BOX:
        return None
    return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


def _bbox_from_mixed(get):
    """混合角点格式 x,y,x2,y2 → 归一化 x,y,w,h"""
    x, y = get("x"), get("y")
    x2, y2 = get("x2", "x_2", "right", "xmax", "x_max"), get("y2", "y_2", "bottom", "ymax", "y_max")
    if None in (x, y, x2, y2):
        return None
    x, x2 = min(x, x2), max(x, x2)
    y, y2 = min(y, y2), max(y, y2)
    x, y = max(0.0, x), max(0.0, y)
    if x2 - x < MIN_VALID_BOX or y2 - y < MIN_VALID_BOX:
        return None
    return {"x": x, "y": y, "w": x2 - x, "h": y2 - y}


def _bbox_from_xywh(get, w, h):
    """x,y,w,h 格式 → 归一化 x,y,w,h（含"其实给的是中心点"的纠偏）"""
    x, y = get("x"), get("y")
    if None in (x, y, w, h):
        return None
    if x < 0 or w <= 0 or h <= 0:
        return None
    # 按左上角解释会超出画面、但按中心点解释在画面内 => 模型实际给的是中心点，纠偏。
    # 仅当越界幅度足够大时才纠偏，否则"靠近右下边缘的正常大目标"会被误判（bug 清单 M3）
    overshoot = max(x + w - NORM_MAX, y + h - NORM_MAX)
    if overshoot >= CENTER_FMT_MIN_OVERSHOOT and (x - w / 2 >= -60) and (y - h / 2 >= -60):
        return _center_box_intersection(x, y, w, h)
    return {"x": x, "y": y, "w": w, "h": h}


def parse_bbox_any(bboxt):
    """把模型输出的 bbox（多种写法）统一为 0-1000 归一化 x,y,w,h；失败返回 None。

    依次尝试：中心点 → 角点 → 混合角点 → xywh。顺序即优先级，与原实现一致。
    """
    if not isinstance(bboxt, dict):
        return None
    get = _make_bbox_getter(bboxt)
    w, h = get("w", "width"), get("h", "height")
    return (_bbox_from_center(get, w, h)
            or _bbox_from_corners(get)
            or _bbox_from_mixed(get)
            or _bbox_from_xywh(get, w, h))


# 人员类违章关键词（只匹配title，不匹配description，防止误判）
# 规则：违章主体必须是人，不是描述中顺便提到人。
# 模块级共享：_calibrate_bbox_with_yolo 与 _apply_entity_refs 都按此判定
MACHINE_KEYWORDS = ("机械作业无监护", "机械作业", "吊装", "起重", "人机混行", "回转半径")

PERSON_KEYWORDS = [
    "安全帽", "未戴安全帽", "佩戴安全帽", "不戴安全帽",
    "工人", "作业人员", "施工人员", "人员未", "人员进",
    "高处作业", "高空作业",
    "安全带", "系安全带", "未系安全带",
    "反光衣", "防护服", "防护用品",
    "未戴帽", "裸头", "未佩戴", "佩戴不规范",
    "作业人员未", "施工人员未",
]

# 全身类违章：主体是整个人（安全带/高处/反光衣/防护服/工人等），
# 必须整体框（person）定位，禁止贴到 head 小框上
FULL_BODY_KEYWORDS = [
    "安全带", "系安全带", "未系安全带",
    "高处作业", "高空作业",
    "反光衣", "防护服", "防护用品",
    "工人", "作业人员", "施工人员", "人员未", "人员进",
    "未佩戴", "佩戴不规范", "作业人员未", "施工人员未",
]

# 安全帽属于多目标高歧义情形，标准标题虽含“作业人员”等全身关键词，
# 也不能因此绕过位置匹配，把唯一检测到的其他人员强行作为违章主体。
HELMET_SPECIFIC_KEYWORDS = ("安全帽", "头盔", "未戴帽", "裸头")


class _CoordMixin:
    """坐标解析 / 网格换算 / YOLO 校准 / 网格自洽（从 api_client.py 拆出）"""

    @staticmethod
    def _apply_entity_refs(result, yolo_prior):
        """实体引用贴框：VLM 在 entity_id 字段引用检测实体时直接采用该实体坐标。

        背景（2026-10-08 斗臂车带电作业案例）：穿绝缘服的坐姿人员约占画面 3%，
        VLM 描述对了违章但坐标估读脱靶（框甩到电杆/绝缘遮蔽罩上），而 YOLO 分块
        检出的 person 实体坐标正确——常规校准靠 IoU 匹配，框脱靶时 IoU=0 匹配失败。
        实体引用把"坐标估读"变成"实体选择"：VLM 只需说明违章主体是哪个实体（如 P2），
        本地贴该实体的像素级精确框，绕过估读与 IoU 匹配两个不可靠环节。

        防护：仅人员类违章生效（实体清单只有 person/helmet/head，区域类违章引用
        实体属于张冠李戴，不贴）；引用的实体不存在时不动作，走原估读+校准链。
        """
        if not yolo_prior:
            return
        entities = {}
        for e in yolo_prior.get("entities", []) or []:
            if isinstance(e, dict) and e.get("id"):
                entities[str(e["id"]).strip()] = e
        if not entities:
            return
        for v in result.get("violations", []) or []:
            if not isinstance(v, dict):
                continue
            if is_lift_occupancy_candidate(v):
                # 人+吊具是关系框，任何单个人或机械实体都不能替换它。
                mark_lift_occupancy_candidate(v)
                continue
            ref = v.get("entity_id")
            if not ref or not isinstance(ref, str):
                continue
            ent = entities.get(ref.strip())
            if not ent:
                continue
            title = str(v.get("title", ""))
            # 机械/吊装类：引用 M*/BT* 时直接采用机械实体框
            if any(kw in title for kw in MACHINE_KEYWORDS) and \
                    (str(ref).upper().startswith("M") or str(ref).upper().startswith("BT")):
                bn = ent.get("bbox_norm") or {}
                x1, y1, x2, y2 = bbox_corners(bn)
                if x2 > x1 and y2 > y1:
                    v["bbox"] = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
                    v["_bound_entity"] = ref
                    continue
            if not any(kw in title for kw in PERSON_KEYWORDS):
                continue
            if is_helmet_candidate(v) or is_person_identity_candidate(v) or is_harness_candidate(v):
                # 模型显式引用也可能选错邻人或误检的桩；引用不是独立真值。
                # 先与校准前的原框比对，不一致时保存两个坐标、保留原框。
                raw = v.get("bbox_raw_1000") or v.get("bbox")
                ref_box = ent.get("bbox_norm") or {}
                rx1, ry1, rx2, ry2 = bbox_corners(ref_box)
                original = _event_box(raw)
                referenced = (rx1, ry1, rx2, ry2) if rx2 > rx1 and ry2 > ry1 else None
                if (ent.get("type") not in ("person", "head", "helmet")
                        or original is None or referenced is None
                        or _overlap_smaller(original, referenced) < 0.5):
                    if original is not None:
                        v["bbox"] = dict(raw)
                    if referenced is not None:
                        v["entity_reference_candidate_bbox_1000"] = {
                            "x": rx1, "y": ry1, "w": rx2 - rx1, "h": ry2 - ry1}
                    v["entity_reference_conflict"] = True
                    v["needs_review"] = True
                    v["needs_box_relocalize"] = True
                    v.setdefault("review_reason", "实体引用与原框位置冲突，需人工重新定位")
                    v.pop("_yolo_calibrated", None)
                    v.pop("_yolo_entity_id", None)
                    continue
            # 全身类违章（安全带/高处作业/反光衣/防护服）必须整体框住人本体：
            # 引用 head/helmet 小实体会把框缩成头部，禁止贴（该类实体只作
            # 头部违章的锚点），此类引用不动作、走原估读+校准链
            if any(kw in title for kw in FULL_BODY_KEYWORDS) and \
                    ent.get("type") in ("head", "helmet"):
                continue
            bn = ent.get("bbox_norm") or {}
            x1, y1, x2, y2 = bbox_corners(bn)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            v["bbox"] = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
            if isinstance(v.get("grid_range"), dict):
                v["grid_range"] = {
                    "col_start": max(1, int(x1 // 50) + 1),
                    "col_end": max(1, min(20, int((x2 - 0.001) // 50) + 1)),
                    "row_start": max(1, int(y1 // 50) + 1),
                    "row_end": max(1, min(20, int((y2 - 0.001) // 50) + 1)),
                }
            v["_yolo_entity_ref"] = True


    @staticmethod
    @staticmethod
    def _draw_entity_marks(img, entities, scale):
        """Set-of-Mark：把检测实体画成带编号的彩色框（就地绘制）。

        颜色：人员=绿、头部=橙、安全帽=蓝；编号用 ASCII 保证 OCR 稳定。
        """
        colors = {"person": (0, 200, 0), "head": (0, 140, 255), "helmet": (255, 120, 0),
                  "bucket_truck": (200, 0, 200)}
        for ent in entities or []:
            bx = ent.get("bbox_px") or {}
            try:
                x1, y1 = int(bx["x1"] * scale), int(bx["y1"] * scale)
                x2, y2 = int(bx["x2"] * scale), int(bx["y2"] * scale)
            except (KeyError, TypeError, ValueError):
                continue
            eid = str(ent.get("id") or "")
            if not eid or x2 <= x1 or y2 <= y1:
                continue
            color = colors.get(ent.get("type"), (0, 200, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), base = cv2.getTextSize(eid, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ly = max(th + 4, y1)
            cv2.rectangle(img, (x1, ly - th - 4), (x1 + tw + 6, ly + base), color, -1)
            cv2.putText(img, eid, (x1 + 3, ly - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _add_coord_grid(img_bgr, inplace=False):
        """在发送给API的图上叠加0-1000坐标网格（20×20主格 + 40×40次格）
        - 每50单位主格线（带刻度数字）：20格/边，估读精度比10格翻倍
        - 每25单位次格线（暗细）：40格/边，辅助插值
        仅用于API输入，标注渲染仍用无网格原图
        inplace=True 时直接在传入数组上绘制（调用方须保证该数组独占、不被复用）"""
        out = img_bgr if inplace else img_bgr.copy()
        h, w = out.shape[:2]

        # 次格线：每25单位，暗黄细线（辅助插值，不标数字）
        for i in range(1, 40):
            if i % 2 == 0:
                continue  # 50单位位置留给主格线画
            u = int(round(i / 40 * w))
            v = int(round(i / 40 * h))
            cv2.line(out, (u, 0), (u, h), (50, 140, 140), 1, cv2.LINE_AA)
            cv2.line(out, (0, v), (w, v), (50, 140, 140), 1, cv2.LINE_AA)

        # 主格线：每50单位，亮黄色细线 + 刻度数字（20格/边）
        for i in range(1, 20):
            u = int(round(i / 20 * w))
            v = int(round(i / 20 * h))
            cv2.line(out, (u, 0), (u, h), (0, 210, 210), 1, cv2.LINE_AA)
            cv2.line(out, (0, v), (w, v), (0, 210, 210), 1, cv2.LINE_AA)

        # 中心十字加粗（500,500位置）
        cv2.line(out, (w // 2, 0), (w // 2, h), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(out, (0, h // 2), (w, h // 2), (0, 255, 255), 2, cv2.LINE_AA)

        # 刻度数字：四边都标（上/下标x刻度，左/右标y刻度）。
        # 20格制：每50单位一个数字（50, 100, 150, ..., 950），共19个。
        # 字体随分辨率缩放，保证小图不挤、大图清晰。
        fs = max(0.35, h / 1080 * 0.45)
        for i in range(1, 20):
            label = str(i * 50)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            u = int(round(i / 20 * w))
            v = int(round(i / 20 * h))
            # x刻度：上边 + 下边
            for ly in (1, h - th - 6):
                x = min(max(u - tw // 2, 1), max(1, w - tw - 1))
                ly = max(1, min(ly, h - th - 4))
                cv2.rectangle(out, (x, ly), (x + tw + 2, ly + th + 3), (0, 0, 0), -1)
                cv2.putText(out, label, (x + 1, ly + th + 1), cv2.FONT_HERSHEY_SIMPLEX, fs,
                            (0, 255, 255), 1, cv2.LINE_AA)
            # y刻度：左边 + 右边
            for lx in (1, w - tw - 6):
                y = min(max(v + th + 1, th + 4), h - 3)
                lx = max(1, min(lx, w - tw - 3))
                cv2.rectangle(out, (lx, y - th - 1), (lx + tw + 3, y + 1), (0, 0, 0), -1)
                cv2.putText(out, label, (lx + 3, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                            (0, 255, 255), 1, cv2.LINE_AA)
        return out


    @staticmethod
    def _parse_bbox(bbox):
        """（薄封装）多种写法 → 0-1000 归一化 x,y,w,h；实现在模块级纯函数里

        拆出模块级函数的理由：格式分支（中心点/角点/混合/xywh）此前挤在一个
        函数里，圈复杂度 29；拆开后每个分支独立可测，主函数只负责"依次尝试"。
        """
        return parse_bbox_any(bbox)

    @staticmethod
    def _grid_cells_to_norm(gr):
        """把模型自报的grid_range换算成0-1000归一化区间 (x_lo, x_hi, y_lo, y_hi)；无效返回None
        自动适配20格制（1-20，每格50单位）和旧10格制（1-10，每格100单位）"""
        if not isinstance(gr, dict):
            return None

        def _int(v):
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        c1, c2 = _int(gr.get("col_start")), _int(gr.get("col_end"))
        r1, r2 = _int(gr.get("row_start")), _int(gr.get("row_end"))
        if None in (c1, c2, r1, r2):
            return None
        # 自动判断是10格制还是20格制：最大值>10则为20格制（每格50单位），否则10格制（每格100单位）
        max_val = max(c1, c2, r1, r2)
        if max_val > 20 or max_val < 1:
            return None
        if max_val <= 10:
            cells_per_side = 10
            units_per_cell = 100.0
        else:
            cells_per_side = 20
            units_per_cell = 50.0
        if not (1 <= c1 <= cells_per_side and 1 <= c2 <= cells_per_side
                and 1 <= r1 <= cells_per_side and 1 <= r2 <= cells_per_side):
            return None
        if c2 < c1:
            c1, c2 = c2, c1
        if r2 < r1:
            r1, r2 = r2, r1
        return ((c1 - 1) * units_per_cell, c2 * units_per_cell,
                (r1 - 1) * units_per_cell, r2 * units_per_cell)


    def _convert_bboxes_to_pixels(self, result, orig_w, orig_h, sent_w=None, sent_h=None):
        """把0-1000归一化bbox换算为原图像素坐标

        坐标歧义消解：小图放大发送后，模型偶尔按"发送图像素"输出坐标（数值≤1000时
        与归一化无法直接区分），误判会把框整体偏移约 sent/1000 倍（如1920宽时约2倍），
        直接落到目标外。这里用模型自报的grid_range裁决：哪种解释的框中心落在
        grid_range格子区间内，就按哪种解释换算。"""
        valid = []

        def keep_unlocated(v, raw_box):
            if is_lift_occupancy_candidate(v):
                def audit_value(value):
                    if isinstance(value, float) and not math.isfinite(value):
                        return str(value)
                    if value is None or isinstance(value, (str, int, float, bool)):
                        return value
                    if isinstance(value, dict):
                        return {str(key): audit_value(item) for key, item in value.items()}
                    if isinstance(value, (tuple, list)):
                        return [audit_value(item) for item in value]
                    return str(type(value).__name__)

                v.setdefault("bbox_unverified_raw", audit_value(raw_box))
                v["bbox"] = None
                mark_lift_occupancy_candidate(v)
                valid.append(v)

        for v in result.get("violations", []) or []:
            # 非法条目直接丢弃，不再塞零尺寸框（bug 清单 M2：零尺寸框会污染报告并错位 frame_map）
            if not isinstance(v, dict):
                continue
            bbox = v.get("bbox")
            if not isinstance(bbox, dict):
                keep_unlocated(v, bbox)
                continue
            if is_lift_occupancy_candidate(v) and any(isinstance(bbox.get(k), bool) for k in ("x", "y", "w", "h")):
                keep_unlocated(v, dict(bbox))
                continue
            try:
                x, y = float(bbox.get("x", 0)), float(bbox.get("y", 0))
                bw, bh = float(bbox.get("w", 0)), float(bbox.get("h", 0))
            except (TypeError, ValueError):
                keep_unlocated(v, dict(bbox))
                continue
            # NaN/Infinity 框直接丢弃：round(nan) 会抛 ValueError 中断整帧分析，
            # inf 则会在后续与 0 比较时"永远成立"而混进报告
            if not all(math.isfinite(v) for v in (x, y, bw, bh)):
                keep_unlocated(v, dict(bbox))
                continue

            max_val = max(x, y, x + bw, y + bh)
            if max_val <= NORM_MAX:
                # 候选歧义：≤1000的数值既可能是归一化，也可能是发送图像素坐标。
                # sent尺寸>1000时像素解释才可能产生≤1000的坐标值；grid_range可用时裁决。
                resolved = False
                cells = self._grid_cells_to_norm(v.get("grid_range"))
                if cells is not None and sent_w and max(sent_w, sent_h) > NORM_MAX:
                    cx_n, cy_n = x + bw / 2.0, y + bh / 2.0  # 归一化解释的框中心
                    cx_p = (x + bw / 2.0) / sent_w * NORM_SCALE  # 像素解释换算回归一化的框中心
                    cy_p = (y + bh / 2.0) / sent_h * NORM_SCALE
                    x_lo, x_hi, y_lo, y_hi = cells
                    in_n = x_lo <= cx_n <= x_hi and y_lo <= cy_n <= y_hi
                    in_p = x_lo <= cx_p <= x_hi and y_lo <= cy_p <= y_hi
                    if in_p and not in_n:
                        # 模型实际给的是发送图像素坐标：按发送图尺寸换算回原图
                        v["bbox"] = _scale_to_orig(
                            x, y, bw, bh, orig_w / sent_w, orig_h / sent_h)
                        print(f"[trace] 坐标歧义消解：按发送图像素解释（grid_range裁决）title={v.get('title', '')[:12]}")
                        resolved = True
                    # 两者都在/都不在格子内：保持归一化解释（原行为）
                elif sent_w and max(sent_w, sent_h) > NORM_MAX:
                    # grid_range 不可用时，用尺寸合理性做补充裁决：
                    # 如果按归一化换算后框占画面 > 70%，明显不合理，判定为像素坐标解释
                    if bw / NORM_SCALE > 0.7 or bh / NORM_SCALE > 0.7:
                        v["bbox"] = _scale_to_orig(
                            x, y, bw, bh, orig_w / sent_w, orig_h / sent_h)
                        print(f"[trace] 坐标歧义消解：按发送图像素解释（尺寸合理性裁决）title={v.get('title', '')[:12]}")
                        resolved = True
                if not resolved:
                    # 归一化0-1000
                    v["bbox"] = _scale_to_orig(
                        x, y, bw, bh, orig_w / NORM_SCALE, orig_h / NORM_SCALE)
            else:
                # 模型返回了像素坐标：按发送图尺寸换算回原图
                sx = orig_w / sent_w if sent_w else 1.0
                sy = orig_h / sent_h if sent_h else 1.0
                v["bbox"] = _scale_to_orig(x, y, bw, bh, sx, sy)

            # 换算后仍无有效尺寸的，丢弃，避免零尺寸框进入事件追踪
            # （同时拦掉 NaN/Infinity：它们与 0 的比较恒为 False，能骗过 w<=0 检查）
            nb = v.get("bbox") or {}
            for _k in ("x", "y", "w", "h"):
                _val = nb.get(_k, 0)
                if not (isinstance(_val, (int, float)) and math.isfinite(_val)):
                    nb = {}
                    break
            if not nb or nb.get("w", 0) <= 0 or nb.get("h", 0) <= 0:
                keep_unlocated(v, dict(bbox))
                continue
            valid.append(v)

        result["violations"] = valid


    @staticmethod
    def _calibrate_bbox_with_yolo(result, yolo_prior):
        """YOLO后向坐标校准（检测引导VLM推理闭环）
        对人员类违章，用YOLO专用检测模型的像素级精确bbox替换VLM的网格估读坐标。
        YOLO检测模型的定位精度远高于VLM通过网格插值估读的坐标，可显著提升人员类违章的定位准确度。

        匹配策略：
        1. 识别人员类违章（仅看title：安全帽、工人、高处作业、安全带、反光衣 等关键词）
           —— 注意：不看description，避免描述里提到"人员"的区域类违章被误匹配
        2. 在YOLO人员实体中找IoU最大的匹配（一对一分配，每人只配一个违章）
        3. 匹配成功则用YOLO的bbox_norm替换，并同步更新grid_range
        """
        if not yolo_prior:
            return
        entities = yolo_prior.get("entities", [])
        if not entities:
            return

        # 人员实体分两类：person（整身，像素级准）与 head（远景裸头小框，仅覆盖头部）。
        # 全身类人员违章只能贴到 person 整框上；贴到 head 小框会把框缩到头顶/甩到一边，
        # 而人员站在车旁时 head 误检还常落在驾驶室上——正是"框跑到车上"的典型来源。
        full_body_entities = [e for e in entities if e.get("type") == "person"
                              and (e.get("conf") or 0) >= YOLO_CALIB_MIN_CONF]
        head_entities = [e for e in entities if e.get("type") == "head"
                         and (e.get("conf") or 0) >= YOLO_CALIB_MIN_CONF]
        # 无 person/head 仍可能有 helmet（绝缘服人员只检出安全帽）：
        # 此时不能提前返回——第四步的"安全帽锚点强校准"仍要用它救偏移的框
        if (not full_body_entities and not head_entities
                and not any(e.get("type") == "helmet" for e in entities)):
            return

        # 人员类违章关键词与全身类关键词统一用模块级 PERSON_KEYWORDS /
        # FULL_BODY_KEYWORDS（与 _apply_entity_refs 共享，防两处判据漂移）

        # 第一步：收集所有人员类违章并计算与每个YOLO实体的匹配分
        candidates = []  # (violation_idx, entity_idx, score, bbox_data)
        violations = result.get("violations", [])

        for vi, v in enumerate(violations):
            # 模型原始输出未经 _convert_bboxes_to_pixels 的非法条目剔除，
            # 此处元素可能是 str/list（非 dict）—— .get 会直接 AttributeError
            if not isinstance(v, dict):
                continue
            if is_lift_occupancy_candidate(v):
                mark_lift_occupancy_candidate(v)
                continue
            # 仅用title判断是否为人员类违章——description中提到"人员"不算
            # 例如"临边防护不足，存在人员坠落风险"不是人员类违章
            title = v.get("title", "")
            is_person_violation = any(kw in title for kw in PERSON_KEYWORDS)
            if not is_person_violation:
                continue

            # 可匹配实体池：安全帽/未戴帽/裸头类允许 person+head；其余仅 person
            is_full_body = any(kw in title for kw in FULL_BODY_KEYWORDS)
            pool = full_body_entities if is_full_body else (full_body_entities + head_entities)
            if not pool:
                continue

            bbox = v.get("bbox")
            if not isinstance(bbox, dict):
                continue
            try:
                vx, vy = float(bbox["x"]), float(bbox["y"])
                vw, vh = float(bbox["w"]), float(bbox["h"])
            except (KeyError, TypeError, ValueError):
                continue
            if vw <= 0 or vh <= 0:
                continue
            # 疑似像素坐标时跳过（大于1000的不是归一化坐标）
            if max(vx, vy, vx + vw, vy + vh) > 1000.5:
                continue

            vcx, vcy = vx + vw / 2, vy + vh / 2
            v_diag = (vw ** 2 + vh ** 2) ** 0.5

            best_eidx = -1
            best_score = -1
            best_iou = 0.0

            for ei, ent in enumerate(pool):
                # 注意 `or {}` 不能省：.get("bbox_norm", {}) 在键存在但值为 None 时
                # 返回 None（默认值只在"缺键"时生效），随后 bn.get 会 AttributeError；
                # 值为 None 的坐标分量也要靠 bbox_corners 吸收，否则 float(None) TypeError。
                bn = ent.get("bbox_norm") or {}
                ex1, ey1, ex2, ey2 = bbox_corners(bn)
                ew, eh = ex2 - ex1, ey2 - ey1
                if ew <= 0 or eh <= 0:
                    continue
                ecx, ecy = (ex1 + ex2) / 2, (ey1 + ey2) / 2

                # 计算IoU
                ix1 = max(vx, ex1)
                iy1 = max(vy, ey1)
                ix2 = min(vx + vw, ex2)
                iy2 = min(vy + vh, ey2)
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                union = vw * vh + ew * eh - inter
                iou = inter / union if union > 0 else 0

                # 中心距离（归一化到框对角线）
                dist = ((vcx - ecx) ** 2 + (vcy - ecy) ** 2) ** 0.5
                dist_ratio = dist / max(v_diag, 1)

                # 综合评分：IoU为主，距离为辅
                score = iou * 2 - dist_ratio * 0.3
                if score > best_score:
                    best_score = score
                    best_iou = iou
                    best_eidx = ei

            # 匹配阈值：IoU 不低于 YOLO_CALIB_MIN_IOU 且 中心位移不超过框尺寸的一部分
            # —— 防止 YOLO 误检（如俯视视角下建材/车辆被识别为 person/head）把框拉到
            # 完全不同的物体上，正是"人员框跑到车上"的根因。
            if best_eidx < 0 or best_score < 0.25 or best_iou < YOLO_CALIB_MIN_IOU:
                continue
            # 位移护栏：YOLO 实体中心与 VLM 框中心距离，不得超过 VLM 框对角线
            # 的 YOLO_CALIB_MAX_SHIFT_RATIO 倍。超过说明两者很可能不是同一目标。
            best_ent = pool[best_eidx]
            bn_best = best_ent.get("bbox_norm") or {}
            _sx1, _sy1, _sx2, _sy2 = bbox_corners(bn_best)
            becx = (_sx1 + _sx2) / 2
            becy = (_sy1 + _sy2) / 2
            dist = ((vcx - becx) ** 2 + (vcy - becy) ** 2) ** 0.5
            if dist > v_diag * YOLO_CALIB_MAX_SHIFT_RATIO:
                continue

            candidates.append((vi, best_ent, best_score))

        # 第二步：一对一贪心分配——按分数从高到低匹配，每个实体只配一个违章
        # 防止多个违章都匹配到同一个人，导致框都叠在同一个人身上
        candidates.sort(key=lambda x: x[2], reverse=True)
        used_entity_ids = set()
        matched = {}  # vi -> entity

        for vi, ent, score in candidates:
            eid = id(ent)
            if eid in used_entity_ids:
                continue
            used_entity_ids.add(eid)
            matched[vi] = ent

        # 第三步：应用匹配结果
        def _apply_entity_box(v, ent, strong=False):
            # 与第一步同样的容缺键取法：实体的 bbox_norm 若缺键/值为 None，
            # 硬取会 KeyError/TypeError 中断整帧分析结果的后处理
            bn = ent.get("bbox_norm") or {}
            nx1, ny1, nx2, ny2 = bbox_corners(bn)
            if nx2 - nx1 <= 0 or ny2 - ny1 <= 0:
                return      # 无效替换框：保留 VLM 原框，别塞一个 0 尺寸框进报告

            # 用YOLO bbox替换VLM的估读bbox
            v["bbox"] = {"x": nx1, "y": ny1, "w": nx2 - nx1, "h": ny2 - ny1}

            # 同步更新grid_range（基于新的bbox重新计算）
            gr = v.get("grid_range")
            if isinstance(gr, dict):
                # 重新计算grid_range：20x20网格，每格50单位
                col_start = max(1, int(nx1 // 50) + 1)
                col_end = max(1, min(20, int((nx2 - 0.001) // 50) + 1))
                row_start = max(1, int(ny1 // 50) + 1)
                row_end = max(1, min(20, int((ny2 - 0.001) // 50) + 1))
                v["grid_range"] = {
                    "col_start": col_start,
                    "col_end": col_end,
                    "row_start": row_start,
                    "row_end": row_end,
                }

            # 标记已校准（方便调试和追踪）
            v["_yolo_calibrated"] = True
            v["_yolo_entity_id"] = ent.get("id", "")
            if strong:
                v["_yolo_strong_calibrated"] = True

        def _strong_apply_all(ent, label):
            """把实体框强校准贴给所有未匹配的全身类违章（安全带/高处/工人等）。"""
            for vi, v in enumerate(violations):
                if vi in matched or not isinstance(v, dict):
                    continue
                title = v.get("title", "")
                if not any(kw in title for kw in FULL_BODY_KEYWORDS):
                    continue
                if is_helmet_candidate(v) or is_person_identity_candidate(v) or is_harness_candidate(v):
                    # 与本函数下方“安全帽等多目标类不启用强校准”的约定一致。
                    # 无空间匹配时保留原框和候选，不把检测不到的人换成另一人。
                    # 即使后续有实体引用，原估读与唯一人员位置不一致仍是
                    # 冲突信号；不静默断言哪个框正确，保留待核和重定位标记。
                    v["needs_review"] = True
                    v["needs_box_relocalize"] = True
                    if not v.get("review_reason"):
                        kind = "安全帽" if is_helmet_candidate(v) else "安全带" if is_harness_candidate(v) else "人员身份"
                        v["review_reason"] = f"{kind}候选与人员位置未可靠匹配，需人工重新定位"
                    continue
                _apply_entity_box(v, ent, strong=True)
                print(f"[trace] YOLO{label}强校准: title={str(title)[:16]}")

        for vi, ent in matched.items():
            _apply_entity_box(violations[vi], ent)

        # 第四步：唯一人员强校准 —— 修"看得到人、框却在设备上"的定位偏移。
        # 场景：单人作业（斗臂车带电作业/登杆等）中，VLM 把人员违章的框估读到了
        # 设备上（与 YOLO person 框 IoU≈0），常规校准因 IoU 门槛不救，错框原样
        # 上报（实测框到了绝缘臂黄色护套上，而人在画面中下方的绝缘斗内）。
        # 画面恰有一个高置信 person 且违章属全身类（安全带/高处作业/工人等）时，
        # 该违章的框必然应框住这个人：直接贴 YOLO person 框——本地检测模型的
        # 定位精度远高于 VLM 网格估读。安全帽等多目标类不启用（画面可能有多人，
        # YOLO 漏检其一会把框挪到另一个人上）。
        strong_pool = [e for e in full_body_entities
                       if (e.get("conf") or 0) >= YOLO_STRONG_CALIB_MIN_CONF]
        if len(strong_pool) == 1:
            _strong_apply_all(strong_pool[0], "唯一人员")
        elif not strong_pool:
            # 锚点强校准（无高置信 person 时的兜底）：远景小人/异形着装人员
            # 整身框常检不出，但头盔类小目标检测稳定。取唯一高置信锚点
            # （helmet 或 head），下扩估算人体范围贴给未匹配的全身类违章。
            # 估框不完美，但远好于 VLM 估读框甩到设备上（框到绝缘臂的案例）。
            anchors = [e for e in entities if e.get("type") == "helmet"
                       and (e.get("conf") or 0) >= YOLO_STRONG_CALIB_MIN_CONF]
            anchor_kind = "安全帽"
            if len(anchors) != 1:
                # head（未戴帽裸头）：远景小人的 person 框常低于置信度门槛，
                # head 是画面里唯一的人员定位证据；且 head 出现本身与
                # "未正确佩戴安全帽"违章同源，定位方向不会跑偏
                anchors = [e for e in entities if e.get("type") == "head"
                           and (e.get("conf") or 0) >= YOLO_STRONG_CALIB_MIN_CONF]
                anchor_kind = "裸头"
            if len(anchors) == 1:
                bn = anchors[0].get("bbox_norm") or {}
                hx1, hy1, hx2, hy2 = bbox_corners(bn)
                hw, hh = hx2 - hx1, hy2 - hy1
                if hw > 0 and hh > 0:
                    est_ent = {
                        "id": anchors[0].get("id", ""),
                        "conf": anchors[0].get("conf"),
                        "bbox_norm": {
                            "x1": max(0.0, hx1 - 0.4 * hw),
                            "y1": hy1,
                            "x2": min(1000.0, hx2 + 0.4 * hw),
                            "y2": min(1000.0, hy2 + 2.2 * hh),
                        },
                    }
                    _strong_apply_all(est_ent, f"{anchor_kind}锚定")


    @staticmethod
    def _reconcile_bbox_with_cells(result):
        """网格区间自洽校验（本地，零API调用）
        模型会同时输出bbox和grid_range（目标跨越的主格区间）。读大刻度标签得到grid_range
        比直接估bbox坐标更可靠，当bbox中心落在grid_range之外时，
        把bbox平移进网格区间内；偏差过大视为grid_range不可信，不动。
        自动适配20格制（每格50单位）和旧10格制（每格100单位）。"""
        for v in result.get("violations", []):
            if is_lift_occupancy_candidate(v):
                mark_lift_occupancy_candidate(v)
                continue
            # 实体引用贴框是像素级精确值，grid_range 也已同步按实体框重算，
            # 网格自洽平移只会把好框推走——直接跳过
            if v.get("_yolo_entity_ref") or v.get("entity_reference_conflict"):
                continue
            bbox = v.get("bbox")
            gr = v.get("grid_range")
            if not isinstance(bbox, dict) or not isinstance(gr, dict):
                continue

            cells = _CoordMixin._grid_cells_to_norm(gr)
            if cells is None:
                continue
            x_lo, x_hi, y_lo, y_hi = cells
            # 根据grid_range自动判断每格多少单位，用于阈值换算
            cell_size = (x_hi - x_lo) / max(1, (gr.get("col_end", 1) - gr.get("col_start", 0) + 1)) \
                if isinstance(gr.get("col_end"), int) and isinstance(gr.get("col_start"), int) else 50.0

            try:
                bx, by = float(bbox["x"]), float(bbox["y"])
                bw, bh = float(bbox["w"]), float(bbox["h"])
            except (KeyError, TypeError, ValueError):
                continue
            if bw <= 0 or bh <= 0:
                continue
            # 疑似像素坐标（放大发送后像素值可超1000）时跳过，交给像素换算分支处理
            if max(bx, by, bx + bw, by + bh) > 1000.5:
                continue

            cx, cy = bx + bw / 2, by + bh / 2
            tx = min(max(cx, x_lo), x_hi)
            ty = min(max(cy, y_lo), y_hi)
            if tx == cx and ty == cy:
                continue
            # 大范围框（基坑/围挡等横跨多格的区域）的grid_range粒度粗、误报率高，
            # 平移极易把本来正确的框推到目标外 —— 只对小框做网格平移
            # 阈值按每格大小动态调整：约3格宽度以上视为大框
            large_box_thresh = cell_size * 3.0
            if bw > large_box_thresh or bh > large_box_thresh:
                continue
            # 平移幅度护栏：超过0.6格说明grid_range多半误读；
            # 超过框自身尺寸更是必然把框推到别的物体上。宁可不平移（VLM估读
            # 通常只差半格以内），也不冒把正确框推走的风险
            max_shift = cell_size * 0.6
            max_shift_x = min(max_shift, max(bw, cell_size * 0.3))
            max_shift_y = min(max_shift, max(bh, cell_size * 0.3))
            if abs(tx - cx) > max_shift_x or abs(ty - cy) > max_shift_y:
                continue
            nx = max(0.0, min(1000.0 - bw, tx - bw / 2))
            ny = max(0.0, min(1000.0 - bh, ty - bh / 2))
            v["bbox"] = {"x": nx, "y": ny, "w": bw, "h": bh}


    def _parse_any_bbox(self, data):
        """从各种可能的JSON结构中提取bbox坐标（x1,y1,x2,y2格式）
        支持格式：
        - {"bbox": {"x1":.., "y1":.., "x2":.., "y2":..}}
        - {"bbox": [x1, y1, x2, y2]}
        - {"bbox_2d": [x1, y1, x2, y2]}
        - [{"bbox": ...}]
        - {"results": [{"bbox": ...}]}
        返回 (x1, y1, x2, y2) 或 None
        """
        import json
        if not data:
            return None

        # 如果是字符串先解析
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (TypeError, ValueError):
                return None

        # 如果是列表，取第一个有效元素
        if isinstance(data, list):
            for item in data:
                result = self._parse_any_bbox(item)
                if result:
                    return result
            return None

        if not isinstance(data, dict):
            return None

        # 尝试各种key
        for key in ["bbox", "bbox_2d", "box", "rect", "region"]:
            val = data.get(key)
            if val is None:
                continue
            if isinstance(val, dict):
                # 各种可能的字段名
                x1 = val.get("x1", val.get("left", val.get("x", val.get("start_x"))))
                y1 = val.get("y1", val.get("top", val.get("y", val.get("start_y"))))
                x2 = val.get("x2", val.get("right", val.get("end_x")))
                y2 = val.get("y2", val.get("bottom", val.get("end_y")))
                # 也可能是 x,y,w,h 格式
                if x2 is None and "w" in val:
                    x2 = (val.get("x", 0) + val.get("w", 0))
                if y2 is None and "h" in val:
                    y2 = (val.get("y", 0) + val.get("h", 0))
                if all(v is not None for v in [x1, y1, x2, y2]):
                    try:
                        return float(x1), float(y1), float(x2), float(y2)
                    except (TypeError, ValueError):
                        pass
            elif isinstance(val, list) and len(val) >= 4:
                # [x1, y1, x2, y2]
                try:
                    return float(val[0]), float(val[1]), float(val[2]), float(val[3])
                except (TypeError, ValueError):
                    pass

        # 嵌套在 results / objects / items 里
        for key in ["results", "objects", "items", "data", "detections"]:
            nested = data.get(key)
            if isinstance(nested, list) and nested:
                result = self._parse_any_bbox(nested[0])
                if result:
                    return result

        return None
