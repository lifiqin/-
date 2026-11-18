# -*- coding: utf-8 -*-
"""
图片标注渲染模块 - 根据坐标画框、标注文字
"""
import math
import os
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.bbox_utils import as_number as _as_number
from core.frame_lookup import normalize_timestamp
from core.report_status import frame_analysis_incomplete
from core.helmet_evidence import helmet_display_label, helmet_evidence_note
from core.person_evidence import person_display_label, person_evidence_note
from core.area_localization import area_display_label, area_evidence_note

# 颜色配置（BGR格式，OpenCV用）
COLORS = {
    "重大隐患": (0, 0, 255),      # 红色
    "一般隐患": (0, 165, 255),    # 橙色
    "轻微违章": (0, 255, 255),    # 黄色
    "default": (0, 255, 0),       # 绿色
}

def get_color(severity):
    """根据严重程度获取颜色"""
    return COLORS.get(severity, COLORS["default"])

# 中文字体候选（按优先级）：Windows 自带字体优先，Linux 兜底 DejaVu
_CJK_FONT_PATHS = (
    "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
    "C:/Windows/Fonts/msyhbd.ttc",    # 微软雅黑粗体
    "C:/Windows/Fonts/simhei.ttf",    # 黑体
    "C:/Windows/Fonts/simsun.ttc",    # 宋体
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

def _load_cjk_font(font_size):
    """按优先级加载中文字体，全部失败退回 PIL 默认字体。"""
    for font_path in _CJK_FONT_PATHS:
        try:
            return ImageFont.truetype(font_path, font_size)
        except (OSError, ValueError):
            continue
    return ImageFont.load_default()

def put_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """
    在图片上绘制中文文字（使用PIL解决OpenCV中文乱码问题）
    
    Args:
        img: OpenCV BGR图片
        text: 文字内容
        position: (x, y) 左上角位置
        font_size: 字体大小
        color: 文字颜色 (B, G, R)
        
    Returns:
        标注后的图片
    """
    # 转换为PIL Image
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = _load_cjk_font(font_size)

    # 转换颜色为RGB
    rgb_color = (color[2], color[1], color[0])
    
    # 绘制文字
    draw.text(position, text, font=font, fill=rgb_color)
    
    # 转换回OpenCV格式
    result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return result

def draw_label_with_bg(img, text, position, font_size=20, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """
    绘制带背景的标签
    
    Args:
        img: OpenCV图片
        text: 标签文字
        position: (x, y) 标签左上角
        font_size: 字体大小
        text_color: 文字颜色
        bg_color: 背景颜色
        
    Returns:
        标注后的图片
    """
    # 先计算文字尺寸
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = _load_cjk_font(font_size)

    # 获取文字尺寸
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    padding_x = 8
    padding_y = 4
    
    x, y = position
    bg_x1 = x
    bg_y1 = y
    bg_x2 = x + text_width + padding_x * 2
    bg_y2 = y + text_height + padding_y * 2
    
    # 绘制半透明背景
    overlay = img.copy()
    cv2.rectangle(overlay, (int(bg_x1), int(bg_y1)), (int(bg_x2), int(bg_y2)), bg_color, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    
    # 绘制文字
    text_x = x + padding_x
    text_y = y + padding_y
    img = put_chinese_text(img, text, (text_x, text_y), font_size, text_color)
    
    return img

def _resolve_bbox(bbox, w, h):
    """把违章 bbox 解析为画面内可绘制的 (bx, by, bw, bh, valid)。

    兼容两种格式（{x1,y1,x2,y2} 与 {x,y,w,h}）；数值一律走安全转换。
    valid=False 的情形（统一由调用方画"?"堆叠标记）：
    - 完全出画（必须在钳制前判定，否则负坐标大框会被错误"拉回"角落）
    - 裁剪后剩余面积不足声明面积的 5%（会画成贴边细线，肉眼不可见）
    - 本身就是零尺寸/缺失
    """
    if not isinstance(bbox, dict) or w <= 0 or h <= 0:
        return 0, 0, 0, 0, False

    def _read(keys):
        values = []
        for key in keys:
            raw = bbox.get(key)
            if isinstance(raw, bool):
                return None
            value = _as_number(raw, None)
            if value is None:
                return None
            values.append(value)
        return values

    has_corners = "x1" in bbox and "x2" in bbox
    has_xywh = bbox.get("x") is not None and bbox.get("w") is not None
    if has_corners and not (has_xywh and bbox.get("w", 0)):
        values = _read(("x1", "y1", "x2", "y2"))
        if values is None:
            return 0, 0, 0, 0, False
        _x1, _y1, _x2, _y2 = values
        _bx, _by, _bw, _bh = _x1, _y1, _x2 - _x1, _y2 - _y1
    else:
        values = _read(("x", "y", "w", "h"))
        if values is None:
            return 0, 0, 0, 0, False
        _bx, _by, _bw, _bh = values

    if _bw <= 0 or _bh <= 0:
        return 0, 0, 0, 0, False

    # 【完全出画判定——必须在钳制前做】负坐标大框（如 x=-2500）若直接钳制，
    # 会被错误地"拉回"画面角落画成一个大框，看起来像画在完全不相干的位置。
    x2r, y2r = _bx + _bw, _by + _bh
    if x2r <= 0 or y2r <= 0 or _bx >= w or _by >= h:
        return 0, 0, 0, 0, False

    # 边界检查：框超出画面时裁到画面范围
    # 负起点的端点先求交再量化。若先把负起点钳到 0 再沿用原宽高，
    # 右/下边缘会被人为平移（例如 x=-20,w=40 会画到 x=40 而非 x=20）。
    # 不越左/上边界时保留旧有 int 起点 + int 宽高的量化方式，避免把
    # 原本正常的浮点框扩大 1px。
    bx, by = int(max(0.0, _bx)), int(max(0.0, _by))
    bw = min(int(_bw), w - bx) if _bx >= 0 else int(min(w, x2r)) - bx
    bh = min(int(_bh), h - by) if _by >= 0 else int(min(h, y2r)) - by

    # 坐标严重越界时（声明面积不小、但被画面裁到只剩零头），钳制后会在边缘
    # 留下一条几乎不可见的细线——按无效处理。
    _decl = _bw * _bh
    if bw > 0 and bh > 0 and bw * bh * 20 < _decl:
        return bx, by, 0, 0, False

    if bw <= 0 or bh <= 0:
        return bx, by, 0, 0, False
    return bx, by, bw, bh, True


def _num_label(index):
    """序号 → 带圈数字（超出 10 用普通数字）"""
    circle_numbers = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]
    return circle_numbers[index] if index < len(circle_numbers) else str(index + 1) + "."


def _draw_box_rect(img, bx, by, bw, bh, color):
    """画矩形框 + 四角角标（不含标签）"""
    line_thickness = max(2, int(min(img.shape[:2]) / 400))
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, line_thickness)
    corner_len = int(min(bw, bh) * 0.15)
    corner_thickness = line_thickness + 1
    # 左上角
    cv2.line(img, (bx, by), (bx + corner_len, by), color, corner_thickness)
    cv2.line(img, (bx, by), (bx, by + corner_len), color, corner_thickness)
    # 右上角
    cv2.line(img, (bx + bw, by), (bx + bw - corner_len, by), color, corner_thickness)
    cv2.line(img, (bx + bw, by), (bx + bw, by + corner_len), color, corner_thickness)
    # 左下角
    cv2.line(img, (bx, by + bh), (bx + corner_len, by + bh), color, corner_thickness)
    cv2.line(img, (bx, by + bh), (bx, by + bh - corner_len), color, corner_thickness)
    # 右下角
    cv2.line(img, (bx + bw, by + bh), (bx + bw - corner_len, by + bh), color, corner_thickness)
    cv2.line(img, (bx + bw, by + bh), (bx + bw, by + bh - corner_len), color, corner_thickness)


def _draw_invalid_marker(img, index, color, marker_size, h):
    """无效框的"?"堆叠标记（左下角，按序号纵向排列）"""
    mx = 8
    my = h - marker_size - 8 - index * (marker_size + 10)
    my = max(4, min(my, h - marker_size - 4))
    cv2.rectangle(img, (mx, my),
                  (mx + marker_size, my + marker_size), color, 2)
    cv2.putText(img, "?", (mx + 6, my + marker_size - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def _rects_overlap(a, b):
    return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0]
                or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])


def _place_label(label_x, label_y, label_w, label_h, placed, w, h):
    """标签避让：与已放置标签重叠时逐级下移，直到无重叠或到达下限。

    返回最终 (x, y)。重叠的框（如同一位置报多项违章）编号会纵向错开，
    每个编号都可见——不再互相覆盖。
    """
    label_x = max(0, min(label_x, max(0, w - label_w)))
    max_y = h - label_h - 2
    y = label_y
    rect = (label_x, y, label_w, label_h)
    tries = 0
    while any(_rects_overlap(rect, p) for p in placed) and tries < 80:
        y += label_h + 6
        if y > max_y:
            # 到底后回到首选位置附近向左列挤（换列：右移一个标签宽）
            y = max(4, label_y + (tries % 2) * (label_h + 6))
            label_x = min(max(0, w - label_w), label_x + label_w + 8)
        rect = (label_x, y, label_w, label_h)
        tries += 1
    placed.append(rect)
    return label_x, y


def _boxes_near_identical(a, b):
    """两个框是否近乎重叠（IoU≥0.5）——近乎重叠的违章合并标签写联合编号（如"①③"）。

    注意用 IoU 而非"交集/小框面积"：临边这类大区域框会套住内部的人员小框，
    按包含关系分组会把不同位置的违章错误合并到一起。
    """
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    if inter <= 0:
        return False
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / max(1, union) >= 0.5


def _strip_event_suffix(title):
    """去掉事件追踪追加的"（事件N）"后缀（联合标签里统一写成 事件1/3）"""
    return re.sub(r"（事件\d+）$", "", title)


def _merged_label_text(group, max_text_w, font_size):
    """一组近乎重叠的违章 → 联合标签文本。

    group 成员为 (index, bx, by, bw, bh, color, title)。正常返回如
    "①③ 作业人员未正确佩戴安全帽 / 高处作业未系挂安全带（事件1/3）"；
    超宽放不下时退化为 "①③ 首个标题 等N项（事件1/3）"。
    """
    group = sorted(group, key=lambda e: e[0])
    nums = "".join(_num_label(e[0]) for e in group)
    titles = [_strip_event_suffix(e[6]) for e in group]
    eids = []
    for e in group:
        m = re.search(r"（事件(\d+)）$", e[6])
        if m:
            eids.append(m.group(1))
    suffix = ("（事件" + "/".join(eids) + "）") if eids else ""
    text = nums + " " + " / ".join(titles) + suffix
    if (len(text) * font_size * 1.05) + 16 > max_text_w and len(group) > 1:
        pending = "含待核 " if any("待核" in title for title in titles) else ""
        text = nums + " " + pending + titles[0] + f" 等{len(group)}项" + suffix
    return text



def draw_violation_bbox(img, violation, index=0):
    """
    绘制单个违章标注框（框 + 标签，无碰撞避让；逐帧全量绘制请用 annotate_image）

    Args:
        img: OpenCV BGR图片
        violation: 违章信息dict，包含bbox, title, severity等
        index: 序号（用于显示①②③等）

    Returns:
        标注后的图片
    """
    h, w = img.shape[:2]
    bbox = violation.get("bbox") or {}
    severity = violation.get("severity", "default")
    title = _annotation_candidate_label(violation)

    if violation.get("text_overlay_check_status") == "suspected_text_overlap":
        _draw_invalid_marker(img, index, get_color(severity), max(24, int(min(w, h) / 50)), h)
        return img

    bx, by, bw, bh, valid = _resolve_bbox(bbox, w, h)

    if not valid:
        # bbox 无效时画一个带问号的标记，确保报告里列出的每一项违规在画面上
        # 都有对应视觉元素。
        # 【位置固定堆叠在左下角，且由 annotate_image 保证最后绘制】此前画在
        # (bx,by)——坐标越界的框被钳到画面边缘后，标记正好卡在边上、绝大部分
        # 在画面外，肉眼完全看不见；用户会以为"报了 N 个违章图上只有 N-1 个框"。
        color = get_color(severity)
        _draw_invalid_marker(img, index, color, max(24, int(min(w, h) / 50)), h)
        return img

    color = get_color(severity)
    _draw_box_rect(img, bx, by, bw, bh, color)

    # 序号标签（默认位置：框上方，上方空间不够则放框内）
    label_text = _num_label(index) + " " + title
    font_size = max(14, int(min(w, h) / 50))
    label_y = by - font_size - 12
    if label_y < 5:
        label_y = by + 5
    approx_label_w = int(len(label_text) * font_size * 1.05) + 16
    label_x = bx
    if label_x + approx_label_w > w:
        label_x = max(0, w - approx_label_w)

    img = draw_label_with_bg(
        img, label_text, (label_x, label_y),
        font_size=font_size,
        text_color=(255, 255, 255),
        bg_color=color
    )
    return img


def annotate_image(img, analysis_result):
    """在图片上标注所有违章。

    绘制顺序（保证重叠框的每个编号都可见）：
    1. 先画所有有效框的矩形+角标（不含标签）
    2. 近乎重叠的框（IoU≥0.5，如同一人身上"未戴安全帽+未系挂安全带"）
       合并为一个联合标签，编号连写（"①③ 标题A / 标题B"）——每个编号都可见
    3. 标签统一碰撞避让（重叠时下移错位）
    4. 最后画无效框的"?"堆叠标记（左下角，永远在最上层）
    """
    if isinstance(img, str):
        img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None

    result_img = img.copy()
    violations = (analysis_result or {}).get("violations") or []

    h, w = result_img.shape[:2]
    font_size = max(14, int(min(w, h) / 50))

    items = [(i, v) for i, v in enumerate(violations) if isinstance(v, dict)]

    invalid = []   # (index, severity)
    valid = []     # (index, bx, by, bw, bh, color, title)
    for i, violation in items:
        bbox = violation.get("bbox") or {}
        severity = violation.get("severity", "default")
        title = _annotation_candidate_label(violation)
        if violation.get("text_overlay_check_status") == "suspected_text_overlap":
            invalid.append((i, severity))
            continue
        bx, by, bw, bh, ok = _resolve_bbox(bbox, w, h)
        if not ok:
            invalid.append((i, severity))
            continue
        valid.append((i, bx, by, bw, bh, get_color(severity), title))

    # 先画所有框矩形+角标
    for i, bx, by, bw, bh, color, title in valid:
        _draw_box_rect(result_img, bx, by, bw, bh, color)

    # 近乎重叠的框分组（贪心：依次并入已有组，否则新建组）
    groups = []
    for entry in valid:
        for g in groups:
            if any(_boxes_near_identical(e[1:5], entry[1:5]) for e in g):
                g.append(entry)
                break
        else:
            groups.append([entry])

    # 每组一个联合标签，统一避让放置
    placed = []
    for g in groups:
        anchor = max(g, key=lambda e: e[3] * e[4])   # 组内最大框为标签锚点
        text = _merged_label_text(g, w - 16, font_size)
        approx_label_w = int(len(text) * font_size * 1.05) + 16
        label_x = max(0, min(anchor[1], max(0, w - approx_label_w)))
        label_y = anchor[2] - font_size - 12
        if label_y < 5:
            label_y = anchor[2] + 5
        fx, fy = _place_label(label_x, label_y, approx_label_w,
                              font_size + 12, placed, w, h)
        result_img = draw_label_with_bg(
            result_img, text, (fx, fy),
            font_size=font_size,
            text_color=(255, 255, 255),
            bg_color=anchor[5]
        )

    # 无效框的"?"堆叠标记最后画：永远在最上层可见
    marker_size = max(24, int(min(w, h) / 50))
    for i, severity in invalid:
        _draw_invalid_marker(result_img, i, get_color(severity), marker_size, h)

    return result_img


# 说明卡只作为附加展示，不改变原帧、原报告或框坐标。
def detail_source_metadata(event):
    """只根据明示来源标志选择措辞；多种互斥来源同时出现时不猜来源。"""
    event = event if isinstance(event, dict) else {}
    is_rule = event.get("rule_based") is True or event.get("track_based") is True
    is_clip = event.get("clip_based") is True
    detail_source = event.get("detail_source")
    is_frame_model = isinstance(detail_source, str) and detail_source in (
        "representative_frame_model_output", "frame_model_output"
    )
    if event.get("detail_provenance_conflict") is True or sum((is_rule, is_clip, is_frame_model)) > 1:
        source_type, source_note, prefix = "conflict", "来源标识冲突，需人工核查", ""
    elif is_rule:
        source_type, source_note, prefix = "rule", "规则输出，需人工核查", "规则"
    elif is_clip:
        source_type, source_note, prefix = (
            "clip_model", "片段模型输出，可能受原视频字幕或标记影响，需人工核查", "片段模型"
        )
    elif is_frame_model:
        source_type, source_note, prefix = (
            "frame_model", "模型输出，可能受原视频字幕或标记影响，需人工核查", "模型"
        )
    else:
        source_type, source_note, prefix = "unknown", "来源未标注，需人工核查", ""
    return {
        "source_type": source_type, "source_note": source_note,
        "position_label": f"{prefix}位置（待核）",
        "description_label": f"{prefix}描述（待核）",
        "evidence_label": f"{prefix}观察（待核）",
    }


def detail_contextual_advice(event):
    """按候选标题生成条件式核查提示，绝不把标题或字幕当画面事实。

    返回稳定顺序的 (标题, 内容) 元组。场景只引用报告显式提供的名称；
    模型描述/证据仍由调用者按原来源单独显示，不从标题杜撰对象、动作或时间。
    """
    event = event if isinstance(event, dict) else {}
    title = event.get("title")
    title = title[:4000] if isinstance(title, str) else ""
    scene = next((value.strip() for value in (event.get("scene_name"), event.get("scene"))
                  if isinstance(value, str) and value.strip()), "")
    # 这些都是条件式通用风险和复核问题，并非在声称该画面确有对应状态。
    profiles = (
        (("安全帽", "头盔"),
         "若确有人员且头部防护不足，可能增加头部受伤风险。",
         "先核实目标是否为人员；再核查头部、帽壳和系带是否清晰可见，遮挡或看不清不能判为未佩戴。"),
        (("安全带",),
         "若确需高处防坠防护且穿戴或系挂不符合要求，可能存在坠落风险。",
         "对照原视频先核实人员及作业位置，再分别核查肩背腰部带体、连接绳、挂钩和可靠锚点；看不清不等于未使用。"),
        (("基坑", "临边", "洞口", "孔洞", "栏杆", "脚手架", "跳板", "高处", "安全带"),
         "若确有需要防护的高差、开口或通道且防护不足，可能存在坠落风险。",
         "核查开口或边界位置、实际高差、人员位置和防护措施；被遮挡或位于画外的部位不能推断缺失。"),
        (("支腿", "垫板", "路基板"),
         "若设备支撑或地基不满足作业要求，可能影响设备稳定性。",
         "核查设备类型、支腿落点和垫板形态，并结合地基及作业方案确认是否适用。"),
        (("动火", "火花", "焊接", "焊割"),
         "若确有动火且防火条件不足，可能存在火灾或灼伤风险。",
         "核查火花或热源及其发生时段，并查证隔离、监护和灭火措施；画外未见不等于不存在。"),
        (("机械作业", "吊装", "起重", "人机混行", "车辆", "交通"),
         "若人员进入运行设备或吊物危险范围且防护不足，可能发生碰撞、挤压或坠物伤害。",
         "先核实人物与设备身份、相对位置及设备状态，再结合连续画面和现场警戒确认。"),
        (("临时用电", "配电", "带电", "电缆", "线路"),
         "若电气设备或线路带电且防护不足，可能存在触电风险。",
         "核查线状物或设备的实际类型、敷设与隔离情况；单凭画面不能确认是否带电。"),
        (("积水", "排水"),
         "若作业区液体影响通行或电气设施，可能增加滑倒或触电风险。",
         "核实液体范围、深度、人员通行和电气设施的位置及现场排水情况。"),
        (("材料堆", "材料堆放", "堆码"),
         "若材料堆放影响通道或稳定性，可能造成绊倒或物件倾落。",
         "核实材料、通道和堆放稳定性；若画面实际是设备支承，还需按支承要求另行核查。"),
    )
    matched = [(risk, check) for words, risk, check in profiles if any(word in title for word in words)][:2]
    risk = " ".join(pair[0] for pair in matched) or "潜在风险需结合现场事实及作业条件核实，不能由标题直接认定。"
    advice = " ".join(pair[1] for pair in matched) or "核对原视频中目标、位置、状态及现场资料；无法看清时保留待复核。"
    moving_words = ("移动", "行走", "驶入", "闯入", "进入", "穿越", "跨越", "人机", "作业半径", "吊装", "抛物")
    moving = event.get("track_based") is True or event.get("clip_based") is True or any(w in title for w in moving_words)
    time_note = ("应结合连续原视频确认同一目标和实际动作；抽样帧、首末跨度或文本不能单独证明移动方向或动作持续。"
                 if moving else "首末采样时间之间未必连续，不能仅凭跨度认定事件持续。")
    return [
        ("场景归属（待核）", scene or "未提供，不能依据候选标题推断现场场景"),
        ("风险提示（条件成立时）", risk),
        ("现场复核重点", advice),
        ("时间与动作证据限制", time_note),
        ("画面与字幕提示", "应核对原画面；视频自带字幕或圈线仅是提示，不是独立视觉证据。"),
    ]


def detail_representative_reference(event):
    """仅读取报告代表帧及其时间；缺失时不借用其他帧或默认到 0 秒。"""
    event = event if isinstance(event, dict) else {}
    raw_frame = event.get("representative_frame")
    if (isinstance(raw_frame, (int, np.integer)) and not isinstance(raw_frame, (bool, np.bool_))
            and 0 <= raw_frame <= 2**63 - 1):
        frame = int(raw_frame)
    else:
        frame = None
    raw_time = event.get("representative_timestamp")
    timestamp = None if isinstance(raw_time, (bool, np.bool_)) else normalize_timestamp(raw_time)
    frame_text = str(frame) if frame is not None else "未知"
    if timestamp is None:
        time_text = "未知"
    elif timestamp == 0:
        time_text = "0 秒"
    else:
        if timestamp >= 1_000_000_000:
            # 非常大的有限秒数用科学计数法，避免固定小数格式生成数百位字符串。
            mantissa, exponent = f"{timestamp:.2e}".split("e")
            rendered = mantissa.rstrip("0").rstrip(".") + "e" + exponent
        else:
            rendered = f"{timestamp:.2f}".rstrip("0").rstrip(".")
        # 只改变显示值；元数据保留未经舍入的原始规范化时间。
        approximate = float(rendered) != timestamp
        time_text = ("约" if approximate else "") + rendered + " 秒"
    return {"frame_index": frame, "timestamp": timestamp,
            "text": f"报告代表帧：{frame_text}；报告代表时间：{time_text}。"}


def detail_review_disposition(event):
    """独立卡片只按已知处置标记展示；未知值不猜测、不改变事件状态。"""
    event = event if isinstance(event, dict) else {}
    value = event.get("review_disposition")
    if value is None or (isinstance(value, str) and value == ""):
        return {"disposition_type": "not_recorded", "disposition_note": "无处置记录，需人工核查"}
    notes = {
        "rejected": "已降级，未确认，不作为已确认违章",
        "box_relocalize": "框需重定位，未确认，不作为已确认违章",
        "dropped": "已移出主复核队列，未确认，不作为已确认违章",
        "low_priority_review": "待人工复核，未确认，不作为已确认违章",
        "low_priority_confirm": "低优先级待确认，未确认，不作为已确认违章",
    }
    if isinstance(value, str) and value in notes:
        if candidate_evidence_notes(event) and value in ("rejected", "box_relocalize", "dropped"):
            return {"disposition_type": value,
                    "disposition_note": "原记录：" + notes[value] + "；须核对当前复核队列"}
        return {"disposition_type": value, "disposition_note": notes[value]}
    return {"disposition_type": "unknown", "disposition_note": "处置状态未知，需人工核查，不作为已确认违章"}


def detail_box_relocalization(event):
    """显示框重定位线索；不覆盖当前复核处置或修改候选。"""
    event = event if isinstance(event, dict) else {}
    flag = event.get("needs_box_relocalize")
    blocked = event.get("analysis_incomplete_blocked_disposition")
    helmet_blocked = event.get("helmet_evidence_blocked_disposition")
    person_blocked = event.get("person_evidence_blocked_disposition")
    area_blocked = event.get("area_localization_blocked_disposition")
    person_evidence = event.get("person_evidence")
    person_conflict = (isinstance(person_evidence, dict)
                       and person_evidence.get("reference_conflict") is True)
    current = event.get("review_disposition")
    blocked_box = any(isinstance(value, str) and value == "box_relocalize"
                      for value in (blocked, helmet_blocked, person_blocked, area_blocked))
    current_box = isinstance(current, str) and current == "box_relocalize"
    if flag is True or blocked_box or current_box or person_conflict:
        return {"box_relocalization_status": "required",
                "box_relocalization_note": "框需重定位，当前复核处置不变，位置未确认"}
    invalid_flag = flag is not None and flag is not False
    invalid_blocked = any(value is not None and
                          (not isinstance(value, str) or value not in ("", "rejected", "dropped"))
                          for value in (blocked, helmet_blocked, person_blocked, area_blocked))
    if invalid_flag or invalid_blocked:
        return {"box_relocalization_status": "unknown",
                "box_relocalization_note": "框重定位标记未知，需人工核查"}
    return {"box_relocalization_status": "not_marked", "box_relocalization_note": ""}


_DETAIL_MAX_WIDTH = 8192
_DETAIL_MAX_HEIGHT = 8192
_DETAIL_MAX_PIXELS = 24_000_000
_DETAIL_MAX_FIELD_CHARS = 4000


def _detail_limit(value, maximum):
    """拒绝布尔、非数值、NaN 及负值，避免参数导致无限尺寸分配。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = int(value)
    except (ValueError, OverflowError):
        return None
    return result if 0 < result <= maximum else None


def _detail_cjk_font(font_size):
    """加载真实支持中文的字体，不能静默把中文画成缺字方框。"""
    paths = _CJK_FONT_PATHS + (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    )
    for font_path in paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            glyph = font.getmask("未")
            missing = font.getmask("\U0010ffff")
            if glyph.getbbox() is not None and (glyph.size, bytes(glyph)) != (missing.size, bytes(missing)):
                return font
        except (OSError, ValueError, UnicodeError):
            continue
    return None


def _detail_text(value, max_chars):
    """仅展示已有字符串；不将数字、对象或缺失字段伪装为事实。"""
    if not isinstance(value, str):
        return "未提供", False, False
    truncated = len(value) > max_chars
    text = value[:max_chars].encode("utf-8", "replace").decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 控制字符和双向覆盖符不能把界面提示伪装成别的顺序；保留换行。
    text = "".join(
        c if c == "\n" or (ord(c) >= 32 and not 0x7f <= ord(c) <= 0x9f
                              and not 0x202a <= ord(c) <= 0x202e
                              and not 0x2066 <= ord(c) <= 0x2069) else " "
        for c in text
    ).strip()
    return (text or "未提供"), truncated, bool(text)


def _helmet_box_detail(box, *, normalized=False):
    """显示已记录的原框，不对畸形坐标造零或猜测位置。"""
    numbers = None
    raw = None
    if isinstance(box, dict):
        raw = [box.get(key) for key in ("x", "y", "w", "h")]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw):
            try:
                numbers = [float(v) for v in raw]
            except (ValueError, OverflowError):
                pass
    valid = (numbers and all(math.isfinite(v) for v in numbers)
             and numbers[2] > 0 and numbers[3] > 0)
    if normalized and valid:
        x, y, w, h = numbers
        valid = x >= 0 and y >= 0 and x + w <= 1000.000001 and y + h <= 1000.000001
    if not valid:
        return "原框坐标无效，无法定位线索，需人工核查"
    coords = ", ".join(f"{key}={value!r}" for key, value in zip(("x", "y", "w", "h"), raw))
    return coords + "；线索来源帧及最终框位置仍需核查"


def harness_review_note(event):
    """待核安全带候选仅作展示提示，不改变原事件、模型判决或审核队列。"""
    if not isinstance(event, dict) or not isinstance(event.get("title"), str):
        return ""
    if "安全带" not in event["title"]:
        return ""
    pending = (event.get("needs_review") is True or event.get("analysis_incomplete") is True
               or event.get("review_disposition") in (
                   "low_priority_review", "low_priority_confirm", "box_relocalize"))
    if not pending:
        return ""
    return ("人员及作业位置、带体、连接绳、挂钩和锚点需分别对照原视频核实；"
            "看不清不等于缺失，模型描述不是独立确认")


def candidate_display_label(event):
    """安全帽、人员与区域定位证据依次提示；旧记录保持原标题。"""
    raw = helmet_display_label(event) or person_display_label(event) or area_display_label(event)
    if not raw and harness_review_note(event):
        raw = "待核：安全带穿戴及系挂状态"
    if not raw:
        from core.review_policy import source_video_review_guard
        if (source_video_review_guard(event) or isinstance(event, dict) and any(
                event.get(key) is True for key in ("frozen_assets_unverified_guard",
                    "height_gate_frame_unverified", "strict_gate_frame_unverified", "box_entity_frame_unverified",
                    "daylight_frame_unverified", "ai_review_frame_unverified")) or isinstance(event, dict) and (
                event.get("height_gate_review_status") in ("uncertain", "support_unverified")
                or event.get("strict_gate_status") == "missing_prerequisite")):
            raw = "待核：" + candidate_original_title(event)
    if not isinstance(raw, str) or not raw.strip():
        return None
    text, _, _ = _detail_text(raw, len(raw))
    return " ".join(text.splitlines()).replace("\ufffe", " ").replace("\uffff", " ")


def _annotation_candidate_label(event):
    """自动生成的图中标记均为模型候选，AI 复核不是人工确认。

    帧图在事件复核前绘制，不能依赖后续的 needs_review 状态；为避免某些
    失败或未复核候选被图中文字表现成事实，统一显示待核，不改事件标题。
    """
    specific = candidate_display_label(event)
    if specific:
        return specific
    original = candidate_original_title(event)
    if original == "未提供":
        original = "违章线索"
    return original if original.startswith("待核：") else "待核：" + original


def candidate_original_title(event):
    """显示用净化；完整原模型标题仍保留在事件及证据包元数据中。"""
    raw = event.get("title") if isinstance(event, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        return "未提供"
    text, _, present = _detail_text(raw, len(raw))
    text = " ".join(text.splitlines()).replace("\ufffe", " ").replace("\uffff", " ").strip()
    return text if present and text else "未提供"


def candidate_evidence_notes(event):
    """不同证据可共存；返回明确来源，不把一种证据覆盖另一种。"""
    notes = []
    for label, value in (("安全帽证据", helmet_evidence_note(event)),
                         ("人员身份证据", person_evidence_note(event)),
                         ("安全带复核提示", harness_review_note(event)),
                         ("区域定位证据", area_evidence_note(event))):
        if value:
            notes.append((label, value))
    from core.review_policy import source_video_review_guard
    if source_video_review_guard(event):
        notes.append(("来源级保守复核", "来源视频候选保守待核；不表示匹配某条冻结样本，也不代表违章成立"))
    if isinstance(event, dict) and event.get("frozen_assets_unverified_guard") is True:
        notes.append(("冻结资产核验", "冻结样本资产无法核验，事件仅按保守规则留待复核，不表示已匹配金标"))
    if isinstance(event, dict) and any(event.get(key) is True for key in (
            "height_gate_frame_unverified", "strict_gate_frame_unverified", "box_entity_frame_unverified",
            "daylight_frame_unverified", "ai_review_frame_unverified")):
        notes.append(("门禁证据核验", "对应原帧、框或门禁证据未能严格核验，相关反证未作为确定结论，需人工核查"))
    if isinstance(event, dict) and (event.get("height_gate_review_status") in ("uncertain", "support_unverified")
                                    or event.get("strict_gate_status") == "missing_prerequisite"):
        notes.append(("门禁结论", "高度或实体前提仍不确定，不能据此确认或排除违章，需人工核查"))
    if isinstance(event, dict) and event.get("height_gate_review_status") == "invalid_vote_threshold":
        notes.append(("高度复核参数", "高度复核票数参数无效，未执行复核，需人工确认"))
    return notes


def candidate_review_diagnostics(event):
    """展示已有线索，不将模型或规则意见升级为人工结论。原始值仍保留在事件中。"""
    if not isinstance(event, dict) or not candidate_evidence_notes(event):
        return []
    fields = []
    if event.get("suspected_false") is True:
        fields.append(("suspected_false", "误报线索", "存在疑似误报线索，仍需人工核查"))
    if event.get("entity_reference_conflict") is True:
        fields.append(("entity_reference_conflict", "框选线索冲突",
                       "模型原框与引用实体候选框不一致，需人工重定位，不能将引用候选当成已确认主体"))
    evidence = event.get("helmet_evidence")
    if isinstance(evidence, dict) and "reference_bbox" in evidence:
        fields.append(("reference_bbox", "线索原框坐标（像素，待核）",
                       _helmet_box_detail(evidence["reference_bbox"])))
    evidence = event.get("person_evidence")
    if isinstance(evidence, dict):
        if "reference_bbox" in evidence:
            fields.append(("person_reference_bbox", "人员线索原框坐标（像素，待核）",
                           _helmet_box_detail(evidence["reference_bbox"])))
        source_notes = {
            "model_reference_only": "未获得可用的本地匹配结果，尚无独立身份核验",
            "no_matching_person_in_local_detector": "未能形成有效的本地匹配结果；不能据此确认对象不是人",
            "same_frame_local_detector_candidate": "同帧本地检测候选不是独立身份确认",
        }
        source = evidence.get("source")
        if isinstance(source, str) and source in source_notes:
            fields.append(("person_source", "人员线索来源（待核）", source_notes[source]))
        elif "source" in evidence:
            fields.append(("person_source", "人员线索来源（待核）", "来源未知，需人工核查"))
        if "source_entity_id" in evidence:
            ref = evidence.get("source_entity_id")
            if isinstance(ref, str) and re.fullmatch(r"P[0-9]+", ref.strip(), re.I):
                ref = ref.strip()
                label = ref if len(ref) <= 40 else ref[:40] + "…（完整编号见元数据）"
                fields.append(("person_source_entity_id", "人员引用编号（待核）",
                               label + "；编号不是独立身份确认"))
            else:
                fields.append(("person_source_entity_id", "人员引用编号（待核）", "编号无效，需人工核查"))
        conflict = evidence.get("reference_conflict")
        if conflict is True:
            fields.append(("person_reference_conflict", "人员引用与框的冲突",
                           "当前帧引用编号与局部框未可靠对应，需人工重定位；编号不是身份确认"))
        elif conflict is not None and conflict is not False:
            fields.append(("person_reference_conflict", "人员引用与框的冲突", "冲突标记未知，需人工核查"))
    area = event.get("area_localization")
    if isinstance(area, dict):
        if "reference_bbox" in area:
            fields.append(("area_reference_bbox", "区域线索原框（像素，待核）",
                           _helmet_box_detail(area["reference_bbox"])))
        if "search_window" in area:
            fields.append(("area_search_window", "人工查找范围（不是违章框）",
                           _helmet_box_detail(area["search_window"])))
        context = area.get("visual_context_candidate")
        if isinstance(context, dict) and context.get("not_violation_box") is True:
            fields.append(("area_visual_context", "未验证背景候选（不是违章框）",
                           _helmet_box_detail(context.get("bbox"))))
    for key, label in (("bbox_raw_1000", "模型原框（归一化0–1000，待核）"),
                       ("entity_reference_candidate_bbox_1000", "引用候选框（归一化0–1000，待核）")):
        if key in event:
            fields.append((key, label, _helmet_box_detail(event[key], normalized=True)))
    blocked_notes = {"rejected": "原记录拟降级",
                     "box_relocalize": "原记录提示框需重定位",
                     "dropped": "原记录拟移出主复核队列"}
    for key, label in (("helmet_evidence_blocked_disposition", "安全帽护栏拦截的原处置"),
                       ("person_evidence_blocked_disposition", "人员护栏拦截的原处置"),
                       ("area_localization_blocked_disposition", "区域定位护栏拦截的原处置")):
        blocked = event.get(key)
        if isinstance(blocked, str) and blocked in blocked_notes:
            fields.append((key, label, blocked_notes[blocked] + "；当前状态仍需人工核查"))
    for key, label in (("review_reason", "既有复核线索（待核）"),
                       ("ai_review_reason", "AI复核意见（待核）"),
                       ("ai_review_note", "复核状态记录（待核）")):
        raw = event.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        value, _, present = _detail_text(raw, len(raw))
        # 不可信文本不可伪造新字段，也不可把非法 XML 字符送入 Word。
        value = " ".join(value.splitlines())
        value = "".join(c for c in value if ord(c) not in (0xfffe, 0xffff))
        if present and value.strip():
            fields.append((key, label, value))
    return fields


def helmet_review_diagnostics(event):
    """兼容既有调用者；无安全帽证据时不添加安全帽相关状态。"""
    return candidate_review_diagnostics(event) if helmet_evidence_note(event) else []


def _detail_wrap(draw, text, font, width):
    """按真实字形宽度逐字换行；输入已受最大字符数约束。"""
    lines, line = [], ""
    for char in text:
        if char == "\n":
            lines.append(line or " ")
            line = ""
            continue
        candidate = line + char
        if line and draw.textlength(candidate, font=font) > width:
            lines.append(line)
            line = char
        else:
            line = candidate
    if line or not lines:
        lines.append(line or " ")
    return lines


def _detail_ellipsis(draw, line, font, width):
    marker = "…"
    while line and draw.textlength(line + marker, font=font) > width:
        line = line[:-1]
    return line + marker


def render_detailed_event_card(annotated_image, event, *, max_width=4096,
                               max_height=8192, max_panel_height=1400,
                               max_field_chars=1500):
    """在已标注图下方附加模型说明卡，返回 ``(图片或 None, 状态元数据)``。

    原图所有像素保持原坐标和原值，不修改入参；卡片来源只按明示标志展示，
    均需人工核查，也不识别/改写原视频字幕。失败时不生成假图。输出像素总量、
    画布尺寸和每个字段字符数均有上限；被截断时明确提示查看完整报告。
    """
    width_limit = _detail_limit(max_width, _DETAIL_MAX_WIDTH)
    height_limit = _detail_limit(max_height, _DETAIL_MAX_HEIGHT)
    panel_limit = _detail_limit(max_panel_height, _DETAIL_MAX_HEIGHT)
    chars_limit = _detail_limit(max_field_chars, _DETAIL_MAX_FIELD_CHARS)
    if None in (width_limit, height_limit, panel_limit, chars_limit):
        return None, {"status": "invalid_limits"}
    if (not isinstance(annotated_image, np.ndarray) or annotated_image.ndim != 3
            or annotated_image.shape[2] != 3 or annotated_image.dtype != np.uint8):
        return None, {"status": "invalid_image"}
    image_h, image_w = annotated_image.shape[:2]
    if image_h <= 0 or image_w <= 0:
        return None, {"status": "invalid_image"}
    source_size = {"width": image_w, "height": image_h}
    if image_w > width_limit or image_h >= height_limit:
        return None, {"status": "canvas_limit_exceeded", "source_image_size": source_size}
    # 避免极窄画布把字段名或“未提供”拆成不可理解的半行；不能扩到最小宽度时不造卡片。
    if width_limit < 320:
        return None, {"status": "canvas_limit_exceeded", "source_image_size": source_size}
    canvas_w = max(image_w, 320)
    available_h = min(panel_limit, height_limit - image_h,
                      _DETAIL_MAX_PIXELS // canvas_w - image_h)
    font_size = min(22, max(12, canvas_w // 42))
    font = _detail_cjk_font(font_size)
    if font is None:
        return None, {"status": "font_unavailable", "source_image_size": source_size}

    record = event if isinstance(event, dict) else {}
    analysis_incomplete = record.get("analysis_incomplete") is True
    source = detail_source_metadata(record)
    reference = detail_representative_reference(record)
    disposition = detail_review_disposition(record)
    box_relocalization = detail_box_relocalization(record)
    evidence_notes = candidate_evidence_notes(record)
    diagnostics = candidate_review_diagnostics(record)
    diagnostic_values = {key: value for key, _, value in diagnostics}
    if evidence_notes:
        diagnostic_values["title"] = candidate_original_title(record)
    detail_fields = (
        ("title", "原模型标题（待核）" if evidence_notes else "项目"),
        ("target_location", source["position_label"]),
        ("description", source["description_label"]),
        ("evidence", source["evidence_label"]),
    ) + tuple((key, label) for key, label, _ in diagnostics)
    try:
        position_valid = _resolve_bbox(record.get("bbox"), image_w, image_h)[-1]
    except (TypeError, ValueError, OverflowError):
        # 非法或超大外部坐标不可让卡片崩溃，更不能被当成已定位。
        position_valid = False
    position_state = "coordinates_unverified" if position_valid else "unavailable"
    margin = max(8, min(20, canvas_w // 20))
    text_width = canvas_w - 2 * margin
    line_h = font_size + 8
    max_lines = (available_h - 2 * margin) // line_h
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    heading = [
        "【分析不完整 · 候选仅供人工核查】" if analysis_incomplete else "【事件详情 · 待人工核查】",
        "复核处置：" + disposition["disposition_note"] + "。",
    ]
    for label, note in evidence_notes:
        heading.append(label + "：" + note + "。")
    if box_relocalization["box_relocalization_note"]:
        heading.append("框状态：" + box_relocalization["box_relocalization_note"] + "。")
    heading.extend([
        "来源：" + source["source_note"] + "。",
        reference["text"],
        "框选：已有坐标，位置仍需核查。" if position_valid else "框选：无有效坐标，不能定位。",
    ])
    heading_lines = [line for item in heading for line in _detail_wrap(probe, item, font, text_width)]
    notices = {
        False: _detail_wrap(probe, "提示：显示内容以报告为准。", font, text_width),
        True: _detail_wrap(probe, "提示：内容已截断，请查看完整报告。", font, text_width),
    }
    reserved_notice = max(map(len, notices.values()))
    budget = max_lines - len(heading_lines) - reserved_notice
    if budget < len(detail_fields):
        return None, {"status": "canvas_limit_exceeded", "source_image_size": source_size}

    wrapped, pre_truncated, missing = {}, set(), []
    for key, label in detail_fields:
        value, truncated, present = _detail_text(
            diagnostic_values[key] if key in diagnostic_values else record.get(key), chars_limit)
        if truncated:
            pre_truncated.add(key)
        if not present:
            missing.append(key)
        wrapped[key] = _detail_wrap(probe, f"{label}：{value}", font, text_width)
    # 每个字段先保留一行，再轮询分配剩余空间，防止长标题/描述挤掉后续字段。
    count = {key: 1 for key, _ in detail_fields}
    remaining = budget - len(count)
    while remaining > 0:
        changed = False
        for key, _ in detail_fields:
            if remaining and count[key] < len(wrapped[key]):
                count[key] += 1
                remaining -= 1
                changed = True
        if not changed:
            break

    # 原有标题、位置、描述及证据优先；只有富余行数足够时才附加场景与核查问题，
    # 绝不因新增文字挤掉原证据或使本可输出的图片变得不可用。
    context = dict(detail_contextual_advice(record))
    optional = []
    if context.get("场景归属（待核）") != "未提供，不能依据候选标题推断现场场景":
        optional.append(("context_scene", "场景归属（待核）", context["场景归属（待核）"]))
    optional.append(("context_review", "现场复核重点", context["现场复核重点"]))
    extra = []
    for key, label, value in optional:
        rendered, was_truncated, _ = _detail_text(value, chars_limit)
        candidate_lines = _detail_wrap(probe, f"{label}：{rendered}", font, text_width)
        if len(candidate_lines) <= remaining:
            wrapped[key] = candidate_lines
            count[key] = len(candidate_lines)
            remaining -= len(candidate_lines)
            if was_truncated:
                pre_truncated.add(key)
            extra.append((key, label))
    detail_fields = detail_fields + tuple(extra)

    body_lines, truncated_fields = [], []
    for key, _ in detail_fields:
        selected = list(wrapped[key][:count[key]])
        if count[key] < len(wrapped[key]) or key in pre_truncated:
            truncated_fields.append(key)
            selected[-1] = _detail_ellipsis(probe, selected[-1], font, text_width)
        body_lines.extend(selected)
    lines = heading_lines + body_lines + notices[bool(truncated_fields)]
    panel_h = len(lines) * line_h + 2 * margin

    try:
        panel = Image.new("RGB", (canvas_w, panel_h), (245, 247, 250))
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, canvas_w - 1, 2), fill=(83, 112, 145))
        for number, line in enumerate(lines):
            color = (42, 65, 94) if number < len(heading_lines) else (35, 40, 48)
            draw.text((margin, margin + number * line_h), line, font=font, fill=color)
        panel_bgr = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
        canvas = np.full((image_h + panel_h, canvas_w, 3), 245, dtype=np.uint8)
        canvas[:image_h, :image_w] = annotated_image
        canvas[image_h:, :] = panel_bgr
    except (OSError, ValueError, UnicodeError, cv2.error, MemoryError):
        return None, {"status": "render_failed", "source_image_size": source_size}
    return canvas, {
        "status": "ok", "source_image_size": source_size,
        "canvas_size": {"width": canvas_w, "height": image_h + panel_h},
        "detail_panel": {"x": 0, "y": image_h, "width": canvas_w, "height": panel_h},
        "bbox_status": position_state, "missing_fields": missing,
        "truncated_fields": truncated_fields,
        "source_type": source["source_type"], "source_note": source["source_note"],
        "representative_reference": reference,
        "analysis_incomplete": analysis_incomplete,
        "disposition_type": disposition["disposition_type"],
        "disposition_note": disposition["disposition_note"],
        "box_relocalization_status": box_relocalization["box_relocalization_status"],
        "box_relocalization_note": box_relocalization["box_relocalization_note"],
    }

def generate_report_text(analysis_result, timestamp=None):
    """
    生成文字报告
    
    Args:
        analysis_result: 分析结果
        timestamp: 时间戳（秒）
        
    Returns:
        str: 报告文本
    """
    lines = []
    analysis_result = analysis_result if isinstance(analysis_result, dict) else {}
    
    if timestamp is not None:
        _ts = None if isinstance(timestamp, (bool, np.bool_)) else normalize_timestamp(timestamp)
        if _ts is None:
            lines.append("【时间点】未知")
        else:
            minutes = int(_ts // 60)
            seconds = int(_ts % 60)
            lines.append(f"【时间点】{minutes:02d}:{seconds:02d}")

    raw_violations = analysis_result.get("violations")
    violations = ([v for v in raw_violations if isinstance(v, dict)]
                  if isinstance(raw_violations, list) else [])
    try:
        has_error = frame_analysis_incomplete(analysis_result)
    except Exception:
        # 畸形错误标志无法判定时保守视为分析不完整，不输出异常正文。
        has_error = True
    if has_error and not violations:
        lines.append("【本帧分析失败，无法据此判断】")
        return "\n".join(lines)
    if has_error:
        # 错误可能含密钥；不显示摘要和异常原文，但保留已有候选供核查。
        lines.append("【本帧分析不完整，以下候选仅供人工核查】")

    summary = analysis_result.get("summary", "")
    has_summary = isinstance(summary, str) and bool(summary.strip())
    if has_summary and not has_error:
        source = detail_source_metadata(analysis_result)
        is_frame_model = (source["source_type"] == "frame_model"
                          and analysis_result.get("detail_source") == "frame_model_output")
        label = "模型概述（待核查）" if is_frame_model else "分析概述（待核查）"
        lines.append(f"【{label}】{summary}")
    
    if violations:
        lines.append(f"【保留 {len(violations)} 项候选（待核查）】" if has_error
                     else f"【输出 {len(violations)} 项结构化违章候选】")

        for i, v in enumerate(violations):
            if not isinstance(v, dict):
                continue
            num = _num_label(i)
            title = v.get("title", "未知违章")
            severity = v.get("severity", "未知")
            regulation = v.get("regulation", "")
            regulation_text = v.get("regulation_text", "")
            
            shown_title = candidate_display_label(v) or title
            lines.append(f"{num} {shown_title} [{severity}]")
            evidence_notes = candidate_evidence_notes(v)
            if evidence_notes:
                raw_title = candidate_original_title(v)
                lines.append(f"   原模型标题（待核）：{raw_title}")
                for label, note in evidence_notes:
                    lines.append(f"   {label}：{note}")
                for _, label, value in candidate_review_diagnostics(v):
                    lines.append(f"   {label}：{value}")
                box_note = detail_box_relocalization(v)["box_relocalization_note"]
                if box_note:
                    lines.append(f"   框状态：{box_note}")
            source = detail_source_metadata(v)
            lines.append(f"   来源：{source['source_note']}。")
            # 文本报告保留完整字符串，不复用说明卡的显示长度上限；不猜缺失事实。
            for key, label in (("target_location", source["position_label"]),
                               ("description", source["description_label"]),
                               ("evidence", source["evidence_label"])):
                raw = v.get(key)
                value = raw if isinstance(raw, str) and raw.strip() else "未提供"
                lines.append(f"   {label}：{value}")
            for label, value in detail_contextual_advice(v):
                lines.append(f"   {label}：{value}")
            if regulation:
                lines.append(f"   法规依据（待核）：{regulation}")
            if regulation_text:
                lines.append(f"   法规条款（待核）：{regulation_text}")
    else:
        if has_summary:
            lines.append("【本帧未输出结构化违章候选；概述中的线索仍需人工核查】")
        else:
            lines.append("【本帧未输出结构化违章候选，且无可用概述；无法据此断言画面不存在违章】")
    
    return "\n".join(lines)

def save_annotated_image(img, output_path):
    """保存标注后的图片（支持中文路径）"""
    # cv2.imwrite 不支持中文路径，用 imencode + tofile 替代
    ext = os.path.splitext(output_path)[1] or ".jpg"
    success, buf = cv2.imencode(ext, img)
    if success:
        buf.tofile(output_path)
    return success
