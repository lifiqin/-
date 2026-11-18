# -*- coding: utf-8 -*-
"""
违章画面显示组件：按分析结果把违章框画在画面上，支持滚轮缩放与悬停高亮
基于 QGraphicsView + QGraphicsScene
"""
import cv2
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsSimpleTextItem, QLabel,
    QGraphicsItem,
)
from PySide6.QtGui import QPixmap, QImage, QColor, QPen, QBrush, QFont, QPainter
from PySide6.QtCore import Qt, QRectF

from ui.theme import DANGER, WARNING, SUCCESS
from core.bbox_utils import bbox_xywh as _bbox_xywh, bbox_corners as _bbox_corners

# 与 ui/theme.py 设计令牌保持一致
SEVERITY_COLORS = {
    "严重违章": DANGER,
    "重大隐患": DANGER,
    "一般违章": WARNING,
    "文明施工": "#0284c7",
    "严重": DANGER,
    "较大": WARNING,
    "一般隐患": WARNING,
    "一般": "#0284c7",
    "轻微违章": SUCCESS,
    "正常": SUCCESS,
}
DEFAULT_BOX_COLOR = WARNING


class ViolationBoxItem(QGraphicsRectItem):
    """可交互的违章框"""

    def __init__(self, violation_id, rect, color, label="", parent=None):
        super().__init__(rect, parent)
        self.violation_id = violation_id
        self._color = QColor(color)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self._update_style()

        # 编号标签：先铺一层同色底片，避免白字压在浅色画面上看不清
        self._chip = QGraphicsRectItem(self)
        self._chip.setBrush(QBrush(self._color))
        self._chip.setPen(QPen(Qt.NoPen))

        self._text_item = QGraphicsSimpleTextItem(str(label), self)
        self._text_item.setBrush(QBrush(QColor("white")))
        font = QFont("Microsoft YaHei", 10, QFont.Bold)
        self._text_item.setFont(font)
        self._position_label()

    def _update_style(self):
        pen = QPen(self._color)
        pen.setWidth(3)
        self.setPen(pen)
        brush = QBrush(QColor(self._color.red(), self._color.green(), self._color.blue(), 40))
        self.setBrush(brush)

    def _position_label(self):
        """把编号标签放在框的左上角（带底片）"""
        r = self.rect()
        tb = self._text_item.boundingRect()
        pad_x, pad_y = 6, 3
        self._chip.setRect(r.x(), r.y(), tb.width() + pad_x * 2, tb.height() + pad_y * 2)
        self._text_item.setPos(r.x() + pad_x, r.y() + pad_y)

    def hoverEnterEvent(self, event):
        pen = self.pen()
        pen.setWidth(5)
        self.setPen(pen)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._update_style()
        super().hoverLeaveEvent(event)


class InteractiveImageView(QGraphicsView):
    """违章画面查看器：载入画面 + 按分析结果画违章框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("imageStage")
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        # 关键：开启平滑缩放，避免预览图因 fitInView 缩放而发虚/锯齿
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        from ui.theme import STAGE
        self.setBackgroundBrush(QBrush(QColor(STAGE)))

        # 空状态占位：未载入画面时给出明确引导，而不是一块纯色画布
        self._placeholder = QLabel(self)
        self._placeholder.setObjectName("emptyState")
        self._placeholder.setWordWrap(True)
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.show_placeholder("尚未载入画面\n\n选择视频并点击「开始分析」\n检测结果与违章框会显示在这里")

        self._pixmap_item = None
        self._box_items = {}  # violation_id -> ViolationBoxItem

    def show_placeholder(self, text=None):
        """显示空状态占位文案"""
        if text:
            self._placeholder.setText(text)
        self._placeholder.show()
        self._placeholder.setGeometry(self.viewport().rect())
        self._placeholder.raise_()

    def hide_placeholder(self):
        self._placeholder.hide()

    def set_image(self, img_bgr):
        """设置图片（BGR numpy数组）"""
        self.scene.clear()
        self._box_items.clear()
        self.hide_placeholder()

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)

        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        # 关键：让 QGraphicsPixmapItem 在视图缩放时使用双线性插值（默认是最近邻，会发虚）
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self.scene.addItem(self._pixmap_item)
        self.scene.setSceneRect(0, 0, w, h)
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def set_violations(self, violations):
        """设置违章列表并画框
        violations: [{id, title, severity, bbox:{x1,y1,x2,y2} 或 {x,y,w,h}}, ...]
        """
        # 清除旧框
        for item in self._box_items.values():
            self.scene.removeItem(item)
        self._box_items.clear()

        for idx, v in enumerate(violations, 1):
            vid = v.get("id", idx)
            bbox = v.get("bbox") or {}
            # 兼容两种格式：{x1,y1,x2,y2} 和 {x,y,w,h}
            # 数值统一走安全转换：报告里的坐标可能是 null（JSON 写盘净化了
            # NaN），直接 float(None) 会抛异常把 UI 槽函数打断。
            # 角点分支必须用 bbox_corners（本分支的语义就是"角点优先"），
            # 用 bbox_xywh 会在混有 xywh 键时选错分支。
            if "x1" in bbox and "y1" in bbox and "x2" in bbox and "y2" in bbox:
                x1, y1, x2, y2 = _bbox_corners(bbox)
            else:
                x1, y1, w, h = _bbox_xywh({"x": bbox.get("x"), "y": bbox.get("y"),
                                           "w": bbox.get("w"), "h": bbox.get("h")})
                x2, y2 = x1 + w, y1 + h
            severity = v.get("severity", "一般隐患")
            color = SEVERITY_COLORS.get(severity, DEFAULT_BOX_COLOR)
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            item = ViolationBoxItem(vid, rect, color, label=idx)
            self.scene.addItem(item)
            self._box_items[vid] = item

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._placeholder.isVisible():
            self._placeholder.setGeometry(self.viewport().rect())
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        # 滚轮缩放
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)
