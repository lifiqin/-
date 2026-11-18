# -*- coding: utf-8 -*-
"""区域事件原帧人工画框；结果只写旁路修订记录，需人工另行确认。"""
import math

import cv2
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QLabel, QLineEdit, QMessageBox,
    QVBoxLayout, QWidget,
)

from core.area_manual_revision import (
    AreaRevisionError, box_from_image_points, list_manual_area_revisions,
    prepare_manual_area_revision, save_manual_area_revision, viewport_to_image,
)


class AreaFrameCanvas(QWidget):
    """等比显示原图，拖拽框以原图像素坐标返回。"""
    box_changed = Signal(object)

    def __init__(self, frame_bgr, original_bbox=None, previous_bbox=None, parent=None,
                 context_search_window=None, context_visual_bbox=None):
        super().__init__(parent)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._height, self._width = rgb.shape[:2]
        self._image = QImage(rgb.data, self._width, self._height, rgb.strides[0],
                             QImage.Format.Format_RGB888).copy()
        self._original = original_bbox
        self._previous = previous_bbox
        self._show_original = True
        self._context_search_window = context_search_window
        self._context_visual_bbox = context_visual_bbox
        self._show_context = False
        self._start = None
        self._end = None
        self._dragging = False
        self.selected_box = None
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)

    def set_original_visible(self, visible):
        self._show_original = bool(visible)
        self.update()

    def set_context_visible(self, visible):
        self._show_context = bool(visible)
        self.update()

    def _geometry(self):
        scale = min(self.width() / self._width, self.height() / self._height)
        left = (self.width() - self._width * scale) / 2.0
        top = (self.height() - self._height * scale) / 2.0
        return left, top, scale

    def _widget_rect(self, box):
        if not isinstance(box, dict):
            return None
        try:
            x, y, w, h = (float(box[k]) for k in ('x', 'y', 'w', 'h'))
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            return None
        left, top, scale = self._geometry()
        return QRectF(left + x * scale, top + y * scale, w * scale, h * scale)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        left, top, scale = self._geometry()
        painter.drawImage(QRectF(left, top, self._width * scale, self._height * scale), self._image)
        if self._show_context:
            for box, colour, style in ((self._context_search_window, QColor(205, 205, 205), Qt.PenStyle.DotLine),
                                       (self._context_visual_bbox, QColor(0, 220, 240), Qt.PenStyle.DashLine)):
                rect = self._widget_rect(box)
                if rect is not None:
                    painter.setPen(QPen(colour, 2, style))
                    painter.drawRect(rect)
        if self._show_original:
            rect = self._widget_rect(self._original)
            if rect is not None:
                painter.setPen(QPen(QColor(255, 180, 0), 2, Qt.PenStyle.DashLine))
                painter.drawRect(rect)
        old = self._widget_rect(self._previous)
        if old is not None:
            painter.setPen(QPen(QColor(225, 80, 225), 2, Qt.PenStyle.DashLine))
            painter.drawRect(old)
        current = self.selected_box
        if self._dragging and self._start is not None and self._end is not None:
            try:
                current = box_from_image_points(self._start, self._end, self._width, self._height)
            except AreaRevisionError:
                current = None
        rect = self._widget_rect(current)
        if rect is not None:
            painter.setPen(QPen(QColor(25, 230, 100), 3))
            painter.drawRect(rect)
        painter.end()

    def _to_image(self, position, clamp=False):
        return viewport_to_image(position.x(), position.y(), self.width(), self.height(),
                                 self._width, self._height, clamp=clamp)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self._to_image(event.position())
        if point is None:
            return
        self._start = point
        self._end = point
        self._dragging = True
        self.update()

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._end = self._to_image(event.position(), clamp=True)
            self.update()

    def mouseReleaseEvent(self, event):
        if not self._dragging or event.button() != Qt.MouseButton.LeftButton:
            return
        self._end = self._to_image(event.position(), clamp=True)
        self._dragging = False
        try:
            self.selected_box = box_from_image_points(self._start, self._end, self._width, self._height)
        except AreaRevisionError:
            self.selected_box = None
        self.box_changed.emit(self.selected_box)
        self.update()


class ManualAreaEditor(QDialog):
    """主窗口可直接创建；调用者只需提供报告目录和 event_id。"""

    def __init__(self, output_dir, event_id, parent=None):
        super().__init__(parent)
        self.context, frame = prepare_manual_area_revision(output_dir, event_id)
        previous = list_manual_area_revisions(self.context)
        rows = previous['records']
        previous_bbox = rows[-1]['record']['manual_bbox'] if rows else None
        self.saved_path = None
        self._has_previous = previous_bbox is not None
        self.setWindowTitle('区域事件人工复核画框')
        self.resize(1024, 800)
        layout = QVBoxLayout(self)
        intro = QLabel(f'事件 {self.context.event_id}：{self.context.title} '
                       f'（原始帧 {self.context.frame_index}，{self.context.timestamp:g} 秒）')
        intro.setTextFormat(Qt.TextFormat.PlainText)
        intro.setWordWrap(True)
        layout.addWidget(intro)
        notice = QLabel('请在原图上拖出需要核查的缺口/区域。橙色虚线为原报告框（可能已自动校准），'
                        '紫色为上次人工候选，绿色为本次新框；'
                        '可选的灰/青色为未经验证的搜索范围/背景暗区，并非缺口框。'
                        '保存只增加旁路修订，不改原报告、原图、证据或金标；人工框不是自动模型结果。')
        notice.setTextFormat(Qt.TextFormat.PlainText)
        notice.setWordWrap(True)
        layout.addWidget(notice)
        self.pixel_source_notice = QLabel('帧来源：' + self.context.frame_pixel_source_note)
        self.pixel_source_notice.setTextFormat(Qt.TextFormat.PlainText)
        self.pixel_source_notice.setWordWrap(True)
        layout.addWidget(self.pixel_source_notice)
        if previous['invalid_count']:
            warning = QLabel(f'有 {previous["invalid_count"]} 条旧修订不匹配或不可读，未显示。')
            warning.setTextFormat(Qt.TextFormat.PlainText)
            layout.addWidget(warning)
        self.show_old = QCheckBox('显示原报告框，可能已自动校准（橙色虚线）')
        self.show_old.setChecked(True)
        layout.addWidget(self.show_old)
        self.show_context = QCheckBox('显示同帧搜索范围/背景暗区（未验证、非缺口框）')
        self.show_context.setChecked(False)
        self.show_context.setEnabled(bool(self.context.context_search_window or self.context.context_visual_bbox))
        self.show_context.setToolTip('仅作查找上下文，可能包含阴影、人员和设备，不能视为缺口或已确认违章。')
        layout.addWidget(self.show_context)
        self.canvas = AreaFrameCanvas(frame, self.context.original_bbox, previous_bbox, self,
                                      context_search_window=self.context.context_search_window,
                                      context_visual_bbox=self.context.context_visual_bbox)
        self.show_old.toggled.connect(self.canvas.set_original_visible)
        self.show_context.toggled.connect(self.canvas.set_context_visible)
        layout.addWidget(self.canvas, 1)
        self.note = QLineEdit(self)
        self.note.setMaxLength(1000)
        self.note.setPlaceholderText('备注（可选，例如可见范围和被遮挡部分）')
        layout.addWidget(self.note)
        self.boxes = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.save_button = self.boxes.button(QDialogButtonBox.StandardButton.Save)
        self.save_button.setText('保存人工候选框')
        self.save_button.setEnabled(False)
        self.boxes.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        self.canvas.box_changed.connect(lambda box: self.save_button.setEnabled(box is not None))
        self.boxes.accepted.connect(self._save)
        self.boxes.rejected.connect(self.reject)
        layout.addWidget(self.boxes)

    def _save(self):
        try:
            shown = {'original_model_box_shown': self.show_old.isChecked(),
                     'visual_context_shown': self.show_context.isChecked(),
                     'previous_manual_candidate_shown': self._has_previous}
            self.saved_path = save_manual_area_revision(self.context, self.canvas.selected_box, self.note.text(), shown)
        except AreaRevisionError as exc:
            QMessageBox.warning(self, '无法保存', str(exc))
            return
        self.accept()


def open_manual_area_editor(output_dir, event_id, parent=None):
    """在已有 QApplication 主线程打开对话框，返回旁路记录路径或 None。"""
    dialog = ManualAreaEditor(output_dir, event_id, parent)
    return dialog.saved_path if dialog.exec() == QDialog.DialogCode.Accepted else None
