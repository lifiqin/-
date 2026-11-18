"""Narrow Qt WebChannel adapter; no arbitrary file, config or network access from JS."""
import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFileDialog, QMessageBox

from application.analysis_session import ACTIVE, AnalysisSession
from application.run_publication import explain_issues, inspect_run, read_run_report
from application.run_publication import within_directory as _inside
from core.config import get_app_dir, get_config
from core.video_processor import VideoProcessor
from ui.workers import AnalysisThread, ConnectionCheckThread, ExportWorkerThread

SCENES = frozenset(('电力建设现场', '电力检修现场', '变电站作业', '输电线路作业',
                    '配电作业', '电缆井作业', '通用安全检查'))


def _complete_report(output, root, rows):
    """Compatibility helper; the controller also checks the selected source."""
    return bool(output and not inspect_run(read_run_report(output, root=root), rows))


def _preview_data(frame, root):
    if not isinstance(frame, dict):
        return ''
    for path in (frame.get('_annotated'), frame.get('_original')):
        try:
            if not path or not _inside(path, root) or Path(path).stat().st_size > 30_000_000:
                continue
            raw = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if image is None or image.shape[0] * image.shape[1] > 32_000_000:
                continue
            h, w = image.shape[:2]
            scale = min(1.0, 1280 / max(w, h))
            if scale < 1:
                image = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
            ok, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok and encoded.size <= 3_000_000:
                return 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode('ascii')
        except (OSError, ValueError, cv2.error, MemoryError):
            continue
    return ''


class _VideoInfoThread(QThread):
    checked = Signal(str, object)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            self.checked.emit(self.path, VideoProcessor().get_video_info(self.path))
        except Exception:
            self.checked.emit(self.path, None)


class WebBridge(QObject):
    stateChanged = Signal(str)
    previewChanged = Signal(str)
    legacyRequested = Signal()
    shutdownReady = Signal()

    def __init__(self, window, *, output_root=None, info_factory=_VideoInfoThread,
                 connection_factory=ConnectionCheckThread, analysis_factory=AnalysisThread,
                 export_factory=ExportWorkerThread):
        super().__init__(window)
        self.window = window
        self.session = AnalysisSession()
        configured = get_config().get('output_dir', 'output') if output_root is None else output_root
        self.output_root = os.path.abspath(os.path.join(get_app_dir(), configured or 'output'))
        self._info_factory = info_factory
        self._connection_factory = connection_factory
        self._analysis_factory = analysis_factory
        self._export_factory = export_factory
        self._info = self._connection = self._analysis = self._export = None
        self._closing = False
        self._scene_hint = '电力建设现场'
        self._export_result_seen = False
        self._pending_preview = False
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.timeout.connect(self._flush_updates)

    def _emit(self):
        snapshot = self.session.snapshot()
        snapshot['export_busy'] = bool(self._export is not None and self._export.isRunning())
        self.stateChanged.emit(json.dumps(snapshot, ensure_ascii=False, allow_nan=False))

    def _queue_update(self, preview=False):
        self._pending_preview = self._pending_preview or preview
        if not self._flush_timer.isActive():
            self._flush_timer.start(200)

    def _flush_updates(self):
        self._emit()
        if self._pending_preview:
            self._pending_preview = False
            self.previewChanged.emit(_preview_data(self.session.frames.get(self.session.selected), self.output_root))

    def _busy(self):
        return any(t is not None and t.isRunning() for t in (self._info, self._connection, self._analysis, self._export))

    def _thread_finished(self):
        if self._closing and not self._busy():
            self.shutdownReady.emit()

    @Slot()
    def ready(self):
        self._emit()

    @Slot()
    def chooseVideo(self):
        if self._closing or self._busy() or self.session.phase in ACTIVE:
            return
        path, _ = QFileDialog.getOpenFileName(self.window, '选择视频文件', '',
                         '视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;所有文件 (*.*)')
        if not path or not os.path.isfile(path) or not self.session.begin_loading():
            return
        self._emit()
        try:
            self._info = self._info_factory(path)
            self._info.checked.connect(self._video_checked)
            self._info.finished.connect(self._info_thread_finished)
            self._info.start()
        except Exception:
            self.session.loading_failed()
            self._emit()

    @Slot(str, object)
    def _video_checked(self, path, info):
        if self._closing:
            return
        if not self.session.video_loaded(path, info):
            self.session.loading_failed()
        self.previewChanged.emit('')
        self._emit()

    @Slot()
    def _info_thread_finished(self):
        if not self._closing and self.session.phase == 'loading':
            self.session.loading_failed()
            self._emit()
        self._thread_finished()

    @Slot(str)
    def startAnalysis(self, scene):
        if self._closing or self._busy() or scene not in SCENES or not os.path.isfile(self.session.video_path):
            return
        if not self.session.begin_check():
            return
        self._scene_hint = scene
        self._emit()
        try:
            self._connection = self._connection_factory()
            self._connection.checked.connect(self._connection_checked)
            self._connection.finished.connect(self._connection_thread_finished)
            self._connection.start()
        except Exception:
            self.session.connection_failed()
            self._emit()

    @Slot(bool, str)
    def _connection_checked(self, ok, _message):
        if self._closing:
            return
        if not ok or not os.path.isfile(self.session.video_path):
            self.session.connection_failed()
            self._emit()
            return
        if not self.session.begin_analysis():
            return
        self.previewChanged.emit('')
        self._emit()
        try:
            self._analysis = self._analysis_factory(self.session.video_path, self._scene_hint)
            self._analysis.progress_updated.connect(self._progress)
            self._analysis.frame_completed.connect(self._frame)
            self._analysis.analysis_finished.connect(self._finished)
            self._analysis.error_occurred.connect(self._failed)
            self._analysis.finished.connect(self._analysis_thread_finished)
            self._analysis.start()
        except Exception:
            self.session.failed()
            self._emit()

    @Slot()
    def _connection_thread_finished(self):
        if not self._closing and self.session.phase == 'checking':
            self.session.connection_failed()
            self._emit()
        self._thread_finished()

    @Slot(int, str)
    def _progress(self, percent, message):
        self.session.set_progress(percent, message)
        self._queue_update()

    @Slot(int, int, float, str, str, dict)
    def _frame(self, index, _total, timestamp, annotated, original, result):
        if self.session.add_frame(index, timestamp, annotated, original, result):
            self._queue_update(preview=self.session.selected == index)

    @Slot(list, str)
    def _finished(self, rows, output):
        allowed = bool(output and _inside(output, self.output_root))
        stopped = self.session.phase == 'stopping'
        report = read_run_report(output, root=self.output_root) if allowed and not stopped else None
        issues = inspect_run(report, rows, expected_video=self.session.video_path) if not stopped else ()
        complete = allowed and not stopped and not issues
        if self.session.finish(rows, output if allowed else '', complete=complete):
            if issues:
                self.session.message += '。' + explain_issues(issues)
            self._flush_timer.stop()
            self._pending_preview = False
            self._emit()
            self.previewChanged.emit(_preview_data(self.session.frames.get(self.session.selected), self.output_root))

    @Slot(str)
    def _failed(self, _safe_message):
        if self.session.failed():
            self._emit()

    @Slot()
    def _analysis_thread_finished(self):
        # Guard against a worker silently exiting without either callback.
        if self.session.phase in ('running', 'paused', 'stopping'):
            self.session.failed()
            self._emit()
        self._thread_finished()

    @Slot()
    def togglePause(self):
        if self._analysis is None or not self._analysis.isRunning():
            return
        try:
            if self.session.phase == 'running':
                self._analysis.pause()
                self.session.set_paused(True)
            elif self.session.phase == 'paused':
                self._analysis.resume()
                self.session.set_paused(False)
            self._emit()
        except Exception:
            self.session.failed()
            self._emit()

    @Slot()
    def stopAnalysis(self):
        if self._analysis is None or not self._analysis.isRunning() or self.session.phase not in ('running', 'paused'):
            return
        if QMessageBox.question(self.window, '停止分析', '确定停止当前分析吗？',
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.session.stop()
        try:
            self._analysis.stop()
        except Exception:
            self.session.message = '停止请求失败，请等待任务结束后检查结果'
        self._emit()

    @Slot(int)
    def selectFrame(self, index):
        if self.session.select_frame(index):
            self._pending_preview = False
            self._emit()
            self.previewChanged.emit(_preview_data(self.session.frames[index], self.output_root))

    @Slot()
    def openOutput(self):
        path = self.session.output_dir
        if path and _inside(path, self.output_root) and Path(path).is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _start_export(self, kind):
        output = self.session.output_dir
        if self._closing or self._busy() or not output or not _inside(output, self.output_root):
            return
        try:
            from core.evidence_exporter import export_evidence_package, generate_rectification_notice
            func = export_evidence_package if kind == 'evidence' else generate_rectification_notice
            self._export = self._export_factory(func, output, self.session.video_path)
            self._export_result_seen = False
            self._export.done.connect(self._export_done)
            self._export.failed.connect(self._export_failed)
            self._export.finished.connect(self._export_thread_finished)
            self._export.start()
            self.session.message = '正在导出文件，候选仍需人工核查'
            self._emit()
        except Exception:
            self.session.message = '导出任务启动失败，请在经典界面查看'
            self._emit()

    @Slot()
    def exportEvidence(self):
        self._start_export('evidence')

    @Slot()
    def exportNotice(self):
        self._start_export('notice')

    @Slot(object)
    def _export_done(self, result):
        self._export_result_seen = True
        candidate = result.get('zip') if isinstance(result, dict) else result
        ok = (isinstance(candidate, str) and _inside(candidate, self.session.output_dir)
              and Path(candidate).is_file())
        self.session.message = '文件已导出，请在结果目录查看' if ok else '导出未生成有效文件，请检查结果'
        self._emit()

    @Slot(str)
    def _export_failed(self, _message):
        self._export_result_seen = True
        self.session.message = '导出失败或没有可导出事件，请检查结果后重试'
        self._emit()

    @Slot()
    def _export_thread_finished(self):
        if not self._export_result_seen:
            self.session.message = '导出线程未返回结果，请检查输出目录'
        self._emit()
        self._thread_finished()

    @Slot()
    def openSettings(self):
        if self._closing or self._busy() or self.session.phase in ACTIVE:
            return
        from ui.settings_dialog import SettingsDialog
        SettingsDialog(self.window).exec()

    @Slot()
    def openLegacy(self):
        if not self._closing and not self._busy() and self.session.phase not in ACTIVE:
            self.legacyRequested.emit()

    def shutdown(self):
        """Never destroy a running QThread; the window closes after workers actually finish."""
        self._closing = True
        if self._analysis is not None and self._analysis.isRunning():
            self.session.stop()
            try:
                self._analysis.stop()
            except Exception:
                pass
        return not self._busy()
