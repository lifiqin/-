# -*- coding: utf-8 -*-
"""界面后台线程集合（QThread workers）

原则：凡是可能阻塞界面超过几十毫秒的工作，一律放到这里的 QThread 中执行，
通过 Signal 把结果发回主线程；主窗口只做信号接线，不写业务逻辑。
"""
from PySide6.QtCore import QThread, Signal

from core.analysis_engine import get_analysis_engine
from core.knowledge_base import get_knowledge_base
from core.api_client import get_ai_client
from core.api_errors import safe_error_info


class AnalysisThread(QThread):
    """后台分析线程"""
    progress_updated = Signal(int, str)
    # 只发轻量元数据（序号/总数/时刻/落盘路径/结果）。画面像素（整帧 numpy
    # 数组 + 标注图）逐帧经信号跨线程投递会被 Qt 事件队列持有，界面来不及
    # 消费时内存持续上涨，编组本身也有开销。引擎已把原图/标注图落盘，
    # 界面按路径按需读入即可。
    frame_completed = Signal(int, int, float, str, str, dict)
    analysis_finished = Signal(list, str)
    error_occurred = Signal(str)

    def __init__(self, video_path, scene_hint):
        super().__init__()
        self.video_path = video_path
        self.scene_hint = scene_hint
        self.engine = get_analysis_engine()

    def run(self):
        try:
            def progress_cb(percent, status):
                self.progress_updated.emit(percent, status)

            def frame_cb(i, total, frame_data, annotated_img, result):
                self.frame_completed.emit(
                    i, total, float(frame_data.get("timestamp", 0) or 0),
                    frame_data.get("annotated_path") or "",
                    frame_data.get("original_path") or "",
                    result)

            def result_cb(results, output_dir):
                self.analysis_finished.emit(results, output_dir)

            self.engine.analyze_video(
                self.video_path,
                scene_hint=self.scene_hint,
                progress_callback=progress_cb,
                frame_callback=frame_cb,
                result_callback=result_cb
            )
        except Exception as e:
            error = safe_error_info(e)
            print(f"[analysis] 后台任务异常: {error}")
            self.error_occurred.emit(error)

    def stop(self):
        self.engine.stop()

    def pause(self):
        self.engine.pause()

    def resume(self):
        self.engine.resume()


class ConnectionCheckThread(QThread):
    """API 连接探测线程。

    原实现在 UI 线程里同步调用 test_connection()，而它会逐个探测 11 个通道，
    网络不佳时界面会长时间无响应（表现为"点开始分析就卡"）。
    这里把探测放到后台线程，UI 始终保持可交互。
    """
    checked = Signal(bool, str)

    def run(self):
        try:
            ok, msg = get_ai_client().test_connection()
        except Exception as e:
            ok, msg = False, safe_error_info(e)
        self.checked.emit(ok, msg)


class ExportWorkerThread(QThread):
    """结果导出后台线程（证据包 / 整改通知书），避免裁剪视频时冻结界面"""
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self._fn = fn
        self._args = args

    def run(self):
        try:
            self.done.emit(self._fn(*self._args))
        except Exception as e:
            error = safe_error_info(e)
            print(f"[export] 后台任务异常: {error}")
            self.failed.emit(error)


class KnowledgeAutoImportThread(QThread):
    """启动时自动导入「知识库」文件夹PDF的后台线程"""
    finished_import = Signal(list)

    def run(self):
        try:
            kb = get_knowledge_base()
            results = kb.auto_import_folder()
            self.finished_import.emit(results or [])
        except Exception as e:
            print(f"[knowledge] 后台任务异常: {safe_error_info(e)}")
            self.finished_import.emit([])


class YoloWarmupThread(QThread):
    """后台预热 YOLO 检测器（低优先级）。

    ONNX 会话创建（GPU/DirectML 后端可达数秒至数十秒）原先在主窗口构造期
    同步执行，导致"一启动就贼卡"。移到启动后空闲期后台加载：既不阻塞界面，
    又能让首次点「开始分析」时模型已就绪。
    """
    warmed = Signal(bool)

    def run(self):
        try:
            ok = get_analysis_engine().warmup_yolo()
        except Exception as e:
            print(f"[yolo] 预热异常: {safe_error_info(e)}")
            ok = False
        self.warmed.emit(ok)


class SceneNameThread(QThread):
    """自定义场景 AI 命名后台线程。

    用户圈选多帧合并为场景后调用视觉模型起名；失败发 failed，
    由主窗口降级为「自定义场景N」兜底名，场景创建不被阻断。
    """
    named = Signal(str)
    failed = Signal(str)

    def __init__(self, image_paths):
        super().__init__()
        self._paths = list(image_paths or [])

    def run(self):
        try:
            from core.custom_scene import ai_name_scene
            name = ai_name_scene(get_ai_client(), self._paths)
            self.named.emit(name or "")
        except Exception as e:
            error = safe_error_info(e)
            print(f"[scene] AI 命名失败，使用兜底名: {error}")
            self.failed.emit(error)


class RoiCheckThread(QThread):
    """ROI 定向检查（可能在网络调用上耗时数秒）"""
    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(self, output_dir, parent=None):
        super().__init__(parent)
        self.output_dir = output_dir

    def run(self):
        try:
            from core.roi_check import run_check
            self.finished_ok.emit(run_check(self.output_dir))
        except Exception as e:
            self.failed.emit(safe_error_info(e))
