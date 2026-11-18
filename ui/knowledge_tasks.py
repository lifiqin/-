# -*- coding: utf-8 -*-
"""
知识库后台任务线程

从 ui/main_window.py 抽出，供主窗口与知识库检索面板共用，
避免两边互相 import 造成循环依赖。
"""
from PySide6.QtCore import QThread, Signal

from core.knowledge_base import get_knowledge_base


def _run_guarded(error_signal, work):
    """执行后台任务体；异常时打印堆栈并通过 error_signal 上报（三个线程共用）。"""
    try:
        work()
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_signal.emit(str(e))


class KnowledgeAddThread(QThread):
    """添加知识库PDF的后台线程"""

    progress_updated = Signal(int, str)
    finished_ok = Signal(dict)
    finished_error = Signal(str)

    def __init__(self, pdf_path, title=None):
        super().__init__()
        self.pdf_path = pdf_path
        self.title = title

    def run(self):
        def work():
            kb = get_knowledge_base()
            result = kb.add_pdf(
                self.pdf_path, self.title,
                lambda percent, status: self.progress_updated.emit(percent, status))
            if result["success"]:
                self.finished_ok.emit(result)
            else:
                self.finished_error.emit(result.get("error", "未知错误"))

        _run_guarded(self.finished_error, work)


class KnowledgeClassifyThread(QThread):
    """AI场景分类的后台线程"""

    progress_updated = Signal(int, str)
    finished_classify = Signal(int)
    finished_error = Signal(str)

    def run(self):
        def work():
            kb = get_knowledge_base()
            n = kb.ai_classify_all(
                progress_callback=lambda p, s: self.progress_updated.emit(p, s))
            self.finished_classify.emit(n)

        _run_guarded(self.finished_error, work)


class KnowledgeSearchThread(QThread):
    """检索后台线程（向量检索需要调用 embedding 接口，不能阻塞界面）"""

    finished_search = Signal(list)
    finished_error = Signal(str)

    def __init__(self, query, top_k=5, scenes=None, titles=None, min_score=0.0):
        super().__init__()
        self.query = query
        self.top_k = top_k
        self.scenes = scenes
        self.titles = titles
        self.min_score = min_score

    def run(self):
        def work():
            kb = get_knowledge_base()
            results = kb.search(
                self.query,
                top_k=self.top_k,
                scenes=self.scenes or None,
                titles=self.titles or None,
                min_score=self.min_score,
            )
            self.finished_search.emit(results or [])

        _run_guarded(self.finished_error, work)
