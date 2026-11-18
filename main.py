# -*- coding: utf-8 -*-
"""
视频违章分析系统 - 主程序入口
"""
import sys
import os
import faulthandler
import tempfile
import traceback

# 打包与源码入口均先定位模块；此操作不读取运行配置。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 仅离线自检：GUI 初始化、崩溃日志打开及运行配置加载之前直接退出。
if __name__ == "__main__" and "--self-check" in sys.argv[1:]:
    import json
    if sys.argv[1:] != ["--self-check"]:
        print(json.dumps({"mode": "offline_package_self_check", "status": "invalid_arguments"}))
        raise SystemExit(2)
    try:
        from core.package_health import run_package_self_check
        result = run_package_self_check()
    except Exception:
        result = {"mode": "offline_package_self_check", "status": "failed"}
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    raise SystemExit(0 if result.get("status") == "ok" else 1)

_crash = open(os.path.join(tempfile.gettempdir(), "vv_crash.log"), "a", buffering=1)
faulthandler.enable(file=_crash)


def _excepthook(t, v, tb):
    traceback.print_exception(t, v, tb, file=_crash)
    _crash.flush()


sys.excepthook = _excepthook

from PySide6.QtWidgets import QApplication, QStyleFactory
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

from core.config import APP_NAME, APP_VERSION
from ui.theme import stylesheet
from ui.main_window import MainWindow


def main():
    # 高DPI适配
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    # Fusion 基础样式：各平台观感一致，再叠加自定义设计系统
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")

    # 设置全局字体（中文优先微软雅黑，回退到系统无衬线）
    font = QFont()
    font.setFamilies([
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "PingFang SC",
        "Source Han Sans SC",
        "sans-serif",
    ])
    font.setPointSize(9)
    font.setHintingPreference(QFont.PreferDefaultHinting)
    app.setFont(font)

    # 应用设计系统样式表（样式集中在 ui/theme.py 维护）
    app.setStyleSheet(stylesheet())

    if "--web-ui" in sys.argv[1:]:
        try:
            from ui.web_window import WebMainWindow
            window = WebMainWindow()
        except (ImportError, OSError):
            # The legacy production spec deliberately excludes WebEngine.
            # Never turn the optional preview flag into a broken application.
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(None, "界面预览不可用", "当前程序未包含 JavaScript 界面组件，将打开经典界面。")
            window = MainWindow()
    else:
        window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
