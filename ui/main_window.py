# -*- coding: utf-8 -*-
"""
主窗口
"""
import json
import os
from datetime import datetime

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QFileDialog, QProgressBar, QTextEdit,
    QListWidget, QListWidgetItem, QSplitter, QMessageBox,
    QGroupBox, QLineEdit, QComboBox, QDialog, QFrame,
    QAbstractItemView, QInputDialog
)
from PySide6.QtCore import Qt, QTimer, QSize, QThread, QUrl
from PySide6.QtGui import QDesktopServices

from application.run_publication import explain_issues, inspect_run, read_run_report
from core.config import APP_NAME, APP_VERSION, get_config
from core.frame_lookup import normalize_timestamp
from core.report_status import frame_analysis_incomplete
from core.analysis_engine import get_analysis_engine
from core.custom_scene import (
    add_scene, fallback_scene_name, load_scenes, remove_scene, save_scenes,
)
from ui.render_html import esc, render_violations_html
from ui.theme import apply_card_shadow, repolish
from ui.workers import (
    AnalysisThread, ConnectionCheckThread, ExportWorkerThread, KnowledgeAutoImportThread,
    YoloWarmupThread, SceneNameThread, RoiCheckThread,
)
from ui.settings_dialog import SettingsDialog


class FrameRow(QWidget):
    """帧列表行：序号、视频时刻和候选/不完整状态。"""

    def __init__(self, frame_no: int, timestamp: float, violation_count: int,
                 analysis_incomplete: bool = False):
        super().__init__()
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # 选中交互交给 QListWidget
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)

        name = QLabel(f"帧 {frame_no}")
        name.setObjectName("frameRowTitle")
        layout.addWidget(name)

        time_label = QLabel(f"{minutes:02d}:{seconds:02d}")
        time_label.setObjectName("frameRowTime")
        time_label.setToolTip(f"该帧位于视频第 {minutes} 分 {seconds} 秒")
        layout.addWidget(time_label)

        layout.addStretch()

        if analysis_incomplete:
            badge = QLabel(f"分析不完整 · {violation_count} 条候选")
            badge.setObjectName("badgeDanger")
        elif violation_count > 0:
            badge = QLabel(f"{violation_count} 条候选")
            badge.setObjectName("badgeDanger")
        else:
            badge = QLabel("未输出候选")
            badge.setObjectName("badgeMuted")
        badge.setFixedHeight(18)
        layout.addWidget(badge)

    def sizeHint(self):
        return QSize(0, 36)


# ==================== 工作线程 ====================

# ==================== 设置对话框 ====================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.analysis_engine = get_analysis_engine()
        self.video_processor = self.analysis_engine.video_processor
        
        self.current_video_path = None
        self.analysis_results = []
        self.current_result_index = -1
        self.analysis_thread = None
        self.knowledge_thread = None
        self.auto_import_thread = None
        self.classify_thread = None
        self.yolo_warmup_thread = None
        self._knowledge_explorer = None   # 延迟到首次切到「知识库检索」页才构建
        self.current_output_dir = None  # 最近一次分析的输出目录（证据包/通知书用）
        self.export_thread = None       # 结果导出后台线程
        self.custom_scenes = []         # 当前输出目录的自定义场景（人工圈选帧合并）
        self.scene_segments = []        # 自动识别的场景段（来自 report.json）
        self.scene_overrides = []       # 自动/手动合并分组（scene_overrides.json）
        self._scene_name_thread = None  # AI 场景命名后台线程
        
        # 批量更新帧列表的缓存
        self._pending_frames = []
        self._frame_update_timer = QTimer()
        self._frame_update_timer.timeout.connect(self._flush_frame_updates)
        self._frame_update_timer.start(200)  # 每200ms刷新一次
        
        # 单帧画面缓存：内存里最多留当前显示的这一帧（按磁盘路径键控）。
        # 整幅画面逐帧驻留是"内存吃满"的元凶，改为显示时按需读盘。
        self._frame_image_cache = {}
        
        self._init_ui()
        # 知识库自动导入延迟到窗口稳定后执行，且低优先级，避免打开软件即触发
        # CPU 密集的 PDF 解析/向量化（内含 38MB 大毕业规程），导致风扇狂转。
        QTimer.singleShot(3000, self._auto_import_knowledge)
        # YOLO 模型预热：窗口稳定后后台低优先级加载，界面零等待（加载期间
        # 若用户已点开始分析，惰性访问会复用同一单例，不会重复加载）
        QTimer.singleShot(2000, self._warmup_yolo)

    def _warmup_yolo(self):
        """后台预热 YOLO 检测器（低优先级线程，避免抢前台 CPU）"""
        self.yolo_warmup_thread = YoloWarmupThread()
        self.yolo_warmup_thread.start()
        self.yolo_warmup_thread.setPriority(QThread.Priority.LowPriority)

    def _auto_import_knowledge(self):
        """后台自动导入「知识库」文件夹中的PDF（低优先级，避免抢前台 CPU）"""
        self.auto_import_thread = KnowledgeAutoImportThread()
        # setPriority 必须在线程运行后调用：start() 之前调用会打印
        # "QThread::setPriority: Cannot set priority, thread is not running"
        self.auto_import_thread.start()
        self.auto_import_thread.setPriority(QThread.Priority.LowPriority)

        def on_done(results):
            if not results:
                return
            self._load_knowledge_list()
            ok = [r for r in results if r.get("success")]
            failed = [r for r in results if not r.get("success")]
            if ok:
                total = sum(r.get("count", 0) for r in ok)
                self.status_label.setText(
                    f"已自动导入 {len(ok)} 个知识库文档（{total} 块），可直接开始分析")
            if failed:
                names = "、".join(r.get("title", "?") for r in failed)
                self.status_label.setText(f"部分知识库文档导入失败：{names}")

        self.auto_import_thread.finished_import.connect(on_done)

    def _init_ui(self):
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        central_widget = QWidget()
        central_widget.setObjectName("centralRoot")
        self.setCentralWidget(central_widget)
        root = QVBoxLayout(central_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 顶栏（材质浮层）与内容区分离：内容区带统一内边距，顶栏通栏
        root.addWidget(self._build_top_bar())

        content = QWidget()
        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(20, 16, 20, 14)
        main_layout.setSpacing(14)

        # 主内容区 - Tab页
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Tab 1: 视频分析
        self._create_analysis_tab()

        # Tab 2: 知识库管理
        self._create_knowledge_tab()

        # Tab 3: AI 复核结果（只读，QA 分级处置）
        self._create_review_tab()

        # 底部状态栏
        separator = QFrame()
        separator.setObjectName("hSeparator")
        separator.setFixedHeight(1)
        main_layout.addWidget(separator)

        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("statusLabel")
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        self.status_hint = QLabel("")
        self.status_hint.setObjectName("metaLabel")
        status_row.addWidget(self.status_hint)
        main_layout.addLayout(status_row)

        root.addWidget(content, 1)

    def _build_top_bar(self):
        """顶栏：品牌标题 + 版本 + 说明 + 设置。

        半透明材质浮层（QFrame#topBar）+ 底部发丝线，内容区在其下，符合
        Apple「导航/工具栏是浮在内容之上的功能层」的分层原则。
        """
        bar = QFrame()
        bar.setObjectName("topBar")
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(20, 14, 20, 12)
        outer.setSpacing(3)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)

        title_label = QLabel(APP_NAME)
        title_label.setObjectName("titleLabel")
        title_row.addWidget(title_label)

        version_chip = QLabel(f"v{APP_VERSION}")
        version_chip.setObjectName("versionChip")
        title_row.addWidget(version_chip)

        title_row.addStretch()

        self.settings_btn = QPushButton("设置")
        self.settings_btn.setObjectName("ghostBtn")
        self.settings_btn.setCursor(Qt.PointingHandCursor)
        self.settings_btn.clicked.connect(self._open_settings)
        title_row.addWidget(self.settings_btn)

        outer.addLayout(title_row)

        subtitle = QLabel("无人机航拍电力施工现场 · 9 类违章自动识别与法规依据追溯")
        subtitle.setObjectName("subTitleLabel")
        outer.addWidget(subtitle)
        return bar

    def _create_analysis_tab(self):
        """创建视频分析Tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # 视频选择卡片
        video_group = QGroupBox("视频文件")
        video_layout = QHBoxLayout()
        video_layout.setSpacing(10)

        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("选择要分析的无人机航拍视频…")
        self.video_path_edit.setReadOnly(True)
        self.video_path_edit.setMinimumHeight(34)
        video_layout.addWidget(self.video_path_edit, 1)

        self.browse_btn = QPushButton("浏览…")
        self.browse_btn.setObjectName("ghostBtn")
        self.browse_btn.setCursor(Qt.PointingHandCursor)
        self.browse_btn.clicked.connect(self._browse_video)
        video_layout.addWidget(self.browse_btn)

        video_group.setLayout(video_layout)
        apply_card_shadow(video_group)
        layout.addWidget(video_group)

        # 分析控制卡片
        control_group = QGroupBox("分析控制")
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)

        scene_label = QLabel("场景类型")
        scene_label.setObjectName("metaLabel")
        control_layout.addWidget(scene_label)

        self.scene_combo = QComboBox()
        self.scene_combo.addItems([
            "电力建设现场", "电力检修现场", "变电站作业",
            "输电线路作业", "配电作业", "电缆井作业", "通用安全检查"
        ])
        self.scene_combo.setMinimumWidth(160)
        self.scene_combo.setMinimumHeight(34)
        control_layout.addWidget(self.scene_combo)

        control_layout.addSpacing(12)

        self.start_btn = QPushButton("开始分析")
        self.start_btn.setObjectName("successBtn")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self._start_analysis)
        control_layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setObjectName("warningBtn")
        self.pause_btn.setCursor(Qt.PointingHandCursor)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._toggle_pause)
        control_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("dangerBtn")
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_analysis)
        control_layout.addWidget(self.stop_btn)

        control_layout.addStretch()

        control_group.setLayout(control_layout)
        apply_card_shadow(control_group)
        layout.addWidget(control_group)

        # 进度行（细进度条 + 百分比）
        progress_row = QHBoxLayout()
        progress_row.setSpacing(10)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setTextVisible(False)
        progress_row.addWidget(self.progress_bar, 1)
        self.progress_label = QLabel("0%")
        self.progress_label.setObjectName("statPill")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setMinimumWidth(48)
        progress_row.addWidget(self.progress_label)
        layout.addLayout(progress_row)

        # 主内容区分栏
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # 左侧：帧列表
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        left_header = QHBoxLayout()
        left_title = QLabel("分析结果")
        left_title.setObjectName("sectionLabel")
        left_header.addWidget(left_title)
        left_header.addStretch()

        # 结果导出：证据包（片段+截图+坐标+依据）与安全违章通知书（Word）
        self.export_pkg_btn = QPushButton("导出证据包")
        self.export_pkg_btn.setObjectName("ghostBtn")
        self.export_pkg_btn.setCursor(Qt.PointingHandCursor)
        self.export_pkg_btn.setEnabled(False)
        self.export_pkg_btn.setToolTip(
            "导出违章事件的视频片段+代表帧+坐标+判定依据，打包为文件夹与zip")
        self.export_pkg_btn.clicked.connect(self._export_evidence_package)
        left_header.addWidget(self.export_pkg_btn)

        self.notice_btn = QPushButton("安全违章通知书")
        self.notice_btn.setObjectName("ghostBtn")
        self.notice_btn.setCursor(Qt.PointingHandCursor)
        self.notice_btn.setEnabled(False)
        self.notice_btn.setToolTip(
            "生成 Word 格式安全违章通知书（含事件截图、判定依据、整改要求）")
        self.notice_btn.clicked.connect(self._generate_rectification_notice)
        left_header.addWidget(self.notice_btn)

        self.roi_btn = QPushButton("ROI框选")
        self.roi_btn.setObjectName("ghostBtn")
        self.roi_btn.setCursor(Qt.PointingHandCursor)
        self.roi_btn.setEnabled(False)
        self.roi_btn.setToolTip(
            "在代表帧上框出脚手架/临边/电缆/外电区域，导出 rois.json 后做定向检查")
        self.roi_btn.clicked.connect(self._open_roi_editor)
        left_header.addWidget(self.roi_btn)

        self.roi_run_btn = QPushButton("ROI检查")
        self.roi_run_btn.setObjectName("ghostBtn")
        self.roi_run_btn.setCursor(Qt.PointingHandCursor)
        self.roi_run_btn.setEnabled(False)
        self.roi_run_btn.setToolTip("读取 rois.json，在框定区域内做三采样检查，结果进「疑似线索」")
        self.roi_run_btn.clicked.connect(self._run_roi_check)
        left_header.addWidget(self.roi_run_btn)

        self.summary_label = QLabel("共 0 帧")
        self.summary_label.setObjectName("statPill")
        left_header.addWidget(self.summary_label)
        left_layout.addLayout(left_header)

        self.frame_list = QListWidget()
        self.frame_list.currentRowChanged.connect(self._on_frame_selected)
        self.frame_list.setUniformItemSizes(True)  # 优化性能
        self.frame_list.setSpacing(2)
        # 多选：Ctrl/Shift 圈选多帧后可合并为一个自定义场景
        self.frame_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.frame_list.itemSelectionChanged.connect(self._on_frame_selection_changed)
        left_layout.addWidget(self.frame_list, 1)

        self.frame_tip = QLabel("点击帧查看画面与违章详情；Ctrl/Shift 多选后可合并为场景")
        self.frame_tip.setObjectName("metaLabel")
        left_layout.addWidget(self.frame_tip)

        # 自定义场景：人工圈选帧合并为场景，报告导出时场景内相似违章合并、AI 自动命名
        scene_row = QHBoxLayout()
        scene_row.setSpacing(8)
        scene_title = QLabel("自定义场景")
        scene_title.setObjectName("sectionLabel")
        scene_row.addWidget(scene_title)
        scene_row.addStretch()

        self.merge_scene_btn = QPushButton("合并为场景")
        self.merge_scene_btn.setObjectName("ghostBtn")
        self.merge_scene_btn.setCursor(Qt.PointingHandCursor)
        self.merge_scene_btn.setEnabled(False)
        self.merge_scene_btn.setToolTip(
            "在左侧列表 Ctrl/Shift 多选至少 2 帧后点击：\n"
            "把这些关键帧合并为一个作业场景，AI 自动为场景命名；\n"
            "生成报告时该场景内相似违章合并为一条展示")
        self.merge_scene_btn.clicked.connect(self._merge_selected_frames_to_scene)
        scene_row.addWidget(self.merge_scene_btn)

        self.remove_scene_btn = QPushButton("移除场景")
        self.remove_scene_btn.setObjectName("ghostBtn")
        self.remove_scene_btn.setCursor(Qt.PointingHandCursor)
        self.remove_scene_btn.setEnabled(False)
        self.remove_scene_btn.setToolTip("删除选中的自定义场景（不影响帧与违章数据）")
        self.remove_scene_btn.clicked.connect(self._remove_selected_scene)
        scene_row.addWidget(self.remove_scene_btn)

        self.auto_merge_btn = QPushButton("自动合并")
        self.auto_merge_btn.setObjectName("ghostBtn")
        self.auto_merge_btn.setCursor(Qt.PointingHandCursor)
        self.auto_merge_btn.setEnabled(False)
        self.auto_merge_btn.setToolTip("把相邻且同名的自动场景合并为一个（可随时撤销）")
        self.auto_merge_btn.clicked.connect(self._auto_merge_scenes)
        scene_row.addWidget(self.auto_merge_btn)
        left_layout.addLayout(scene_row)

        self.scene_list = QListWidget()
        self.scene_list.setFixedHeight(170)
        self.scene_list.itemSelectionChanged.connect(self._on_scene_selection_changed)
        left_layout.addWidget(self.scene_list)

        splitter.addWidget(left_widget)

        # 右侧：画面 + 详情
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        from ui.interactive_image import InteractiveImageView
        self.image_view = InteractiveImageView()
        self.image_view.setMinimumSize(600, 420)
        right_layout.addWidget(self.image_view, 3)

        detail_header = QHBoxLayout()
        detail_title = QLabel("违章详情")
        detail_title.setObjectName("sectionLabel")
        detail_header.addWidget(detail_title)
        detail_header.addStretch()
        self.detail_count_label = QLabel("未开始")
        self.detail_count_label.setObjectName("statPill")
        detail_header.addWidget(self.detail_count_label)
        right_layout.addLayout(detail_header)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMinimumHeight(150)
        right_layout.addWidget(self.detail_text, 1)

        splitter.addWidget(right_widget)
        splitter.setSizes([320, 980])

        layout.addWidget(splitter, 1)

        self.tab_widget.addTab(tab, "视频分析")

    def _create_knowledge_tab(self):
        """创建知识库检索页（占位）。

        真正的 KnowledgeExplorer 构建耗时约 0.3s，且会触发 15.7MB 索引解析
        （约 0.2s）。这些都只在用户切到该页时才需要，故延迟到首次切换时构建，
        避免拖慢启动与首屏响应。
        """
        self._knowledge_tab = QWidget()
        self._knowledge_tab_index = self.tab_widget.addTab(self._knowledge_tab, "知识库检索")
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def _create_review_tab(self):
        """复核结果页签：原报告只读，人工框另存旁路记录。

        只展示 report.json 的 events 与 disposition 分组；不引入框编辑 / 人工
        校正控件直接改写结果。「一键恢复远景待核」与
        ``tools/ai_review_restore.py`` 共用 ``core.ai_review`` 的恢复函数。
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("AI 复核结果（只读）")
        title.setObjectName("cardTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.manual_area_btn = QPushButton("区域人工纠框")
        self.manual_area_btn.setObjectName("ghostBtn")
        self.manual_area_btn.setToolTip("选择分析目录和区域事件，在确切原始代表帧上画框；另存人工候选，不改原报告")
        self.manual_area_btn.clicked.connect(self._open_manual_area_review)
        header.addWidget(self.manual_area_btn)
        self.review_restore_btn = QPushButton("一键恢复远景待核")
        self.review_restore_btn.setObjectName("ghostBtn")
        self.review_restore_btn.setEnabled(False)
        self.review_restore_btn.clicked.connect(self._on_restore_low_priority_review)
        header.addWidget(self.review_restore_btn)
        layout.addLayout(header)

        self.review_hint = QLabel("本次未启用 AI 复核")
        self.review_hint.setWordWrap(True)
        layout.addWidget(self.review_hint)

        self.review_list = QListWidget()
        self.review_list.setSelectionMode(QAbstractItemView.NoSelection)
        layout.addWidget(self.review_list, 1)

        self._review_report_path = ""
        self._review_tab_index = self.tab_widget.addTab(tab, "复核结果")

    def _open_manual_area_review(self):
        """选择本次或历史分析目录；人工框只存旁路，不覆盖原报告。"""
        folder = QFileDialog.getExistingDirectory(
            self, "选择含 report.json 和 original 原图的分析结果目录", self.current_output_dir or "")
        if not folder:
            return
        try:
            from core.area_localization import AREA_TITLES
            from ui.area_manual_editor import open_manual_area_editor
            with open(os.path.join(folder, "report.json"), encoding="utf-8-sig") as handle:
                report = json.load(handle)
            events = report.get("events") if isinstance(report, dict) else None
            if not isinstance(events, list):
                raise ValueError("报告缺少事件列表")
            choices = []
            ids = set()
            for event in events:
                if not isinstance(event, dict):
                    continue
                eid, title = event.get("event_id"), event.get("title")
                if (type(eid) is not int or eid < 0 or eid in ids
                        or not isinstance(title, str) or title not in AREA_TITLES):
                    continue
                ids.add(eid)
                ts = normalize_timestamp(event.get("representative_timestamp"))
                label = f"#{eid}  {title}  {'时间未知' if ts is None else f'{ts:g} 秒'}"
                choices.append((label, eid))
            if not choices:
                QMessageBox.information(self, "区域人工纠框", "该报告没有支持人工定位的区域事件。")
                return
            selected, ok = QInputDialog.getItem(self, "区域人工纠框", "选择需要核查的事件：",
                                                 [item[0] for item in choices], 0, False)
            if not ok:
                return
            selected_id = next((eid for label, eid in choices if label == selected), None)
            if selected_id is None:
                raise ValueError("所选事件无效")
            saved = open_manual_area_editor(folder, selected_id, self)
            if saved:
                QMessageBox.information(
                    self, "已保存人工候选框",
                    "人工框已另存，未覆盖原报告、原图或证据。现有报告和导出不会自动替换为人工框。\n\n"
                    f"修订记录：{saved}")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "区域人工纠框", f"无法打开该事件：{type(exc).__name__}。请核查报告和原图。")

    def _refresh_review_tab(self, output_dir=None):
        """按 disposition 分组刷新只读列表；未启用 AI 复核时给出明确提示。"""
        from core.review_policy import source_video_review_guard
        if not hasattr(self, "review_list"):
            return
        folder = output_dir or self.current_output_dir or ""
        report_path = os.path.join(folder, "report.json") if folder else ""
        events = []
        enabled = bool(get_config().get("ai_review_enabled", False))
        if report_path and os.path.isfile(report_path):
            try:
                with open(report_path, encoding="utf-8") as handle:
                    report = json.load(handle)
                if isinstance(report, dict):
                    events = [e for e in (report.get("events") or [])
                              if isinstance(e, dict)]
            except Exception:
                events = []

        self.review_list.clear()
        has_disposition = any(str(e.get("review_disposition") or "")
                              for e in events)
        source_count = sum(source_video_review_guard(e) for e in events)
        if (not enabled or not has_disposition) and not source_count:
            self.review_hint.setText("本次未启用 AI 复核")
            self.review_restore_btn.setEnabled(False)
            self._review_report_path = ""
            return

        self._review_report_path = report_path
        groups = [
            ("待复核（主队列）",
             lambda e: not source_video_review_guard(e) and not str(e.get("review_disposition") or "")),
            ("已处置（未确认）",
             lambda e: not source_video_review_guard(e) and str(e.get("review_disposition") or "") in
             ("rejected", "dropped", "low_priority_confirm")),
            ("远景待核（仍在主队列，可一键恢复）",
             lambda e: not source_video_review_guard(e) and str(e.get("review_disposition") or "") ==
             "low_priority_review"),
            ("框需重定位（未确认）",
             lambda e: not source_video_review_guard(e) and str(e.get("review_disposition") or "") ==
             "box_relocalize"),
        ]
        if source_count:
            groups.insert(0, ("来源级保守待核（不代表匹配金标）", source_video_review_guard))
        for label, predicate in groups:
            picked = [e for e in events if predicate(e)]
            head = QListWidgetItem("【%s】%d 条" % (label, len(picked)))
            head.setFlags(head.flags() & ~Qt.ItemIsSelectable)
            self.review_list.addItem(head)
            for event in picked:
                try:
                    secs = float(event.get("representative_timestamp") or 0)
                except (TypeError, ValueError):
                    secs = 0.0
                stamp = "%02d:%02d" % (int(secs // 60), int(secs % 60))
                disposition = str(event.get("review_disposition") or "-")
                line = "  #%s | %s | %s | %s | %s" % (
                    event.get("event_id"), str(event.get("title") or ""),
                    str(event.get("severity") or ""), disposition, stamp)
                self.review_list.addItem(QListWidgetItem(line))
            self.review_list.addItem(QListWidgetItem(""))

        remote = sum(1 for e in events
                     if not source_video_review_guard(e) and str(e.get("review_disposition") or "") ==
                     "low_priority_review")
        main_queue = sum(1 for e in events if e.get("needs_review"))
        if source_count:
            prefix = "AI 复核未启用；" if not enabled else ""
            self.review_hint.setText(
                prefix + "共 %d 条事件｜主待复核 %d 条｜来源保守待核 %d 条｜远景待核 %d 条"
                "（均未确认，仍在主队列可见）" % (len(events), main_queue, source_count, remote))
        else:
            self.review_hint.setText(
                "共 %d 条事件｜主待复核 %d 条｜远景待核 %d 条（均未确认，"
                "仍在主队列可见）" % (len(events), main_queue, remote))
        self.review_restore_btn.setEnabled(remote > 0)

    def _on_restore_low_priority_review(self):
        """一键把 C 组事件改回主待复核队列（纯本地，不重跑 API）。"""
        path = getattr(self, "_review_report_path", "") or ""
        if not path or not os.path.isfile(path):
            QMessageBox.information(self, "一键恢复", "没有可恢复的报告")
            return
        try:
            from core.ai_review import restore_low_priority_review
            from core.atomic_io import atomic_write_json
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
            result = restore_low_priority_review(report)
            if result.get("restored"):
                atomic_write_json(path, report)
            QMessageBox.information(
                self, "一键恢复",
                "已恢复 %d 条远景待核事件到主待复核队列。" %
                int(result.get("restored", 0) or 0))
            self._refresh_review_tab(os.path.dirname(path))
        except Exception as exc:
            QMessageBox.warning(self, "一键恢复失败", str(exc))

    def _on_tab_changed(self, index):
        """切到知识库检索页时才构建该页（一次性）"""
        if index == getattr(self, "_knowledge_tab_index", -1) and self._knowledge_explorer is None:
            self._build_knowledge_tab()

    def _build_knowledge_tab(self):
        from ui.knowledge_explorer import KnowledgeExplorer

        layout = QVBoxLayout(self._knowledge_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        self._knowledge_explorer = KnowledgeExplorer(self)
        layout.addWidget(self._knowledge_explorer)

    # ==================== 视频分析相关 ====================

    def _browse_video(self):
        """浏览选择视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;所有文件 (*.*)"
        )
        if file_path:
            self.current_video_path = file_path
            self.video_path_edit.setText(file_path)
            self._show_video_info(file_path)

    def _show_video_info(self, video_path):
        """读取并显示视频信息（分辨率、时长、帧率）"""
        try:
            info = self.video_processor.get_video_info(video_path)
        except Exception:
            info = None
        if not info:
            self.status_label.setText("警告：无法读取该视频，请确认文件未损坏")
            return
        duration = int(info.get("duration", 0))
        dur_text = f"{duration // 60}分{duration % 60:02d}秒" if duration >= 60 else f"{duration}秒"
        fps = info.get("fps", 0) or 0
        self.status_label.setText(
            f"已选择视频：{info.get('width', '?')}x{info.get('height', '?')}，"
            f"时长 {dur_text}，{fps:.0f} fps"
        )

    def _start_analysis(self):
        """开始分析：连接探测放到后台线程，避免界面卡住"""
        if not self.current_video_path:
            QMessageBox.warning(self, "提示", "请先选择视频文件！")
            return

        if not os.path.exists(self.current_video_path):
            QMessageBox.warning(self, "提示", "视频文件不存在！")
            return

        # 先检查 API 连接（视频分析必须依赖视觉模型），但探测在后台线程执行
        self.start_btn.setEnabled(False)
        self.status_label.setText("正在检查API连接...")
        self._conn_check_thread = ConnectionCheckThread()
        self._conn_check_thread.checked.connect(self._on_connection_checked)
        self._conn_check_thread.start()

    def _on_connection_checked(self, ok, msg):
        """连接探测回调（主线程）"""
        if not ok:
            self.status_label.setText("API连接失败")
            self.start_btn.setEnabled(True)
            reply = QMessageBox.question(
                self, "API连接失败",
                f"API连接失败：{msg[:300]}\n\n"
                f"视频违章分析必须调用视觉大模型，API不可用无法分析。\n"
                f"常见原因：密钥是中转站的，但API地址还是默认的OpenAI官方地址。\n\n"
                f"是否现在打开「设置」配置API？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self._open_settings()
            return

        self._begin_analysis()

    def _begin_analysis(self):
        """连接正常后真正启动分析线程"""
        self.status_label.setText("API连接正常，开始分析...")

        # 清空之前的结果
        self.analysis_results = []
        self.current_result_index = -1
        self._pending_frames = []
        self._frame_image_cache.clear()
        self.frame_list.clear()
        self.detail_text.clear()
        self.progress_bar.setValue(0)
        # 清空图片显示
        self.image_view.scene.clear()
        # 清空自定义场景（新分析对应新输出目录，旧场景不沿用）
        self.custom_scenes = []
        self.scene_list.clear()
        self.merge_scene_btn.setEnabled(False)
        self.remove_scene_btn.setEnabled(False)

        # 更新按钮状态
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("暂停")
        self.stop_btn.setEnabled(True)
        self.browse_btn.setEnabled(False)
        self.current_output_dir = None
        self.export_pkg_btn.setEnabled(False)
        self.notice_btn.setEnabled(False)

        # 启动分析线程
        scene_hint = self.scene_combo.currentText()
        self.analysis_thread = AnalysisThread(self.current_video_path, scene_hint)
        self.analysis_thread.progress_updated.connect(self._on_progress)
        self.analysis_thread.frame_completed.connect(self._on_frame_completed)
        self.analysis_thread.analysis_finished.connect(self._on_analysis_finished)
        self.analysis_thread.error_occurred.connect(self._on_error)
        # 线程结束兜底：无论正常完成、异常还是提前返回，都恢复按钮可用
        self.analysis_thread.finished.connect(self._on_thread_finished)
        self.analysis_thread.start()

    def _on_thread_finished(self):
        """分析线程结束（任何路径）后恢复按钮状态"""
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)

    def _toggle_pause(self):
        """暂停/继续"""
        if self.analysis_thread:
            if self.analysis_thread.engine.is_paused():
                self.analysis_thread.resume()
                self.pause_btn.setText("暂停")
                self.status_label.setText("分析中...")
            else:
                self.analysis_thread.pause()
                self.pause_btn.setText("继续")
                self.status_label.setText("已暂停")

    def _stop_analysis(self):
        """停止分析"""
        if self.analysis_thread:
            reply = QMessageBox.question(
                self, "确认停止", "确定要停止分析吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.analysis_thread.stop()
                self.status_label.setText("正在停止...")

    def _on_progress(self, percent, status):
        """进度更新"""
        pct = max(0, min(100, int(percent)))
        self.progress_bar.setValue(pct)
        self.progress_label.setText(f"{pct}%")
        self.status_label.setText(status)

    def _on_frame_completed(self, index, total, timestamp, annotated_path, original_path, result):
        """单帧分析完成（只登记轻量元数据，不保留画面像素）

        画面像素逐帧驻留或逐帧经信号跨线程投递，都会让内存随帧数上涨。引擎
        在回调前已把原图/标注图落盘，这里只留时刻 + 路径 + 结果，显示某帧时
        再由 _load_frame_image 按路径读入，内存占用与帧数无关。
        """
        entry = {
            "index": index,
            "timestamp": timestamp,
            "annotated_path": annotated_path or None,
            "original_path": original_path or None,
            "result": result,
        }
        self._pending_frames.append(entry)
        self.analysis_results.append(entry)

    def _flush_frame_updates(self):
        """批量刷新帧列表"""
        if not self._pending_frames:
            return
        
        # 批量添加到列表
        items = []
        first_violation_row = -1
        
        for data in self._pending_frames:
            index = data["index"]
            result = data["result"]

            timestamp = data.get("timestamp", 0)
            violation_count = len(result.get("violations") or [])
            
            item = QListWidgetItem()
            item.setSizeHint(QSize(0, 38))
            items.append((item, index + 1, timestamp, violation_count, frame_analysis_incomplete(result)))
        
        # 一次性添加所有（行控件在这里挂上，避免每行重复布局开销）
        for item, frame_no, ts, vc, failed in items:
            row = self.frame_list.count()
            self.frame_list.addItem(item)
            self.frame_list.setItemWidget(item, FrameRow(frame_no, ts, vc, failed))
            if vc > 0 and first_violation_row < 0:
                first_violation_row = row
        
        # 更新统计
        total_violations = sum(
            len(r["result"].get("violations") or []) 
            for r in self.analysis_results
        )
        self.summary_label.setText(
            f"{len(self.analysis_results)} 帧 · {total_violations} 条候选（按帧计）"
        )
        
        # 自动选中第一个有违章的帧
        if first_violation_row >= 0 and self.current_result_index < 0:
            self.frame_list.setCurrentRow(first_violation_row)
        
        self._pending_frames.clear()

    def _on_analysis_finished(self, results, output_dir):
        """保留回调结果，按共享报告契约展示完成或不完整。"""
        # 确保剩余帧都刷新了
        self._flush_frame_updates()

        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)

        # 用事件校准后的结果重建帧列表（校准框、补漏帧）
        if results and output_dir:
            self._rebuild_results_from(results)

        self.current_output_dir = output_dir or None
        # QA 分级处置：分析完成后刷新只读「复核结果」页签（未启用时显示提示）。
        self._refresh_review_tab(output_dir)
        self.export_pkg_btn.setEnabled(bool(output_dir))
        self.notice_btn.setEnabled(bool(output_dir))
        self.roi_btn.setEnabled(bool(output_dir))
        self.roi_run_btn.setEnabled(bool(output_dir))
        # 载入该输出目录已保存的自定义场景（上次会话圈选的场景可继续用于导出）
        self.custom_scenes = load_scenes(self.current_output_dir) if output_dir else []
        self.scene_segments = []
        self.scene_overrides = []
        report = read_run_report(output_dir) if output_dir else None
        segments = report.get("scene_segments") if report is not None else None
        if isinstance(segments, list):
            self.scene_segments = [segment for segment in segments if isinstance(segment, dict)]
        if output_dir:
            try:
                from core.scene_merge import load_overrides
                self.scene_overrides = load_overrides(output_dir)
            except Exception:
                self.scene_overrides = []
        self.auto_merge_btn.setEnabled(bool(self.scene_segments))
        self._refresh_scene_list()

        total_violations = sum(
            len(r["result"].get("violations") or [])
            for r in self.analysis_results
        )
        incomplete_count = sum(
            frame_analysis_incomplete(r.get("result")) for r in self.analysis_results
        )
        self.progress_bar.setValue(100)

        if not output_dir:
            self.status_label.setText("分析终止，未生成完整结果")
            from core.analysis_engine import get_analysis_engine
            reason = (getattr(get_analysis_engine(), "last_fail_reason", "")
                      or "无法读取视频或未抽取到帧")
            QMessageBox.warning(
                self, "分析终止",
                f"未能分析该视频：{reason}\n"
                "请确认视频文件可正常播放后重试。"
            )
            return

        issues = inspect_run(report, results, expected_video=getattr(self, "current_video_path", None))
        if issues:
            frame_failures = f"{incomplete_count} 帧失败，" if incomplete_count else ""
            frame_note = f"，其中 {incomplete_count} 帧失败或不完整" if incomplete_count else ""
            self.status_label.setText(
                f"分析不完整：{frame_failures}{total_violations} 条候选（按帧计）"
            )
            QMessageBox.warning(
                self, "分析不完整",
                f"分析不完整：共处理 {len(self.analysis_results)} 帧{frame_note}。\n"
                f"原因：{explain_issues(issues)}。\n"
                f"保留 {total_violations} 条候选（按帧计），不能据此判定无违章，需人工核查。\n\n"
                f"结果已保存至：\n{output_dir}"
            )
        else:
            self.status_label.setText(
                f"逐帧分析完成：{len(self.analysis_results)} 帧，{total_violations} 条候选（按帧计，需人工核查）"
            )
            QMessageBox.information(
                self, "逐帧分析完成",
                f"共分析 {len(self.analysis_results)} 帧\n"
                f"输出 {total_violations} 条候选（按帧计，需人工核查）\n\n"
                f"结果已保存至：\n{output_dir}"
            )

    # ==================== 结果导出：证据包 / 安全违章通知书 ====================

    def _set_export_buttons_enabled(self, enabled):
        self.export_pkg_btn.setEnabled(enabled)
        self.notice_btn.setEnabled(enabled)

    def _start_export_job(self, func, status_text, done_slot):
        """结果导出的公共前置（校验输出目录/互斥）→ 起后台线程并接线回调。"""
        if not self.current_output_dir:
            QMessageBox.warning(self, "提示", "请先完成一次视频分析")
            return
        if self.export_thread and self.export_thread.isRunning():
            return
        self._set_export_buttons_enabled(False)
        self.status_label.setText(status_text)
        self.export_thread = ExportWorkerThread(
            func, self.current_output_dir, self.current_video_path)
        self.export_thread.done.connect(done_slot)
        self.export_thread.failed.connect(self._on_export_failed)
        self.export_thread.start()

    def _open_roi_editor(self):
        """生成并打开 ROI 框选页"""
        if not self.current_output_dir:
            return
        try:
            from core.roi_check import generate_editor
            path = generate_editor(self.current_output_dir)
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            QMessageBox.information(
                self, "ROI 框选",
                "已在浏览器打开框选页：\n画框 → 选类型 → 导出 rois.json\n\n"
                "请把导出的 rois.json 保存到：\n" + self.current_output_dir +
                "\n\n然后点「运行ROI检查」。")
        except Exception as e:
            QMessageBox.warning(self, "打开失败", str(e))

    def _run_roi_check(self):
        """后台执行 ROI 定向检查"""
        if not self.current_output_dir:
            return
        self.roi_run_btn.setEnabled(False)
        self.roi_run_btn.setText("检查中…")
        self._roi_thread = RoiCheckThread(self.current_output_dir, self)
        self._roi_thread.finished_ok.connect(self._on_roi_checked)
        self._roi_thread.failed.connect(self._on_roi_failed)
        self._roi_thread.start()

    def _on_roi_checked(self, clues):
        self.roi_run_btn.setEnabled(True)
        self.roi_run_btn.setText("ROI检查")
        if not clues:
            QMessageBox.information(self, "ROI 检查", "未形成三次一致的疑似线索；这不代表现场没有违章。")
            return
        try:
            from core.roi_check import generate_review_page
            page = generate_review_page(self.current_output_dir)
            QDesktopServices.openUrl(QUrl.fromLocalFile(page))
        except Exception:
            pass
        lines = "\n".join("· %s（%s）" % (c["title"], c["state"]) for c in clues)
        QMessageBox.information(self, "ROI 检查完成",
                                "发现 %d 条疑似线索（不计入违章数，需人工复核）：\n\n%s" % (len(clues), lines))

    def _on_roi_failed(self, msg):
        self.roi_run_btn.setEnabled(True)
        self.roi_run_btn.setText("ROI检查")
        QMessageBox.warning(self, "ROI 检查失败", msg)

    def _export_evidence_package(self):
        """导出违章证据包（后台线程）"""
        from core.evidence_exporter import export_evidence_package
        self._start_export_job(
            export_evidence_package, "正在导出证据包（裁剪视频片段）...",
            self._on_export_done)

    def _generate_rectification_notice(self):
        """生成电子安全违章通知书（Word，后台线程）"""
        from core.evidence_exporter import generate_rectification_notice
        self._start_export_job(
            generate_rectification_notice, "正在生成安全违章通知书...",
            self._on_notice_done)

    def _on_export_done(self, res):
        self._set_export_buttons_enabled(True)
        if isinstance(res, dict) and res.get("dir"):
            size_txt = f"（{res.get('size_mb', 0):.1f} MB）" if res.get("size_mb") else ""
            self.status_label.setText(f"证据包导出完成：{res['count']} 个事件{size_txt}")
            cut_txt = ""
            if res.get("truncated"):
                cut_txt = (f"\n其中 {res['truncated']} 条事件跨度较长，"
                           f"片段已按锚点截取为 {res.get('clip_max_sec', 60):.0f} 秒。")
            QMessageBox.information(
                self, "证据包导出完成",
                f"已导出 {res['count']} 个违章事件的证据包{size_txt}：\n\n"
                f"文件夹：{res['dir']}\n"
                f"压缩包：{res['zip']}\n"
                f"片段编码：{res.get('codec', '')}{cut_txt}\n\n"
                f"每个事件包含：视频片段、代表帧、坐标、判定依据。"
            )
            self._reveal_in_explorer(res["dir"])
        else:
            QMessageBox.warning(self, "导出异常", f"导出结果异常：{res}")

    def _on_notice_done(self, path):
        self._set_export_buttons_enabled(True)
        if isinstance(path, str) and path.endswith(".docx"):
            self.status_label.setText(f"文书已生成：{os.path.basename(path)}")
            QMessageBox.information(
                self, "文书已生成",
                f"已生成文书：\n{path}"
            )
            self._reveal_in_explorer(path)
        else:
            QMessageBox.warning(self, "导出异常", f"生成结果异常：{path}")

    def _on_export_failed(self, msg):
        self._set_export_buttons_enabled(True)
        self.status_label.setText("导出失败")
        QMessageBox.warning(
            self, "导出失败",
            f"未能完成导出：\n{msg[:400]}\n\n"
            f"常见原因：输出目录没有 report.json、没有违章事件，"
            f"或原始视频已被移动/删除。"
        )

    @staticmethod
    def _reveal_in_explorer(path):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _rebuild_results_from(self, results):
        """分析结束后用校准结果重建帧列表（只登记路径，画面按需读盘）"""
        self.analysis_results = []
        self.frame_list.clear()
        self.current_result_index = -1
        self._frame_image_cache.clear()
        first_violation_row = -1

        for pos, r in enumerate(results):
            ts = r.get("timestamp", 0)
            vc = r.get("violation_count", 0)
            self.analysis_results.append({
                "index": pos,
                # 视频帧号：与事件 frame_indices 同一口径，场景归属靠它对齐
                "frame_index": r.get("frame_index"),
                "timestamp": ts,
                "annotated_path": r.get("annotated_path"),
                "original_path": r.get("original_path"),
                "result": r.get("analysis", {}),
            })

            item = QListWidgetItem()
            item.setSizeHint(QSize(0, 38))
            self.frame_list.addItem(item)
            result = r.get("analysis") if isinstance(r.get("analysis"), dict) else {}
            self.frame_list.setItemWidget(item, FrameRow(pos + 1, ts, vc, frame_analysis_incomplete(result)))
            if vc > 0 and first_violation_row < 0:
                first_violation_row = pos

        if first_violation_row >= 0:
            self.frame_list.setCurrentRow(first_violation_row)

        total_violations = sum(r.get("violation_count", 0) for r in results)
        self.summary_label.setText(
            f"{len(results)} 帧 · {total_violations} 条候选（按帧计）"
        )

    def _on_error(self, error_msg):
        """发生错误"""
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)
        self.status_label.setText(f"错误：{error_msg}")
        QMessageBox.critical(self, "错误", f"分析失败：{error_msg}")

    # ==================== 自定义场景（人工圈选帧合并） ====================

    def _on_frame_selection_changed(self):
        """多选帧数 ≥2 时才允许合并为场景"""
        self.merge_scene_btn.setEnabled(len(self._selected_frame_rows()) >= 2)

    def _on_scene_selection_changed(self):
        n = len(self.scene_list.selectedItems())
        self.remove_scene_btn.setEnabled(n == 1)
        self.merge_scene_btn.setEnabled(n >= 2 or len(self._selected_frame_rows()) >= 2)

    def _selected_frame_rows(self):
        rows = sorted({self.frame_list.row(it) for it in self.frame_list.selectedItems()})
        return [r for r in rows if 0 <= r < len(self.analysis_results)]

    def _merge_selected_frames_to_scene(self):
        """优先合并选中的场景；否则把选中的多帧合并为自定义场景"""
        picked = [it.data(Qt.UserRole) for it in self.scene_list.selectedItems()]
        if len(picked) >= 2 and self.current_output_dir:
            from core.scene_merge import save_overrides
            ids = [d.get("segment_id") for d in picked if isinstance(d, dict)]
            name = next((d.get("name") for d in picked if isinstance(d, dict)), "")
            self.scene_overrides.append({"ids": ids, "name": name, "source": "manual"})
            save_overrides(self.current_output_dir, self.scene_overrides)
            self._refresh_scene_list()
            self.status_label.setText(f"已合并 {len(ids)} 个场景")
            return
        rows = self._selected_frame_rows()
        if len(rows) < 2:
            QMessageBox.information(
                self, "提示",
                "请先在左侧帧列表中多选至少 2 帧（Ctrl 点选 / Shift 连选），再点「合并为场景」")
            return
        if not self.current_output_dir:
            QMessageBox.warning(self, "提示", "请先完成一次视频分析")
            return

        entries = [self.analysis_results[r] for r in rows]
        # 场景帧号用视频帧号（与事件 frame_indices 同口径），归属性才对得上
        frame_indices = sorted({
            int(e.get("frame_index") if e.get("frame_index") is not None
                else e.get("index") or 0)
            for e in entries})
        ts_list = [float(e.get("timestamp") or 0) for e in entries]
        scene = {
            "name": "",
            "frame_indices": frame_indices,
            "time_span": [min(ts_list), max(ts_list)],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "manual",
        }
        self.custom_scenes = add_scene(self.current_output_dir, scene)
        scene_id = self.custom_scenes[-1].get("scene_id")

        # 取圈选帧画面给 AI 命名（等间隔采样由 ai_name_scene 内部处理）
        image_paths = []
        for e in entries:
            p = e.get("original_path") or e.get("annotated_path")
            if p and os.path.exists(p):
                image_paths.append(p)

        self.status_label.setText(
            f"已合并 {len(frame_indices)} 帧为场景，AI 正在命名…")
        self._refresh_scene_list(select_id=scene_id)

        self._scene_name_thread = SceneNameThread(image_paths)
        self._scene_name_thread.named.connect(
            lambda name, sid=scene_id: self._on_scene_named(sid, name))
        self._scene_name_thread.failed.connect(
            lambda _msg, sid=scene_id: self._on_scene_named(sid, ""))
        self._scene_name_thread.start()
        self._scene_name_thread.setPriority(QThread.Priority.LowPriority)

    def _on_scene_named(self, scene_id, name):
        """AI 命名结果回填（失败时 name 为空 → 用「自定义场景N」兜底）"""
        name = (name or "").strip()
        for s in self.custom_scenes:
            if str(s.get("scene_id")) == str(scene_id):
                if not name:
                    name = fallback_scene_name(len(self.custom_scenes) - 1)
                s["name"] = name
                break
        if self.current_output_dir:
            save_scenes(self.current_output_dir, self.custom_scenes)
        self._refresh_scene_list(select_id=scene_id)
        self.status_label.setText(f"场景「{name}」已创建，生成报告时相似违章将合并")

    def _remove_selected_scene(self):
        item = self.scene_list.currentItem()
        if not item or not self.current_output_dir:
            return
        data = item.data(Qt.UserRole)
        if isinstance(data, dict):
            QMessageBox.information(self, "提示", "自动场景不支持删除；可用「自动合并」重新分组。")
            return
        self.custom_scenes = remove_scene(
            self.current_output_dir, item.data(Qt.UserRole))
        self._refresh_scene_list()
        self.remove_scene_btn.setEnabled(False)
        self.status_label.setText("已移除该自定义场景")

    def _auto_merge_scenes(self):
        if not self.current_output_dir or not self.scene_segments:
            return
        from core.scene_merge import auto_groups, save_overrides
        self.scene_overrides = auto_groups(self.scene_segments, max_gap=30)
        save_overrides(self.current_output_dir, self.scene_overrides)
        self._refresh_scene_list()
        self.status_label.setText(f"自动合并完成：{len(self.scene_overrides)} 组")

    def _refresh_scene_list(self, select_id=None):
        """刷新场景列表：先显示自动场景（应用合并结果），再显示自定义场景"""
        self.scene_list.clear()
        try:
            from core.scene_merge import apply_overrides
            merged = apply_overrides(self.scene_segments, self.scene_overrides) if self.scene_segments else []
        except Exception:
            merged = self.scene_segments
        merged = sorted(merged, key=lambda x: float(x.get("start_time", 0)))
        for i, s in enumerate(merged, 1):
            item = QListWidgetItem(f"第{i}场景：{s.get('name') or '未分段'}  "
                                   f"{float(s.get('start_time', 0)):.0f}-{float(s.get('end_time', 0)):.0f}秒"
                                   + (f"  [合并{len(s.get('seg_ids') or [])}段]" if len(s.get("seg_ids") or []) > 1 else ""))
            item.setData(Qt.UserRole, {"kind": "auto", "segment_id": s.get("segment_id"),
                                       "name": s.get("name"), "seg_ids": s.get("seg_ids")})
            self.scene_list.addItem(item)
        for s in self.custom_scenes:
            name = (s.get("name") or "").strip() or "（AI命名中…）"
            n_frames = len(s.get("frame_indices") or [])
            item = QListWidgetItem(f"{name}（{n_frames} 帧）")
            item.setData(Qt.UserRole, s.get("scene_id"))
            self.scene_list.addItem(item)
            if select_id is not None and str(s.get("scene_id")) == str(select_id):
                self.scene_list.setCurrentItem(item)

    def _load_frame_image(self, entry):
        """按需从磁盘读入一帧画面（单帧缓存）。

        返回 (img_bgr, is_original)：优先原图（可交互画框），无原图则用标注图
        （框已烘焙在图上）。内存里只保留最近读过的那一张，避免逐帧驻留。
        """
        for key, is_original in (("original_path", True), ("annotated_path", False)):
            path = entry.get(key)
            if not path or not os.path.exists(path):
                continue
            cached = self._frame_image_cache.get(path)
            if cached is None:
                cached = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if cached is None:
                    continue
                # 只留一张：换帧即释放上一张，内存占用与帧数无关
                self._frame_image_cache.clear()
                self._frame_image_cache[path] = cached
            return cached, is_original
        return None, False

    def _on_frame_selected(self, row):
        """选中某帧（画面按需从磁盘读入）"""
        if row < 0 or row >= len(self.analysis_results):
            return

        self.current_result_index = row
        result_data = self.analysis_results[row]
        result = result_data["result"]

        display_img, is_original = self._load_frame_image(result_data)
        if display_img is None:
            self.image_view.scene.clear()
            self.image_view.show_placeholder("该帧无可用画面")
            self.detail_text.clear()
            return

        # 显示图片
        self.image_view.set_image(display_img)
        # 有原图时由交互组件画框；标注图模式下框已在图上，不重复画
        if is_original:
            self.image_view.set_violations(result.get("violations") or [])
        else:
            self.image_view.set_violations([])

        # 填充违章详情文字（标注本帧在视频中的时刻）
        self._update_detail_text(result, result_data.get("timestamp"))

    def _scene_banner(self, frame_ts):
        segs = [s for s in (self.scene_segments or []) if isinstance(s, dict)
                and normalize_timestamp(s.get("start_time")) is not None
                and normalize_timestamp(s.get("end_time")) is not None
                and normalize_timestamp(s.get("start_time")) <= normalize_timestamp(s.get("end_time"))]
        segs.sort(key=lambda s: normalize_timestamp(s.get("start_time")))
        ts = normalize_timestamp(frame_ts)
        if ts is None or not segs:
            return ""
        for i, s in enumerate(segs, 1):
            if float(s.get("start_time", 0)) <= ts <= float(s.get("end_time", 0)):
                break
        else:
            s = min(segs, key=lambda x: min(abs(ts - float(x.get("start_time", 0))),
                                            abs(ts - float(x.get("end_time", 0)))))
            i = segs.index(s) + 1
        n = len(s.get("seg_ids") or [])
        extra = f"（合并 {n} 段）" if n > 1 else ""
        return ("<div style='background:#eef4ff;border-left:4px solid #1a73e8;padding:6px 8px;"
                "margin:0 0 8px 0;border-radius:4px;font-size:12px;color:#1f2937'>"
                f"<b>第{i}场景：{esc(s.get('name') or '未命名场景')}</b>"
                f"　第 {float(s.get('start_time', 0)):.0f} 秒 – 第 {float(s.get('end_time', 0)):.0f} 秒{extra}</div>")

    def _update_detail_text(self, result, frame_ts=None):
        """更新违章详情（卡片式富文本，替代原来的纯文本拼接）"""
        raw_violations = (result or {}).get("violations") or []
        violations = [v for v in raw_violations if isinstance(v, dict)] if isinstance(raw_violations, (list, tuple)) else []
        html = render_violations_html(result, frame_ts=frame_ts)
        banner = self._scene_banner(frame_ts)
        self.detail_text.setHtml((banner + html) if banner else html)

        if frame_analysis_incomplete(result):
            self.detail_count_label.setText(f"分析不完整 · {len(violations)} 项候选" if violations else "分析失败")
            self.detail_count_label.setObjectName("dangerPill")
        elif violations:
            self.detail_count_label.setText(f"{len(violations)} 项候选")
            self.detail_count_label.setObjectName("dangerPill")
        else:
            self.detail_count_label.setText("未输出结构化候选")
            self.detail_count_label.setObjectName("statPill")
        # 运行时改了 objectName，必须 repolish 才会重新套用样式表
        repolish(self.detail_count_label)

    def resizeEvent(self, event):
        """窗口大小改变时重新缩放图片（由image_view自行处理）"""
        super().resizeEvent(event)

    # ==================== 知识库相关 ====================

    def _load_knowledge_list(self):
        """刷新知识库检索面板（添加/删除/分类后调用）"""
        explorer = getattr(self, "_knowledge_explorer", None)
        if explorer is not None:
            explorer.refresh()

    # ==================== 设置相关 ====================

    def _open_settings(self):
        """打开设置对话框"""
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.Accepted:
            # 设置保存后刷新知识库状态显示
            self._load_knowledge_list()

    # ==================== 关闭收尾 ====================

    def closeEvent(self, event):
        """关闭窗口收尾：先中断在跑的后台任务再退出。

        不加这个处理会踩两个坑：
        1) QThread 在运行中被析构 → Qt 报 "QThread: Destroyed while thread is
           still running"，可能直接崩溃；
        2) 帧分析线程池是非守护线程，解释器退出时会等在途 API 调用结束，
           界面看起来"卡死"（每次等待可达数十秒）。
        """
        # YOLO 预热线程属非关键任务：不弹"任务进行中"确认框，只短暂等它收尾，
        # 避免 QThread 运行中被析构的警告；超时则放行（进程退出会清理）
        warm = self.yolo_warmup_thread
        if warm is not None and getattr(warm, "isRunning", lambda: False)():
            warm.wait(1000)

        candidates = [
            ("视频分析", self.analysis_thread),
            ("知识库导入", self.knowledge_thread),
            ("知识库自动导入", self.auto_import_thread),
            ("知识库分类", self.classify_thread),
            ("结果导出", self.export_thread),
            ("场景命名", self._scene_name_thread),
            ("连接测试", getattr(self, "_conn_check_thread", None)),
        ]
        # 知识库检索面板自己持有一组线程（检索/导入/分类），与上面的不是同一批对象；
        # 漏掉它们同样会在关窗时触发 "QThread: Destroyed while thread is still running"
        _explorer = getattr(self, "_knowledge_explorer", None)
        if _explorer is not None and hasattr(_explorer, "active_threads"):
            try:
                candidates.extend(_explorer.active_threads())
            except Exception as e:
                print(f"[ui] 收集知识库面板线程失败（忽略）: {e}")
        running = [(name, t) for name, t in candidates
                   if t is not None and getattr(t, "isRunning", lambda: False)()]

        if running:
            names = "、".join(name for name, _ in running)
            reply = QMessageBox.question(
                self, "任务进行中",
                f"以下任务正在运行：{names}。" + chr(10) + "退出会中断它们，确定退出吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return

            # 先让分析引擎置停止位：主循环不再提交新帧，在途请求尽快返回
            try:
                get_analysis_engine().stop()
            except Exception as e:
                print(f"[ui] 停止分析引擎时出错（忽略）: {e}")

            for _, t in running:
                try:
                    if hasattr(t, "stop"):
                        t.stop()
                except Exception as e:
                    print(f"[ui] 请求线程停止时出错（忽略）: {e}")

            for name, t in running:
                try:
                    if not t.wait(3000):
                        print(f"[ui] {name}线程未在 3 秒内退出，直接退出程序")
                except Exception:
                    pass

        event.accept()
