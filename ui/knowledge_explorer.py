# -*- coding: utf-8 -*-
"""
知识库智能检索查看系统

左侧：知识库概览、文档列表、场景标签（点选即筛选）
顶部：语义/关键词智能检索 + 高级筛选（文档、场景、条数、相似度门槛）
右侧：结果列表 + 条款原文详情（关键词高亮、复制、导出）

检索走后台线程（向量检索需要调用 embedding 接口），浏览为本地内存操作。
"""
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QLineEdit, QListWidget, QListWidgetItem, QTextEdit, QSplitter,
    QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog,
    QMessageBox, QProgressDialog, QApplication, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, QSize, QTimer

from core.knowledge_base import get_knowledge_base
from ui.knowledge_tasks import (
    KnowledgeAddThread, KnowledgeClassifyThread, KnowledgeSearchThread,
)
from ui.render_html import render_chunk_detail_html, split_keywords, display_name
from ui.theme import INK_4, FONT_STACK, apply_card_shadow


class ChunkRow(QWidget):
    """检索/浏览结果行：来源 + 条款 + 相似度 + 场景 + 摘要"""

    def __init__(self, item, index):
        super().__init__()
        self.item = item
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 10)
        root.setSpacing(4)

        scenes = self.item.get("scenes") or []
        # 强制固定高度，绕开 QListWidget+setItemWidget 路径下 sizeHint 偶发不生效导致的「行被压底」
        self.setFixedHeight(88 if scenes else 60)

        # 第一行：序号 + 来源 + 相似度
        top = QHBoxLayout()
        top.setSpacing(6)

        name = display_name(item)
        article = item.get("article", "")
        page = item.get("page", "")

        src = f"{index}. 《{name}》"
        if article:
            src += f" {article}"
        elif page:
            src += f" · P{page}"

        title_label = QLabel(src)
        title_label.setObjectName("listTitle")
        title_label.setMinimumHeight(20)
        top.addWidget(title_label)

        top.addStretch()

        sim = item.get("similarity", 0) or 0
        if item.get("mode") != "browse" and sim:
            badge = QLabel(f"{float(sim):.3f}")
            badge.setObjectName("simBadge")
            badge.setFixedHeight(18)
            top.addWidget(badge)

        root.addLayout(top)

        # 第二行：摘要
        content = " ".join(str(item.get("content", "")).split())
        summary = content[:88] + ("…" if len(content) > 88 else "")
        summary_label = QLabel(summary if summary else "（无内容）")
        summary_label.setObjectName("listSummary")
        summary_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        summary_label.setMinimumHeight(20)
        summary_label.setWordWrap(True)
        root.addWidget(summary_label)

        # 第三行：场景标签
        scenes = item.get("scenes") or []
        if scenes:
            scene_text = " · ".join(str(s) for s in scenes[:4])
            scene_label = QLabel(scene_text)
            scene_label.setObjectName("listScene")
            scene_label.setMinimumHeight(16)
            scene_label.setWordWrap(True)
            root.addWidget(scene_label)

    def sizeHint(self):
        # 标题(18) + 摘要(16) + 场景(15) + 边距(14) + 间距(6) ≈ 69，留 buffer
        scenes = self.item.get("scenes") or []
        return QSize(0, 74 if scenes else 54)


class KnowledgeExplorer(QWidget):
    """知识库智能检索查看面板"""

    BROWSE_LIMIT = 200  # 浏览模式单次取块上限

    def __init__(self, parent=None):
        super().__init__(parent)
        self.kb = get_knowledge_base()

        self._mode = "browse"        # browse / search
        self._items = []             # 当前页展示的条目
        self._all_results = []       # 检索模式下的全部结果
        self._browse_total = 0
        self._page = 0
        self._current = None
        self._search_thread = None
        self._add_thread = None
        self._classify_thread = None

        self._build_ui()
        self.refresh()

    def active_threads(self):
        """本面板正在运行的后台线程 → [(名称, QThread)]

        供主窗口 closeEvent 统一中断：面板**自己持有**这些 QThread，主窗口的
        knowledge_thread/classify_thread 是另一组对象。关窗时若不带上它们，
        运行中的线程会被析构 → "QThread: Destroyed while thread is still running"。
        """
        pairs = (
            ("知识库检索", self._search_thread),
            ("知识库导入(面板)", self._add_thread),
            ("知识库分类(面板)", self._classify_thread),
        )
        out = []
        for name, t in pairs:
            try:
                if t is not None and t.isRunning():
                    out.append((name, t))
            except RuntimeError:
                # 底层 C++ 对象已释放（PySide 包装层还在）→ 视为未运行
                continue
        return out
        # 首次进入默认浏览全部条款，让界面一打开就有内容
        QTimer.singleShot(0, lambda: self._browse(reset_page=True))

    # ==================== 界面搭建 ====================

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        # 先建筛选面板：检索栏的「筛选」按钮要绑定它
        self.filter_panel = self._build_filters()
        root.addWidget(self._build_search_bar())
        root.addWidget(self.filter_panel)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_result_area())
        splitter.setSizes([330, 1010])
        root.addWidget(splitter, 1)

    def _build_search_bar(self):
        box = QGroupBox("智能检索")
        layout = QVBoxLayout(box)
        layout.setSpacing(10)

        row = QHBoxLayout()
        row.setSpacing(10)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(
            "输入问题或关键词，例如：基坑临边防护有哪些要求？（回车检索）")
        self.search_edit.setMinimumHeight(36)
        self.search_edit.returnPressed.connect(self.run_search)
        row.addWidget(self.search_edit, 1)

        self.search_btn = QPushButton("检索")
        self.search_btn.setCursor(Qt.PointingHandCursor)
        self.search_btn.setMinimumHeight(36)
        self.search_btn.clicked.connect(self.run_search)
        row.addWidget(self.search_btn)

        self.advanced_btn = QPushButton("筛选")
        self.advanced_btn.setObjectName("chipBtn")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.setCursor(Qt.PointingHandCursor)
        self.advanced_btn.setMinimumHeight(36)
        self.advanced_btn.toggled.connect(self.filter_panel.setVisible)
        row.addWidget(self.advanced_btn)

        layout.addLayout(row)

        status_row = QHBoxLayout()
        self.status_label = QLabel("输入关键词开始检索，或在左侧点击文档浏览全部条款")
        self.status_label.setObjectName("metaLabel")
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        layout.addLayout(status_row)

        apply_card_shadow(box)
        return box

    def _build_filters(self):
        self.filter_panel = QWidget()
        layout = QVBoxLayout(self.filter_panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # 文档筛选
        doc_row = QHBoxLayout()
        doc_row.setSpacing(8)
        doc_title = QLabel("文档")
        doc_title.setObjectName("metaLabel")
        doc_title.setFixedWidth(56)
        doc_row.addWidget(doc_title)
        self.doc_combo = QComboBox()
        self.doc_combo.setMinimumHeight(32)
        self.doc_combo.currentIndexChanged.connect(lambda _: self._on_filter_changed())
        doc_row.addWidget(self.doc_combo, 1)

        doc_row.addSpacing(16)

        page_title = QLabel("每页")
        page_title.setObjectName("metaLabel")
        doc_row.addWidget(page_title)
        self.page_spin = QSpinBox()
        self.page_spin.setRange(5, 100)
        self.page_spin.setSingleStep(5)
        self.page_spin.setValue(20)
        self.page_spin.setFixedWidth(70)
        self.page_spin.setMinimumHeight(32)
        self.page_spin.valueChanged.connect(lambda _: self._on_filter_changed())
        doc_row.addWidget(self.page_spin)

        doc_row.addSpacing(16)

        score_title = QLabel("相似度≥")
        score_title.setObjectName("metaLabel")
        doc_row.addWidget(score_title)
        self.score_spin = QDoubleSpinBox()
        self.score_spin.setRange(0.0, 1.0)
        self.score_spin.setSingleStep(0.05)
        self.score_spin.setValue(0.0)
        self.score_spin.setDecimals(2)
        self.score_spin.setFixedWidth(80)
        self.score_spin.setMinimumHeight(32)
        doc_row.addWidget(self.score_spin)

        doc_row.addStretch()
        layout.addLayout(doc_row)

        # 场景筛选（标签式多选）
        scene_row = QHBoxLayout()
        scene_row.setSpacing(8)
        scene_title = QLabel("场景")
        scene_title.setObjectName("metaLabel")
        scene_title.setFixedWidth(56)
        scene_title.setAlignment(Qt.AlignTop)
        scene_row.addWidget(scene_title)

        self.scene_grid = QWidget()
        self.scene_grid_layout = QGridLayout(self.scene_grid)
        self.scene_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.scene_grid_layout.setSpacing(6)
        scene_row.addWidget(self.scene_grid, 1)
        layout.addLayout(scene_row)

        sep = QFrame()
        sep.setObjectName("hSeparator")
        sep.setFixedHeight(1)
        layout.addWidget(sep)

        self.filter_panel.setVisible(False)
        return self.filter_panel

    def _build_sidebar(self):
        box = QGroupBox("知识库")
        layout = QVBoxLayout(box)
        layout.setSpacing(10)

        # 概览
        self.stats_label = QLabel("—")
        self.stats_label.setObjectName("statPill")
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)

        doc_title = QLabel("文档（点击查看全部条款）")
        doc_title.setObjectName("sectionLabel")
        layout.addWidget(doc_title)

        self.doc_list = QListWidget()
        self.doc_list.setSpacing(2)
        self.doc_list.currentTextChanged.connect(self._on_doc_clicked)
        layout.addWidget(self.doc_list, 2)

        scene_title = QLabel("场景分类（点击筛选）")
        scene_title.setObjectName("sectionLabel")
        layout.addWidget(scene_title)

        self.scene_list = QListWidget()
        self.scene_list.setSpacing(2)
        self.scene_list.currentTextChanged.connect(self._on_scene_clicked)
        layout.addWidget(self.scene_list, 2)

        # 管理按钮
        btn_grid = QGridLayout()
        btn_grid.setSpacing(8)

        add_btn = QPushButton("添加PDF")
        add_btn.setObjectName("ghostBtn")
        add_btn.setCursor(Qt.PointingHandCursor)
        add_btn.clicked.connect(self.add_pdf)
        btn_grid.addWidget(add_btn, 0, 0)

        classify_btn = QPushButton("AI场景分类")
        classify_btn.setObjectName("ghostBtn")
        classify_btn.setCursor(Qt.PointingHandCursor)
        classify_btn.clicked.connect(self.classify_scenes)
        btn_grid.addWidget(classify_btn, 0, 1)

        browse_all_btn = QPushButton("浏览全部")
        browse_all_btn.setObjectName("ghostBtn")
        browse_all_btn.setCursor(Qt.PointingHandCursor)
        browse_all_btn.clicked.connect(lambda: self._browse(reset_page=True))
        btn_grid.addWidget(browse_all_btn, 1, 0)

        remove_btn = QPushButton("删除文档")
        remove_btn.setObjectName("dangerBtn")
        remove_btn.setCursor(Qt.PointingHandCursor)
        remove_btn.clicked.connect(self.remove_document)
        btn_grid.addWidget(remove_btn, 1, 1)

        layout.addLayout(btn_grid)
        return box

    def _build_result_area(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # 标题行 + 分页
        header = QHBoxLayout()
        header.setSpacing(8)
        self.result_title = QLabel("全部条款")
        self.result_title.setObjectName("sectionLabel")
        header.addWidget(self.result_title)
        header.addStretch()

        self.prev_btn = QPushButton("上一页")
        self.prev_btn.setObjectName("ghostBtn")
        self.prev_btn.setCursor(Qt.PointingHandCursor)
        self.prev_btn.clicked.connect(lambda: self._goto_page(self._page - 1))
        header.addWidget(self.prev_btn)

        self.page_label = QLabel("第 1 页")
        self.page_label.setObjectName("statPill")
        header.addWidget(self.page_label)

        self.next_btn = QPushButton("下一页")
        self.next_btn.setObjectName("ghostBtn")
        self.next_btn.setCursor(Qt.PointingHandCursor)
        self.next_btn.clicked.connect(lambda: self._goto_page(self._page + 1))
        header.addWidget(self.next_btn)

        layout.addLayout(header)

        body = QSplitter(Qt.Vertical)
        body.setChildrenCollapsible(False)

        self.result_list = QListWidget()
        self.result_list.setSpacing(4)
        # 关闭统一行高，否则每条 item 的 setSizeHint 失效，摘要/场景被压底
        self.result_list.setUniformItemSizes(False)
        self.result_list.currentRowChanged.connect(self._on_result_selected)
        body.addWidget(self.result_list)

        detail_box = QWidget()
        detail_layout = QVBoxLayout(detail_box)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(8)

        detail_header = QHBoxLayout()
        detail_title = QLabel("条款原文")
        detail_title.setObjectName("sectionLabel")
        detail_header.addWidget(detail_title)
        detail_header.addStretch()

        copy_btn = QPushButton("复制原文")
        copy_btn.setObjectName("ghostBtn")
        copy_btn.setCursor(Qt.PointingHandCursor)
        copy_btn.clicked.connect(self.copy_current)
        detail_header.addWidget(copy_btn)

        export_btn = QPushButton("导出结果")
        export_btn.setObjectName("ghostBtn")
        export_btn.setCursor(Qt.PointingHandCursor)
        export_btn.clicked.connect(self.export_results)
        detail_header.addWidget(export_btn)

        detail_layout.addLayout(detail_header)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setHtml(self._empty_detail_html())
        detail_layout.addWidget(self.detail_text, 1)

        body.addWidget(detail_box)
        body.setSizes([420, 300])
        layout.addWidget(body, 1)

        return widget

    # ==================== 数据刷新 ====================

    def refresh(self):
        """重新载入文档、场景、统计（添加/删除/分类后调用）"""
        self._reload_docs()
        self._reload_scenes()
        self._reload_stats()

    def _reload_docs(self):
        self.doc_list.blockSignals(True)
        self.doc_combo.blockSignals(True)
        current = self.doc_combo.currentData()
        self.doc_list.clear()
        self.doc_combo.clear()
        self.doc_combo.addItem("全部文档", None)
        for title in sorted(self.kb.list_documents()):
            info = self.kb.get_document_info(title) or {}
            # 显式设行高，避免 Qt sizeHint 不按 \n 换行把第二行条款数压掉
            li = QListWidgetItem(f"{title}\n{info.get('chunks', 0)} 个条款块")
            li.setSizeHint(QSize(0, 46))
            self.doc_list.addItem(li)
            self.doc_combo.addItem(title, title)
        idx = self.doc_combo.findData(current)
        self.doc_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.doc_list.blockSignals(False)
        self.doc_combo.blockSignals(False)

    def _reload_scenes(self):
        scenes = self.kb.all_scenes()

        # 侧栏场景列表
        self.scene_list.blockSignals(True)
        self.scene_list.clear()
        for name, count in scenes.items():
            li = QListWidgetItem(f"{name}  ({count})")
            li.setSizeHint(QSize(0, 32))
            self.scene_list.addItem(li)
        self.scene_list.blockSignals(False)

        # 顶部筛选区的可切换标签
        while self.scene_grid_layout.count():
            item = self.scene_grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._scene_chips = {}
        cols = 4
        for i, (name, count) in enumerate(scenes.items()):
            chip = QPushButton(f"{name} {count}")
            chip.setObjectName("chipBtn")
            chip.setCheckable(True)
            chip.setCursor(Qt.PointingHandCursor)
            chip.toggled.connect(lambda _c, n=name: self._on_filter_changed())
            self.scene_grid_layout.addWidget(chip, i // cols, i % cols)
            self._scene_chips[name] = chip

    def _reload_stats(self):
        st = self.kb.stats()
        cover = 0
        if st["chunks"]:
            cover = int(st["vector_chunks"] / st["chunks"] * 100)
        mode_text = "语义检索" if st["mode"] == "vector" else "关键词检索"
        self.stats_label.setText(
            f"{st['documents']} 份规程 · {st['chunks']} 个条款块\n"
            f"{st['scenes']} 类场景 · 向量覆盖 {cover}%（{mode_text}）"
        )

    # ==================== 检索与浏览 ====================

    def run_search(self):
        """执行检索（空查询则回到浏览模式）"""
        query = self.search_edit.text().strip()
        if not query:
            self._browse(reset_page=True)
            return

        scenes = self._checked_scenes()
        titles = self._selected_titles()
        top_k = max(20, self.page_spin.value() * 5)
        min_score = float(self.score_spin.value())

        self.search_btn.setEnabled(False)
        self.status_label.setText("检索中…")
        self._t0 = time.time()

        self._search_thread = KnowledgeSearchThread(
            query, top_k=top_k, scenes=scenes, titles=titles, min_score=min_score)
        self._search_thread.finished_search.connect(self._on_search_done)
        self._search_thread.finished_error.connect(self._on_search_error)
        self._search_thread.start()

    def _on_search_done(self, results):
        self.search_btn.setEnabled(True)
        cost = time.time() - getattr(self, "_t0", time.time())
        self._mode = "search"
        self._all_results = results
        self._page = 0
        self.result_title.setText(f"检索结果：{self.search_edit.text().strip()}")
        mode_text = "语义检索" if results and results[0].get("mode") == "vector" else "关键词检索"
        self.status_label.setText(
            f"命中 {len(results)} 条 · {mode_text} · 耗时 {cost:.1f}s")
        self._render_page()

    def _on_search_error(self, msg):
        self.search_btn.setEnabled(True)
        self.status_label.setText(f"检索失败：{msg}")
        QMessageBox.critical(self, "检索失败", f"知识库检索出错：\n{msg}")

    def _browse(self, reset_page=True):
        """浏览模式：按文档/场景列出条款块"""
        if reset_page:
            self._page = 0
        self._mode = "browse"
        limit = self.page_spin.value()
        title = self.doc_combo.currentData()
        scenes = self._checked_scenes()
        scene = scenes[0] if scenes else None

        res = self.kb.list_chunks(title=title, scene=scene,
                                  offset=self._page * limit, limit=limit)
        self._items = res["items"]
        self._browse_total = res["total"]

        name = title or "全部文档"
        if scene:
            name += f" · {scene}"
        self.result_title.setText(f"条款浏览：{name}")
        self.status_label.setText(
            f"共 {res['total']} 个条款块 · 第 {self._page + 1} 页")
        self._paint_items()

    def _render_page(self):
        """检索模式下按页切片展示"""
        limit = self.page_spin.value()
        start = self._page * limit
        self._items = self._all_results[start:start + limit]
        self._paint_items()

    def _goto_page(self, page):
        if page < 0:
            return
        total_pages = self._total_pages()
        if total_pages and page >= total_pages:
            return
        self._page = page
        if self._mode == "browse":
            self._browse(reset_page=False)
        else:
            self.status_label.setText(
                f"命中 {len(self._all_results)} 条 · 第 {self._page + 1}/{total_pages} 页")
            self._render_page()

    def _total_pages(self):
        limit = max(1, self.page_spin.value())
        total = self._browse_total if self._mode == "browse" else len(self._all_results)
        return max(1, (total + limit - 1) // limit)

    def _paint_items(self):
        """把当前页条目渲染到结果列表"""
        self.result_list.blockSignals(True)
        self.result_list.clear()
        keywords = split_keywords(self.search_edit.text()) if self._mode == "search" else []

        base = self._page * self.page_spin.value()
        for i, item in enumerate(self._items):
            row_widget = ChunkRow(item, base + i + 1)
            # 把 item 高度同步成 widget 实际高度，绕开 setItemWidget 下 sizeHint 偶发被吞
            row_height = row_widget.sizeHint().height()
            target_h = max(row_height, 88 if (item.get("scenes") or []) else 60)
            row_item = QListWidgetItem()
            row_item.setSizeHint(QSize(0, target_h))
            self.result_list.addItem(row_item)
            self.result_list.setItemWidget(row_item, row_widget)

        self.result_list.blockSignals(False)

        total_pages = self._total_pages()
        self.page_label.setText(f"第 {self._page + 1}/{total_pages} 页")
        self.prev_btn.setEnabled(self._page > 0)
        self.next_btn.setEnabled(self._page + 1 < total_pages)

        if self._items:
            self.result_list.setCurrentRow(0)
            self._on_result_selected(0)
        else:
            self._current = None
            self.detail_text.setHtml(
                "<div style='font-family:%s; font-size:13px; color:%s; padding:16px 0;'>"
                "没有匹配的条款，换个关键词或调整筛选条件</div>"
                % (FONT_STACK, INK_4))
        self._keywords = keywords

    def _on_result_selected(self, row):
        if row < 0 or row >= len(self._items):
            return
        item = self._items[row]
        self._current = item
        keywords = getattr(self, "_keywords", None)
        if self._mode == "search":
            keywords = keywords or split_keywords(self.search_edit.text())
        self.detail_text.setHtml(render_chunk_detail_html(item, keywords))

    # ==================== 交互事件 ====================

    def _checked_scenes(self):
        return [name for name, chip in getattr(self, "_scene_chips", {}).items()
                if chip.isChecked()]

    def _selected_titles(self):
        title = self.doc_combo.currentData()
        return [title] if title else None

    def _on_filter_changed(self):
        """筛选条件变化：检索态重新检索，浏览态重新取数"""
        if self._mode == "search" and self.search_edit.text().strip():
            self.run_search()
        else:
            self._browse(reset_page=True)

    def _on_doc_clicked(self, text):
        if not text:
            return
        title = text.split("\n")[0]
        idx = self.doc_combo.findData(title)
        if idx >= 0:
            self.doc_combo.setCurrentIndex(idx)
            self._browse(reset_page=True)

    def _on_scene_clicked(self, text):
        if not text:
            return
        name = text.rsplit("  (", 1)[0]
        chip = getattr(self, "_scene_chips", {}).get(name)
        if chip is not None:
            chip.setChecked(not chip.isChecked())

    # ==================== 管理操作 ====================

    def add_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择PDF文件", "", "PDF文件 (*.pdf);;所有文件 (*.*)")
        if not file_path:
            return

        progress = QProgressDialog("正在准备…", "取消", 0, 100, self)
        progress.setWindowTitle("添加知识库")
        progress.setMinimumDuration(0)
        progress.setModal(True)

        def on_ok(result):
            progress.close()
            self.refresh()
            mode_text = "语义检索" if result.get("mode") == "embedding" else "关键词检索"
            msg = (f"PDF添加成功！\n\n文档块数：{result.get('count', 0)}\n"
                   f"检索模式：{mode_text}")
            if result.get("ocr_pages", 0):
                msg += f"\nOCR识别：{result['ocr_pages']}页（扫描件已识别）"
            if result.get("embedding_failed", 0) > 0:
                msg += (f"\n\n注意：{result['embedding_failed']}个块生成向量失败，"
                        f"已降级为关键词检索。")
            QMessageBox.information(self, "成功", msg)
            self._browse(reset_page=True)

        def on_error(error):
            progress.close()
            QMessageBox.critical(self, "错误", f"添加PDF失败：\n{error}")

        self._add_thread = KnowledgeAddThread(file_path)
        self._add_thread.progress_updated.connect(
            lambda p, s: (progress.setValue(p), progress.setLabelText(s)))
        self._add_thread.finished_ok.connect(on_ok)
        self._add_thread.finished_error.connect(on_error)
        self._add_thread.start()
        progress.exec()

    def classify_scenes(self):
        if self.kb.count() == 0:
            QMessageBox.information(self, "提示", "知识库为空，请先添加PDF文件")
            return

        progress = QProgressDialog("AI场景分类中…", "取消", 0, 100, self)
        progress.setWindowTitle("AI场景分类")
        progress.setMinimumDuration(0)
        progress.setModal(True)

        def on_ok(n):
            progress.close()
            self.refresh()
            if n == 0:
                QMessageBox.information(
                    self, "完成",
                    "所有文档块此前已完成AI场景分类，本次没有新增块需要处理。")
            else:
                QMessageBox.information(
                    self, "完成",
                    f"已对 {n} 个文档块完成AI场景分类。\n\n"
                    f"现在可以按场景筛选条款，分析视频时也会优先检索对应场景的法规。")

        def on_error(error):
            progress.close()
            QMessageBox.critical(self, "错误", f"AI场景分类失败：\n{error}")

        self._classify_thread = KnowledgeClassifyThread()
        self._classify_thread.progress_updated.connect(
            lambda p, s: (progress.setValue(p), progress.setLabelText(s)))
        self._classify_thread.finished_classify.connect(on_ok)
        self._classify_thread.finished_error.connect(on_error)
        self._classify_thread.start()
        progress.exec()

    def remove_document(self):
        current = self.doc_list.currentItem()
        if not current:
            QMessageBox.information(self, "提示", "请先在左侧选择要删除的文档")
            return
        title = current.text().split("\n")[0]

        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除文档「{title}」吗？\n\n"
            f"将同时删除「知识库」文件夹中对应的PDF源文件，该操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            count = self.kb.remove_document(title)
            self.refresh()
            self._browse(reset_page=True)
            self.status_label.setText(f"已删除 {count} 个文档块")

    # ==================== 结果操作 ====================

    def copy_current(self):
        if not self._current:
            return
        QApplication.clipboard().setText(str(self._current.get("content", "")))
        self.status_label.setText("条款原文已复制到剪贴板")

    def export_results(self):
        if not self._items:
            QMessageBox.information(self, "提示", "当前没有可导出的结果")
            return

        default_name = "知识库检索结果.md"
        if self._mode == "search":
            default_name = f"检索_{self.search_edit.text().strip()[:20]}.md"

        path, _ = QFileDialog.getSaveFileName(
            self, "导出检索结果", default_name, "Markdown 文件 (*.md);;文本文件 (*.txt)")
        if not path:
            return

        lines = ["# 知识库检索结果", ""]
        if self._mode == "search":
            lines.append(f"- 检索式：{self.search_edit.text().strip()}")
        scenes = self._checked_scenes()
        if scenes:
            lines.append(f"- 场景筛选：{'、'.join(scenes)}")
        if self.doc_combo.currentData():
            lines.append(f"- 文档筛选：{self.doc_combo.currentData()}")
        lines.append(f"- 导出时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- 条目数：{len(self._items)}")
        lines.append("")

        for i, item in enumerate(self._items, 1):
            name = item.get("official_title") or item.get("title", "")
            name = str(name)
            if name.lower().endswith(".pdf"):
                name = name[:-4]
            src = f"《{name}》"
            if item.get("article"):
                src += f" {item['article']}"
            if item.get("page"):
                src += f" · 第{item['page']}页"
            lines.append(f"## {i}. {src}")
            if item.get("scenes"):
                lines.append(f"场景：{'、'.join(item['scenes'])}")
            if item.get("similarity") and item.get("mode") != "browse":
                lines.append(f"相关度：{float(item['similarity']):.3f}")
            lines.append("")
            lines.append(str(item.get("content", "")).strip())
            lines.append("")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self.status_label.setText(f"已导出 {len(self._items)} 条到：{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"写入文件失败：\n{e}")

    @staticmethod
    def _empty_detail_html():
        return (
            "<div style='font-family:%s; font-size:13px; color:%s; "
            "padding:18px 4px;'>在上方结果列表中点击任意条款，这里会显示完整原文"
            "并高亮命中的关键词</div>" % (FONT_STACK, INK_4)
        )
