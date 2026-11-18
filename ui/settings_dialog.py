# -*- coding: utf-8 -*-
"""设置对话框（API 通道 / 参数配置），从 main_window.py 拆出"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QLineEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QPushButton, QCheckBox,
    QFormLayout, QDialogButtonBox, QApplication,
)
from PySide6.QtCore import Qt

from core.config import get_config
from core.api_client import get_ai_client
from core.api_errors import safe_error_info
from ui.theme import WARNING_TEXT, SUCCESS_TEXT, DANGER_TEXT


class SettingsDialog(QDialog):
    """设置对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("系统设置")
        self.setMinimumWidth(520)
        self.config = get_config()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 18)
        layout.setSpacing(14)
        form = QFormLayout()
        form.setSpacing(10)

        self.api_key_edit = QLineEdit(self.config.get("api_key", ""))
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        form.addRow("API密钥：", self.api_key_edit)

        # API地址：下拉可编辑，内置常见中转站
        self.base_url_combo = QComboBox()
        self.base_url_combo.setEditable(True)
        base_urls = [
            "https://api.duckcoding.ai/v1",
            "https://api.openai.com/v1",
            "https://api.chatanywhere.tech/v1",
            "https://api.chatanywhere.com.cn/v1",
            "https://api.gptsapi.net/v1",
            "https://api.api2d.net/v1",
            "https://api.ohmygpt.com/v1",
        ]
        current_url = self.config.get("base_url", "https://api.openai.com/v1")
        for u in base_urls:
            self.base_url_combo.addItem(u)
        if current_url not in base_urls:
            self.base_url_combo.addItem(current_url)
        self.base_url_combo.setCurrentText(current_url)
        form.addRow("API地址：", self.base_url_combo)
        
        url_hint = QLabel("提示：中转站密钥必须填中转站自己的地址（一般以 /v1 结尾），\n用官方OpenAI地址会报401密钥错误")
        url_hint.setObjectName("hintLabel")
        form.addRow("", url_hint)

        self.vision_model_combo = QComboBox()
        self.vision_model_combo.setEditable(True)
        default_models = [
            "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-high", "gpt-5.4-low",
            "gpt-5.5", "gpt-5.5-medium",
            "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
            "claude-3-5-sonnet-20241022",
            "qwen-vl-plus", "glm-4v",
        ]
        current_model = self.config.get("vision_model", "gpt-5.4")
        for m in default_models:
            self.vision_model_combo.addItem(m)
        if current_model not in default_models:
            self.vision_model_combo.addItem(current_model)
        self.vision_model_combo.setCurrentText(current_model)
        form.addRow("视觉模型：", self.vision_model_combo)

        model_hint = QLabel("提示：mini类模型速度快但目标定位精度明显偏低（标注框容易偏移），\n正式分析建议用 gpt-5.4 等完整模型")
        model_hint.setObjectName("hintLabel")
        form.addRow("", model_hint)

        self.embedding_model_edit = QLineEdit(self.config.get("embedding_model", ""))
        self.embedding_model_edit.setPlaceholderText("留空则自动使用关键词检索（DuckCoding无embedding模型，留空即可）")
        form.addRow("向量模型：", self.embedding_model_edit)

        # 视频分析设置
        self.frame_interval_spin = QDoubleSpinBox()
        self.frame_interval_spin.setRange(0.5, 60.0)
        self.frame_interval_spin.setSingleStep(0.5)
        self.frame_interval_spin.setValue(self.config.get("frame_interval", 5.0))
        self.frame_interval_spin.setSuffix(" 秒")
        form.addRow("抽帧间隔：", self.frame_interval_spin)

        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(1, 500)
        self.max_frames_spin.setValue(self.config.get("max_frames", 100))
        form.addRow("最大帧数：", self.max_frames_spin)

        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(640, 2560)
        self.img_size_spin.setSingleStep(64)
        self.img_size_spin.setSuffix(" px")
        self.img_size_spin.setValue(self.config.get("max_image_size", 2048))
        form.addRow("识别分辨率上限：", self.img_size_spin)

        size_hint = QLabel("大图超过此值会压缩发送；小于1920的帧会自动放大发送以提升小目标定位精度")
        size_hint.setObjectName("hintLabel")
        form.addRow("", size_hint)

        # 场景优化模式
        self.scene_optimized_check = QCheckBox("启用场景优化（节省API，自动过滤高空远景）")
        self.scene_optimized_check.setChecked(bool(self.config.get("scene_optimized", False)))
        form.addRow("", self.scene_optimized_check)

        scene_hint = QLabel("开启后先用大模型判断场景，只分析代表帧，可节省60-80% API调用")
        scene_hint.setObjectName("hintLabel")
        form.addRow("", scene_hint)

        layout.addLayout(form)

        # 测试连接按钮
        test_btn = QPushButton("测试连接")
        test_btn.setObjectName("ghostBtn")
        test_btn.setCursor(Qt.PointingHandCursor)
        test_btn.clicked.connect(self._test_connection)
        layout.addWidget(test_btn)

        self.test_result_label = QLabel("")
        self.test_result_label.setWordWrap(True)
        self.test_result_label.setMinimumHeight(40)
        layout.addWidget(self.test_result_label)

        # 确定/取消按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("保存")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.button(QDialogButtonBox.Cancel).setObjectName("ghostBtn")
        buttons.accepted.connect(self._save_and_close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _test_connection(self):
        """测试API连接"""
        self.test_result_label.setText(
            f"<span style='color: {WARNING_TEXT};'>正在测试连接…</span>")
        QApplication.processEvents()
        
        try:
            api_key = self.api_key_edit.text().strip()
            base_url = self.base_url_combo.currentText().strip()
            
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=10)
            models = client.models.list()
            model_list = [m.id for m in models.data]
            
            # 找视觉相关模型
            vision_models = [m for m in model_list if any(
                k in m.lower() for k in ['vision', 'gpt-4o', 'gpt-4-turbo', 'qwen-vl', 
                                          'glm-4v', 'claude-3', 'moonshot', 'gemini']
            )]
            
            display_models = vision_models[:8] if vision_models else model_list[:8]
            
            self.test_result_label.setText(
                f"<span style='color: {SUCCESS_TEXT}; font-weight: bold;'>连接成功！</span><br>"
                f"可用模型总数：{len(model_list)}<br>"
                f"视觉模型：{', '.join(display_models)}"
            )
        except Exception as e:
            self.test_result_label.setText(
                f"<span style='color: {DANGER_TEXT}; font-weight: bold;'>连接失败</span><br>"
                f"错误信息：{safe_error_info(e)}"
            )

    def _save_and_close(self):
        """保存设置并关闭"""
        saved = self.config.update({
            "api_key": self.api_key_edit.text().strip(),
            "base_url": self.base_url_combo.currentText().strip(),
            "vision_model": self.vision_model_combo.currentText().strip(),
            "embedding_model": self.embedding_model_edit.text().strip(),
            "frame_interval": self.frame_interval_spin.value(),
            "max_frames": self.max_frames_spin.value(),
            "max_image_size": self.img_size_spin.value(),
            "scene_optimized": self.scene_optimized_check.isChecked(),
        })
        if saved is False:
            self.test_result_label.setText(
                f"<span style='color: {DANGER_TEXT};'>配置未保存，请先修复原配置或检查保存目录。</span>")
            return
        try:
            get_ai_client().update_config()
        except Exception as exc:
            self.test_result_label.setText(
                f"<span style='color: {DANGER_TEXT};'>配置已保存，但客户端更新失败：{safe_error_info(exc)}</span>")
            return
        self.accept()


# ==================== 主窗口 ====================
