# -*- coding: utf-8 -*-
"""
界面设计系统（Design System）

把颜色、圆角、间距、组件样式集中到一处，main.py 加载本模块的样式表，
界面代码只使用语义化的 objectName（primaryBtn / cardTitle 等），
不再到处写内联样式，改版只需改这一个文件。

视觉语言参考 Apple 的人机界面规范（Apple Design）：
- 颜色：Apple 系统色板（systemBlue/Red/Orange/Green + label / background 分层）
- 材质：凹陷画布（systemGroupedBackground）上浮起白色卡片（material），
  用描边 + 投影表达层级，而不是靠重边框
- 排版：默认平台系统字体，用「字号 + 字重」建立层级；大字号收紧字距
- 控件：统一 8px 圆角、克制的按下反馈、清晰的聚焦态

Qt Widgets 不支持 CSS 的 blur / box-shadow / transition，因此：
- 「毛玻璃」用半透明色近似（顶栏 material）
- 「投影」用 QGraphicsDropShadowEffect 实现（见 apply_card_shadow）
- 「按下反馈」用 :pressed 的色阶变化表达
"""

# ============================ 设计 Tokens ============================
# ---- 强调色（Apple systemBlue）----
BRAND = "#0071e3"
BRAND_HOVER = "#0077ed"
BRAND_PRESS = "#0059b3"
BRAND_SOFT = "#e8f1fd"

# ---- 语义色：无后缀为填充/图形色；_TEXT 为浅底上的文字色（保证可读性）----
DANGER = "#ff3b30"
DANGER_TEXT = "#d70015"
DANGER_SOFT = "#ffeceb"
WARNING = "#ff9500"
WARNING_TEXT = "#b25000"
WARNING_SOFT = "#fff4e5"
SUCCESS = "#34c759"
SUCCESS_TEXT = "#1a7f37"
SUCCESS_SOFT = "#eaf9ef"

# ---- 文字层级（Apple label hierarchy）----
INK = "#1d1d1f"       # label
INK_2 = "#3a3a3c"     # 正文
INK_3 = "#6e6e73"     # secondaryLabel
INK_4 = "#aeaeb2"     # tertiaryLabel
INK_5 = "#c7c7cc"     # quaternaryLabel

# ---- 背景层级（Apple background hierarchy）----
CANVAS = "#f2f2f7"    # systemGroupedBackground：凹陷画布
SURFACE = "#ffffff"   # 卡片：抬起
SURFACE_2 = "#f7f7fa" # 次级填充（只读输入框等）
STAGE = "#1c1c1e"     # 图片画布底色（深色，让航拍画面更突出）

# ---- 描边 / 分隔（Apple separator）----
LINE = "#e5e5ea"      # 卡片描边
LINE_SOFT = "#f0f0f4" # 弱分隔
SEPARATOR = "#d1d1d6" # 强分隔

# ---- 排版 / 形状 ----
# 默认平台系统字体：Windows 用 Segoe UI Variable / 微软雅黑 UI，macOS 用苹方
FONT_STACK = ("'Segoe UI Variable Text','Segoe UI','Microsoft YaHei UI',"
              "'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',sans-serif")

RADIUS = "12px"       # 卡片
RADIUS_SM = "8px"     # 控件
RADIUS_PILL = "999px" # 药丸标签


def stylesheet() -> str:
    """返回全局样式表。

    PySide6 的 QSS 不支持 CSS 变量，这里用 Python 字符串插值保证 token 单一来源。
    """
    return f"""
/* ===================== 基础 ===================== */
QWidget {{
    font-family: {FONT_STACK};
    font-size: 13px;
    color: {INK_2};
}}
QMainWindow, QWidget#centralRoot {{
    background-color: {CANVAS};
}}

/* 顶栏材质：半透明浮层 + 底部发丝线（内容在其下滚动） */
QFrame#topBar {{
    background-color: rgba(255, 255, 255, 0.78);
    border: none;
    border-bottom: 1px solid {LINE};
}}

QToolTip {{
    background-color: {INK};
    color: #ffffff;
    border: none;
    padding: 6px 9px;
    border-radius: 7px;
    font-size: 12px;
}}

/* ===================== 卡片 ===================== */
QGroupBox {{
    background-color: {SURFACE};
    border: 1px solid {LINE};
    border-radius: {RADIUS};
    margin-top: 16px;
    padding: 16px;
    font-weight: 600;
    color: {INK};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: {INK};
    background-color: transparent;
}}

/* ===================== 文本 ===================== */
QLabel {{
    background: transparent;
}}
QLabel#titleLabel {{
    font-size: 22px;
    font-weight: 700;
    color: {INK};
    letter-spacing: -0.4px;
}}
QLabel#subTitleLabel {{
    font-size: 13px;
    color: {INK_3};
    letter-spacing: 0.1px;
}}
QLabel#sectionLabel {{
    font-size: 12px;
    font-weight: 600;
    color: {INK_3};
    padding-bottom: 2px;
}}
QLabel#metaLabel {{
    font-size: 12px;
    color: {INK_3};
}}
QLabel#hintLabel {{
    font-size: 11px;
    color: {INK_3};
}}
QLabel#statusLabel {{
    font-size: 12px;
    color: {INK_3};
    padding: 4px 2px 0 2px;
}}
QLabel#versionChip {{
    font-size: 11px;
    font-weight: 600;
    color: {BRAND};
    background-color: {BRAND_SOFT};
    border-radius: {RADIUS_PILL};
    padding: 3px 10px;
}}
QLabel#emptyState {{
    font-size: 14px;
    color: {INK_4};
    qproperty-alignment: AlignCenter;
}}
QLabel#statPill {{
    font-size: 12px;
    font-weight: 600;
    color: {INK_2};
    background-color: {LINE_SOFT};
    border-radius: {RADIUS_PILL};
    padding: 4px 11px;
}}

/* ---- 语义标签（药丸）：填充色 + 白字 / 浅底 + 深字 ---- */
QLabel#dangerPill {{
    font-size: 12px;
    font-weight: 600;
    color: #ffffff;
    background-color: {DANGER};
    border-radius: {RADIUS_PILL};
    padding: 4px 11px;
}}
QLabel#successPill {{
    font-size: 12px;
    font-weight: 600;
    color: {SUCCESS_TEXT};
    background-color: {SUCCESS_SOFT};
    border-radius: {RADIUS_PILL};
    padding: 4px 11px;
}}

/* ---- 帧列表行 ---- */
QLabel#frameRowTitle {{
    font-size: 13px;
    font-weight: 600;
    color: {INK};
}}
QLabel#frameRowTime {{
    font-size: 13px;
    font-weight: 600;
    color: {INK};
}}
QLabel#badgeDanger {{
    font-size: 11px;
    font-weight: 600;
    color: #ffffff;
    background-color: {DANGER};
    border-radius: {RADIUS_PILL};
    padding: 2px 9px;
}}
QLabel#badgeMuted {{
    font-size: 11px;
    color: {INK_4};
    background-color: {LINE_SOFT};
    border-radius: {RADIUS_PILL};
    padding: 2px 9px;
}}

/* ---- 知识库列表行 ---- */
QLabel#listTitle {{
    font-size: 12px;
    font-weight: 600;
    color: {INK};
}}
QLabel#listSummary {{
    font-size: 12px;
    color: {INK_3};
}}
QLabel#listScene {{
    font-size: 11px;
    color: {INK_4};
}}
QLabel#simBadge {{
    font-size: 11px;
    font-weight: 600;
    color: {BRAND};
    background-color: {BRAND_SOFT};
    border-radius: {RADIUS_PILL};
    padding: 2px 8px;
}}

/* ---- 发丝分隔线 ---- */
QFrame#hSeparator {{
    background-color: {LINE};
    border: none;
    max-height: 1px;
}}

/* ===================== 按钮 ===================== */
QPushButton {{
    background-color: {BRAND};
    color: #ffffff;
    border: none;
    border-radius: {RADIUS_SM};
    padding: 8px 18px;
    font-weight: 600;
    min-height: 16px;
}}
QPushButton:hover {{ background-color: {BRAND_HOVER}; }}
QPushButton:pressed {{ background-color: {BRAND_PRESS}; }}
QPushButton:focus {{ outline: none; border: 2px solid #a8ccf7; padding: 6px 16px; }}
QPushButton:disabled {{
    background-color: {LINE};
    color: {INK_4};
}}

QPushButton#successBtn {{ background-color: {SUCCESS}; }}
QPushButton#successBtn:hover {{ background-color: #2fb454; }}
QPushButton#successBtn:pressed {{ background-color: #279a48; }}

QPushButton#warningBtn {{ background-color: {WARNING}; }}
QPushButton#warningBtn:hover {{ background-color: #f08a00; }}
QPushButton#warningBtn:pressed {{ background-color: #d97e00; }}

QPushButton#dangerBtn {{ background-color: {DANGER}; }}
QPushButton#dangerBtn:hover {{ background-color: #f03028; }}
QPushButton#dangerBtn:pressed {{ background-color: #d92a22; }}

/* 次级按钮：白底描边，克制而清晰 */
QPushButton#ghostBtn {{
    background-color: {SURFACE};
    color: {INK_2};
    border: 1px solid {LINE};
}}
QPushButton#ghostBtn:hover {{
    background-color: {SURFACE_2};
    border-color: {SEPARATOR};
    color: {INK};
}}
QPushButton#ghostBtn:pressed {{ background-color: {LINE_SOFT}; }}
QPushButton#ghostBtn:disabled {{
    background-color: {SURFACE};
    color: {INK_4};
    border-color: {LINE};
}}
QPushButton#ghostBtn:focus {{ border: 2px solid #a8ccf7; padding: 7px 17px; }}

/* 可切换筛选标签（药丸） */
QPushButton#chipBtn {{
    background-color: {SURFACE};
    color: {INK_3};
    border: 1px solid {LINE};
    border-radius: {RADIUS_PILL};
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 500;
}}
QPushButton#chipBtn:hover {{
    background-color: {SURFACE_2};
    color: {INK_2};
    border-color: {SEPARATOR};
}}
QPushButton#chipBtn:checked {{
    background-color: {BRAND_SOFT};
    color: {BRAND};
    border-color: #bcdcfb;
    font-weight: 600;
}}

/* ===================== 输入 ===================== */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {SURFACE};
    border: 1px solid {LINE};
    border-radius: {RADIUS_SM};
    padding: 7px 10px;
    selection-background-color: {BRAND};
    selection-color: #ffffff;
}}
/* 大块只读正文（违章详情 / 条款原文）按卡片规格处理，与列表、画面同级 */
QTextEdit {{
    background-color: {SURFACE};
    border: 1px solid {LINE};
    border-radius: {RADIUS};
    padding: 10px 12px;
    selection-background-color: {BRAND};
    selection-color: #ffffff;
}}
QLineEdit:hover, QTextEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {SEPARATOR};
}}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {BRAND};
    background-color: {SURFACE};
}}
QLineEdit:read-only {{
    background-color: {SURFACE_2};
    color: {INK_2};
}}
QLineEdit[echoMode="2"] {{ letter-spacing: 2px; }}
/* 下拉箭头：不写 ::drop-down 规则，交回 Fusion 原生三角；
   自绘（border:none + 边框折线）会得到残缺直角，实测更差 */
/* 数字框步进按钮：必须给出 subcontrol-origin，否则箭头被压成横杠 */
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: padding;
    width: 18px;
}}
QComboBox QAbstractItemView {{
    border: 1px solid {LINE};
    border-radius: {RADIUS_SM};
    background-color: {SURFACE};
    selection-background-color: {BRAND_SOFT};
    selection-color: {INK};
    padding: 4px;
    outline: none;
}}

/* ===================== 列表 ===================== */
QListWidget, QTreeWidget {{
    background-color: {SURFACE};
    border: 1px solid {LINE};
    border-radius: {RADIUS};
    padding: 4px;
    outline: none;
}}
/* 卡片内的列表不再自带外框，避免「卡片套卡片」的双层描边 */
QGroupBox QListWidget, QGroupBox QTreeWidget {{
    border: none;
    background-color: transparent;
    padding: 0px;
}}
QListWidget::item {{
    border-radius: {RADIUS_SM};
    padding: 6px 8px;
    margin: 1px 0px;
}}
QListWidget::item:hover {{ background-color: {SURFACE_2}; }}
QListWidget::item:selected {{
    background-color: {BRAND_SOFT};
    color: {INK};
}}
QListView::item:selected:!active {{ background-color: {BRAND_SOFT}; color: {INK}; }}

/* ===================== 进度条 ===================== */
QProgressBar {{
    border: none;
    background-color: {LINE};
    border-radius: 3px;
    height: 6px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {BRAND};
    border-radius: 3px;
}}

/* ===================== Tab ===================== */
QTabWidget::pane {{
    border: none;
    background-color: transparent;
}}
QTabBar {{
    background-color: transparent;
    qproperty-drawBase: 0;
}}
QTabBar::tab {{
    background-color: transparent;
    color: {INK_3};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 9px 18px;
    font-size: 13px;
    font-weight: 500;
}}
QTabBar::tab:hover {{ color: {INK_2}; }}
QTabBar::tab:selected {{
    color: {BRAND};
    font-weight: 600;
    border-bottom: 2px solid {BRAND};
}}

/* ===================== 分割器 ===================== */
QSplitter::handle {{
    background-color: transparent;
    width: 12px;
}}

/* ===================== 图片画布 ===================== */
QGraphicsView#imageStage {{
    background-color: {STAGE};
    border: 1px solid {LINE};
    border-radius: {RADIUS};
}}

/* ===================== 滚动条（悬浮细条） ===================== */
QScrollArea {{ background: transparent; border: none; }}

QScrollBar:vertical {{
    background-color: transparent;
    width: 10px;
    margin: 2px 2px 2px 0px;
}}
QScrollBar::handle:vertical {{
    background-color: {INK_5};
    border-radius: 5px;
    min-height: 36px;
}}
QScrollBar::handle:vertical:hover {{ background-color: {INK_4}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}

QScrollBar:horizontal {{
    background-color: transparent;
    height: 10px;
    margin: 0px 2px 2px 2px;
}}
QScrollBar::handle:horizontal {{
    background-color: {INK_5};
    border-radius: 5px;
    min-width: 36px;
}}
QScrollBar::handle:horizontal:hover {{ background-color: {INK_4}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}

/* ===================== 对话框 ===================== */
QDialog {{
    background-color: {CANVAS};
}}
QDialogButtonBox QPushButton {{
    min-width: 84px;
}}
QProgressDialog {{
    background-color: {SURFACE};
}}
"""


# ============================ 视觉工具 ============================
def apply_card_shadow(widget, blur=18, dy=2, alpha=26):
    """给卡片加一层柔和投影（Apple 材质：抬起的层）。

    Qt 的 QSS 不支持 box-shadow，只能用 QGraphicsDropShadowEffect。
    Qt 在此惰性导入，保证本模块被纯 HTML 渲染（render_html）引用时不强制依赖 Qt。
    """
    from PySide6.QtWidgets import QGraphicsDropShadowEffect
    from PySide6.QtGui import QColor

    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setOffset(0, dy)
    effect.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(effect)
    return effect


def repolish(widget):
    """objectName 变更后强制重新套用样式。

    Qt 只在 polish 时按当时的 objectName 求值一次样式表；运行时改名后必须
    unpolish/polish 才会重新匹配，否则控件会停留在旧样式（例如违章数量徽章
    从「有违章」切到「无违章」时颜色不跟着变）。
    """
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
