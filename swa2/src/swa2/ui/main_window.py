"""主窗口 — 左侧导航栏 + 右侧内容区"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QListWidgetItem, QStackedWidget, QLabel,
    QFrame,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QIcon

from swa2.ui.pages.page1 import Page1
from swa2.ui.pages.page2 import Page2
from swa2.ui.pages.page3 import Page3


_NAV_ITEMS = [
    ("数据库连接", "🔗"),
    ("项目管理",   "📊"),
    ("统计信息",   "📈"),
]

_PAGES = [Page1, Page2, Page3]


class Sidebar(QListWidget):
    """左侧导航栏"""

    def __init__(self):
        super().__init__()
        self.setObjectName("sidebar")
        self.setFixedWidth(180)
        self.setSpacing(0)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        font = QFont("Microsoft YaHei", 13)

        for label, icon in _NAV_ITEMS:
            item = QListWidgetItem(f"  {icon}  {label}")
            item.setFont(font)
            item.setSizeHint(QSize(180, 52))
            self.addItem(item)

        self.setCurrentRow(0)

    def _apply_style(self):
        self.setStyleSheet("""
            QListWidget#sidebar {
                background-color: #1e1e2e;
                border: none;
                outline: none;
            }
            QListWidget#sidebar::item {
                color: #cdd6f4;
                padding: 10px 16px;
                border-left: 3px solid transparent;
            }
            QListWidget#sidebar::item:selected {
                background-color: #313244;
                color: #cdd6f4;
                border-left: 3px solid #89b4fa;
            }
            QListWidget#sidebar::item:hover:!selected {
                background-color: #2a2a3c;
                color: #cdd6f4;
            }
        """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SWA2 — 波形分析系统")
        self.resize(1200, 780)

        # 中央部件
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── 左侧导航 ──
        self.sidebar = Sidebar()
        self.sidebar.currentRowChanged.connect(self._switch_page)
        layout.addWidget(self.sidebar)

        # ── 分割线 ──
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #313244;")
        layout.addWidget(sep)

        # ── 右侧内容 ──
        self.stack = QStackedWidget()
        self._pages = []
        for PageCls in _PAGES:
            page = PageCls()
            self._pages.append(page)
            self.stack.addWidget(page)
        layout.addWidget(self.stack, 1)

        # 全局样式
        self._apply_styles()

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: "Microsoft YaHei";
            }
            QLabel {
                background: transparent;
            }
            QFrame {
                background: transparent;
            }
            QStackedWidget {
                background-color: #181825;
            }
        """)

    def _switch_page(self, index: int):
        if 0 <= index < len(self._pages):
            self.stack.setCurrentIndex(index)

    def closeEvent(self, event):
        """窗口关闭时，安全停止所有页面的后台任务。"""
        for page in self._pages:
            stop = getattr(page, "stop_download", None)
            if stop:
                stop()
        super().closeEvent(event)
