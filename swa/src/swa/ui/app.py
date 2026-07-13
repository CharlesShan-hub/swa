"""
GUI 入口 — 主窗口

页面:
  1. 数据下载  →  local.jsonl（原始备份）
  2. 项目管理  →  SQLite 项目（标签替换 + 导入）
  3. 数据探索  →  图表 + 质量筛选
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QStackedWidget,
)
from PySide6.QtCore import Qt

from swa.ui.pages.download_page import DownloadPage
from swa.ui.pages.project_page import ProjectPage
from swa.ui.pages.explorer_page import ExplorerPage


PAGES = ["数据下载", "项目管理", "数据探索"]

APP_STYLE = """
QWidget {
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
    background: #ffffff;
    color: #000000;
}

/* 导航栏 */
QListWidget#navList {
    background: #ffffff;
    border-right: 1px solid #ddd;
    font-size: 14px;
    padding: 8px 0;
    color: #000;
}
QListWidget#navList::item {
    padding: 12px 20px;
    border: none;
    background: #ffffff;
    color: #000;
}
QListWidget#navList::item:selected {
    background: #f0f0f0;
    font-weight: 600;
}

/* 堆叠页面 */
QStackedWidget {
    background: #ffffff;
}

/* 分组框 */
QGroupBox {
    font-weight: 600;
    border: 1px solid #ddd;
    margin-top: 8px;
    padding: 14px 12px 10px;
    background: #ffffff;
    color: #000;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 10px;
    color: #000;
}

/* 按钮 */
QPushButton {
    border: 1px solid #ccc;
    padding: 6px 18px;
    background: #ffffff;
    color: #000;
    font-size: 13px;
}
QPushButton:hover {
    background: #f5f5f5;
}

/* 输入框 */
QLineEdit, QSpinBox, QDoubleSpinBox {
    border: 1px solid #ccc;
    padding: 5px 8px;
    background: #ffffff;
    color: #000;
}

/* 表格 */
QTableWidget {
    border: 1px solid #ddd;
    gridline-color: #eee;
    background: #ffffff;
    color: #000;
}
QTableWidget::item {
    padding: 6px 10px;
}
QHeaderView::section {
    background: #fafafa;
    border: none;
    border-bottom: 1px solid #ddd;
    padding: 8px 10px;
    font-weight: 600;
    color: #000;
}

/* 日志框 */
QPlainTextEdit {
    border: 1px solid #ddd;
    background: #ffffff;
    padding: 8px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
    color: #000;
}

/* 进度条 */
QProgressBar {
    border: 1px solid #ddd;
    background: #f5f5f5;
    height: 12px;
    text-align: center;
    font-size: 11px;
    color: #000;
}
QProgressBar::chunk {
    background: #999;
}

/* 列表 */
QListWidget {
    border: 1px solid #ddd;
    background: #ffffff;
    color: #000;
}
QListWidget::item {
    padding: 6px 10px;
    color: #000;
}
QListWidget::item:selected {
    background: #f0f0f0;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("swa — 场磨电压检测分析")
        self.resize(1200, 840)
        self.setStyleSheet(APP_STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 左侧导航
        self.nav = QListWidget()
        self.nav.setObjectName("navList")
        self.nav.setFixedWidth(180)
        self.nav.addItems(PAGES)
        self.nav.setCurrentRow(0)

        # 右侧页面
        self.pages = QStackedWidget()
        self.pages.addWidget(DownloadPage())
        self.pages.addWidget(ProjectPage())
        self.pages.addWidget(ExplorerPage())

        layout.addWidget(self.nav)
        layout.addWidget(self.pages, 1)

        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
