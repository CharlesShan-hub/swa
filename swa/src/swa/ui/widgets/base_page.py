"""
通用页面模板 — 所有页面统一标题和布局
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QFont


class BasePage(QWidget):
    """所有页面的基类，统一标题和边距。"""

    def __init__(self, title: str):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        self.title = QLabel(title)
        self.title.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        layout.addWidget(self.title)
        layout.addSpacing(4)

        # 子类把内容加到这里
        self.content = QVBoxLayout()
        layout.addLayout(self.content, 1)
