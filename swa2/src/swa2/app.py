"""SWA2 入口"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from swa2.ui.main_window import MainWindow


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)

    # 全局样式
    app.setStyleSheet("""
        QToolTip {
            background-color: #2d2d2d;
            color: #e0e0e0;
            border: 1px solid #555;
            padding: 4px 8px;
            font-size: 12px;
        }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
