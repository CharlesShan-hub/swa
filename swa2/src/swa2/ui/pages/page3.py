"""页面 3 — 统计信息（按钮 + 弹窗表单模式）"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from swa2.ui.widgets.form_dialog import FormDialog
from swa2.ui.widgets.styles import STYLES


class Page3(QWidget):
    def __init__(self):
        super().__init__()
        self._config = {
            "chart_type": "电压曲线",
            "window_size": "8",
            "show_raw": "是",
        }

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)

        # ── 标题 ──
        title = QLabel("统计信息")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        desc = QLabel("多子图展示电压/温湿度曲线")
        desc.setStyleSheet("color: #a6adc8; font-size: 13px;")
        layout.addWidget(desc)

        layout.addSpacing(20)

        # ── 配置卡片 ──
        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet("""
            QFrame#card {
                background-color: #1e1e2e;
                border: 1px solid #313244;
                border-radius: 8px;
                padding: 20px;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(12)

        self.config_label = QLabel(self._format_config())
        self.config_label.setStyleSheet("color: #a6adc8; font-size: 13px;")
        card_layout.addWidget(self.config_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #313244;")
        card_layout.addWidget(sep)

        btn_row = QHBoxLayout()
        self.config_btn = QPushButton("图表配置")
        self.config_btn.setFixedHeight(34)
        self.config_btn.setStyleSheet(STYLES["btn_primary"])
        self.config_btn.clicked.connect(self._open_config_dialog)
        btn_row.addWidget(self.config_btn)
        btn_row.addStretch()
        card_layout.addLayout(btn_row)

        layout.addWidget(card)

        # ── 占位区域 ──
        layout.addSpacing(20)
        placeholder = QFrame()
        placeholder.setStyleSheet("""
            background-color: #1e1e2e;
            border: 1px dashed #313244;
            border-radius: 8px;
        """)
        ph_layout = QVBoxLayout(placeholder)
        ph_layout.setAlignment(Qt.AlignCenter)
        lbl = QLabel("图表区域（待实现）")
        lbl.setStyleSheet("color: #585b70; font-size: 15px; background: transparent;")
        ph_layout.addWidget(lbl)
        layout.addWidget(placeholder, 1)

    def _format_config(self) -> str:
        return (f"图表类型: {self._config['chart_type']}  |  "
                f"滑动窗口: {self._config['window_size']}  |  "
                f"显示原始: {self._config['show_raw']}")

    def _open_config_dialog(self):
        dialog = FormDialog("图表配置", [
            ("图表类型:",      "chart_type",  self._config["chart_type"]),
            ("滑动窗口:",      "window_size", self._config["window_size"]),
            ("显示原始波形:",  "show_raw",    self._config["show_raw"]),
        ], parent=self)

        if dialog.exec():
            self._config = dialog.values()
            self.config_label.setText(self._format_config())
