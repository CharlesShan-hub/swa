"""通用表单弹窗 — 按钮点击弹出，节省界面空间

用法:
    dialog = FormDialog("连接配置", [
        ("主机:", "host", "10.15.10.1"),
        ("端口:", "port", "5256"),
        ("用户:", "user", "SYSDBA"),
        ("密码:", "password", "", "password"),
    ])
    if dialog.exec():
        values = dialog.values()  # {"host": "10.15.10.1", "port": "5256", ...}
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFormLayout, QWidget, QDialogButtonBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from swa2.ui.widgets.styles import STYLES


class FormDialog(QDialog):
    """通用表单弹窗。

    fields: list of (label, key, default, [echo_mode])
        - label: 显示标签
        - key: 值的字典键名
        - default: 默认值
        - echo_mode: 可选 "password" / "normal"（默认 normal）
    """

    def __init__(self, title: str, fields: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(380)
        self.setModal(True)
        self._keys = []
        self._widgets = {}

        self.setStyleSheet(STYLES["dialog"])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        # 标题
        lbl = QLabel(title)
        lbl.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        layout.addWidget(lbl)

        # 表单
        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        for item in fields:
            label = item[0]
            key = item[1]
            default = item[2] if len(item) > 2 else ""
            echo_mode = item[3] if len(item) > 3 else "normal"

            self._keys.append(key)

            if echo_mode == "password":
                w = QLineEdit()
                w.setEchoMode(QLineEdit.EchoMode.Password)
            else:
                w = QLineEdit()

            w.setText(str(default))
            w.setStyleSheet(STYLES["input"])
            w.setMinimumHeight(30)
            form.addRow(QLabel(label), w)
            self._widgets[key] = w

        layout.addLayout(form)
        layout.addSpacing(12)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(STYLES["btn_secondary"])
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("确定")
        ok_btn.setStyleSheet(STYLES["btn_primary"])
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)

        layout.addLayout(btn_layout)

    def values(self) -> dict:
        """返回 {key: value, ...}"""
        return {k: self._widgets[k].text().strip() for k in self._keys}

    def int_value(self, key: str, default: int = 0) -> int:
        """安全获取整数值"""
        try:
            return int(self._widgets[key].text().strip())
        except (ValueError, KeyError):
            return default
