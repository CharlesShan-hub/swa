"""共享样式常量 — 统一全应用视觉效果"""

INPUT = """
    QLineEdit {
        background-color: #313244;
        color: #cdd6f4;
        border: 1px solid #45475a;
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 13px;
    }
    QLineEdit:focus {
        border-color: #89b4fa;
    }
    QLineEdit:disabled {
        background-color: #252536;
        color: #585b70;
    }
"""

COMBO = """
    QComboBox {
        background-color: #313244;
        color: #cdd6f4;
        border: 1px solid #45475a;
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 13px;
        min-width: 80px;
    }
    QComboBox:focus {
        border-color: #89b4fa;
    }
    QComboBox::drop-down {
        border: none;
        width: 24px;
    }
    QComboBox::down-arrow {
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid #a6adc8;
        margin-right: 6px;
    }
    QComboBox QAbstractItemView {
        background-color: #313244;
        color: #cdd6f4;
        border: 1px solid #45475a;
        selection-background-color: #45475a;
    }
"""

SPINBOX = """
    QSpinBox {
        background-color: #313244;
        color: #cdd6f4;
        border: 1px solid #45475a;
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 13px;
    }
    QSpinBox:focus {
        border-color: #89b4fa;
    }
    QSpinBox::up-button, QSpinBox::down-button {
        border: none;
        background: transparent;
        width: 20px;
    }
    QSpinBox::up-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-bottom: 6px solid #a6adc8;
        margin: 2px auto;
    }
    QSpinBox::down-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 6px solid #a6adc8;
        margin: 2px auto;
    }
"""

BTN_PRIMARY = """
    QPushButton {
        background-color: #89b4fa;
        color: #1e1e2e;
        border: none;
        border-radius: 4px;
        padding: 6px 20px;
        font-size: 13px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #74c7ec;
    }
    QPushButton:pressed {
        background-color: #89b4fa;
    }
    QPushButton:disabled {
        background-color: #45475a;
        color: #6c7086;
    }
"""

BTN_SECONDARY = """
    QPushButton {
        background-color: #45475a;
        color: #cdd6f4;
        border: none;
        border-radius: 4px;
        padding: 6px 20px;
        font-size: 13px;
    }
    QPushButton:hover {
        background-color: #585b70;
    }
    QPushButton:pressed {
        background-color: #45475a;
    }
"""

DIALOG = """
    QDialog {
        background-color: #1e1e2e;
    }
    QLabel {
        color: #cdd6f4;
        font-size: 13px;
        background: transparent;
    }
"""

STYLES = {
    "input": INPUT,
    "combo": COMBO,
    "spinbox": SPINBOX,
    "btn_primary": BTN_PRIMARY,
    "btn_secondary": BTN_SECONDARY,
    "dialog": DIALOG,
}
