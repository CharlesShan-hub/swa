"""
第二页：波形分析 — 单条波形查看 + 刷新
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QPlainTextEdit,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib
import numpy as np

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


class WavePage(QWidget):
    """波形分析页面 — 按偏移量查看波形 + 自动刷新。"""

    def __init__(self):
        super().__init__()
        self.conn = None
        self.current_offset = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        # 标题
        title = QLabel("波形分析")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        layout.addWidget(title)
        layout.addSpacing(12)

        # 控制栏
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("偏移量:"))
        self.offset_edit = QLineEdit("0")
        self.offset_edit.setFixedWidth(120)
        ctrl.addWidget(self.offset_edit)
        ctrl.addWidget(QLabel(" / 总计 67370"))

        ctrl.addSpacing(16)

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self._do_refresh)
        ctrl.addWidget(self.refresh_btn)

        self.auto_btn = QPushButton("自动刷新(8s)")
        self.auto_btn.setCheckable(True)
        self.auto_btn.toggled.connect(self._toggle_auto)
        ctrl.addWidget(self.auto_btn)

        layout.addLayout(ctrl)
        layout.addSpacing(8)

        # matplotlib 画布
        self.fig = Figure(figsize=(8, 3))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas, 1)

        # 信息栏
        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setMaximumHeight(80)
        layout.addWidget(self.info)

        # 定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._do_refresh)

    def set_connection(self, conn):
        self.conn = conn
        self.info.appendPlainText("数据库已连接")

    def _do_refresh(self):
        if self.conn is None:
            self.info.appendPlainText("请先连接数据库")
            return

        try:
            text = self.offset_edit.text().strip()
            offset = int(text) if text else 0
        except ValueError:
            offset = 0

        self.current_offset = offset
        self._fetch_and_plot(offset)

    def _fetch_and_plot(self, offset: int):
        """按偏移量获取波形并绘制。"""
        cur = self.conn.cursor()
        dameng_offset = offset + 1

        try:
            cur.execute(f"""
                SELECT ACTUAL_VOLTAGE, SYSTEM_TIME,
                       RTU_REGS_P00_ENV_TEMP, RTU_REGS_P00_ENV_HUMIDITY,
                       RTU_REGS_P00_ROTOR_RPM, RTU_REGS_P00_WAVE_DATA
                FROM (
                    SELECT a.*, ROWNUM rn FROM (
                        SELECT * FROM YS_DB.TB_MODBUS_DEV_POINT
                        ORDER BY SYSTEM_TIME DESC
                    ) a WHERE ROWNUM <= {dameng_offset}
                )
                WHERE rn = {dameng_offset}
            """)
            row = cur.fetchone()
            if row is None:
                self.info.appendPlainText(f"偏移量 {offset} 无数据")
                return

            voltage, sys_time, temp, humid, rpm, wave_str = row

            # 解析波形
            try:
                wave = np.array([float(x) for x in wave_str.split(",")][:512])
            except (ValueError, TypeError):
                self.info.appendPlainText("波形数据解析失败")
                return

            # 绘制
            self.ax.clear()
            self.ax.plot(wave, linewidth=0.8, color="#0078d4")
            self.ax.set_xlabel("采样点")
            self.ax.set_ylabel("幅值")
            self.ax.set_title(f"偏移量 #{offset} — 电压: {voltage}V")
            self.ax.grid(True, alpha=0.3)
            self.fig.tight_layout()
            self.canvas.draw()

            # 信息
            self.info.clear()
            self.info.appendPlainText(
                f"电压: {voltage}V  |  时间: {sys_time}  |  "
                f"温度: {temp}°C  |  湿度: {humid}%  |  RPM: {rpm}"
            )
            # 更新 offset 显示
            self.offset_edit.setText(str(offset))

        except Exception as e:
            self.info.appendPlainText(f"查询失败: {e}")

    def _toggle_auto(self, checked):
        if checked:
            self.auto_btn.setText("停止刷新")
            self.timer.start(8000)
        else:
            self.auto_btn.setText("自动刷新(8s)")
            self.timer.stop()
