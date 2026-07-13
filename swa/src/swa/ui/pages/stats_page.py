"""
第三页：统计信息 — 最近 500 条记录的电压/温度/湿度曲线
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib
import numpy as np

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


class StatsPage(QWidget):
    """统计页面 — 最近 500 条数据的趋势图。"""

    def __init__(self):
        super().__init__()
        self.conn = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("统计信息")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        layout.addWidget(title)
        layout.addSpacing(8)

        # 刷新按钮
        ctrl_layout = QVBoxLayout()
        self.refresh_btn = QPushButton("刷新统计 (最近 500 条)")
        self.refresh_btn.setFixedWidth(200)
        self.refresh_btn.clicked.connect(self._do_refresh)
        ctrl_layout.addWidget(self.refresh_btn)
        layout.addLayout(ctrl_layout)
        layout.addSpacing(12)

        # matplotlib 画布（3 个子图）
        self.fig = Figure(figsize=(10, 8))
        self.ax_volt = self.fig.add_subplot(311)
        self.ax_temp = self.fig.add_subplot(312)
        self.ax_humid = self.fig.add_subplot(313)
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas, 1)

    def set_connection(self, conn):
        self.conn = conn

    def _do_refresh(self):
        if self.conn is None:
            return

        cur = self.conn.cursor()
        try:
            cur.execute("""
                SELECT ACTUAL_VOLTAGE, SYSTEM_TIME,
                       RTU_REGS_P00_ENV_TEMP, RTU_REGS_P00_ENV_HUMIDITY,
                       RTU_REGS_P00_ROTOR_RPM
                FROM YS_DB.TB_MODBUS_DEV_POINT
                ORDER BY SYSTEM_TIME DESC
                LIMIT 500
            """)
            rows = cur.fetchall()

            if not rows:
                return

            # 倒序排列（按时间正序绘图）
            rows = list(reversed(rows))
            voltages = [float(r[0]) if r[0] else 0 for r in rows]
            temps = [float(r[2]) / 10 if r[2] else 0 for r in rows]
            humids = [float(r[3]) / 10 if r[3] else 0 for r in rows]
            times = [str(r[1])[-8:] if r[1] else "" for r in rows]
            x = np.arange(len(rows))

            # 电压
            self.ax_volt.clear()
            self.ax_volt.plot(x, voltages, color="#d43f3a", linewidth=1)
            self.ax_volt.set_ylabel("电压 (V)")
            self.ax_volt.set_xticks(x[::50])
            self.ax_volt.set_xticklabels(times[::50], rotation=30, fontsize=8)
            self.ax_volt.grid(True, alpha=0.3)

            # 温度
            self.ax_temp.clear()
            self.ax_temp.plot(x, temps, color="#e68a2e", linewidth=1)
            self.ax_temp.set_ylabel("温度 (°C)")
            self.ax_temp.set_xticks(x[::50])
            self.ax_temp.set_xticklabels(times[::50], rotation=30, fontsize=8)
            self.ax_temp.grid(True, alpha=0.3)

            # 湿度
            self.ax_humid.clear()
            self.ax_humid.plot(x, humids, color="#4caf50", linewidth=1)
            self.ax_humid.set_xlabel("时间")
            self.ax_humid.set_ylabel("湿度 (%)")
            self.ax_humid.set_xticks(x[::50])
            self.ax_humid.set_xticklabels(times[::50], rotation=30, fontsize=8)
            self.ax_humid.grid(True, alpha=0.3)

            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            print(f"查询失败: {e}")
