"""
数据探索页面 — 嵌入式波形 + 数据筛选 + 行级质量开关
"""

import sys, os, tempfile, webbrowser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QAbstractItemView, QSpinBox, QLineEdit,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QShortcut, QKeySequence

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib
matplotlib.use("QtAgg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyecharts.charts import Line, Grid
from pyecharts import options as opts
from pyecharts.globals import ThemeType

from swa.data.manager import DataManager
from swa.ui.widgets.base_page import BasePage

import json

LOCAL_JSONL = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "data", "local.jsonl"
)


class ExplorerPage(BasePage):
    """数据探索页面 — 嵌入式波形 + 数据筛选。"""

    def __init__(self):
        super().__init__("数据探索")
        self.dm = DataManager()
        self.current_df: pd.DataFrame = None

        # ── 顶部：项目选择 + 摘要 ───────────────────────────────
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("项目:"))
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(200)
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        top_row.addWidget(self.project_combo)
        self.reload_btn = QPushButton("刷新")
        self.reload_btn.clicked.connect(self._refresh_project_list)
        top_row.addWidget(self.reload_btn)
        top_row.addSpacing(16)
        self.summary_label = QLabel("")
        top_row.addWidget(self.summary_label, 1)
        self.content.addLayout(top_row)

        # ── 中部：分割器（波形图 + 表格）─────────────────────────
        splitter = QSplitter(Qt.Vertical)

        # 波形图区
        wave_widget = QWidget()
        wave_layout = QVBoxLayout(wave_widget)
        wave_layout.setContentsMargins(0, 0, 0, 0)

        wave_ctrl = QHBoxLayout()
        wave_ctrl.addWidget(QLabel("波形 ID:"))
        self.wave_id_edit = QLineEdit()
        self.wave_id_edit.setPlaceholderText("输入 ID 回车查看")
        self.wave_id_edit.setFixedWidth(120)
        self.wave_id_edit.returnPressed.connect(self._show_waveform)
        wave_ctrl.addWidget(self.wave_id_edit)
        self.wave_btn = QPushButton("查看")
        self.wave_btn.clicked.connect(self._show_waveform)
        wave_ctrl.addWidget(self.wave_btn)
        wave_ctrl.addStretch()
        self.open_chart_btn = QPushButton("打开趋势图")
        self.open_chart_btn.clicked.connect(self._open_chart)
        wave_ctrl.addWidget(self.open_chart_btn)
        self.quality_btn = QPushButton("质量检测")
        self.quality_btn.clicked.connect(self._run_quality_check)
        wave_ctrl.addWidget(self.quality_btn)
        self.backfill_btn = QPushButton("回填谐波")
        self.backfill_btn.clicked.connect(self._backfill_harmonics)
        wave_ctrl.addWidget(self.backfill_btn)
        wave_layout.addLayout(wave_ctrl)

        # 内嵌 matplotlib 画布
        self.fig = Figure(figsize=(8, 2.5))
        self.fig.set_tight_layout(True)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("采样点")
        self.ax.set_ylabel("幅值")
        self.ax.grid(True, alpha=0.3)
        self.canvas = FigureCanvasQTAgg(self.fig)
        wave_layout.addWidget(self.canvas, 1)
        splitter.addWidget(wave_widget)

        # 数据表格区
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0, 0, 0, 0)

        # 筛选栏
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("电压:"))
        self.volt_filter = QComboBox()
        self.volt_filter.addItem("全部")
        self.volt_filter.currentTextChanged.connect(self._on_filter_changed)
        self.volt_filter.setMinimumWidth(100)
        filter_row.addWidget(self.volt_filter)
        self.volt_disable_btn = QPushButton("禁用整组")
        self.volt_disable_btn.setFixedHeight(24)
        self.volt_disable_btn.clicked.connect(self._disable_voltage_group)
        filter_row.addWidget(self.volt_disable_btn)
        self.volt_enable_btn = QPushButton("启用整组")
        self.volt_enable_btn.setFixedHeight(24)
        self.volt_enable_btn.clicked.connect(self._enable_voltage_group)
        filter_row.addWidget(self.volt_enable_btn)
        filter_row.addSpacing(12)

        filter_row.addWidget(QLabel("状态:"))
        self.status_filter = QComboBox()
        self.status_filter.addItems(["全部", "启用", "禁用"])
        self.status_filter.currentTextChanged.connect(self._on_filter_changed)
        self.status_filter.setFixedWidth(70)
        filter_row.addWidget(self.status_filter)
        filter_row.addStretch()
        table_layout.addLayout(filter_row)

        # 表格操作
        table_ops = QHBoxLayout()
        self.disable_btn = QPushButton("禁用选中行 [2]")
        self.disable_btn.clicked.connect(self._disable_selected)
        table_ops.addWidget(self.disable_btn)
        self.enable_btn = QPushButton("启用选中行 [1]")
        self.enable_btn.clicked.connect(self._enable_selected)
        table_ops.addWidget(self.enable_btn)
        self.sync_btn = QPushButton("同步到 jsonl")
        self.sync_btn.clicked.connect(self._sync_to_jsonl)
        table_ops.addWidget(self.sync_btn)
        self.status_label = QLabel("")
        table_ops.addWidget(self.status_label)
        table_ops.addStretch()
        table_layout.addLayout(table_ops)

        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        table_layout.addWidget(self.table, 1)
        splitter.addWidget(table_widget)

        # ── 键盘快捷键 ──────────────────────────────────────────
        QShortcut(QKeySequence("Return"), self.table, self._key_next_row)
        QShortcut(QKeySequence("1"), self.table, self._key_enable)
        QShortcut(QKeySequence("2"), self.table, self._key_disable)

        splitter.setSizes([250, 400])
        self.content.addWidget(splitter, 1)

        self._refresh_project_list()

    def showEvent(self, event):
        super().showEvent(event)
        old = self.project_combo.currentText()
        self._refresh_project_list()
        if old:
            idx = self.project_combo.findText(old)
            if idx >= 0:
                self.project_combo.setCurrentIndex(idx)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.dm.close()

    # ── 项目 ────────────────────────────────────────────────────

    def _refresh_project_list(self):
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        for p in self.dm.list_projects():
            name = p.get("name", "?")
            total = p.get("total_records", 0)
            self.project_combo.addItem(f"{name} ({total} 条)", name)
        self.project_combo.blockSignals(False)
        if self.project_combo.count() > 0:
            self.project_combo.setCurrentIndex(0)
            self._load_project(self.project_combo.currentData())
        else:
            self.summary_label.setText("暂无项目，请先导入")

    def _on_project_changed(self, text):
        if text:
            name = self.project_combo.currentData()
            if name:
                self._load_project(name)

    def _load_project(self, name):
        try:
            self.dm.load_project(name)
            s = self.dm.summary()
            self.summary_label.setText(
                f"总 {s['total']}  |  启用 {s['enabled']}  |  禁用 {s['disabled']}"
            )
            # 动态更新电压下拉（按连续段分组，按时间排序）
            self.volt_filter.blockSignals(True)
            self.volt_filter.clear()
            self.volt_filter.addItem("全部")
            cur = self.dm._conn.cursor()
            cur.execute("SELECT id, actual_voltage, system_time FROM records ORDER BY id")
            rows = cur.fetchall()
            segments = []  # [(v, id_start, id_end, time_start), ...]
            for rid, v, ts in rows:
                if segments and segments[-1][0] == v:
                    segments[-1] = (v, segments[-1][1], rid, segments[-1][3])
                else:
                    segments.append((v, rid, rid, ts))
            from collections import Counter
            seg_counter = Counter()
            for v, _, _, _ in segments:
                seg_counter[v] += 1
            seg_index = Counter()
            for v, id_start, id_end, first_ts in segments:
                seg_index[v] += 1
                time_str = str(first_ts)[:16] if first_ts else ""
                label = f"{v:+.0f}V ({seg_index[v]}/{seg_counter[v]}) {time_str}"
                where = f"actual_voltage = {v} AND id BETWEEN {id_start} AND {id_end}"
                self.volt_filter.addItem(label, where)
            self.volt_filter.blockSignals(False)
            self.volt_filter.setCurrentIndex(0)
            self._refresh_table()
            self._clear_wave()
        except Exception as e:
            self.summary_label.setText(f"加载失败: {e}")

    # ── 筛选 ────────────────────────────────────────────────────

    def _on_filter_changed(self):
        self._refresh_table()

    def _build_where(self) -> str:
        conds = []
        where = self.volt_filter.currentData()
        if where is not None:
            conds.append(f"({where})")
        s = self.status_filter.currentText()
        if s == "启用":
            conds.append("enabled = 1")
        elif s == "禁用":
            conds.append("enabled = 0")
        return " AND ".join(conds) if conds else "1=1"

    def _get_selected_where(self) -> str:
        where = self.volt_filter.currentData()
        if where is None:
            raise ValueError("未选择具体电压")
        return where

    def _disable_voltage_group(self):
        try:
            where = self._get_selected_where()
        except ValueError:
            self.status_label.setText("请先选择具体电压")
            return
        self.dm.disable_records(where)
        self.status_label.setText(f"已禁用该段数据")
        self._refresh_summary_and_table()

    def _enable_voltage_group(self):
        try:
            where = self._get_selected_where()
        except ValueError:
            self.status_label.setText("请先选择具体电压")
            return
        self.dm.enable_records(where)
        self.status_label.setText(f"已启用该段数据")
        self._refresh_summary_and_table()

    def _refresh_summary_and_table(self):
        s = self.dm.summary()
        self.summary_label.setText(f"总 {s['total']}  |  启用 {s['enabled']}  |  禁用 {s['disabled']}")
        self._refresh_table()

    # ── 波形 ────────────────────────────────────────────────────

    def _clear_wave(self):
        self.ax.clear()
        self.ax.set_xlabel("采样点")
        self.ax.set_ylabel("幅值")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("")
        self.canvas.draw()

    def _show_waveform(self, record_id=None):
        if record_id is None:
            text = self.wave_id_edit.text().strip()
            if not text:
                return
            try:
                record_id = int(text)
            except ValueError:
                self.status_label.setText("ID 必须是数字")
                return

        wave = self.dm.get_waveform(record_id)
        if wave is None:
            self.status_label.setText(f"ID {record_id} 波形未找到")
            return

        # 计算拟合正弦波
        fitted = None
        try:
            n = len(wave)
            wc = wave - np.mean(wave)
            fft_vals = np.fft.rfft(wc)
            mag = np.abs(fft_vals[1:])
            if len(mag) > 3:
                fund_idx = np.argmax(mag[: n // 6]) + 1
                a1 = mag[fund_idx - 1]
                phase = np.angle(fft_vals[fund_idx])
                fitted = 2 * a1 / n * np.cos(2 * np.pi * fund_idx * np.arange(n) / n + phase)
        except Exception:
            pass

        # 查记录信息
        df = self.dm.query(
            fields=["id", "system_time", "actual_voltage", "temperature", "humidity",
                    "harm_a1", "harm_a2", "harm_error"],
            where=f"id={record_id}", enabled_only=False,
        )

        self.ax.clear()
        self.ax.plot(wave, linewidth=0.8, color="#333", label="原始")
        if fitted is not None:
            self.ax.plot(fitted + np.mean(wave), linewidth=1.2, color="#e74c3c",
                         linestyle="--", label="拟合 (基频)")
        self.ax.set_xlabel("采样点")
        self.ax.set_ylabel("幅值")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(fontsize=9)

        if not df.empty:
            r = df.iloc[0]
            title = f"ID={record_id}  电压={r['actual_voltage']}V"
            if pd.notna(r.get('harm_cycles')):
                title += (
                    f"  {r['harm_cycles']:.1f}周期"
                    f"  A1={r['harm_a1']:.0f}  A2={r['harm_a2']:.0f}"
                    f"  THD={r['harm_thd']:.2f}"
                    f"  噪声={r['harm_noise_pct']:.0%}"
                )
            if pd.notna(r['system_time']):
                title += f"  {r['system_time']}"
            self.ax.set_title(title)
            self.wave_id_edit.setText(str(record_id))
        else:
            self.ax.set_title(f"波形 ID={record_id}")

        self.canvas.draw()
        self.status_label.setText(f"显示波形 ID={record_id}")

    def _on_table_selection_changed(self):
        rows = self.table.selectionModel().selectedRows()
        # 单选时才显示波形
        if len(rows) == 1:
            item = self.table.item(rows[0].row(), 1)  # ID 在第 2 列
            if item:
                try:
                    self._show_waveform(int(item.text()))
                except ValueError:
                    pass

    # ── pyecharts 趋势图 ────────────────────────────────────────

    def _get_data(self):
        df = self.dm.query(
            fields=["id", "system_time", "actual_voltage", "temperature", "humidity",
                    "enabled", "harm_a1", "harm_a2", "harm_error",
                    "harm_cycles", "harm_thd", "harm_noise_pct"],
            where=self._build_where(),
            enabled_only=False, order_by="system_time",
        )
        return df if not df.empty else None

    def _open_chart(self):
        df = self._get_data()
        if df is None:
            self.status_label.setText("无数据可显示")
            return
        n = len(df)
        x = [str(i) for i in range(1, n + 1)]

        line_volt = (Line().add_xaxis(x)
            .add_yaxis("电压 (V)", df["actual_voltage"].round(1).tolist(), is_smooth=True, symbol="none")
            .set_global_opts(title_opts=opts.TitleOpts(title="电压"),
                yaxis_opts=opts.AxisOpts(name="电压 (V)"),
                datazoom_opts=[opts.DataZoomOpts(xaxis_index=[0, 1, 2])],
                tooltip_opts=opts.TooltipOpts(trigger="axis")))
        line_temp = (Line().add_xaxis(x)
            .add_yaxis("温度 (°C)", df["temperature"].round(1).tolist(), is_smooth=True, symbol="none")
            .set_global_opts(title_opts=opts.TitleOpts(title="温度"),
                yaxis_opts=opts.AxisOpts(name="温度 (°C)"),
                tooltip_opts=opts.TooltipOpts(trigger="axis")))
        line_humid = (Line().add_xaxis(x)
            .add_yaxis("湿度 (%)", df["humidity"].round(1).tolist(), is_smooth=True, symbol="none")
            .set_global_opts(title_opts=opts.TitleOpts(title="湿度"),
                yaxis_opts=opts.AxisOpts(name="湿度 (%)"),
                tooltip_opts=opts.TooltipOpts(trigger="axis")))

        grid = (Grid(init_opts=opts.InitOpts(width="1200px", height="700px"))
            .add(line_volt, grid_opts=opts.GridOpts(pos_top="3%", pos_bottom="68%"))
            .add(line_temp, grid_opts=opts.GridOpts(pos_top="35%", pos_bottom="35%"))
            .add(line_humid, grid_opts=opts.GridOpts(pos_top="67%", pos_bottom="3%")))

        tmp = os.path.join(tempfile.gettempdir(), "swa_explorer.html")
        grid.render(tmp)
        webbrowser.open(f"file://{tmp}")
        self.status_label.setText(f"趋势图已打开 ({n} 条)")

    def _run_quality_check(self):
        """运行波形质量检测。"""
        if self.dm._conn is None:
            self.status_label.setText("请先加载项目")
            return
        self.status_label.setText("质量检测中...")
        self.quality_btn.setEnabled(False)
        # 让 UI 刷新
        from PySide6.QtCore import QTimer
        QTimer.singleShot(50, self._do_quality_check)

    def _do_quality_check(self):
        try:
            n = self.dm.run_quality_check()
            s = self.dm.summary()
            self.summary_label.setText(f"总 {s['total']}  |  启用 {s['enabled']}  |  禁用 {s['disabled']}")
            self.status_label.setText(f"质量检测完成，已禁用 {n} 条坏数据")
            self._refresh_table()
        except Exception as e:
            self.status_label.setText(f"质量检测失败: {e}")
        finally:
            self.quality_btn.setEnabled(True)

    def _backfill_harmonics(self):
        """回填已有项目中的谐波字段。"""
        if self.dm._conn is None:
            return
        self.status_label.setText("回填中...")
        self.backfill_btn.setEnabled(False)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(50, self._do_backfill)

    def _do_backfill(self):
        try:
            n = self.dm.backfill_harmonics()
            self.status_label.setText(f"回填完成: {n} 条")
            self._refresh_table()
        except Exception as e:
            self.status_label.setText(f"回填失败: {e}")
        finally:
            self.backfill_btn.setEnabled(True)

    # ── 表格 ────────────────────────────────────────────────────

    def _refresh_table(self):
        df = self._get_data()
        if df is None:
            self.table.setRowCount(0)
            return
        cols = ["enabled", "id", "system_time", "actual_voltage", "temperature", "humidity",
                "harm_cycles", "harm_a1", "harm_a2", "harm_error",
                "harm_thd", "harm_noise_pct"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(["启用", "ID", "时间", "电压(V)", "温度", "湿度",
                                             "周期", "A1", "A2", "误差",
                                             "THD", "噪声%"])
        self.table.setRowCount(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            for j, col in enumerate(cols):
                val = row.get(col)
                if col in ("harm_a1", "harm_a2", "harm_error", "harm_thd"):
                    text = f"{val:.2f}" if pd.notna(val) else ""
                elif col == "harm_noise_pct":
                    text = f"{val:.1%}" if pd.notna(val) else ""
                elif col == "harm_cycles":
                    text = f"{val:.1f}" if pd.notna(val) else ""
                elif col == "enabled":
                    text = str(int(val)) if pd.notna(val) else ""
                else:
                    text = str(val) if val is not None else ""
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if col == "enabled":
                    item.setBackground(Qt.green if val == 1 else Qt.red)
                self.table.setItem(i, j, item)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)

    # ── 行级质量操作 ────────────────────────────────────────────

    def _get_selected_ids(self) -> list[int]:
        rows = set()
        for item in self.table.selectedItems():
            rows.add(item.row())
        ids = []
        for r in rows:
            item = self.table.item(r, 1)  # ID 在第 2 列（索引 1）
            if item:
                try:
                    ids.append(int(item.text()))
                except ValueError:
                    pass
        return ids

    def _disable_selected(self):
        ids = self._get_selected_ids()
        if not ids:
            return
        self.dm.disable_records(f"id IN ({','.join(map(str, ids))})")
        self.status_label.setText(f"已禁用 {len(ids)} 条")
        s = self.dm.summary()
        self.summary_label.setText(f"总 {s['total']}  |  启用 {s['enabled']}  |  禁用 {s['disabled']}")
        self._refresh_table()

    def _enable_selected(self):
        ids = self._get_selected_ids()
        if not ids:
            return
        self.dm.enable_records(f"id IN ({','.join(map(str, ids))})")
        self.status_label.setText(f"已启用 {len(ids)} 条")
        s = self.dm.summary()
        self.summary_label.setText(f"总 {s['total']}  |  启用 {s['enabled']}  |  禁用 {s['disabled']}")
        self._refresh_table()

    # ── 键盘快捷键 ────────────────────────────────────────────

    def _key_next_row(self):
        """回车 → 下一行。"""
        row = self.table.currentRow()
        if row < self.table.rowCount() - 1:
            self.table.selectRow(row + 1)

    def _key_enable(self):
        """1 → 启用选中行。"""
        self._enable_selected()

    def _key_disable(self):
        """2 → 禁用选中行。"""
        self._disable_selected()

    # ── 同步到 jsonl ─────────────────────────────────────────

    def _sync_to_jsonl(self):
        """将当前项目的启用状态写回 local.jsonl。"""
        if self.dm._conn is None or not os.path.exists(LOCAL_JSONL):
            self.status_label.setText("请先加载项目")
            return

        # 读取数据库所有记录的 id 和 enabled 状态（按 id 排序）
        cur = self.dm._conn.cursor()
        cur.execute("SELECT id, enabled FROM records ORDER BY id")
        id_enabled = dict(cur.fetchall())

        # 读取 jsonl，按行号映射到 record id
        with open(LOCAL_JSONL, "r", encoding="utf-8") as f:
            lines = f.readlines()

        updated = 0
        for i, line in enumerate(lines):
            record_id = i + 1  # 第 0 行 → id=1
            if record_id in id_enabled:
                rec = json.loads(line)
                new_enabled = id_enabled[record_id]
                if rec.get("ENABLED") != new_enabled:
                    rec["ENABLED"] = new_enabled
                    lines[i] = json.dumps(rec, ensure_ascii=False) + "\n"
                    updated += 1

        if updated > 0:
            with open(LOCAL_JSONL, "w", encoding="utf-8") as f:
                f.writelines(lines)
            self.status_label.setText(f"已同步 {updated} 条到 local.jsonl")
        else:
            self.status_label.setText("无需同步")
