"""
检测页面 — 选择项目/检测方法/电压分配，运行检测并查看结果。
"""

import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QListWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QSplitter, QAbstractItemView,
    QMessageBox, QPlainTextEdit, QSpinBox, QListWidgetItem,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from matplotlib.figure import Figure
import matplotlib
matplotlib.use("QtAgg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from swa.data.manager import DataManager
from swa.detection import METHODS
from swa.ui.widgets.base_page import BasePage

import numpy as np


class DetectionPage(BasePage):
    """检测页面 — 选择方法、分配电压、运行检测。"""

    def __init__(self):
        super().__init__("电压检测")
        self.dm = DataManager()
        self._all_voltages: list[float] = []
        self._project_dir: str | None = None
        self._last_result: dict | None = None

        # ── 顶栏：项目 + 方法选择 ──────────────────────────────
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("项目:"))
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(200)
        top_row.addWidget(self.project_combo)
        top_row.addSpacing(16)

        top_row.addWidget(QLabel("检测方法:"))
        self.method_combo = QComboBox()
        self.method_combo.addItem("最小二乘法 (LS)", "least_squares")
        self.method_combo.setMinimumWidth(160)
        top_row.addWidget(self.method_combo)
        top_row.addSpacing(16)

        self.load_btn = QPushButton("加载项目")
        self.load_btn.clicked.connect(self._load_project)
        top_row.addWidget(self.load_btn)
        top_row.addStretch()
        self.content.addLayout(top_row)

        # ── 电压分配 ────────────────────────────────────────────
        split = QSplitter(Qt.Horizontal)

        # 训练集
        train_group = QGroupBox("训练电压")
        train_layout = QVBoxLayout(train_group)
        self.train_list = QListWidget()
        self.train_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        train_layout.addWidget(self.train_list)
        split.addWidget(train_group)

        # 控制按钮1（训练↔测试）
        ctrl1 = QGroupBox("操作")
        c1 = QVBoxLayout(ctrl1)
        c1.setAlignment(Qt.AlignCenter)
        btn = QPushButton("→ 测试")
        btn.setFixedWidth(100)
        btn.clicked.connect(lambda: self._move_items(self.train_list, self.test_list))
        c1.addWidget(btn)
        btn = QPushButton("← 训练")
        btn.setFixedWidth(100)
        btn.clicked.connect(lambda: self._move_items(self.test_list, self.train_list))
        c1.addWidget(btn)
        c1.addStretch()
        split.addWidget(ctrl1)

        # 测试集
        test_group = QGroupBox("测试电压")
        test_layout = QVBoxLayout(test_group)
        self.test_list = QListWidget()
        self.test_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        test_layout.addWidget(self.test_list)
        split.addWidget(test_group)

        # 控制按钮2（测试↔禁用）
        ctrl2 = QGroupBox("操作")
        c2 = QVBoxLayout(ctrl2)
        c2.setAlignment(Qt.AlignCenter)
        btn = QPushButton("→ 禁用")
        btn.setFixedWidth(100)
        btn.clicked.connect(lambda: self._move_items(self.test_list, self.discard_list))
        c2.addWidget(btn)
        btn = QPushButton("← 测试")
        btn.setFixedWidth(100)
        btn.clicked.connect(lambda: self._move_items(self.discard_list, self.test_list))
        c2.addWidget(btn)
        c2.addStretch()
        split.addWidget(ctrl2)

        # 禁用集
        discard_group = QGroupBox("禁用电压 (不参与计算)")
        discard_layout = QVBoxLayout(discard_group)
        self.discard_list = QListWidget()
        self.discard_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        discard_layout.addWidget(self.discard_list)
        split.addWidget(discard_group)

        split.setSizes([180, 80, 180, 80, 180])
        self.content.addWidget(split, 2)

        # ── 参数 + 运行 ────────────────────────────────────────
        param_row = QHBoxLayout()

        param_row.addWidget(QLabel("滑动窗口大小:"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 200)
        self.window_spin.setValue(8)
        self.window_spin.setToolTip(
            ">1 时对同电压连续记录做滑动窗口平均，窗口内各条指标取平均作为一条样本"
        )
        self.window_spin.setFixedWidth(70)
        param_row.addWidget(self.window_spin)
        param_row.addSpacing(8)
        self.window_info = QLabel("(8条→1个窗口)")
        param_row.addWidget(self.window_info)
        self.window_spin.valueChanged.connect(
            lambda v: self.window_info.setText(f"({v}条→1个窗口)" if v > 1 else "(1=不启用)")
        )

        param_row.addSpacing(24)
        param_row.addWidget(QLabel("每电压上限:"))
        self.max_samples_spin = QSpinBox()
        self.max_samples_spin.setRange(0, 99999)
        self.max_samples_spin.setValue(0)
        self.max_samples_spin.setToolTip("每个电压等级最多使用的样本数（0=不限）")
        self.max_samples_spin.setFixedWidth(80)
        param_row.addWidget(self.max_samples_spin)
        param_row.addSpacing(8)
        self.max_samples_info = QLabel("(0=不限)")
        param_row.addWidget(self.max_samples_info)
        self.max_samples_spin.valueChanged.connect(
            lambda v: self.max_samples_info.setText(f"(上限{v}条)" if v > 0 else "(0=不限)")
        )

        param_row.addSpacing(24)
        param_row.addWidget(QLabel("设备:"))
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(100)
        self.device_combo.setToolTip("按设备 ID 过滤，选择具体设备时只分析该设备的数据")
        param_row.addWidget(self.device_combo)

        self.mapping_cb = QPushButton("映射校准")
        self.mapping_cb.setCheckable(True)
        self.mapping_cb.setChecked(True)
        self.mapping_cb.setFixedHeight(24)
        self.mapping_cb.setToolTip("启用后，将非基准设备的 A1 通过电压映射校正到基准设备水平")
        self.mapping_cb.toggled.connect(self._on_mapping_toggled)
        param_row.addWidget(self.mapping_cb)

        self.ref_device_combo = QComboBox()
        self.ref_device_combo.setMinimumWidth(100)
        self.ref_device_combo.setToolTip("选择作为基准的设备")
        self.ref_device_combo.setVisible(False)
        param_row.addWidget(self.ref_device_combo)

        param_row.addSpacing(12)

        param_row.addWidget(QLabel("湿度校正:"))
        self.hum_corr_combo = QComboBox()
        self.hum_corr_combo.addItems(["禁用", "启用"])
        self.hum_corr_combo.setCurrentIndex(0)
        self.hum_corr_combo.setFixedWidth(80)
        self.hum_corr_combo.setToolTip("启用后对 A1 进行湿度漂移校正（实验性）")
        param_row.addWidget(self.hum_corr_combo)

        param_row.addSpacing(8)

        param_row.addWidget(QLabel("噪声校正:"))
        self.noise_corr_combo = QComboBox()
        self.noise_corr_combo.addItems(["禁用", "启用"])
        self.noise_corr_combo.setCurrentIndex(0)
        self.noise_corr_combo.setFixedWidth(80)
        self.noise_corr_combo.setToolTip("启用后对 A1 进行噪声校正: A1_clean = A1 × (1-noise_pct)")
        param_row.addWidget(self.noise_corr_combo)

        param_row.addSpacing(8)

        param_row.addWidget(QLabel("削波:"))
        self.clip_device_list = QListWidget()
        self.clip_device_list.setMaximumHeight(56)
        self.clip_device_list.setToolTip("勾选需要启用削波矫正的设备（仅选定的设备进行 A1 矫正）")
        param_row.addWidget(self.clip_device_list)

        param_row.addSpacing(12)

        param_row.addWidget(QLabel("回填到:"))
        self.buffer_combo = QComboBox()
        for i in range(1, 6):
            self.buffer_combo.addItem(f"缓冲区 {i}", i)
        self.buffer_combo.setFixedWidth(100)
        self.buffer_combo.setToolTip("选择将预测结果回填到哪个缓冲区（1-5）")
        param_row.addWidget(self.buffer_combo)

        self.backfill_btn = QPushButton("回填预测值")
        self.backfill_btn.setFixedHeight(24)
        self.backfill_btn.setEnabled(False)
        self.backfill_btn.setToolTip("将预测结果写回数据库指定缓冲区，可在数据探索页查看")
        self.backfill_btn.clicked.connect(self._backfill_predicted)
        param_row.addWidget(self.backfill_btn)

        param_row.addStretch()

        self.run_btn = QPushButton("运行检测")
        self.run_btn.setFixedHeight(36)
        self.run_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.run_btn.clicked.connect(self._run_detection)
        param_row.addWidget(self.run_btn)
        self.content.addLayout(param_row)

        # ── 结果 ────────────────────────────────────────────────
        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout(result_group)

        self.result_table = QTableWidget(0, 7)
        self.result_table.setHorizontalHeaderLabels([
            "指标", "训练集", "测试集", "", "", "", ""
        ])
        self.result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 7):
            self.result_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
        result_layout.addWidget(self.result_table, 1)

        # 详细日志
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setPlaceholderText("运行日志...")
        result_layout.addWidget(self.log_text)

        self.content.addWidget(result_group, 2)

        # ── 初始化 ──────────────────────────────────────────────
        self._refresh_projects()

    # ── 项目列表刷新 ────────────────────────────────────────────

    def _refresh_projects(self):
        self.project_combo.clear()
        projects = self.dm.list_projects()
        for p in projects:
            name = p.get("name", "?")
            total = p.get("total_records", 0)
            enabled = p.get("enabled_records", 0)
            self.project_combo.addItem(f"{name}  ({enabled}/{total})", name)

    # ── 加载项目 ────────────────────────────────────────────────

    def _on_mapping_toggled(self, checked: bool):
        self.ref_device_combo.setVisible(checked)

    def _load_project(self):
        name = self.project_combo.currentData()
        if not name:
            QMessageBox.warning(self, "提示", "请先选择项目")
            return

        try:
            self.dm.load_project(name)
            meta = self.dm.summary()
            self._project_dir = os.path.join(self.dm.projects_dir, name)

            # 读取所有电压等级及其条数
            cur = self.dm._conn.cursor()
            cur.execute(
                "SELECT actual_voltage, COUNT(*) AS cnt FROM records WHERE enabled=1 GROUP BY actual_voltage ORDER BY actual_voltage"
            )
            rows = cur.fetchall()
            self._all_voltages = [r[0] for r in rows]
            self._voltage_counts = {r[0]: r[1] for r in rows}

            # 填充设备下拉
            self.device_combo.clear()
            self.device_combo.addItem("全部设备", None)
            cur.execute("SELECT DISTINCT device_id FROM records WHERE enabled=1 AND device_id IS NOT NULL ORDER BY device_id")
            devices = [r[0] for r in cur.fetchall()]
            for did in devices:
                self.device_combo.addItem(f"设备 {did}", did)

            # 填充削波设备列表（多选，默认勾选 2539）
            self.clip_device_list.clear()
            for did in devices:
                item = QListWidgetItem(f"设备 {did[-4:]}")
                item.setData(Qt.UserRole, did)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if "2539" in did else Qt.Unchecked)
                self.clip_device_list.addItem(item)

            # 填充基准设备下拉
            self.ref_device_combo.clear()
            for did in devices:
                self.ref_device_combo.addItem(f"设备 {did}", did)

            # 填充电压列表：默认全部进训练集
            self.train_list.clear()
            self.test_list.clear()
            self.discard_list.clear()
            for v in self._all_voltages:
                cnt = self._voltage_counts.get(v, 0)
                self.train_list.addItem(f"{v:+.0f} V ({cnt}条)")

            # 恢复上次的电压分配（若保存过）
            self._load_voltage_config()

            self.log_text.setPlainText(
                f"已加载项目: {name}\n"
                f"总记录: {meta['total_records']}, 启用: {meta['enabled_records']}\n"
                f"电压等级: {len(self._all_voltages)} 个\n"
                f"请分配训练/测试电压后点击「运行检测」"
            )
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))
            self.log_text.setPlainText(f"加载失败: {e}")

    # ── 电压移动 ────────────────────────────────────────────────

    def _move_items(self, src: QListWidget, dst: QListWidget):
        items = src.selectedItems()
        if not items:
            return
        for item in items:
            dst.addItem(item.text())
            src.takeItem(src.row(item))
        self._save_voltage_config()

    # ── 电压分配持久化 ──────────────────────────────────────────

    def _save_voltage_config(self):
        """将当前电压 train/test/discard 分配保存到项目目录。"""
        if not self._project_dir:
            return
        config = {
            "train": [self.train_list.item(i).text() for i in range(self.train_list.count())],
            "test": [self.test_list.item(i).text() for i in range(self.test_list.count())],
            "discard": [self.discard_list.item(i).text() for i in range(self.discard_list.count())],
        }
        path = os.path.join(self._project_dir, "voltage_config.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # 静默失败，不影响主流程

    def _load_voltage_config(self):
        """从项目目录恢复电压 train/test/discard 分配。"""
        if not self._project_dir:
            return
        path = os.path.join(self._project_dir, "voltage_config.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            return

        self.train_list.clear()
        self.test_list.clear()
        self.discard_list.clear()

        for text in config.get("train", []):
            self.train_list.addItem(text)
        for text in config.get("test", []):
            self.test_list.addItem(text)
        for text in config.get("discard", []):
            self.discard_list.addItem(text)

    # ── 运行检测 ────────────────────────────────────────────────

    def _run_detection(self):
        if not self._project_dir:
            QMessageBox.warning(self, "提示", "请先加载项目")
            return

        if self.train_list.count() == 0:
            QMessageBox.warning(self, "提示", "训练电压不能为空")
            return

        method_key = self.method_combo.currentData()
        if method_key not in METHODS:
            QMessageBox.critical(self, "错误", f"未知检测方法: {method_key}")
            return

        # 解析电压值
        def parse_voltages(lst: QListWidget) -> list[float]:
            result = []
            for i in range(lst.count()):
                text = lst.item(i).text()
                # 格式: "+110 V (42条)" → 取 "V" 前面的部分
                v_str = text.split("V")[0].strip()
                result.append(float(v_str))
            return result

        window_size = self.window_spin.value()
        max_samples = self.max_samples_spin.value()
        device_id = self.device_combo.currentData()
        device_mapping = self.mapping_cb.isChecked()
        ref_device_id = self.ref_device_combo.currentData() if device_mapping else None
        train_v = parse_voltages(self.train_list)
        test_v = parse_voltages(self.test_list)
        discard_v = parse_voltages(self.discard_list)

        device_str = f"全部设备" if device_id is None else f"设备 {device_id}"
        mapping_str = f"  映射到基准 {ref_device_id}" if device_mapping else ""

        # 收集削波设备
        clip_devices = []
        for i in range(self.clip_device_list.count()):
            item = self.clip_device_list.item(i)
            if item.checkState() == Qt.Checked:
                clip_devices.append(item.data(Qt.UserRole))
        clip_str = ", ".join(d[-4:] for d in clip_devices) if clip_devices else "禁用"

        self.log_text.setPlainText(
            f"训练电压: {', '.join(f'{v:+.0f}V' for v in sorted(train_v))}\n"
            f"测试电压: {', '.join(f'{v:+.0f}V' for v in sorted(test_v)) if test_v else '(无)'}\n"
            f"禁用电压: {', '.join(f'{v:+.0f}V' for v in sorted(discard_v)) if discard_v else '(无)'}\n"
            f"设备: {device_str}{mapping_str}\n"
            f"滑动窗口: {'不启用' if window_size <= 1 else f'{window_size}条→1个窗口'}\n"
            f"每电压上限: {'不限' if max_samples <= 0 else f'{max_samples}条'}\n"
            f"噪声校正: {'启用' if self.noise_corr_combo.currentIndex() == 1 else '禁用'}\n"
            f"削波: {clip_str}\n"
            "运行中，请稍候...\n"
        )
        self.run_btn.setEnabled(False)

        try:
            result = METHODS[method_key].run(
                self._project_dir, train_v, test_v,
                window_size=window_size,
                max_samples_per_voltage=max_samples,
                device_id=device_id,
                device_mapping=device_mapping,
                ref_device_id=ref_device_id,
                humidity_correction=self.hum_corr_combo.currentIndex() == 1,
                noise_correction=self.noise_corr_combo.currentIndex() == 1,
                clip_correction=clip_devices if clip_devices else False,
            )

            if "error" in result:
                self.log_text.setPlainText(f"错误: {result['error']}")
                self._last_result = None
                self.backfill_btn.setEnabled(False)
                return

            self._last_result = result
            self.backfill_btn.setEnabled(True)
            self._show_results(result)
        except Exception as e:
            import traceback
            self.log_text.setPlainText(f"运行失败:\n{traceback.format_exc()}")
            QMessageBox.critical(self, "错误", str(e))
        finally:
            self.run_btn.setEnabled(True)

    # ── 显示结果 ────────────────────────────────────────────────

    def _show_results(self, result: dict):
        metrics = result["metrics"]
        train_m = metrics["train"]
        test_m = metrics["test"]

        # 填结果表
        rows = [
            ("样本数", f"{metrics['train_count']}", f"{metrics['test_count']}"),
            ("MAE (V)", f"{train_m['mae']:.3f}", f"{test_m['mae']:.3f}"),
            ("RMSE (V)", f"{train_m['rmse']:.3f}", f"{test_m['rmse']:.3f}"),
            ("R²", f"{train_m['r2']:.4f}", f"{test_m['r2']:.4f}"),
            ("MAPE (%)", f"{train_m['mape']:.2f}", f"{test_m['mape']:.2f}"),
        ]

        self.result_table.setRowCount(len(rows) + 1)  # +1 标题行
        for i, (label, train_val, test_val) in enumerate(rows):
            self.result_table.setItem(i, 0, QTableWidgetItem(label))
            self.result_table.setItem(i, 1, QTableWidgetItem(train_val))
            self.result_table.setItem(i, 2, QTableWidgetItem(test_val))
            for j in range(3, 7):
                self.result_table.setItem(i, j, QTableWidgetItem(""))

        # 系数行
        coeff = result.get("coefficients", {})
        intercept = result.get("intercept", 0)
        coeff_str = " + ".join(
            f"{v:.4f}×{k}" for k, v in sorted(coeff.items())
        )
        self.result_table.setItem(len(rows), 0, QTableWidgetItem("回归方程"))
        item = QTableWidgetItem(f"V = {intercept:.4f} + {coeff_str}")
        self.result_table.setSpan(len(rows), 1, 1, 6)
        self.result_table.setItem(len(rows), 1, item)

        # 窗口信息
        log_lines = []
        ws = result.get("window_size", 1)
        ms = result.get("max_samples_per_voltage", 0)
        dm = result.get("device_mapping", False)
        rd = result.get("ref_device_id", None)
        if ws > 1 or ms > 0 or dm:
            parts = []
            if ws > 1:
                parts.append(f"滑动窗口: {ws}条→1个窗口")
            if ms > 0:
                parts.append(f"每电压上限: {ms}条")
            if dm and rd:
                parts.append(f"设备映射 → {rd}")
            parts.append(f"样本数: {metrics['train_count']}(训练) / {metrics['test_count']}(测试)")
            log_lines.append(" | ".join(parts))
            log_lines.append("")

        log_lines.append("── 各测试电压 MAE ──")
        v_pred_mean = result.get("voltage_pred_mean", {})
        for v_label, v_mae in sorted(result.get("voltage_mae", {}).items(), key=lambda x: float(x[0].rstrip("V"))):
            mean_str = f"  预测均值={v_pred_mean.get(v_label, 0):.2f}V" if v_label in v_pred_mean else ""
            log_lines.append(f"  {v_label:>6s}: MAE={v_mae:.3f}V{mean_str}")

        log_lines.append("")
        log_lines.append("── 回归系数 ──")
        log_lines.append(f"  截距: {intercept:.4f}")
        for name, val in sorted(coeff.items()):
            log_lines.append(f"  {name}: {val:.4f}")

        # ── 弹出小提琴图 ─────────────────────────────────────
        test_results = result.get("test_results", [])
        train_results = result.get("train_results", [])
        if test_results or train_results:
            self._show_violin_popup(train_results, test_results)

        norm_params = result.get("norm_params", {})
        if norm_params:
            log_lines.append("")
            log_lines.append("── 归一化参数 (均值±标准差) ──")
            for name in sorted(norm_params.keys()):
                p = norm_params[name]
                log_lines.append(f"  {name}: {p['mean']:.4f} ± {p['std']:.4f}")

        self.log_text.setPlainText("\n".join(log_lines))

    def _backfill_predicted(self):
        """将预测结果回填到数据库的指定缓冲区。"""
        if self._last_result is None:
            return
        train_results = self._last_result.get("train_results", [])
        test_results = self._last_result.get("test_results", [])
        all_results = train_results + test_results
        if not all_results:
            QMessageBox.information(self, "提示", "没有可回填的预测结果")
            return
        buffer = self.buffer_combo.currentData()
        try:
            self.dm.backfill_predicted(all_results, buffer_index=buffer)
            n = len(all_results)
            self.log_text.appendPlainText(f"\n✓ 已回填 {n} 条记录到缓冲区 {buffer}")
            self.log_text.appendPlainText("可在「数据探索」页查看")
        except Exception as e:
            QMessageBox.critical(self, "回填失败", str(e))

    def _show_violin_popup(self, train_results: list[dict], test_results: list[dict]):
        """弹出独立窗口显示小提琴图 — N+1 个子图（每设备 + 总图）。"""
        from PySide6.QtWidgets import QDialog, QVBoxLayout
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.gridspec import GridSpec

        all_results = train_results + test_results
        if not all_results:
            return

        # 收集设备
        dev_ids = set()
        for r in all_results:
            did = r.get("device_id")
            if did:
                dev_ids.add(did)
        dev_ids = sorted(dev_ids)
        n_dev = len(dev_ids)
        n_plots = n_dev + 1  # 每个设备 + 总图

        dialog = QDialog(self)
        dialog.setWindowTitle("各电压预测分布 (小提琴图)")
        dialog.resize(1200, 300 * n_plots)

        layout = QVBoxLayout(dialog)

        fig = Figure(figsize=(11, 3.5 * n_plots))
        gs = GridSpec(n_plots, 1, figure=fig, hspace=0.4)

        def _plot_violin(ax, tr, te, title):
            """在指定 ax 上画小提琴图。"""
            plot_data = []
            for t in tr:
                plot_data.append({"电压": f"{t['actual']:.0f}V", "预测值(V)": t["pred"], "数据集": "训练"})
            for t in te:
                plot_data.append({"电压": f"{t['actual']:.0f}V", "预测值(V)": t["pred"], "数据集": "测试"})
            if not plot_data:
                ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
                return

            vf = pd.DataFrame(plot_data)
            cats = sorted(vf["电压"].unique(), key=lambda x: float(x.rstrip("V")))
            vf["电压"] = pd.Categorical(vf["电压"], categories=cats, ordered=True)
            sns.violinplot(data=vf, x="电压", y="预测值(V)", hue="数据集", ax=ax,
                           palette={"训练": "#A0D8F1", "测试": "#F4A582"},
                           split=False, density_norm="width")

            # 真值线和均值线
            for i, cat in enumerate(cats):
                v = float(cat.rstrip("V"))
                ax.hlines(v, i - 0.4, i + 0.4, colors="gray", linestyles="--",
                          linewidth=1.5, label="真值" if i == 0 else "")
                test_sub = [t["pred"] for t in te if abs(t["actual"] - v) < 1e-6]
                if test_sub:
                    ax.hlines(np.mean(test_sub), i - 0.4, i + 0.4,
                              colors="#D62728", linewidth=2, label="测试均值" if i == 0 else "")
                train_sub = [t["pred"] for t in tr if abs(t["actual"] - v) < 1e-6]
                if train_sub:
                    ax.hlines(np.mean(train_sub), i - 0.4, i + 0.4,
                              colors="#1F77B4", linewidth=2, label="训练均值" if i == 0 else "")

            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), fontsize=8)
            ax.set_title(title)
            ax.set_ylabel("预测电压 (V)")
            ax.tick_params(axis="x", labelsize=8)

        # ── 总图 ──
        ax_total = fig.add_subplot(gs[0])
        _plot_violin(ax_total, train_results, test_results, "所有设备 (汇总)")

        # ── 各设备图 ──
        for idx, dev in enumerate(dev_ids):
            short_dev = dev[-4:]
            ax = fig.add_subplot(gs[idx + 1])
            tr_dev = [r for r in train_results if r.get("device_id") == dev]
            te_dev = [r for r in test_results if r.get("device_id") == dev]
            _plot_violin(ax, tr_dev, te_dev, f"设备 {short_dev} ({len(tr_dev)+len(te_dev)} 条)")

        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)

        toolbar = NavigationToolbar2QT(canvas, dialog)
        layout.addWidget(toolbar)

        dialog.exec()
