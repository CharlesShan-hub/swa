"""
检测页面 — 选择项目/检测方法/电压分配，运行检测并查看结果。
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QListWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QSplitter, QAbstractItemView,
    QMessageBox, QPlainTextEdit, QSpinBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

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
        self.window_spin.setValue(1)
        self.window_spin.setToolTip(
            ">1 时对同电压连续记录做滑动窗口平均，窗口内各条指标取平均作为一条样本"
        )
        self.window_spin.setFixedWidth(70)
        param_row.addWidget(self.window_spin)
        param_row.addSpacing(8)
        self.window_info = QLabel("(1=不启用)")
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

            # 填充电压列表：默认全部进训练集
            self.train_list.clear()
            self.test_list.clear()
            self.discard_list.clear()
            for v in self._all_voltages:
                cnt = self._voltage_counts.get(v, 0)
                self.train_list.addItem(f"{v:+.0f} V ({cnt}条)")

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
        train_v = parse_voltages(self.train_list)
        test_v = parse_voltages(self.test_list)
        discard_v = parse_voltages(self.discard_list)

        self.log_text.setPlainText(
            f"训练电压: {', '.join(f'{v:+.0f}V' for v in sorted(train_v))}\n"
            f"测试电压: {', '.join(f'{v:+.0f}V' for v in sorted(test_v)) if test_v else '(无)'}\n"
            f"禁用电压: {', '.join(f'{v:+.0f}V' for v in sorted(discard_v)) if discard_v else '(无)'}\n"
            f"滑动窗口: {'不启用' if window_size <= 1 else f'{window_size}条→1个窗口'}\n"
            f"每电压上限: {'不限' if max_samples <= 0 else f'{max_samples}条'}\n"
            "运行中，请稍候...\n"
        )
        self.run_btn.setEnabled(False)

        try:
            result = METHODS[method_key].run(
                self._project_dir, train_v, test_v,
                window_size=window_size,
                max_samples_per_voltage=max_samples,
            )

            if "error" in result:
                self.log_text.setPlainText(f"错误: {result['error']}")
                return

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
        if ws > 1 or ms > 0:
            parts = []
            if ws > 1:
                parts.append(f"滑动窗口: {ws}条→1个窗口")
            if ms > 0:
                parts.append(f"每电压上限: {ms}条")
            parts.append(f"样本数: {metrics['train_count']}(训练) / {metrics['test_count']}(测试)")
            log_lines.append(" | ".join(parts))
            log_lines.append("")

        log_lines.append("── 各测试电压 MAE ──")
        for v_label, v_mae in sorted(result.get("voltage_mae", {}).items(), key=lambda x: float(x[0].rstrip("V"))):
            log_lines.append(f"  {v_label:>6s}: {v_mae:.3f} V")

        log_lines.append("")
        log_lines.append("── 回归系数 ──")
        log_lines.append(f"  截距: {intercept:.4f}")
        for name, val in sorted(coeff.items()):
            log_lines.append(f"  {name}: {val:.4f}")

        # 归一化参数
        norm_params = result.get("norm_params", {})
        if norm_params:
            log_lines.append("")
            log_lines.append("── 归一化参数 (均值±标准差) ──")
            for name in sorted(norm_params.keys()):
                p = norm_params[name]
                log_lines.append(f"  {name}: {p['mean']:.4f} ± {p['std']:.4f}")

        self.log_text.setPlainText("\n".join(log_lines))
