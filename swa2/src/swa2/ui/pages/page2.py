"""页面 2 — 项目管理"""

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QListWidget, QListWidgetItem, QSplitter,
    QAbstractItemView, QInputDialog, QMessageBox, QProgressDialog,
    QCheckBox,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from swa2.ui.widgets.styles import STYLES
from swa2.data.project_manager import ProjectManager
from swa2.data.local_db import LocalDB


class Page2(QWidget):
    def __init__(self):
        super().__init__()
        self.pm = ProjectManager()
        self._creating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)

        # ── 标题 ──
        title = QLabel("项目管理")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        desc = QLabel("从本地数据库创建项目，筛选数据并计算特征")
        desc.setStyleSheet("color: #a6adc8; font-size: 13px;")
        layout.addWidget(desc)

        layout.addSpacing(16)

        # ── 分割布局 ──
        split = QSplitter(Qt.Horizontal)

        # 左侧：项目列表
        left_widget = QFrame()
        left_widget.setObjectName("left")
        left_widget.setStyleSheet("""
            QFrame#left {
                background-color: #1e1e2e;
                border: 1px solid #313244;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(8)

        left_layout.addWidget(QLabel("已有项目"))

        self.project_list = QListWidget()
        self.project_list.setAlternatingRowColors(False)
        self.project_list.setStyleSheet("""
            QListWidget {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 4px;
                padding: 4px;
            }
            QListWidget::item {
                color: #cdd6f4;
                padding: 8px 10px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #313244;
            }
            QListWidget::item:hover:!selected {
                background-color: #252536;
            }
        """)
        self.project_list.itemClicked.connect(self._on_project_clicked)
        left_layout.addWidget(self.project_list, 1)

        btn_row = QHBoxLayout()
        self.new_btn = QPushButton("新建项目")
        self.new_btn.setFixedHeight(32)
        self.new_btn.setStyleSheet(STYLES["btn_primary"])
        self.new_btn.clicked.connect(self._on_new_project)
        btn_row.addWidget(self.new_btn)

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setFixedHeight(32)
        self.refresh_btn.setStyleSheet(STYLES["btn_secondary"])
        self.refresh_btn.clicked.connect(self._refresh_list)
        btn_row.addWidget(self.refresh_btn)

        left_layout.addLayout(btn_row)

        del_row = QHBoxLayout()
        self.del_btn = QPushButton("删除项目")
        self.del_btn.setFixedHeight(28)
        self.del_btn.setStyleSheet("QPushButton { background-color: #b84a4a; color: #cdd6f4; border: none; border-radius: 4px; padding: 4px 12px; font-size: 12px; } QPushButton:hover { background-color: #d05c5c; }")
        self.del_btn.clicked.connect(self._delete_project)
        del_row.addWidget(self.del_btn)
        del_row.addStretch()
        left_layout.addLayout(del_row)
        split.addWidget(left_widget)

        # 右侧：项目详情
        right_widget = QFrame()
        right_widget.setObjectName("right")
        right_widget.setStyleSheet("""
            QFrame#right {
                background-color: #1e1e2e;
                border: 1px solid #313244;
                border-radius: 8px;
                padding: 20px;
            }
        """)
        right_layout = QVBoxLayout(right_widget)

        self.detail_title = QLabel("选择左侧项目查看详情")
        self.detail_title.setFont(QFont("Microsoft YaHei", 14))
        self.detail_title.setStyleSheet("color: #a6adc8; background: transparent;")
        right_layout.addWidget(self.detail_title)

        self.detail_info = QLabel("")
        self.detail_info.setStyleSheet("color: #cdd6f4; font-size: 13px; background: transparent;")
        right_layout.addWidget(self.detail_info)

        right_layout.addStretch()

        split.addWidget(right_widget)
        split.setSizes([260, 500])
        layout.addWidget(split, 1)

        # ── 替换策略按钮 ──
        policy_row = QHBoxLayout()
        self.policy_btn = QPushButton("替换策略")
        self.policy_btn.setFixedHeight(28)
        self.policy_btn.setStyleSheet(STYLES["btn_secondary"])
        self.policy_btn.clicked.connect(self._open_rule_dialog)
        policy_row.addWidget(self.policy_btn)
        self.policy_status = QLabel("4 条规则")
        self.policy_status.setStyleSheet("color: #a6adc8; font-size: 12px;")
        policy_row.addWidget(self.policy_status)
        policy_row.addStretch()
        left_layout.addLayout(policy_row)

        # 默认替换规则
        self._label_map = {
            "未知3": 72.0, "未知2": 36.0,
            "未知1": -43.0, "未知": -87.0,
        }

        self._refresh_list()

    def _refresh_list(self):
        self.project_list.clear()
        projects = self.pm.list_projects()
        for p in projects:
            name = p.get("name", "?")
            total = p.get("total_records", 0)
            created = p.get("created_at", "")[:10]
            item = QListWidgetItem(f"{name}  ({total}条)  {created}")
            item.setData(Qt.UserRole, name)
            self.project_list.addItem(item)

    def _delete_project(self):
        item = self.project_list.currentItem()
        if not item:
            QMessageBox.warning(self, "提示", "请先在左侧选择一个项目")
            return
        name = item.data(Qt.UserRole)
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除项目「{name}」吗？\n此操作不可恢复！",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.pm.delete_project(name)
        self.detail_title.setText("选择左侧项目查看详情")
        self.detail_info.setText("")
        self._refresh_list()

    def _on_project_clicked(self, item):
        name = item.data(Qt.UserRole)
        try:
            self.pm.load_project(name)
            s = self.pm.summary()
            meta_path = os.path.join(self.pm._project_dir, "meta.json")
            import json
            with open(meta_path) as f:
                meta = json.load(f)

            filters = meta.get("filters", {})
            opts = meta.get("options", {})

            skip_n = meta.get("skip_first_n", 0)
            label_map = meta.get("label_map", {}) or {}

            lines = [
                f"总记录: {s['total']} 条",
                f"启用: {s['enabled']}  |  禁用: {s['disabled']}",
                f"创建时间: {meta.get('created_at', '')[:19]}",
                "",
                "导入配置:",
                f"  跳过前 {skip_n} 条",
                f"  替换规则: {len(label_map)} 条",
                "",
                "筛选条件:",
                f"  设备: {filters.get('device_ids', '全部')}",
                f"  电压: {filters.get('voltage_list', '全部')}",
                "",
                "计算选项:",
                f"  谐波特征: {'✓' if opts.get('compute_harmonics') else '✗'}",
                f"  最小二乘评分: {'✓' if opts.get('compute_score') else '✗'}",
                f"  中值滤波: {'✓' if opts.get('median_filter') else '✗'}",
                f"  削波矫正: {'✓' if opts.get('clip_correction') else '✗'}",
            ]
            self.detail_title.setText(f"📁  {name}")
            self.detail_info.setText("\n".join(lines))
        except Exception as e:
            self.detail_info.setText(f"加载失败: {e}")

    # ── 新建项目 ──

    def _on_new_project(self):
        if self._creating:
            return

        # 检查 local.db 是否有数据
        db = LocalDB()
        db.connect()
        total = db.count()
        db.close()
        if total == 0:
            QMessageBox.warning(self, "提示", "本地数据库为空，请先下载数据")
            return

        # 获取设备和电压信息
        devices = self.pm.get_devices()
        voltages = self.pm.get_voltages()

        # 弹窗输入项目名
        name, ok = QInputDialog.getText(self, "新建项目", "项目名称：")
        if not ok or not name.strip():
            return
        name = name.strip()

        if not self._show_feature_dialog(name, devices, voltages):
            return

    def _show_feature_dialog(self, name: str, devices: list[str], voltages: list[float]) -> bool:
        """特征选择对话框。"""
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QDialogButtonBox, QGroupBox,
            QSpinBox, QScrollArea, QWidget,
        )

        dialog = QDialog(self)
        dialog.setWindowTitle(f"新建项目: {name}")
        dialog.setMinimumWidth(460)
        dialog.setMinimumHeight(500)
        dialog.setStyleSheet(STYLES["dialog"])

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        title = QLabel(f"项目: {name}")
        title.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        # ── 滚动区域 ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(12)

        # ── 筛选条件 ──
        filter_group = QGroupBox("数据筛选")
        filter_group.setStyleSheet("""
            QGroupBox {
                color: #cdd6f4;
                border: 1px solid #313244;
                border-radius: 6px;
                margin-top: 8px;
                padding: 12px 8px 8px 8px;
                font-size: 13px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 6px;
            }
        """)
        fg_layout = QVBoxLayout(filter_group)

        self._filter_device_cbs = []
        self._filter_device_cb_all = QCheckBox("全选")
        self._filter_device_cb_all.setChecked(True)
        self._filter_device_cb_all.toggled.connect(
            lambda checked: [cb.setChecked(checked) for cb in self._filter_device_cbs]
        )
        fg_layout.addWidget(self._filter_device_cb_all)

        for d in devices:
            cb = QCheckBox(f"设备 {d[-4:]}")
            cb.setChecked(True)
            cb.setProperty("userData", d)
            self._filter_device_cbs.append(cb)
            fg_layout.addWidget(cb)

        self._filter_voltage_cbs = []
        self._filter_voltage_cb_all = QCheckBox("全选")
        self._filter_voltage_cb_all.setChecked(True)
        self._filter_voltage_cb_all.toggled.connect(
            lambda checked: [cb.setChecked(checked) for cb in self._filter_voltage_cbs]
        )
        fg_layout.addWidget(self._filter_voltage_cb_all)

        for v in voltages:
            cb = QCheckBox(f"{v:+.0f}V")
            cb.setChecked(True)
            cb.setProperty("userData", v)
            self._filter_voltage_cbs.append(cb)
            fg_layout.addWidget(cb)

        scroll_layout.addWidget(filter_group)

        # ── 跳过条数 ──
        skip_row = QHBoxLayout()
        skip_row.addWidget(QLabel("跳过前"))
        self._skip_spin = QSpinBox()
        self._skip_spin.setRange(0, 9999999)
        self._skip_spin.setValue(85510)
        self._skip_spin.setFixedWidth(130)
        self._skip_spin.setStyleSheet(STYLES["spinbox"])
        skip_row.addWidget(self._skip_spin)
        skip_row.addWidget(QLabel("条记录，只导入之后的新数据"))
        skip_row.addStretch()
        scroll_layout.addLayout(skip_row)

        scroll_layout.addSpacing(4)

        # ── 计算选项 ──
        calc_group = QGroupBox("特征计算")
        calc_group.setStyleSheet(filter_group.styleSheet())
        cg_layout = QVBoxLayout(calc_group)

        self._opt_harmonics = QCheckBox("谐波特征 (A1, A2, error, cycles, noise)")
        self._opt_harmonics.setChecked(True)
        cg_layout.addWidget(self._opt_harmonics)

        self._opt_score = QCheckBox("最小二乘评分 (score)")
        self._opt_score.setChecked(True)
        cg_layout.addWidget(self._opt_score)

        self._opt_median = QCheckBox("中值滤波 (扫两次，窗口5)")
        self._opt_median.setChecked(False)
        cg_layout.addWidget(self._opt_median)

        self._opt_clip = QCheckBox("削波矫正 (检测削波并补偿 A1)")
        self._opt_clip.setChecked(True)
        cg_layout.addWidget(self._opt_clip)

        scroll_layout.addWidget(calc_group)
        scroll_layout.addStretch()

        # 滚动区域放入主布局
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        buttons.button(QDialogButtonBox.Ok).setText("开始创建")
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return False

        # 收集选中的设备和电压
        selected_devices = [
            cb.property("userData") for cb in self._filter_device_cbs if cb.isChecked()
        ]
        selected_voltages = [
            cb.property("userData") for cb in self._filter_voltage_cbs if cb.isChecked()
        ]

        # 如果没有筛选条件，传 None（全部）
        device_ids = selected_devices if len(selected_devices) < len(devices) else None
        voltage_list = selected_voltages if len(selected_voltages) < len(voltages) else None

        # 开始创建
        self._do_create_project(
            name=name,
            device_ids=device_ids,
            voltage_list=voltage_list,
            skip_first_n=self._skip_spin.value(),
            label_map=dict(self._label_map) if self._label_map else None,
            compute_harmonics=self._opt_harmonics.isChecked(),
            compute_score=self._opt_score.isChecked(),
            median_filter=self._opt_median.isChecked(),
            clip_correction=self._opt_clip.isChecked(),
        )
        return True

    # ── 替换规则弹窗 ──

    def _open_rule_dialog(self):
        """弹出替换规则管理对话框。"""
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QLabel,
            QTableWidget, QTableWidgetItem, QHeaderView,
            QPushButton, QInputDialog, QMessageBox,
        )

        dialog = QDialog(self)
        dialog.setWindowTitle("电压标签替换策略")
        dialog.setMinimumSize(400, 350)
        dialog.setStyleSheet(STYLES["dialog"])

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        title = QLabel("当 ACTUAL_VOLTAGE 是文字标签时，替换为电压值:")
        title.setStyleSheet("color: #a6adc8; font-size: 13px;")
        layout.addWidget(title)

        table = QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["标签文字", "替换电压 (V)"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.setStyleSheet("""
            QTableWidget {
                background-color: #181825;
                border: 1px solid #313244;
                color: #cdd6f4;
                gridline-color: #313244;
            }
            QTableWidget::item { padding: 4px; }
            QHeaderView::section {
                background-color: #181825;
                color: #a6adc8;
                border: none;
                padding: 6px;
            }
        """)
        layout.addWidget(table, 1)

        # 填充数据
        def refresh():
            table.setRowCount(0)
            for label, voltage in sorted(self._label_map.items(), key=lambda x: -len(x[0])):
                row = table.rowCount()
                table.insertRow(row)
                table.setItem(row, 0, QTableWidgetItem(label))
                item = QTableWidgetItem(f"{voltage:+.0f}")
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row, 1, item)
            # 更新规则数状态
            self.policy_status.setText(f"{len(self._label_map)} 条规则")
        refresh()

        # 按钮行
        btn_row = QHBoxLayout()

        def add_rule():
            label, ok = QInputDialog.getText(dialog, "添加规则", "输入标签文字:")
            if not ok or not label.strip():
                return
            label = label.strip()
            if label in self._label_map:
                QMessageBox.warning(dialog, "提示", f"标签 '{label}' 已存在")
                return
            voltage_str, ok = QInputDialog.getText(dialog, "添加规则", "输入替换电压值:")
            if not ok or not voltage_str.strip():
                return
            try:
                voltage = float(voltage_str.strip())
            except ValueError:
                QMessageBox.warning(dialog, "错误", "电压值必须是数字")
                return
            self._label_map[label] = voltage
            refresh()

        def delete_rule():
            row = table.currentRow()
            if row < 0:
                return
            label = table.item(row, 0).text()
            if label in self._label_map:
                del self._label_map[label]
            refresh()

        def reset_rules():
            self._label_map = {
                "未知3": 72.0, "未知2": 36.0,
                "未知1": -43.0, "未知": -87.0,
            }
            refresh()

        add_btn = QPushButton("+ 添加规则")
        add_btn.setStyleSheet(STYLES["btn_secondary"])
        add_btn.clicked.connect(add_rule)
        btn_row.addWidget(add_btn)

        del_btn = QPushButton("删除选中")
        del_btn.setStyleSheet(STYLES["btn_secondary"])
        del_btn.clicked.connect(delete_rule)
        btn_row.addWidget(del_btn)

        reset_btn = QPushButton("恢复默认")
        reset_btn.setStyleSheet(STYLES["btn_secondary"])
        reset_btn.clicked.connect(reset_rules)
        btn_row.addWidget(reset_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # 关闭按钮
        from PySide6.QtWidgets import QDialogButtonBox
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.exec()

    def _do_create_project(self, **kwargs):
        self._creating = True
        self.new_btn.setEnabled(False)

        # 获取总数
        db = LocalDB()
        db.connect()
        total = db.count()
        db.close()

        from PySide6.QtWidgets import QApplication

        progress = QProgressDialog("创建项目中...", "取消", 0, total, self)
        progress.setWindowTitle("新建项目")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.setStyleSheet("""
            QProgressDialog { background-color: #1e1e2e; color: #cdd6f4; }
            QProgressBar { background-color: #313244; border: none; border-radius: 4px; text-align: center; color: #cdd6f4; }
            QProgressBar::chunk { background-color: #89b4fa; border-radius: 4px; }
        """)

        def on_progress(current, total):
            progress.setValue(min(current, total))
            progress.setLabelText(f"正在创建... {current}/{total}")
            QApplication.processEvents()

        # 延迟执行让 UI 刷新
        QTimer.singleShot(100, lambda: self._do_create_worker(progress, kwargs, on_progress))

    def _do_create_worker(self, progress, kwargs, on_progress):
        try:
            meta = self.pm.create_project(
                name=kwargs["name"],
                device_ids=kwargs["device_ids"],
                voltage_list=kwargs["voltage_list"],
                skip_first_n=kwargs.get("skip_first_n", 85510),
                label_map=kwargs.get("label_map"),
                compute_harmonics=kwargs["compute_harmonics"],
                compute_score=kwargs["compute_score"],
                median_filter=kwargs["median_filter"],
                clip_correction=kwargs["clip_correction"],
                progress_callback=on_progress,
                canceled_check=lambda: progress.wasCanceled(),
            )
            progress.setValue(progress.maximum())
            QMessageBox.information(
                self, "创建完成",
                f"项目 '{kwargs['name']}' 创建成功\n"
                f"共 {meta['total_records']} 条记录"
            )
            self._refresh_list()
        except RuntimeError as e:
            if "用户取消了创建" in str(e):
                QMessageBox.information(self, "已取消", "项目创建已取消")
            else:
                QMessageBox.critical(self, "创建失败", str(e))
        except Exception as e:
            QMessageBox.critical(self, "创建失败", str(e))
        finally:
            self._creating = False
            self.new_btn.setEnabled(True)
