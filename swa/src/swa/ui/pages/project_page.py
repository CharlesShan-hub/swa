"""
第三页：项目管理 — 从 local.jsonl 导入到 SQLite 项目
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QPlainTextEdit, QLineEdit, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QInputDialog, QSpinBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from swa.data import DataManager
from swa.ui.widgets.base_page import BasePage

LOCAL_JSONL = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "data", "local.jsonl"
)

# 默认标签替换规则
DEFAULT_LABEL_MAP = {
    "未知3": "72",
    "未知2": "36",
    "未知1": "-43",
    "未知": "-87",
}


class ProjectPage(BasePage):
    """项目管理页面 — 从 local.jsonl 导入 SQLite 项目。"""

    def __init__(self):
        super().__init__("项目管理")
        self.dm = DataManager()
        self.label_map: dict[str, float] = {}

        # 标签替换规则
        rule_group = QGroupBox("电压标签替换规则")
        rule_layout = QVBoxLayout(rule_group)
        rule_layout.addWidget(QLabel("当 ACTUAL_VOLTAGE 是文字标签时，替换为电压值:"))
        self.rule_table = QTableWidget(0, 2)
        self.rule_table.setHorizontalHeaderLabels(["标签文字", "替换电压(V)"])
        self.rule_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.rule_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        rule_layout.addWidget(self.rule_table, 1)

        rule_btn_layout = QHBoxLayout()
        self.add_rule_btn = QPushButton("+ 添加规则")
        self.add_rule_btn.clicked.connect(self._add_rule)
        rule_btn_layout.addWidget(self.add_rule_btn)
        self.del_rule_btn = QPushButton("删除选中")
        self.del_rule_btn.clicked.connect(self._delete_rule)
        rule_btn_layout.addWidget(self.del_rule_btn)
        self.reset_rule_btn = QPushButton("恢复默认")
        self.reset_rule_btn.clicked.connect(self._reset_rules)
        rule_btn_layout.addWidget(self.reset_rule_btn)
        rule_btn_layout.addStretch()
        rule_layout.addLayout(rule_btn_layout)
        self.content.addWidget(rule_group)

        # 导入区
        import_group = QGroupBox("从 local.jsonl 导入")
        import_layout = QHBoxLayout(import_group)
        import_layout.addWidget(QLabel("项目名称:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("例如: week1")
        import_layout.addWidget(self.name_edit, 1)
        import_layout.addSpacing(16)
        import_layout.addWidget(QLabel("跳过前N条:"))
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(0, 100)
        self.skip_spin.setValue(10)
        self.skip_spin.setToolTip("每个电压等级前 N 条自动禁用（数据不稳定期）")
        self.skip_spin.setFixedWidth(60)
        import_layout.addWidget(self.skip_spin)
        self.import_btn = QPushButton("导入")
        self.import_btn.setFixedHeight(32)
        self.import_btn.clicked.connect(self._do_import)
        import_layout.addWidget(self.import_btn)
        self.content.addWidget(import_group)

        # 项目列表
        list_group = QGroupBox("已有项目")
        list_layout = QVBoxLayout(list_group)
        self.project_list = QListWidget()
        self.project_list.currentItemChanged.connect(self._on_project_selected)
        list_layout.addWidget(self.project_list)

        op_layout = QHBoxLayout()
        self.detail_btn = QPushButton("查看详情")
        self.detail_btn.clicked.connect(self._show_detail)
        op_layout.addWidget(self.detail_btn)
        self.backup_btn = QPushButton("备份项目")
        self.backup_btn.clicked.connect(self._backup_project)
        op_layout.addWidget(self.backup_btn)
        self.delete_btn = QPushButton("删除项目")
        self.delete_btn.clicked.connect(self._delete_project)
        op_layout.addWidget(self.delete_btn)
        op_layout.addStretch()
        list_layout.addLayout(op_layout)
        self.content.addWidget(list_group, 1)

        # 详情
        self.detail_text = QPlainTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(100)
        self.content.addWidget(self.detail_text)

        self._reset_label_map()
        self._refresh_list()

    # ── 标签规则管理 ────────────────────────────────────────────

    def _reset_label_map(self):
        """重置为默认规则。"""
        self.label_map = {k: float(v) for k, v in DEFAULT_LABEL_MAP.items()}
        self._refresh_rule_table()

    def _refresh_rule_table(self):
        self.rule_table.setRowCount(0)
        for label, voltage in sorted(self.label_map.items(), key=lambda x: -len(x[0])):
            row = self.rule_table.rowCount()
            self.rule_table.insertRow(row)
            self.rule_table.setItem(row, 0, QTableWidgetItem(label))
            item = QTableWidgetItem(f"{voltage:+.0f}")
            item.setTextAlignment(Qt.AlignCenter)
            self.rule_table.setItem(row, 1, item)

    def _add_rule(self):
        label, ok = QInputDialog.getText(self, "添加规则", "输入标签文字:")
        if not ok or not label.strip():
            return
        label = label.strip()

        if label in self.label_map:
            QMessageBox.warning(self, "提示", f"标签 '{label}' 已存在")
            return

        voltage_str, ok = QInputDialog.getText(self, "添加规则", "输入替换电压值:")
        if not ok or not voltage_str.strip():
            return
        try:
            voltage = float(voltage_str.strip())
        except ValueError:
            QMessageBox.warning(self, "错误", "电压值必须是数字")
            return

        self.label_map[label] = voltage
        self._refresh_rule_table()

    def _delete_rule(self):
        row = self.rule_table.currentRow()
        if row < 0:
            return
        label = self.rule_table.item(row, 0).text()
        reply = QMessageBox.question(
            self, "确认删除",
            f"删除规则「{label}」?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            del self.label_map[label]
            self._refresh_rule_table()

    def _reset_rules(self):
        reply = QMessageBox.question(
            self, "确认恢复",
            "恢复默认规则？\n（自定义规则将丢失）",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._reset_label_map()

    # ── 导入 ────────────────────────────────────────────────────

    def _do_import(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入项目名称")
            return
        if not os.path.exists(LOCAL_JSONL):
            QMessageBox.warning(self, "提示", "data/local.jsonl 不存在，请先下载")
            return

        try:
            meta = self.dm.create_project(
                name, LOCAL_JSONL,
                description="从 local.jsonl 导入",
                label_map=self.label_map,
                skip_first_n=self.skip_spin.value(),
            )
            QMessageBox.information(
                self, "导入完成",
                f"项目 '{name}' 导入成功\n"
                f"总记录: {meta['total_records']} 条\n"
                f"自动禁用: {self.skip_spin.value()} 条/电压\n"
                f"标签规则: {len(self.label_map)} 条"
            )
            self._refresh_list()
            self.name_edit.clear()
        except FileExistsError:
            QMessageBox.warning(self, "提示", f"项目 '{name}' 已存在")
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    # ── 项目操作 ────────────────────────────────────────────────

    def _refresh_list(self):
        self.project_list.clear()
        projects = self.dm.list_projects()
        for p in projects:
            name = p.get("name", "?")
            total = p.get("total_records", 0)
            enabled = p.get("enabled_records", 0)
            self.project_list.addItem(f"{name}  ({enabled}/{total})")

    def _on_project_selected(self, curr, prev):
        if curr:
            self.detail_text.setPlainText("选中项目后点击「查看详情」")

    def _show_detail(self):
        item = self.project_list.currentItem()
        if not item:
            return
        name = item.text().split("  (")[0]
        try:
            self.dm.load_project(name)
            s = self.dm.summary()
            lines = [
                f"项目: {s.get('name', 'N/A')}",
                f"描述: {s.get('description', 'N/A')}",
                f"来源: {s.get('source', 'N/A')}",
                f"创建时间: {s.get('created_at', 'N/A')}",
                f"总记录: {s['total']}",
                f"启用: {s['enabled']} ({s['enabled_pct']}%)",
                f"禁用: {s['disabled']}",
            ]
            # 显示标签映射
            lm = s.get("label_map", {})
            if lm:
                lines.append(f"标签规则: {len(lm)} 条")
                for k, v in lm.items():
                    lines.append(f"  {k} → {v}V")
            self.detail_text.setPlainText("\n".join(lines))
        except Exception as e:
            self.detail_text.setPlainText(f"加载失败: {e}")

    def _backup_project(self):
        item = self.project_list.currentItem()
        if not item:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return
        name = item.text().split("  (")[0]

        backup_name, ok = QInputDialog.getText(
            self, "备份项目",
            f"项目「{name}」将备份为:",
            text=f"{name}_bak",
        )
        if not ok or not backup_name.strip():
            return
        backup_name = backup_name.strip()

        try:
            self.dm.load_project(name)
            meta = self.dm.backup(backup_name)
            QMessageBox.information(
                self, "备份完成",
                f"项目 '{name}' 已备份为 '{backup_name}'\n"
                f"共 {meta.get('total_records', 0)} 条"
            )
            self._refresh_list()
        except FileExistsError:
            QMessageBox.warning(self, "提示", f"备份名 '{backup_name}' 已存在")
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def _delete_project(self):
        item = self.project_list.currentItem()
        if not item:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return
        name = item.text().split("  (")[0]
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除项目 '{name}' 吗？\n"
            f"此操作不可恢复，项目目录将被完全移除！",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # 二次确认
        confirm = QInputDialog.getText(
            self, "二次确认",
            f"请输入项目名称「{name}」以确认删除:",
        )
        if confirm[1] and confirm[0].strip() == name:
            self.dm.delete_project(name)
            self._refresh_list()
            self.detail_text.clear()
        else:
            QMessageBox.warning(self, "提示", "名称不匹配，已取消删除")
