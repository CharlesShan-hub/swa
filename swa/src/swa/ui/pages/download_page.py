"""
第一页：数据下载 — 数据库连接 + 下载到 local.jsonl
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QPlainTextEdit, QGroupBox, QSpinBox, QDoubleSpinBox,
    QFormLayout, QLineEdit,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

from swa.ui.widgets.base_page import BasePage

LOCAL_JSONL = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "data", "local.jsonl"
)


class DownloadWorker(QThread):
    progress = Signal(int, int)
    log = Signal(str)
    finished = Signal(int)

    def __init__(self, conn, offset, limit, batch, sleep):
        super().__init__()
        self.conn = conn
        self.offset = offset
        self.limit = limit
        self.batch = batch
        self.sleep = sleep

    def run(self):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM YS_DB.TB_MODBUS_DEV_POINT")
        total = cur.fetchone()[0]
        self.log.emit(f"数据库总记录数: {total}")

        actual_limit = min(self.limit, total - self.offset)
        if actual_limit <= 0:
            self.log.emit("无新增数据")
            self.finished.emit(0)
            return

        cur.execute("SELECT MIN(ROWID) FROM YS_DB.TB_MODBUS_DEV_POINT")
        min_rowid = cur.fetchone()[0]

        fields = [
            "TEST_CASE_CODE", "SYSTEM_TIME", "RTU_REGS_SLAVE_ID",
            "RTU_REGS_P00_ROTOR_RPM", "RTU_REGS_P00_ENV_TEMP",
            "RTU_REGS_P00_ENV_HUMIDITY", "ACTUAL_VOLTAGE",
            "RTU_REGS_P00_WAVE_DATA",
        ]
        field_csv = ", ".join(fields)
        exported = 0
        total_batches = (actual_limit + self.batch - 1) // self.batch

        os.makedirs(os.path.dirname(LOCAL_JSONL) or ".", exist_ok=True)
        write_mode = "a" if self.offset > 0 else "w"

        with open(LOCAL_JSONL, write_mode, encoding="utf-8") as f:
            for batch_no in range(total_batches):
                this_batch = min(self.batch, actual_limit - exported)
                if this_batch <= 0:
                    break
                start_rowid = min_rowid + self.offset + exported - 1
                sql = f"SELECT {field_csv} FROM YS_DB.TB_MODBUS_DEV_POINT WHERE ROWID > ? ORDER BY ROWID LIMIT ?"
                cur.execute(sql, (start_rowid, this_batch))
                rows = cur.fetchall()
                if not rows:
                    break
                col_names = [desc[0] for desc in cur.description]
                for row in rows:
                    record = dict(zip(col_names, row))
                    if record.get("SYSTEM_TIME"):
                        record["SYSTEM_TIME"] = str(record["SYSTEM_TIME"])
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += len(rows)
                self.progress.emit(exported, actual_limit)
                self.log.emit(f"  第 {batch_no+1}/{total_batches} 批: {exported}/{actual_limit}")
                time.sleep(self.sleep)
        self.finished.emit(exported)


class DownloadPage(BasePage):
    """数据下载页面 — 连接数据库 + 下载到 local.jsonl。"""

    def __init__(self):
        super().__init__("数据下载")
        self.conn = None

        # 数据库连接
        conn_group = QGroupBox("数据库连接")
        conn_form = QFormLayout(conn_group)
        conn_form.setSpacing(6)

        ip_row = QHBoxLayout()
        self.ip_edit = QLineEdit("10.15.10.1")
        self.ip_edit.setFixedWidth(150)
        self.port_edit = QLineEdit("5256")
        self.port_edit.setFixedWidth(70)
        ip_row.addWidget(self.ip_edit)
        ip_row.addWidget(QLabel(":"))
        ip_row.addWidget(self.port_edit)
        ip_row.addStretch()
        conn_form.addRow("主机:", ip_row)

        self.user_edit = QLineEdit("SYSDBA")
        self.user_edit.setFixedWidth(150)
        conn_form.addRow("用户:", self.user_edit)

        self.pwd_edit = QLineEdit()
        self.pwd_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.pwd_edit.setFixedWidth(150)
        conn_form.addRow("密码:", self.pwd_edit)

        self.connect_btn = QPushButton("连接")
        self.connect_btn.setFixedHeight(26)
        self.connect_btn.clicked.connect(self._do_connect)
        conn_form.addRow("", self.connect_btn)

        self.conn_status = QLabel("未连接")
        self.conn_status.setStyleSheet("color: #888; font-size: 12px;")
        conn_form.addRow("", self.conn_status)

        self.content.addWidget(conn_group)

        # 本地文件信息
        info_group = QGroupBox("本地文件 (data/local.jsonl)")
        info_layout = QVBoxLayout(info_group)
        self.file_info = QLabel("文件: 不存在")
        self.file_info.setStyleSheet("color: #555;")
        info_layout.addWidget(self.file_info)
        self.content.addWidget(info_group)

        # 下载参数
        param_group = QGroupBox("下载参数")
        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("每批条数:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(10, 5000)
        self.batch_spin.setValue(500)
        self.batch_spin.setFixedWidth(90)
        param_row.addWidget(self.batch_spin)
        param_row.addSpacing(16)
        param_row.addWidget(QLabel("间隔(秒):"))
        self.sleep_spin = QDoubleSpinBox()
        self.sleep_spin.setRange(0, 10)
        self.sleep_spin.setSingleStep(0.1)
        self.sleep_spin.setValue(0.5)
        self.sleep_spin.setFixedWidth(70)
        param_row.addWidget(self.sleep_spin)
        param_row.addSpacing(16)
        param_row.addWidget(QLabel("条数:"))
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(0, 200000)
        self.limit_spin.setValue(0)
        self.limit_spin.setSpecialValueText("全部")
        self.limit_spin.setFixedWidth(90)
        param_row.addWidget(self.limit_spin)
        param_row.addStretch()
        param_group_layout = QVBoxLayout(param_group)
        param_group_layout.addLayout(param_row)
        self.content.addWidget(param_group)

        # 操作按钮
        btn_row = QHBoxLayout()
        self.download_btn = QPushButton("开始下载")
        self.download_btn.setFixedHeight(32)
        self.download_btn.clicked.connect(self._do_download)
        btn_row.addWidget(self.download_btn)
        self.refresh_btn = QPushButton("刷新文件信息")
        self.refresh_btn.setFixedHeight(32)
        self.refresh_btn.clicked.connect(self._refresh_file_info)
        btn_row.addWidget(self.refresh_btn)
        btn_row.addStretch()
        self.content.addLayout(btn_row)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.content.addWidget(self.progress)

        # 日志
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        self.content.addWidget(self.log)

        self._refresh_file_info()

    def _do_connect(self):
        ip = self.ip_edit.text().strip()
        port = int(self.port_edit.text().strip())
        user = self.user_edit.text().strip()
        pwd = self.pwd_edit.text()
        self.log.appendPlainText(f"正在连接 {user}@{ip}:{port} ...")
        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("连接中...")
        try:
            import dmPython
            self.conn = dmPython.connect(user=user, password=pwd, server=ip, port=port, autoCommit=True)
            self.log.appendPlainText("数据库连接成功！")
            self.conn_status.setText(f"已连接 {ip}:{port}")
            self.conn_status.setStyleSheet("color: #2e7d32; font-size: 12px;")
        except Exception as e:
            self.log.appendPlainText(f"连接失败: {e}")
            self.conn_status.setText("连接失败")
            self.conn_status.setStyleSheet("color: #d43f3a; font-size: 12px;")
        finally:
            self.connect_btn.setEnabled(True)
            self.connect_btn.setText("连接")

    def _refresh_file_info(self):
        if os.path.exists(LOCAL_JSONL):
            size_mb = os.path.getsize(LOCAL_JSONL) / 1024 / 1024
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(LOCAL_JSONL)))
            with open(LOCAL_JSONL, encoding="utf-8") as f:
                count = sum(1 for _ in f if _.strip())
            self.file_info.setText(f"大小: {size_mb:.1f} MB  |  记录数: {count} 条  |  最后更新: {mtime}")
        else:
            self.file_info.setText("文件: 不存在")

    def _get_params(self):
        return {
            "batch": self.batch_spin.value(),
            "sleep": self.sleep_spin.value(),
            "limit": self.limit_spin.value() or 200000,
        }

    def _do_download(self):
        if self.conn is None:
            self.log.appendPlainText("请先连接数据库")
            return
        offset = 0
        if os.path.exists(LOCAL_JSONL):
            with open(LOCAL_JSONL, encoding="utf-8") as f:
                offset = sum(1 for _ in f if _.strip())
        self.download_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.log.clear()
        p = self._get_params()
        if offset > 0:
            self.log.appendPlainText(f"本地已有 {offset} 条，从第 {offset+1} 条开始增量下载")
        else:
            self.log.appendPlainText("全量下载（本地无文件）")
        self.log.appendPlainText(f"参数: 每批 {p['batch']} 条, 间隔 {p['sleep']}s, 最多 {p['limit']} 条")
        self.worker = DownloadWorker(self.conn, offset=offset, limit=p["limit"], batch=p["batch"], sleep=p["sleep"])
        self.worker.progress.connect(lambda c, t: (self.progress.setMaximum(t), self.progress.setValue(c)))
        self.worker.log.connect(self.log.appendPlainText)
        self.worker.finished.connect(self._on_download_finished)
        self.worker.start()

    def _on_download_finished(self, count):
        self.download_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.log.appendPlainText(f"\n下载完成: {count} 条")
        self._refresh_file_info()
