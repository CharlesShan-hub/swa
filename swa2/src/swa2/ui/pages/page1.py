"""页面 1 — 数据库连接 + 下载到 local.db"""

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QProgressBar, QMessageBox, QInputDialog,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from swa2.ui.widgets.form_dialog import FormDialog
from swa2.ui.widgets.styles import STYLES
from swa2.data.local_db import LocalDB
from swa2.data.downloader import DownloadWorker
from swa2.data.config import load_config, save_config


class Page1(QWidget):
    def __init__(self):
        super().__init__()
        self._config = load_config()
        self._local_db = LocalDB()
        self._dm_conn = None
        self._worker = None
        self._remote_total = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)

        # ── 标题 ──
        title = QLabel("数据库连接")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        desc = QLabel("连接达梦数据库 → 下载波形数据到本地 SQLite")
        desc.setStyleSheet("color: #a6adc8; font-size: 13px;")
        layout.addWidget(desc)

        layout.addSpacing(20)

        # ── 配置卡片 ──
        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet("""
            QFrame#card {
                background-color: #1e1e2e;
                border: 1px solid #313244;
                border-radius: 8px;
                padding: 20px;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(10)

        # 配置摘要
        summary_row = QHBoxLayout()
        summary_row.addWidget(QLabel("当前配置:"))
        self.summary_label = QLabel(self._format_summary())
        self.summary_label.setStyleSheet("color: #a6adc8; font-size: 13px;")
        summary_row.addWidget(self.summary_label, 1)
        card_layout.addLayout(summary_row)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setStyleSheet("color: #313244;")
        card_layout.addWidget(sep1)

        # 连接按钮组
        btn_row1 = QHBoxLayout()
        self.config_btn = QPushButton("下载配置")
        self.config_btn.setFixedHeight(32)
        self.config_btn.setStyleSheet(STYLES["btn_primary"])
        self.config_btn.clicked.connect(self._open_config_dialog)
        btn_row1.addWidget(self.config_btn)

        self.connect_btn = QPushButton("连接达梦")
        self.connect_btn.setFixedHeight(32)
        self.connect_btn.setStyleSheet(STYLES["btn_secondary"])
        self.connect_btn.clicked.connect(self._on_connect)
        btn_row1.addWidget(self.connect_btn)

        self.conn_status = QLabel("未连接")
        self.conn_status.setStyleSheet("color: #585b70; font-size: 12px;")
        btn_row1.addWidget(self.conn_status, 1)
        card_layout.addLayout(btn_row1)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color: #313244;")
        card_layout.addWidget(sep2)

        # 数据库统计摘要
        stats_row = QHBoxLayout()
        stats_row.addWidget(QLabel("本地数据库:"))
        self.local_label = QLabel("—")
        self.local_label.setStyleSheet("color: #a6adc8; font-size: 13px;")
        stats_row.addWidget(self.local_label, 1)
        card_layout.addLayout(stats_row)

        stats_row2 = QHBoxLayout()
        stats_row2.addWidget(QLabel("远程剩余:"))
        self.remote_label = QLabel("(未连接)")
        self.remote_label.setStyleSheet("color: #585b70; font-size: 13px;")
        stats_row2.addWidget(self.remote_label, 1)
        card_layout.addLayout(stats_row2)

        sep3 = QFrame()
        sep3.setFrameShape(QFrame.HLine)
        sep3.setStyleSheet("color: #313244;")
        card_layout.addWidget(sep3)

        # 下载按钮 + 进度
        btn_row2 = QHBoxLayout()
        self.download_btn = QPushButton("下载数据")
        self.download_btn.setFixedHeight(32)
        self.download_btn.setStyleSheet(STYLES["btn_secondary"])
        self.download_btn.setEnabled(False)
        self.download_btn.clicked.connect(self._on_download)
        btn_row2.addWidget(self.download_btn)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setFixedHeight(32)
        self.cancel_btn.setStyleSheet(STYLES["btn_secondary"])
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._on_cancel_download)
        btn_row2.addWidget(self.cancel_btn)

        self.dl_status = QLabel("")
        self.dl_status.setStyleSheet("color: #585b70; font-size: 12px;")
        btn_row2.addWidget(self.dl_status, 1)
        card_layout.addLayout(btn_row2)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #313244;
                border: none;
                border-radius: 4px;
                text-align: center;
                font-size: 11px;
                color: #cdd6f4;
            }
            QProgressBar::chunk {
                background-color: #89b4fa;
                border-radius: 4px;
            }
        """)
        card_layout.addWidget(self.progress_bar)

        # 删除按钮
        del_row = QHBoxLayout()
        del_row.addStretch()
        self.delete_btn = QPushButton("删除本地数据库")
        self.delete_btn.setFixedHeight(28)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #f38ba8;
                border: 1px solid #f38ba8;
                border-radius: 4px;
                padding: 4px 14px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #f38ba8;
                color: #1e1e2e;
            }
        """)
        self.delete_btn.clicked.connect(self._on_delete_db)
        del_row.addWidget(self.delete_btn)
        card_layout.addLayout(del_row)

        layout.addWidget(card)
        layout.addStretch()

        # 加载本地状态
        self._refresh_local()

    def _format_summary(self) -> str:
        pw = "****" if self._config.get("password") else "(空)"
        batch = self._config.get("batch_size", "400")
        sleep = self._config.get("sleep_sec", "1.0")
        return (f"{self._config['host']}:{self._config['port']} / "
                f"{self._config['user']} / {pw}  |  "
                f"每批{batch}条 间隔{sleep}秒")

    def _refresh_local(self):
        try:
            self._local_db.connect()
            s = self._local_db.summary()
            self.local_label.setText(
                f"共 {s['total']} 条  |  启用 {s['enabled']}  |  禁用 {s['disabled']}"
            )
            self._local_db.close()
        except Exception:
            self.local_label.setText("(暂无数据)")

    # ── 配置弹窗 ──

    def _open_config_dialog(self):
        dialog = FormDialog("下载配置", [
            ("主机:",        "host",       self._config.get("host", "10.15.10.1")),
            ("端口:",        "port",       self._config.get("port", "5256")),
            ("用户:",        "user",       self._config.get("user", "SYSDBA")),
            ("密码:",        "password",   self._config.get("password", "SYSDBA"), "password"),
            ("每批条数:",     "batch_size", self._config.get("batch_size", "400")),
            ("间隔(秒):",     "sleep_sec",  self._config.get("sleep_sec", "1.0")),
        ], parent=self)

        if dialog.exec():
            self._config = dialog.values()
            save_config(self._config)
            self.summary_label.setText(self._format_summary())

    # ── 连接达梦 ──

    def _on_connect(self):
        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("连接中...")
        self.conn_status.setText(f"正在连接 {self._config['host']}...")
        self.conn_status.setStyleSheet("color: #f9e2af; font-size: 12px;")
        QTimer.singleShot(200, self._do_connect)

    def _do_connect(self):
        try:
            import dmPython as dm
            self._dm_conn = dm.connect(
                server=self._config["host"],
                port=int(self._config["port"]),
                user=self._config["user"],
                password=self._config["password"],
                autoCommit=False,
            )
            # 查询远程总数
            cur = self._dm_conn.cursor()
            cur.execute("SELECT COUNT(*) FROM YS_DB.TB_MODBUS_DEV_POINT")
            self._remote_total = cur.fetchone()[0]

            # 计算剩余
            s = self._local_db.summary() if self._local_db._conn else {"total": 0}
            remaining = self._remote_total - s["total"]

            self.conn_status.setText(f"✓ 已连接 ({self._remote_total} 条)")
            self.conn_status.setStyleSheet("color: #a6e3a1; font-size: 12px;")
            self.remote_label.setText(
                f"{remaining} 条 (共 {self._remote_total} 条)"
            )
            self.remote_label.setStyleSheet("color: #f9e2af; font-size: 13px;")
            self.download_btn.setEnabled(True)
        except ImportError:
            self.conn_status.setText("✗ 未安装 dmpython")
            self.conn_status.setStyleSheet("color: #f38ba8; font-size: 12px;")
        except Exception as e:
            self.conn_status.setText(f"✗ {e}")
            self.conn_status.setStyleSheet("color: #f38ba8; font-size: 12px;")
        finally:
            self.connect_btn.setEnabled(True)
            self.connect_btn.setText("连接达梦")

    # ── 下载 ──

    def _on_download(self):
        if self._dm_conn is None:
            return

        self._local_db.connect()
        offset = self._local_db.max_dm_rowid()

        self.download_btn.setEnabled(False)
        self.download_btn.setVisible(False)
        self.cancel_btn.setVisible(True)
        self.dl_status.setText("下载中...")
        self.dl_status.setStyleSheet("color: #f9e2af; font-size: 12px;")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        batch_size = int(self._config.get("batch_size", "400"))
        sleep_sec = float(self._config.get("sleep_sec", "1.0"))

        local_db = LocalDB()
        self._worker = DownloadWorker(
            self._dm_conn, local_db,
            offset=offset,
            batch=batch_size,
            sleep=sleep_sec,
        )
        self._worker.progress.connect(self._on_download_progress)
        self._worker.log.connect(self._on_download_log)
        self._worker.finished.connect(self._on_download_finished)
        self._worker.error.connect(self._on_download_error)
        self._worker.start()

    def _on_cancel_download(self):
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("正在停止...")
            self.dl_status.setText("正在停止下载...")

    def _on_download_progress(self, current: int, total: int):
        pct = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.dl_status.setText(f"下载中 {current}/{total}")

    def _on_download_log(self, msg: str):
        pass

    def _on_download_finished(self, count: int):
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setText("取消")
        self.download_btn.setVisible(True)
        self.download_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        if count == 0 and self._worker and self._worker.isInterruptionRequested():
            self.dl_status.setText("已取消")
            self.dl_status.setStyleSheet("color: #f9e2af; font-size: 12px;")
        else:
            self.dl_status.setText(f"✓ 完成，共 {count} 条")
            self.dl_status.setStyleSheet("color: #a6e3a1; font-size: 12px;")
            self._refresh_local()
            # 更新远程剩余（重新连接，因为 _refresh_local 会关闭连接）
            if self._dm_conn:
                self._local_db.connect()
                s = self._local_db.summary()
                remaining = self._remote_total - s["total"]
                self._local_db.close()
                self.remote_label.setText(
                    f"{remaining} 条 (共 {self._remote_total} 条)"
                )

    def _on_download_error(self, msg: str):
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setText("取消")
        self.download_btn.setVisible(True)
        self.download_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.dl_status.setText(f"✗ {msg}")
        self.dl_status.setStyleSheet("color: #f38ba8; font-size: 12px;")

    def stop_download(self):
        """外部调用（如窗口关闭时）安全停止下载线程。"""
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(5000)  # 最多等5秒

    # ── 删除数据库 ──

    def _on_delete_db(self):
        """两步确认：弹窗 → 输入框 → 删除。"""
        # 第一步：确认弹窗
        reply = QMessageBox.question(
            self, "删除确认",
            "确定要删除本地数据库吗？\n"
            "此操作不可撤销，所有本地数据将被清除。",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # 第二步：输入确认
        text, ok = QInputDialog.getText(
            self, "二次确认",
            "狠心要删？那请输入「我是笨蛋」以继续：",
        )
        if not ok or text.strip() != "我是笨蛋":
            return

        # 执行删除
        try:
            self._local_db.close()
            db_path = self._local_db.db_path
            wal_path = db_path + "-wal"
            shm_path = db_path + "-shm"

            import time
            for path in [db_path, wal_path, shm_path]:
                if os.path.exists(path):
                    os.remove(path)
                    time.sleep(0.1)

            self.local_label.setText("(已删除)")
            self.conn_status.setText("未连接")
            self.conn_status.setStyleSheet("color: #585b70; font-size: 12px;")
            self.remote_label.setText("(未连接)")
            self.download_btn.setEnabled(False)
            self.dl_status.setText("")
            QMessageBox.information(self, "完成", "本地数据库已删除")
        except Exception as e:
            QMessageBox.critical(self, "删除失败", str(e))
