"""后台线程下载 — 从达梦数据库直接下载到 local.db"""

import time
from PySide6.QtCore import QThread, Signal

from swa2.data.local_db import LocalDB


class DownloadWorker(QThread):
    """在后台线程执行下载，通过信号汇报进度。"""

    progress = Signal(int, int)   # (current, total)
    log = Signal(str)
    finished = Signal(int)        # 下载条数
    error = Signal(str)

    def __init__(self, dm_conn, local_db: LocalDB,
                 offset: int = 0, limit: int = 0,
                 batch: int = 500, sleep: float = 0.0):
        super().__init__()
        self.dm_conn = dm_conn          # 达梦数据库连接
        self.local_db = local_db        # 本地数据库实例
        self.offset = offset
        self.limit = limit
        self.batch = batch
        self.sleep = sleep

    def run(self):
        try:
            self._do_download()
        except Exception as e:
            self.error.emit(str(e))

    def _do_download(self):
        cur = self.dm_conn.cursor()
        cur.execute("SELECT COUNT(*) FROM YS_DB.TB_MODBUS_DEV_POINT")
        total = cur.fetchone()[0]
        self.log.emit(f"数据库总记录数: {total}")

        actual_limit = self.limit if self.limit > 0 else total - self.offset
        actual_limit = min(actual_limit, total - self.offset)
        if actual_limit <= 0:
            self.log.emit("无新增数据")
            self.finished.emit(0)
            return

        fields = [
            "YS_DB.TB_MODBUS_DEV_POINT.ROWID",
            "TEST_CASE_CODE", "SYSTEM_TIME", "RTU_REGS_SLAVE_ID",
            "RTU_REGS_P00_ROTOR_RPM", "RTU_REGS_P00_ENV_TEMP",
            "RTU_REGS_P00_ENV_HUMIDITY", "ACTUAL_VOLTAGE",
            "RTU_REGS_P00_WAVE_DATA", "DEVICE_ID",
        ]
        field_csv = ", ".join(fields)

        exported = 0
        total_batches = (actual_limit + self.batch - 1) // self.batch

        # 连接 local.db
        self.local_db.connect()

        for batch_no in range(total_batches):
            if self.isInterruptionRequested():
                break

            this_batch = min(self.batch, actual_limit - exported)
            if this_batch <= 0:
                break

            batch_offset = self.offset + exported
            sql = (
                f"SELECT {field_csv} FROM YS_DB.TB_MODBUS_DEV_POINT "
                f"ORDER BY YS_DB.TB_MODBUS_DEV_POINT.ROWID "
                f"LIMIT ? OFFSET ?"
            )
            cur.execute(sql, (this_batch, batch_offset))
            rows = cur.fetchall()
            if not rows:
                break

            col_names = [desc[0] for desc in cur.description]

            for row in rows:
                record = dict(zip(col_names, row))
                system_time = str(record.get("SYSTEM_TIME", "")) if record.get("SYSTEM_TIME") else None

                # 插入 records 表（含 dm_rowid）
                rid = self.local_db.insert_record(
                    system_time=system_time,
                    actual_voltage=record.get("ACTUAL_VOLTAGE"),
                    temperature=record.get("RTU_REGS_P00_ENV_TEMP"),
                    humidity=record.get("RTU_REGS_P00_ENV_HUMIDITY"),
                    rpm=record.get("RTU_REGS_P00_ROTOR_RPM"),
                    slave_id=record.get("RTU_REGS_SLAVE_ID"),
                    device_id=record.get("DEVICE_ID"),
                    test_case_code=record.get("TEST_CASE_CODE"),
                    dm_rowid=record.get("ROWID"),
                )

                # 插入 waveforms 表
                wave = record.get("RTU_REGS_P00_WAVE_DATA")
                if wave is not None:
                    self.local_db.insert_waveform(rid, str(wave))

                exported += 1

            # 分批提交
            self.local_db.commit()

            self.progress.emit(exported, actual_limit)
            self.log.emit(
                f"  第 {batch_no+1}/{total_batches} 批: {exported}/{actual_limit}"
            )

            if self.sleep > 0:
                time.sleep(self.sleep)

        self.local_db.close()
        self.finished.emit(exported)
