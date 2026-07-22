"""本地 SQLite 数据库管理 — 替代 JSONL 中间文件"""

import os
import sqlite3
from typing import Optional


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
DB_PATH = os.path.join(DATA_DIR, "local.db")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    system_time     TEXT,
    actual_voltage  REAL,
    temperature     REAL,
    humidity        REAL,
    rpm             REAL,
    slave_id        INTEGER,
    device_id       TEXT,
    test_case_code  TEXT,
    dm_rowid        INTEGER,
    enabled         INTEGER DEFAULT 1,
    downloaded_at   TEXT DEFAULT (datetime('now', 'localtime')),
    created_at      TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS waveforms (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id   INTEGER UNIQUE NOT NULL,
    wave_data   TEXT,
    FOREIGN KEY (record_id) REFERENCES records(id)
);

CREATE INDEX IF NOT EXISTS idx_records_device  ON records(device_id);
CREATE INDEX IF NOT EXISTS idx_records_voltage ON records(actual_voltage);
CREATE INDEX IF NOT EXISTS idx_records_time    ON records(system_time);
CREATE INDEX IF NOT EXISTS idx_records_enabled ON records(enabled);
"""


class LocalDB:
    """local.db 操作封装"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self):
        """打开/创建数据库并初始化 schema。"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA_SQL)
        # 迁移：已有数据库增加 dm_rowid 列
        try:
            self._conn.execute("ALTER TABLE records ADD COLUMN dm_rowid INTEGER")
        except Exception:
            pass  # 列已存在
        self._conn.commit()
        return self

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("LocalDB 未连接，请先调用 connect()")
        return self._conn

    # ── 写入 ──

    def insert_record(self, **kwargs) -> int:
        """插入一条记录，返回 record_id。"""
        fields = {
            "system_time": kwargs.get("system_time"),
            "actual_voltage": kwargs.get("actual_voltage"),
            "temperature": kwargs.get("temperature"),
            "humidity": kwargs.get("humidity"),
            "rpm": kwargs.get("rpm"),
            "slave_id": kwargs.get("slave_id"),
            "device_id": kwargs.get("device_id"),
            "test_case_code": kwargs.get("test_case_code"),
            "dm_rowid": kwargs.get("dm_rowid"),
            "enabled": 1,
        }
        cols = ", ".join(fields.keys())
        placeholders = ", ".join("?" for _ in fields)
        cur = self.conn.execute(
            f"INSERT INTO records ({cols}) VALUES ({placeholders})",
            list(fields.values()),
        )
        return cur.lastrowid

    def insert_waveform(self, record_id: int, wave_data: str):
        """插入波形数据。"""
        self.conn.execute(
            "INSERT OR REPLACE INTO waveforms (record_id, wave_data) VALUES (?, ?)",
            (record_id, wave_data),
        )

    def commit(self):
        self.conn.commit()

    # ── 查询 ──

    def count(self, enabled_only: bool = False) -> int:
        where = "WHERE enabled = 1" if enabled_only else ""
        return self.conn.execute(
            f"SELECT COUNT(*) FROM records {where}"
        ).fetchone()[0]

    def max_dm_rowid(self) -> int:
        """返回已下载的最大 dm_rowid（断点续传用），无数据返回 0。"""
        row = self.conn.execute("SELECT MAX(dm_rowid) FROM records").fetchone()
        return row[0] if row[0] is not None else 0

    def summary(self) -> dict:
        total = self.count()
        enabled = self.count(enabled_only=True)
        return {
            "total": total,
            "enabled": enabled,
            "disabled": total - enabled,
        }

    def get_record_ids_by_voltage(self) -> list[tuple[int, float]]:
        """返回所有 (id, actual_voltage) 列表，按时间排序。"""
        cur = self.conn.execute(
            "SELECT id, actual_voltage FROM records ORDER BY id"
        )
        return cur.fetchall()

    def get_waveform(self, record_id: int) -> Optional[str]:
        cur = self.conn.execute(
            "SELECT wave_data FROM waveforms WHERE record_id = ?", (record_id,)
        )
        row = cur.fetchone()
        return row[0] if row else None
