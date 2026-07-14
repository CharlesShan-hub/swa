"""
数据项目管理 — SQLite 存储、行级质量开关、查询、备份

使用方式:
    from swa.data.manager import DataManager

    # 从 JSONL 创建项目
    dm = DataManager()
    dm.create_project("week1", "data/raw/2026_7_6.jsonl")
    dm.create_project("week2", "data/raw/new_data.jsonl")

    # 加载已有项目
    dm.load_project("week1")
    print(dm.summary())

    # 按条件禁用质量差的数据
    dm.disable_records("actual_voltage IS NULL")
    dm.disable_records("humidity > 47")

    # 只查启用的数据
    df = dm.query(enabled_only=True, fields=["actual_voltage", "humidity"])

    # 备份
    dm.backup("week1_bak")
"""

import json
import os
import sqlite3
import shutil
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd

from swa.data.loader import load_jsonl, parse_wave, parse_voltage

PROJECTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "projects")


class DataManager:
    """数据项目管理器。"""

    def __init__(self, projects_dir: str = PROJECTS_DIR):
        self.projects_dir = projects_dir
        self.current_project: Optional[str] = None
        self._conn: Optional[sqlite3.Connection] = None
        os.makedirs(projects_dir, exist_ok=True)

    # ── 项目生命周期 ────────────────────────────────────────────

    def list_projects(self) -> list[dict]:
        """列出所有项目及其摘要。"""
        projects = []
        if not os.path.isdir(self.projects_dir):
            return projects
        for name in sorted(os.listdir(self.projects_dir)):
            meta_path = os.path.join(self.projects_dir, name, "meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                meta["name"] = name
                projects.append(meta)
        return projects

    def create_project(
        self,
        name: str,
        source_jsonl: str,
        description: str = "",
        label_map: Optional[dict[str, float]] = None,
        skip_first_n: int = 0,
    ) -> dict:
        """从 JSONL 文件创建新项目。

        Args:
            name: 项目名称
            source_jsonl: 源 JSONL 文件路径
            description: 项目描述
            label_map: 电压标签替换映射 {标签: 电压值}
            skip_first_n: 每个电压等级前 N 条自动禁用（默认 0）
        """
        project_dir = os.path.join(self.projects_dir, name)
        if os.path.exists(project_dir):
            raise FileExistsError(f"项目 '{name}' 已存在")

        os.makedirs(project_dir)

        # 加载并清洗（使用自定义标签映射）
        df = load_jsonl(source_jsonl)
        df["actual_voltage"] = df["actual_voltage"].apply(
            lambda v: parse_voltage(v, label_map)
        )
        df = df.dropna(subset=["actual_voltage"])

        # 写入 SQLite
        db_path = os.path.join(project_dir, "data.db")
        conn = sqlite3.connect(db_path, timeout=5)
        self._create_tables(conn)
        self._import_dataframe(conn, df)

        # 每个电压等级前 N 条自动禁用（数据不稳定）
        if skip_first_n > 0:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT actual_voltage FROM records ORDER BY actual_voltage")
            voltages = [r[0] for r in cur.fetchall()]
            total_disabled = 0
            for v in voltages:
                # 按时间排序，禁用前 N 条
                cur.execute(
                    "SELECT id FROM records WHERE actual_voltage = ? ORDER BY system_time LIMIT ?",
                    (v, skip_first_n),
                )
                ids = [r[0] for r in cur.fetchall()]
                if ids:
                    placeholders = ",".join("?" for _ in ids)
                    cur.execute(
                        f"UPDATE records SET enabled = 0 WHERE id IN ({placeholders})",
                        ids,
                    )
                    total_disabled += len(ids)
            conn.commit()

        # 元信息（保存标签映射）
        meta = {
            "name": name,
            "description": description,
            "source": source_jsonl,
            "created_at": datetime.now().isoformat(),
            "total_records": len(df),
            "enabled_records": len(df) - (skip_first_n if skip_first_n > 0 else 0),
            "label_map": label_map or {},
        }
        with open(os.path.join(project_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        conn.close()
        self.load_project(name)
        return meta

    def load_project(self, name: str):
        """加载已有项目。"""
        project_dir = os.path.join(self.projects_dir, name)
        db_path = os.path.join(project_dir, "data.db")
        meta_path = os.path.join(project_dir, "meta.json")

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"项目 '{name}' 不存在")

        if self._conn:
            self._conn.close()

        self._conn = sqlite3.connect(db_path, timeout=5)
        self._conn.row_factory = sqlite3.Row
        self.current_project = name

        # 已有项目自动回填谐波字段（不阻塞主流程）
        try:
            self.backfill_harmonics()
        except Exception:
            pass

        # 更新 meta 中的统计
        meta = self._load_meta()
        meta["total_records"] = self._count("1=1")
        meta["enabled_records"] = self._count("enabled=1")
        self._save_meta(meta)

    def close(self):
        """关闭当前项目。"""
        if self._conn:
            self._conn.close()
            self._conn = None
        self.current_project = None

    # ── 查询 ────────────────────────────────────────────────────

    def query(
        self,
        fields: Optional[list[str]] = None,
        where: str = "1=1",
        enabled_only: bool = True,
        limit: Optional[int] = None,
        order_by: str = "rowid",
    ) -> pd.DataFrame:
        """
        查询数据。

        Args:
            fields: 字段列表（None = 全部）
            where: SQL WHERE 条件
            enabled_only: 是否只查启用的行
            limit: 最大行数
            order_by: 排序字段

        Returns:
            DataFrame（不含波形数据，波形通过 get_waveform 单独获取）
        """
        if self._conn is None:
            raise RuntimeError("请先 load_project()")

        if enabled_only:
            where = f"({where}) AND enabled=1"

        field_str = ", ".join(fields) if fields else "*"
        sql = f"SELECT {field_str} FROM records WHERE {where} ORDER BY {order_by}"
        if limit:
            sql += f" LIMIT {limit}"

        return pd.read_sql_query(sql, self._conn)

    def get_waveform(self, record_id: int) -> Optional[np.ndarray]:
        """获取单条波形数据。"""
        if self._conn is None:
            raise RuntimeError("请先 load_project()")

        cur = self._conn.cursor()
        cur.execute("SELECT wave_data FROM waveforms WHERE record_id = ?", (record_id,))
        row = cur.fetchone()
        if row is None:
            return None
        # SQLite 存的是逗号分隔的文本
        try:
            return np.array([float(x) for x in row[0].split(",")])
        except (ValueError, TypeError):
            return None

    def get_waveforms_batch(
        self, record_ids: list[int]
    ) -> dict[int, Optional[np.ndarray]]:
        """批量获取波形。"""
        if self._conn is None:
            raise RuntimeError("请先 load_project()")

        cur = self._conn.cursor()
        placeholders = ",".join("?" * len(record_ids))
        cur.execute(
            f"SELECT record_id, wave_data FROM waveforms WHERE record_id IN ({placeholders})",
            record_ids,
        )
        result = {}
        for rid, wd in cur.fetchall():
            try:
                result[rid] = np.array([float(x) for x in wd.split(",")])
            except (ValueError, TypeError):
                result[rid] = None
        return result

    # ── 行级质量开关 ───────────────────────────────────────────

    def enable_records(self, where: str):
        """启用符合条件的行。"""
        self._update_enabled(where, 1)

    def disable_records(self, where: str):
        """禁用符合条件的行。"""
        self._update_enabled(where, 0)

    def toggle_record(self, record_id: int):
        """切换单条记录的启用/禁用状态。"""
        cur = self._conn.cursor()
        cur.execute("UPDATE records SET enabled = NOT enabled WHERE id = ?", (record_id,))
        self._conn.commit()

    def run_quality_check(self, batch_size: int = 500) -> int:
        """对当前项目运行波形质量检测，自动禁用坏数据。

        判断依据（使用导入时已算好的字段）:
            - harm_a1 IS NULL → 坏数据（无有效波形）
            - harm_cycles NOT BETWEEN 6.5 AND 7.5 → 周期数不对
            - harm_thd > 0.30 → 总谐波失真过大（方波、削波等）
            - harm_noise_pct > 0.50 → 超过一半能量不在基频（强噪声）
            - harm_a2 / harm_a1 > 0.25 → 二次谐波过大
            - harm_error / harm_a1 > 0.30 → 拟合误差过大

        Returns:
            禁用条数
        """
        cur = self._conn.cursor()
        cur.execute("""
            UPDATE records SET enabled = 0 WHERE
                enabled = 1 AND (
                    harm_a1 IS NULL
                    OR harm_a1 < 1
                    OR harm_cycles < 6.5
                    OR harm_cycles > 7.5
                    OR harm_thd > 0.30
                    OR harm_noise_pct > 0.50
                    OR (harm_a2 / harm_a1) > 0.25
                    OR (harm_error / harm_a1) > 0.30
                )
        """)
        n = cur.rowcount
        self._conn.commit()
        return n

    def backfill_harmonics(self, batch_size: int = 500) -> int:
        """回填已有项目的谐波字段（对新导入的数据自动计算）。"""
        from swa.data.loader import compute_harmonics
        cur = self._conn.cursor()
        total = 0
        offset = 0
        while True:
            cur.execute(
                "SELECT r.id, w.wave_data FROM records r "
                "JOIN waveforms w ON w.record_id = r.id "
                "WHERE r.harm_cycles IS NULL LIMIT ? OFFSET ?",
                (batch_size, offset),
            )
            rows = cur.fetchall()
            if not rows:
                break
            for row in rows:
                a1, a2, err, cycles, thd, noise_pct = compute_harmonics(row["wave_data"])
                cur.execute(
                    "UPDATE records SET harm_a1=?, harm_a2=?, harm_error=?, harm_cycles=?, harm_thd=?, harm_noise_pct=? WHERE id=?",
                    (a1, a2, err, cycles, thd, noise_pct, row["id"]),
                )
                total += 1
            offset += batch_size
        self._conn.commit()
        return total

    def quality_summary(self) -> dict:
        """质量统计。"""
        total = self._count("1=1")
        enabled = self._count("enabled=1")
        disabled = total - enabled
        return {
            "total": total,
            "enabled": enabled,
            "disabled": disabled,
            "enabled_pct": round(enabled / total * 100, 1) if total else 0,
        }

    # ── 摘要与备份 ─────────────────────────────────────────────

    def summary(self) -> dict:
        """项目摘要信息。"""
        meta = self._load_meta()
        q = self.quality_summary()
        return {
            **meta,
            **q,
        }

    def backup(self, backup_name: str):
        """备份当前项目。"""
        src = os.path.join(self.projects_dir, self.current_project)
        dst = os.path.join(self.projects_dir, backup_name)
        if os.path.exists(dst):
            raise FileExistsError(f"备份 '{backup_name}' 已存在")
        shutil.copytree(src, dst)
        # 更新备份的 meta
        meta_path = os.path.join(dst, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        meta["backup_of"] = self.current_project
        meta["backup_at"] = datetime.now().isoformat()
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return meta

    def delete_project(self, name: str):
        """删除项目（先关闭连接再删除目录）。"""
        self.close()
        import gc, time
        gc.collect()
        time.sleep(0.5)

        project_dir = os.path.join(self.projects_dir, name)
        if not os.path.exists(project_dir):
            return

        # 带重试的删除（应对 SQLite 锁）
        for attempt in range(5):
            try:
                shutil.rmtree(project_dir)
                return
            except PermissionError:
                time.sleep(1)
                gc.collect()
        raise PermissionError(f"无法删除项目 '{name}'，文件被占用")

    def export_jsonl(self, output_path: str, enabled_only: bool = True):
        """导出为 JSONL。"""
        df = self.query(enabled_only=enabled_only)
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)

    # ── 内部方法 ────────────────────────────────────────────────

    def _create_tables(self, conn: sqlite3.Connection):
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS records (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                system_time     TEXT,
                actual_voltage  REAL,
                temperature     REAL,
                humidity        REAL,
                rpm             REAL,
                slave_id        INTEGER,
                test_case_code  TEXT,
                enabled         INTEGER DEFAULT 1,
                harm_a1         REAL,
                harm_a2         REAL,
                harm_error      REAL,
                harm_cycles     REAL,
                harm_thd        REAL,
                harm_noise_pct  REAL,
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS waveforms (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id   INTEGER UNIQUE NOT NULL,
                wave_data   TEXT,
                FOREIGN KEY (record_id) REFERENCES records(id)
            );

            CREATE INDEX IF NOT EXISTS idx_records_enabled ON records(enabled);
            CREATE INDEX IF NOT EXISTS idx_records_voltage ON records(actual_voltage);
            CREATE INDEX IF NOT EXISTS idx_records_time ON records(system_time);
        """)

    def _import_dataframe(self, conn: sqlite3.Connection, df: pd.DataFrame):
        """将 DataFrame 导入 SQLite。"""
        # 字段映射
        col_map = {
            "actual_voltage": "actual_voltage",
            "system_time": "system_time",
            "temperature": "temperature",
            "humidity": "humidity",
            "rpm": "rpm",
            "slave_id": "slave_id",
            "test_case_code": "test_case_code",
        }
        # 波形列
        wave_col = None
        for c in ["wave_data", "RTU_REGS_P00_WAVE_DATA", "WAVE_DATA"]:
            if c in df.columns:
                wave_col = c
                break

        from swa.data.loader import compute_harmonics

        cur = conn.cursor()
        insert_sql = """
            INSERT INTO records
                (system_time, actual_voltage, temperature, humidity,
                 rpm, slave_id, test_case_code, harm_a1, harm_a2, harm_error,
                 harm_cycles, harm_thd, harm_noise_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        wave_sql = "INSERT INTO waveforms (record_id, wave_data) VALUES (?, ?)"

        for _, row in df.iterrows():
            vals = list(row.get(c) if pd.notna(row.get(c)) else None for c in ["system_time", "actual_voltage", "temperature", "humidity", "rpm", "slave_id", "test_case_code"])

            # 计算谐波参数
            wave_str = str(row.get(wave_col)) if wave_col and row.get(wave_col) else None
            if wave_str:
                a1, a2, err, cycles, thd, noise_pct = compute_harmonics(wave_str)
                vals.extend([a1, a2, err, cycles, thd, noise_pct])
            else:
                vals.extend([None, None, None, None, None, None])

            cur.execute(insert_sql, vals)
            record_id = cur.lastrowid

            if wave_str:
                cur.execute(wave_sql, (record_id, str(row[wave_col])))

        conn.commit()

    def _update_enabled(self, where: str, value: int):
        cur = self._conn.cursor()
        cur.execute(f"UPDATE records SET enabled = ? WHERE {where}", (value,))
        self._conn.commit()

    def _count(self, where: str = "1=1") -> int:
        cur = self._conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM records WHERE {where}")
        return cur.fetchone()[0]

    def _load_meta(self) -> dict:
        meta_path = os.path.join(self.projects_dir, self.current_project, "meta.json")
        with open(meta_path) as f:
            return json.load(f)

    def _save_meta(self, meta: dict):
        meta_path = os.path.join(self.projects_dir, self.current_project, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
