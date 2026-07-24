"""项目管理 — 从 local.db 创建项目，独立拷贝数据 + 计算特征"""

import json
import os
import sqlite3
import shutil
from datetime import datetime
from typing import Optional

import numpy as np

from swa2.data.local_db import LocalDB

PROJECTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "projects"
)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_record_id INTEGER,
    system_time     TEXT,
    actual_voltage  REAL,
    temperature     REAL,
    humidity        REAL,
    rpm             REAL,
    slave_id        INTEGER,
    device_id       TEXT,
    test_case_code  TEXT,
    enabled         INTEGER DEFAULT 1,
    harm_a1         REAL,
    harm_a2         REAL,
    harm_error      REAL,
    harm_cycles     REAL,
    harm_noise_pct  REAL,
    harm_clip_ratio REAL,
    harm_clip_corrected INTEGER DEFAULT 0,
    harm_a1_corrected REAL,
    score           REAL,
    predicted_voltage_1 REAL,
    predicted_voltage_2 REAL,
    predicted_voltage_3 REAL,
    predicted_voltage_4 REAL,
    predicted_voltage_5 REAL,
    created_at      TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS waveforms (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id   INTEGER UNIQUE NOT NULL,
    wave_data   TEXT,
    FOREIGN KEY (record_id) REFERENCES records(id)
);

CREATE INDEX IF NOT EXISTS idx_records_enabled ON records(enabled);
CREATE INDEX IF NOT EXISTS idx_records_voltage ON records(actual_voltage);
CREATE INDEX IF NOT EXISTS idx_records_device  ON records(device_id);
"""


class ProjectManager:
    """项目创建和管理。"""

    def __init__(self):
        self._conn: Optional[sqlite3.Connection] = None
        self.current_project: Optional[str] = None
        self._project_dir: Optional[str] = None

    @property
    def projects_dir(self) -> str:
        os.makedirs(PROJECTS_DIR, exist_ok=True)
        return PROJECTS_DIR

    # ── 项目列表 ──

    def list_projects(self) -> list[dict]:
        """列出所有项目及摘要。"""
        projects = []
        if not os.path.exists(self.projects_dir):
            return projects
        for name in sorted(os.listdir(self.projects_dir)):
            meta_path = os.path.join(self.projects_dir, name, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                projects.append(meta)
        return projects

    def load_project(self, name: str):
        """加载已有项目，返回 self。"""
        project_dir = os.path.join(self.projects_dir, name)
        db_path = os.path.join(project_dir, "data.db")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"项目 '{name}' 不存在")
        if self._conn:
            self._conn.close()
        self._conn = sqlite3.connect(db_path, timeout=10)
        self._conn.row_factory = sqlite3.Row
        self.current_project = name
        self._project_dir = project_dir
        return self

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
        self.current_project = None
        self._project_dir = None

    def summary(self) -> dict:
        """返回当前项目的摘要信息。"""
        total = self._conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        enabled = self._conn.execute(
            "SELECT COUNT(*) FROM records WHERE enabled=1"
        ).fetchone()[0]
        return {
            "total": total,
            "enabled": enabled,
            "disabled": total - enabled,
        }

    # ── 查询 local.db ──

    def get_devices(self) -> list[str]:
        """查询 local.db 中所有设备 ID。"""
        db = LocalDB()
        db.connect()
        rows = db.conn.execute(
            "SELECT DISTINCT device_id FROM records WHERE device_id IS NOT NULL ORDER BY device_id"
        ).fetchall()
        db.close()
        return [r[0] for r in rows]

    def get_voltages(self) -> list[float]:
        """查询 local.db 中所有电压值。"""
        db = LocalDB()
        db.connect()
        rows = db.conn.execute(
            "SELECT DISTINCT actual_voltage FROM records WHERE actual_voltage IS NOT NULL ORDER BY actual_voltage"
        ).fetchall()
        db.close()
        result = []
        for r in rows:
            try:
                result.append(float(r[0]))
            except (ValueError, TypeError):
                pass
        return result

    def get_time_range(self) -> tuple:
        """查询 local.db 的时间范围。"""
        db = LocalDB()
        db.connect()
        row = db.conn.execute(
            "SELECT MIN(system_time), MAX(system_time) FROM records"
        ).fetchone()
        db.close()
        return (row[0] or "", row[1] or "")

    def get_record_count(self) -> int:
        db = LocalDB()
        db.connect()
        n = db.count()
        db.close()
        return n

    # ── 创建项目 ──

    def create_project(
        self,
        name: str,
        device_ids: Optional[list[str]] = None,
        voltage_list: Optional[list[float]] = None,
        skip_first_n: int = 85510,
        label_map: Optional[dict[str, float]] = None,
        compute_harmonics: bool = True,
        compute_score: bool = True,
        median_filter: bool = False,
        clip_correction: bool = True,
        progress_callback=None,
        canceled_check=None,
    ) -> dict:
        """从 local.db 创建新项目，独立拷贝数据并计算特征。

        Args:
            name: 项目名称
            device_ids: 包含的设备 ID 列表（None=全部）
            voltage_list: 包含的电压值列表（None=全部）
            compute_harmonics: 是否计算 FFT 谐波特征
            compute_score: 是否计算最小二乘评分
            median_filter: 是否应用中值滤波（扫两次）后再算特征
            clip_correction: 是否启用削波矫正
            progress_callback: f(current, total)

        Returns:
            meta dict
        """
        project_dir = os.path.join(self.projects_dir, name)
        if os.path.exists(project_dir):
            # 清理同名残留目录（上次创建失败遗留的）
            import shutil
            shutil.rmtree(project_dir, ignore_errors=True)
        os.makedirs(project_dir)

        try:
            return self._do_create_project(
                name=name,
                project_dir=project_dir,
                label_map=label_map,
                skip_first_n=skip_first_n,
                device_ids=device_ids,
                voltage_list=voltage_list,
                compute_harmonics=compute_harmonics,
                compute_score=compute_score,
                median_filter=median_filter,
                clip_correction=clip_correction,
                progress_callback=progress_callback,
                canceled_check=canceled_check,
            )
        except Exception:
            # 创建失败时清理残留目录
            import shutil
            shutil.rmtree(project_dir, ignore_errors=True)
            raise

    def _do_create_project(self, *, name, project_dir, **kwargs):
        """create_project 实际逻辑，失败时由上层清理目录。"""
        db_path = os.path.join(project_dir, "data.db")
        conn = sqlite3.connect(db_path, timeout=10)
        conn.executescript(_SCHEMA_SQL)

        label_map = kwargs.get("label_map")
        skip_first_n = kwargs.get("skip_first_n", 85510)
        device_ids = kwargs.get("device_ids")
        voltage_list = kwargs.get("voltage_list")
        compute_harmonics = kwargs.get("compute_harmonics", True)
        compute_score = kwargs.get("compute_score", True)
        median_filter = kwargs.get("median_filter", False)
        clip_correction = kwargs.get("clip_correction", True)
        progress_callback = kwargs.get("progress_callback")
        canceled_check = kwargs.get("canceled_check")

        # 构建 WHERE 条件筛选 local.db
        conds = ["1=1"]
        if skip_first_n > 0:
            conds.append("id > ?")
        if device_ids:
            placeholders = ",".join("?" for _ in device_ids)
            conds.append(f"device_id IN ({placeholders})")
        if voltage_list:
            placeholders = ",".join("?" for _ in voltage_list)
            conds.append(f"actual_voltage IN ({placeholders})")
        where = " AND ".join(conds)

        params = []
        if skip_first_n > 0:
            params.append(skip_first_n)
        if device_ids:
            params.extend(device_ids)
        if voltage_list:
            params.extend(voltage_list)

        # 从 local.db 读取
        local_db = LocalDB()
        local_db.connect()
        rows = local_db.conn.execute(
            f"SELECT id, system_time, actual_voltage, temperature, humidity, "
            f"rpm, slave_id, device_id, test_case_code "
            f"FROM records WHERE {where} ORDER BY id",
            params,
        ).fetchall()
        total_records = len(rows)

        from swa.data.loader import compute_harmonics, _median_filter, parse_voltage
        from swa.core.scoring import compute_score as calc_score

        insert_rec_sql = """
            INSERT INTO records
                (source_record_id, system_time, actual_voltage, temperature,
                 humidity, rpm, slave_id, device_id, test_case_code,
                 harm_a1, harm_a1_corrected, harm_a2, harm_error,
                 harm_cycles, harm_noise_pct, harm_clip_ratio,
                 harm_clip_corrected, score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        insert_wave_sql = "INSERT INTO waveforms (record_id, wave_data) VALUES (?, ?)"

        # 预加载所有波形
        record_ids = [row[0] for row in rows]
        waveforms_map = {}
        if record_ids:
            placeholders = ",".join("?" for _ in record_ids)
            wave_rows = local_db.conn.execute(
                f"SELECT record_id, wave_data FROM waveforms "
                f"WHERE record_id IN ({placeholders})",
                record_ids,
            ).fetchall()
            waveforms_map = {w[0]: w[1] for w in wave_rows}

        # 逐条处理：电压替换 + 特征计算（纯 CPU，不写数据库）
        record_params = []
        waveform_params = []
        total = total_records
        for idx, row in enumerate(rows):
            rid, st, volt, temp, humid, rpm, slave, dev_id, tcc = row
            if label_map and volt is not None:
                volt = parse_voltage(str(volt), label_map)

            a1_orig = a1_corrected = a2 = err = cycles = noise_pct = clip_ratio = None
            clip_flag = 0
            score_val = None

            wave_str = waveforms_map.get(rid)
            if wave_str:
                # 只解析一次：转数组用于 score/滤波，转回字符串给 compute_harmonics
                if compute_score or median_filter:
                    wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
                if median_filter:
                    wave = _median_filter(wave, 5)
                    wave = _median_filter(wave, 5)
                    # 滤波后重新生成字符串（避免 compute_harmonics 再解析原始字符串）
                    wave_str = ",".join(f"{v:.2f}" for v in wave)
                if compute_harmonics:
                    a1_orig, a1_corrected, a2, err, cycles, thd, noise_pct, clip_ratio = \
                        compute_harmonics(wave_str, clip_correction=clip_correction)
                if compute_score:
                    score_val = calc_score(wave)
                if clip_ratio and clip_ratio > 0:
                    clip_flag = 1

            record_params.append((
                rid, st, volt, temp, humid, rpm, slave, dev_id, tcc,
                a1_orig, a1_corrected, a2, err, cycles, noise_pct,
                clip_ratio, clip_flag, score_val,
            ))
            waveform_params.append(wave_str)

            if progress_callback and (idx + 1) % 500 == 0:
                progress_callback(idx + 1, total)
            if canceled_check and canceled_check():
                raise RuntimeError("用户取消了创建")

        # 批量写入 records（executemany 比逐条 execute 快很多）
        conn.executemany(insert_rec_sql, record_params)
        conn.commit()

        # 查询新生成的 id（按插入顺序，和 waveform_params 一一对应）
        new_ids = [r[0] for r in conn.execute("SELECT id FROM records ORDER BY id").fetchall()]

        # 批量写入 waveforms
        wave_batch = [(nid, ws) for nid, ws in zip(new_ids, waveform_params) if ws]
        if wave_batch:
            conn.executemany(insert_wave_sql, wave_batch)
        conn.commit()

        local_db.close()

        if progress_callback:
            progress_callback(total, total)

        # 写入元信息
        # 确保所有值为原生 Python 类型，兼容 numpy 类型
        def _to_native(v):
            if isinstance(v, dict):
                return {k: _to_native(val) for k, val in v.items()}
            if isinstance(v, (list, tuple)):
                return [_to_native(x) for x in v]
            if hasattr(v, "item"):
                return v.item()
            return v

        meta = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "total_records": int(total_records),
            "source_db": "local.db",
            "filters": {
                "device_ids": _to_native(device_ids),
                "voltage_list": _to_native(voltage_list),
            },
            "skip_first_n": int(skip_first_n),
            "label_map": {k: float(v) for k, v in (label_map or {}).items()},
            "options": {
                "compute_harmonics": bool(compute_harmonics),
                "compute_score": bool(compute_score),
                "median_filter": bool(median_filter),
                "clip_correction": bool(clip_correction),
            },
        }
        with open(os.path.join(project_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        conn.close()
        if progress_callback:
            progress_callback(processed, total_records)
        return meta

    # ── 删除项目 ──

    def delete_project(self, name: str):
        """删除项目。"""
        self.close()
        project_dir = os.path.join(self.projects_dir, name)
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
