"""
数据库连接与查询模块

提供达梦 DM8 数据库的连接管理和常用查询。

实际表结构（从 TB_MODBUS_DEV_POINT 确认）：
    - TEST_CASE_CODE, DEVICE_ID, SYSTEM_TIME, ACTUAL_VOLTAGE
    - ENV_TEMPERATURE, ENV_HUMIDITY, RUNNING_HOURS, INSTALLATION_ANGLE, LINE_TYPE
    - RTU_REGS_BUILD_TIME ~ RTU_REGS_P00_WAVE_DATA, SESSION_START_TIME
    - 波形字段：RTU_REGS_P00_WAVE_DATA（逗号分隔的电压值，如 "1.332,1.351,..."）
"""

from typing import Optional

from ..config.settings import config


# ============================================================
# 连接管理
# ============================================================

def get_connection():
    """
    获取达梦数据库连接。

    连接参数从 .env 文件 → os.environ → config.db 读取。
    """
    db = config.db

    if not db.password:
        raise ValueError(
            "数据库密码未配置。\n"
            "请在项目根目录的 .env 文件中设置 DM8_PASSWORD=你的密码"
        )

    try:
        import dmPython
    except ImportError:
        raise ImportError("请先安装 dmPython: uv pip install dmPython")

    conn = dmPython.connect(
        user=db.user,
        password=db.password,
        server=db.host,
        port=db.port,
        autoCommit=True,
    )
    return conn


# ============================================================
# 查询
# ============================================================

TABLE_NAME = "YS_DB.TB_MODBUS_DEV_POINT"


def fetch_records(
    conn,
    slave_id: Optional[int] = None,
    limit: int = 20,
    since: Optional[str] = None,
) -> list[dict]:
    """
    查询 TB_MODBUS_DEV_POINT 报文记录。

    Args:
        conn: 数据库连接
        slave_id: 按 RTU_REGS_SLAVE_ID 筛选
        limit: 最大记录数
        since: 起始时间，如 '2026-01-01'

    Returns:
        记录列表
    """
    sql = f"SELECT * FROM {TABLE_NAME} WHERE 1=1"
    params = []

    if slave_id is not None:
        sql += " AND RTU_REGS_SLAVE_ID = ?"
        params.append(slave_id)
    if since:
        sql += " AND SYSTEM_TIME >= ?"
        params.append(since)

    sql += " ORDER BY SYSTEM_TIME DESC"
    if limit:
        sql += f" LIMIT {limit}"

    cur = conn.cursor()
    cur.execute(sql, params)
    cols = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    return [dict(zip(cols, row)) for row in rows]


def extract_wave(record: dict) -> Optional[str]:
    """
    从记录中提取 WAVE_DATA 波形字符串。

    Args:
        record: 从 fetch_records 返回的单条记录

    Returns:
        逗号分隔的电压值字符串，如 "1.332,1.351,..."
    """
    return record.get("RTU_REGS_P00_WAVE_DATA")
