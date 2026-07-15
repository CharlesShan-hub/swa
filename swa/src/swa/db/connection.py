"""
数据库层 — DM8 连接与查询
"""

from typing import Optional


DEFAULT_HOST = "10.15.10.1"
DEFAULT_PORT = 5256
DEFAULT_USER = "SYSDBA"


def get_connection(
    host=None,
    port=None,
    user=None,
    password=None,
):
    """
    获取达梦数据库连接。

    参数为 None 时使用默认值或环境变量。
    """
    h = host or DEFAULT_HOST
    p = port or DEFAULT_PORT
    u = user or DEFAULT_USER

    if password is None:
        import os as _os
        password = _os.environ.get("DM_PASSWORD")
    if password is None:
        import getpass
        password = getpass.getpass(f"请输入 {u}@{h} 的密码: ")

    try:
        import dmPython
    except ImportError:
        raise ImportError("dmPython 未安装")

    return dmPython.connect(
        user=u, password=password, server=h, port=p, autoCommit=True
    )


def fetch_total(conn) -> int:
    """获取总记录数。"""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM YS_DB.TB_MODBUS_DEV_POINT")
    return cur.fetchone()[0]


def fetch_page(conn, offset: int, limit: int, fields: Optional[list[str]] = None):
    """
    按 ROWID 分页获取数据。

    Args:
        conn: 数据库连接
        offset: 跳过前 N 条
        limit: 获取条数

    Returns:
        (列名列表, 行列表)
    """
    if fields is None:
        fields = [
            "TEST_CASE_CODE", "SYSTEM_TIME", "RTU_REGS_SLAVE_ID",
            "RTU_REGS_P00_ROTOR_RPM", "RTU_REGS_P00_ENV_TEMP",
            "RTU_REGS_P00_ENV_HUMIDITY", "ACTUAL_VOLTAGE",
            "RTU_REGS_P00_WAVE_DATA",
        ]
    field_csv = ", ".join(fields)

    cur = conn.cursor()
    sql = f"SELECT {field_csv} FROM YS_DB.TB_MODBUS_DEV_POINT ORDER BY ROWID LIMIT ? OFFSET ?"
    cur.execute(sql, (limit, offset))
    rows = cur.fetchall()
    col_names = [desc[0] for desc in cur.description]
    return col_names, rows
