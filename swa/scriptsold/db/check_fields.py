"""
查看 DEVICE_ID / LINE_TYPE / INSTALLATION_ANGLE 的变化情况。

用法:
    uv run python scripts/db/check_fields.py --password pwd
"""

import click

DEFAULT_HOST = "10.15.10.1"
DEFAULT_PORT = 5256
DEFAULT_USER = "SYSDBA"


def get_connection(host, port, user, password):
    try:
        import dmPython
    except ImportError:
        raise ImportError("dmPython 未安装，请运行: uv pip install dmPython")
    return dmPython.connect(user=user, password=password,
                            server=host, port=port, autoCommit=True)


@click.command()
@click.option("--host", default=DEFAULT_HOST)
@click.option("--port", type=int, default=DEFAULT_PORT)
@click.option("--user", default=DEFAULT_USER)
@click.option("--password", default=None)
def main(host, port, user, password):
    if password is None:
        password = click.prompt("Password", hide_input=True)

    conn = get_connection(host, port, user, password)
    cur = conn.cursor()

    # 1. DEVICE_ID 有多少种
    cur.execute("""
        SELECT DEVICE_ID, COUNT(*) AS cnt,
               MIN(SYSTEM_TIME) AS first_seen,
               MAX(SYSTEM_TIME) AS last_seen
        FROM YS_DB.TB_MODBUS_DEV_POINT
        WHERE DEVICE_ID IS NOT NULL
        GROUP BY DEVICE_ID
        ORDER BY cnt DESC
    """)
    rows = cur.fetchall()
    print(f"\n=== DEVICE_ID 分布 ({len(rows)} 种) ===")
    print(f"  {'DEVICE_ID':>50} | {'数量':>8} | {'首次出现':>19} | {'最后出现':>19}")
    print(f"  {'-'*50}-+-{'-'*8}-+-{'-'*19}-+-{'-'*19}")
    for r in rows:
        print(f"  {r[0]:>50} | {r[1]:>8} | {str(r[2]):>19} | {str(r[3]):>19}")

    # 2. LINE_TYPE 有多少种
    cur.execute("""
        SELECT LINE_TYPE, COUNT(*) AS cnt,
               MIN(SYSTEM_TIME) AS first_seen,
               MAX(SYSTEM_TIME) AS last_seen
        FROM YS_DB.TB_MODBUS_DEV_POINT
        WHERE LINE_TYPE IS NOT NULL
        GROUP BY LINE_TYPE
        ORDER BY cnt DESC
    """)
    rows = cur.fetchall()
    print(f"\n=== LINE_TYPE 分布 ({len(rows)} 种) ===")
    print(f"  {'LINE_TYPE':>30} | {'数量':>8} | {'首次出现':>19} | {'最后出现':>19}")
    print(f"  {'-'*30}-+-{'-'*8}-+-{'-'*19}-+-{'-'*19}")
    for r in rows:
        print(f"  {r[0]:>30} | {r[1]:>8} | {str(r[2]):>19} | {str(r[3]):>19}")

    # 3. INSTALLATION_ANGLE 有多少种
    cur.execute("""
        SELECT INSTALLATION_ANGLE, COUNT(*) AS cnt,
               MIN(SYSTEM_TIME) AS first_seen,
               MAX(SYSTEM_TIME) AS last_seen
        FROM YS_DB.TB_MODBUS_DEV_POINT
        WHERE INSTALLATION_ANGLE IS NOT NULL
        GROUP BY INSTALLATION_ANGLE
        ORDER BY cnt DESC
    """)
    rows = cur.fetchall()
    print(f"\n=== INSTALLATION_ANGLE 分布 ({len(rows)} 种) ===")
    print(f"  {'INSTALLATION_ANGLE':>30} | {'数量':>8} | {'首次出现':>19} | {'最后出现':>19}")
    print(f"  {'-'*30}-+-{'-'*8}-+-{'-'*19}-+-{'-'*19}")
    for r in rows:
        print(f"  {r[0]:>30} | {r[1]:>8} | {str(r[2]):>19} | {str(r[3]):>19}")

    # 4. DEVICE_ID × LINE_TYPE × INSTALLATION_ANGLE 交叉
    cur.execute("""
        SELECT DEVICE_ID, LINE_TYPE, INSTALLATION_ANGLE, COUNT(*) AS cnt
        FROM YS_DB.TB_MODBUS_DEV_POINT
        GROUP BY DEVICE_ID, LINE_TYPE, INSTALLATION_ANGLE
        ORDER BY cnt DESC
    """)
    rows = cur.fetchall()
    print(f"\n=== DEVICE_ID × LINE_TYPE × INSTALLATION_ANGLE ({len(rows)} 种组合) ===")
    print(f"  {'DEVICE_ID':>50} | {'LINE_TYPE':>20} | {'角度':>10} | {'数量':>8}")
    print(f"  {'-'*50}-+-{'-'*20}-+-{'-'*10}-+-{'-'*8}")
    for r in rows:
        print(f"  {r[0]:>50} | {str(r[1] or ''):>20} | {str(r[2] or ''):>10} | {r[3]:>8}")

    conn.close()


if __name__ == "__main__":
    main()
