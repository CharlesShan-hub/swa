"""
查看数据库表结构和一条完整数据的所有字段。

用法:
    uv run python scripts/db/inspect_record.py --password pwd
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
@click.option("--limit", type=int, default=1, help="显示多少条 (默认 1)")
def main(host, port, user, password, limit):
    if password is None:
        password = click.prompt("Password", hide_input=True)

    conn = get_connection(host, port, user, password)
    cur = conn.cursor()

    # 取一条完整数据
    cur.execute(f"SELECT * FROM YS_DB.TB_MODBUS_DEV_POINT WHERE ROWNUM <= {limit}")
    rows = cur.fetchall()
    col_names = [desc[0] for desc in cur.description]

    print(f"\n表字段 ({len(col_names)} 个):")
    print(f"  {'#':>3} | {'字段名':40s} | {'非空?'}")
    print(f"  {'-'*3}-+-{'-'*40}-+-{'-'*6}")
    for i, name in enumerate(col_names):
        print(f"  {i+1:>3} | {name:40s} |")

    print(f"\n{'='*80}")
    print(f"  最新 {limit} 条完整记录（所有字段）")
    print(f"{'='*80}")
    for ri, row in enumerate(rows):
        print(f"\n  ── 记录 #{ri+1} ──")
        for i, name in enumerate(col_names):
            val = row[i]
            if val is None:
                display = "(NULL)"
            elif name == "RTU_REGS_P00_WAVE_DATA":
                # 波形只显示前 5 个点 + 长度
                parts = str(val).split(",")
                display = f"[{parts[0]}, {parts[1]}, {parts[2]}, ... {len(parts)} 点]"
            elif isinstance(val, str) and len(val) > 80:
                display = val[:77] + "..."
            else:
                display = str(val)
            print(f"    {name:35s} = {display}")

    conn.close()


if __name__ == "__main__":
    main()
