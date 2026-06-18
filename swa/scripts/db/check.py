"""
检查数据库连接及温湿度分布。

用法:
    uv run python scripts/db/check.py                  # 弹窗输入密码
    uv run python scripts/db/check.py --password pwd   # 直接传密码
    uv run python scripts/db/check.py --help           # 所有参数
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import click
from lib.db import get_connection, DEFAULT_HOST, DEFAULT_PORT, DEFAULT_USER


def bucket_stats(values, bucket_fn, label_unit="", width=6):
    """按 bucket_fn 分组统计并打印"""
    buckets = {}
    for v in values:
        k = bucket_fn(v)
        buckets[k] = buckets.get(k, 0) + 1
    total = len(values)
    for k in sorted(buckets):
        n = buckets[k]
        pct = n / total * 100
        click.echo(f"    {k:>6}{label_unit}: {n:>{width}} ({pct:>5.1f}%)")


def voltage_bucket(v):
    return round(v / 10) * 10

def temp_bucket(t):
    return round(t / 5) * 5

def humid_bucket(h):
    return round(h / 5) * 5


@click.command(help="检查 DM8 数据库连接及数据分布。")
@click.option("--host", default=None, help=f"数据库地址 (默认: {DEFAULT_HOST})")
@click.option("--port", type=int, default=None, help=f"数据库端口 (默认: {DEFAULT_PORT})")
@click.option("--user", default=None, help=f"数据库用户 (默认: {DEFAULT_USER})")
@click.option("--head", type=int, default=3, show_default=True, help="显示最新 N 条")
@click.option("--password", default=None, help="密码（不传则弹窗输入）")
@click.option("--analyze/--no-analyze", default=True, help="是否显示分布分析")
def main(host, port, user, head, password, analyze):
    if password is None:
        import getpass
        password = getpass.getpass("请输入密码: ")

    click.echo("=" * 50)
    click.echo("  数据库连接检查")
    click.echo("=" * 50)

    conn = get_connection(host, port, user, password)
    cur = conn.cursor()
    click.echo("  [OK] 连接成功\n")

    try:
        cur.execute("SELECT COUNT(*) FROM YS_DB.TB_MODBUS_DEV_POINT")
        total = cur.fetchone()[0]
        click.echo(f"  [INFO] TB_MODBUS_DEV_POINT: {total} 条记录")

        # 最新 N 条
        cur.execute(f"""
            SELECT TEST_CASE_CODE, SYSTEM_TIME, ACTUAL_VOLTAGE,
                   RTU_REGS_P00_ENV_TEMP, RTU_REGS_P00_ENV_HUMIDITY
            FROM (
                SELECT TEST_CASE_CODE, SYSTEM_TIME, ACTUAL_VOLTAGE,
                       RTU_REGS_P00_ENV_TEMP, RTU_REGS_P00_ENV_HUMIDITY
                FROM YS_DB.TB_MODBUS_DEV_POINT
                ORDER BY SYSTEM_TIME DESC
            )
            WHERE ROWNUM <= {head}
        """)
        rows = cur.fetchall()
        click.echo(f"\n  最新 {head} 条记录:")
        click.echo(f"  {'测试点':>10} | {'时间':>19} | {'电压':>6} | {'温度':>5} | {'湿度':>4}")
        click.echo(f"  {'-'*10}-+-{'-'*19}-+-{'-'*6}-+-{'-'*5}-+-{'-'*4}")
        for r in rows:
            code = r[0] or ""
            t_str = str(r[3]) if r[3] is not None else ""
            h_str = str(r[4]) if r[4] is not None else ""
            click.echo(f"  {code:>10} | {r[1]} | {str(r[2]):>6} | {t_str:>5} | {h_str:>4}")

        # 分布分析
        if analyze and total > 0:
            click.echo("\n" + "=" * 50)
            click.echo("  数据分布分析")
            click.echo("=" * 50)

            cur.execute("SELECT ACTUAL_VOLTAGE FROM YS_DB.TB_MODBUS_DEV_POINT")
            all_v = [row[0] for row in cur.fetchall() if row[0] is not None]
            click.echo(f"\n  电压分布 (n={len(all_v)}):")
            from collections import Counter
            for val, cnt in Counter(all_v).most_common():
                click.echo(f"    {str(val):>10}: {cnt} 条")

            cur.execute("SELECT RTU_REGS_P00_ENV_TEMP FROM YS_DB.TB_MODBUS_DEV_POINT")
            all_t = []
            for row in cur.fetchall():
                v = row[0]
                if v is not None:
                    try:
                        all_t.append(float(v))
                    except (ValueError, TypeError):
                        pass
            click.echo(f"\n  温度分布 (n={len(all_t)}):")
            bucket_stats(all_t, temp_bucket, "°C")
            if all_t:
                click.echo(f"    Range: {min(all_t):.1f} ~ {max(all_t):.1f} °C")
                click.echo(f"    Mean:  {sum(all_t)/len(all_t):.1f} °C")

            cur.execute("SELECT RTU_REGS_P00_ENV_HUMIDITY FROM YS_DB.TB_MODBUS_DEV_POINT")
            all_h = []
            for row in cur.fetchall():
                v = row[0]
                if v is not None:
                    try:
                        all_h.append(float(v))
                    except (ValueError, TypeError):
                        pass
            click.echo(f"\n  湿度分布 (n={len(all_h)}):")
            bucket_stats(all_h, humid_bucket, "%")
            if all_h:
                click.echo(f"    Range: {min(all_h):.1f} ~ {max(all_h):.1f} %")
                click.echo(f"    Mean:  {sum(all_h)/len(all_h):.1f} %")

    except Exception as e:
        click.echo(f"  [FAIL] 查询出错: {e}")

    conn.close()
    click.echo("\n  [OK] 连接已关闭")


if __name__ == "__main__":
    main()
