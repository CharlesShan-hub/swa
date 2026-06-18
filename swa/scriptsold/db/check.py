"""
Check database connectivity and analyze voltage/temperature/humidity distribution.

Standalone script - no project imports required.
Tests connection to the Dameng DM8 database.

Usage:
    uv run python scripts/db/check.py                  # prompts for password
    uv run python scripts/db/check.py --password pwd   # supply password directly
    uv run python scripts/db/check.py --help           # show all options
"""

import click


# ── Default connection config ──
DEFAULT_HOST = "10.15.10.1"
DEFAULT_PORT = 5256
DEFAULT_USER = "SYSDBA"


def get_connection(host, port, user, password):
    """Create a Dameng DM8 database connection."""
    try:
        import dmPython
    except ImportError:
        raise ImportError(
            "dmPython is not installed. Run: uv pip install dmPython"
        )

    return dmPython.connect(
        user=user,
        password=password,
        server=host,
        port=port,
        autoCommit=True,
    )


def bucket_stats(values, bucket_fn, label_unit="", width=6):
    """Group values by bucket_fn, sort by bucket key, show count and percentage."""
    buckets = {}
    for v in values:
        k = bucket_fn(v)
        buckets[k] = buckets.get(k, 0) + 1
    total = len(values)
    for k in sorted(buckets):
        n = buckets[k]
        pct = n / total * 100
        click.echo(f"    {k:>6}{label_unit}: {n:>{width}}条 ({pct:>5.1f}%)")


def voltage_bucket(v):
    """Round voltage to nearest 10V."""
    return round(v / 10) * 10


def temp_bucket(t):
    """Round temperature to nearest 5°C."""
    return round(t / 5) * 5


def humid_bucket(h):
    """Round humidity to nearest 5%."""
    return round(h / 5) * 5


@click.command(help="Check Dameng DM8 database connectivity and data distribution.")
@click.option("--host", default=DEFAULT_HOST, show_default=True,
              help="Database host address")
@click.option("--port", type=int, default=DEFAULT_PORT, show_default=True,
              help="Database port")
@click.option("--user", default=DEFAULT_USER, show_default=True,
              help="Database user")
@click.option("--head", type=int, default=3, show_default=True,
              help="显示最新 N 条记录")
@click.option("--password", default=None,
              help="Database password (will be prompted if omitted)")
@click.option("--analyze/--no-analyze", default=True,
              help="是否显示分布分析 (默认显示)")
def main(host, port, user, head, password, analyze):
    """Check connectivity and inspect the TB_MODBUS_DEV_POINT table."""
    if password is None:
        password = click.prompt("Password", hide_input=True)
    click.echo("=" * 50)
    click.echo("  Database Connectivity Check")
    click.echo("=" * 50)
    click.echo(f"  Host: {host}")
    click.echo(f"  Port: {port}")
    click.echo(f"  User: {user}")
    click.echo()

    try:
        conn = get_connection(host, port, user, password)
        cur = conn.cursor()
        click.echo("  [OK] Connection established\n")
    except Exception as e:
        click.echo(f"  [FAIL] Cannot connect: {e}")
        raise SystemExit(1)

    try:
        cur.execute("SELECT COUNT(*) FROM YS_DB.TB_MODBUS_DEV_POINT")
        total = cur.fetchone()[0]
        click.echo(f"  [INFO] TB_MODBUS_DEV_POINT: {total} records")

        # ── Latest N records ──
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
        click.echo(f"\n  Latest {head} records:")
        click.echo(f"  {'测试点':>10} | {'时间':>19} | {'电压':>6} | {'温度':>5} | {'湿度':>4}")
        click.echo(f"  {'-'*10}-+-{'-'*19}-+-{'--'*3}--+-{'---'*2}-+-{'--'*2}--")
        for r in rows:
            code = r[0] or ""
            click.echo(f"  {code:>10} | {r[1]} | {str(r[2]):>6} | {str(r[3] or ''):>5} | {str(r[4] or ''):>4}")

        # ── Distribution analysis ──
        if analyze and total > 0:
            click.echo("\n" + "=" * 50)
            click.echo("  Data Distribution Analysis")
            click.echo("=" * 50)

            # Voltage distribution
            cur.execute("SELECT ACTUAL_VOLTAGE FROM YS_DB.TB_MODBUS_DEV_POINT")
            all_v = [row[0] for row in cur.fetchall() if row[0] is not None]
            click.echo(f"\n  Voltage distribution (n={len(all_v)}):")
            bucket_stats(all_v, voltage_bucket, "V")

            # Temperature distribution
            cur.execute("SELECT RTU_REGS_P00_ENV_TEMP FROM YS_DB.TB_MODBUS_DEV_POINT")
            all_t = [float(row[0]) for row in cur.fetchall() if row[0] is not None]
            click.echo(f"\n  Temperature distribution (n={len(all_t)}):")
            bucket_stats(all_t, temp_bucket, "°C")
            click.echo(f"    Range: {min(all_t):.1f} ~ {max(all_t):.1f} °C")
            click.echo(f"    Mean:  {sum(all_t)/len(all_t):.1f} °C")

            # Humidity distribution
            cur.execute("SELECT RTU_REGS_P00_ENV_HUMIDITY FROM YS_DB.TB_MODBUS_DEV_POINT")
            all_h = [float(row[0]) for row in cur.fetchall() if row[0] is not None]
            click.echo(f"\n  Humidity distribution (n={len(all_h)}):")
            bucket_stats(all_h, humid_bucket, "%")
            click.echo(f"    Range: {min(all_h):.1f} ~ {max(all_h):.1f} %")
            click.echo(f"    Mean:  {sum(all_h)/len(all_h):.1f} %")

    except Exception as e:
        click.echo(f"  [FAIL] Query error: {e}")

    conn.close()
    click.echo("\n  [OK] Connection closed")


if __name__ == "__main__":
    main()
