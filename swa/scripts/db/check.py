"""
Check database connectivity.

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


@click.command(help="Check Dameng DM8 database connectivity.")
@click.option("--host", default=DEFAULT_HOST, show_default=True,
              help="Database host address")
@click.option("--port", type=int, default=DEFAULT_PORT, show_default=True,
              help="Database port")
@click.option("--user", default=DEFAULT_USER, show_default=True,
              help="Database user")
@click.option("--password", default=None,
              help="Database password (will be prompted if omitted)")
def main(host, port, user, password):
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

        cur.execute("""
            SELECT TEST_CASE_CODE, SYSTEM_TIME, ACTUAL_VOLTAGE
            FROM YS_DB.TB_MODBUS_DEV_POINT
            WHERE ROWNUM <= 3
            ORDER BY SYSTEM_TIME DESC
        """)
        rows = cur.fetchall()
        click.echo("\n  Latest 3 records:")
        for r in rows:
            click.echo(f"    {r[0]:>10} | {r[1]} | {r[2]}")
    except Exception as e:
        click.echo(f"  [FAIL] Query error: {e}")

    conn.close()
    click.echo("\n  [OK] Connection closed")


if __name__ == "__main__":
    main()
