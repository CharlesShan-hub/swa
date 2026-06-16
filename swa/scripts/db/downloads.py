"""
Export waveform data from Dameng DM8 database to JSONL file.

Uses ROWID cursor pagination for efficient batch export.
Standalone script - no project imports required.

Usage:
    uv run python scripts/db/downloads.py                          # full interactive mode
    uv run python scripts/db/downloads.py --limit 5000 --password pwd  # partial args
    uv run python scripts/db/downloads.py --help                   # show all options
"""

import json
import os
import time
import click


# ── Default connection config ──
DEFAULT_HOST = "10.15.10.1"
DEFAULT_PORT = 5256
DEFAULT_USER = "SYSDBA"

# Fields to export
FIELDS = [
    "TEST_CASE_CODE", "SYSTEM_TIME", "RTU_REGS_SLAVE_ID",
    "RTU_REGS_P00_ROTOR_RPM", "RTU_REGS_P00_ENV_TEMP",
    "RTU_REGS_P00_ENV_HUMIDITY", "ACTUAL_VOLTAGE",
    "RTU_REGS_P00_WAVE_DATA",
]
FIELD_CSV = ", ".join(FIELDS)


def get_connection(host, port, user, password):
    """Create a Dameng DM8 database connection."""
    try:
        import dmPython
    except ImportError:
        raise ImportError("dmPython is not installed. Run: uv pip install dmPython")

    return dmPython.connect(
        user=user,
        password=password,
        server=host,
        port=port,
        autoCommit=True,
    )


@click.command(help="Export waveform data from Dameng DM8 to JSONL.")
@click.option("--host", default=DEFAULT_HOST, show_default=True, help="Database host")
@click.option("--port", type=int, default=DEFAULT_PORT, show_default=True, help="Database port")
@click.option("--user", default=DEFAULT_USER, show_default=True, help="Database user")
@click.option("--password", default=None, help="Database password (prompted if omitted)")
@click.option("--limit", type=int, default=None, help="Number of records to export (prompted if omitted)")
@click.option("--batch", "batch_size", type=int, default=None, help="Records per batch (prompted if omitted)")
@click.option("--sleep", type=float, default=None, help="Sleep seconds between batches (prompted if omitted)")
@click.option("--offset", type=int, default=None, help="Skip first N records (prompted if omitted)")
@click.option("--append", is_flag=True, default=False, help="Append to existing file")
@click.option("--output", default=None, help="Output file path (prompted if omitted)")
def main(host, port, user, password, limit, batch_size, sleep, offset, append, output):
    """Export waveform data from Dameng database to JSONL file."""
    if password is None:
        password = click.prompt("Password", hide_input=True)
    if limit is None:
        limit = click.prompt("Records to export", type=int, default=38000)
    if batch_size is None:
        batch_size = click.prompt("Records per batch", type=int, default=500)
    if sleep is None:
        sleep = click.prompt("Sleep seconds between batches", type=float, default=0.5)
    if offset is None:
        offset = click.prompt("Skip first N records", type=int, default=0)
    if output is None:
        output = click.prompt("Output file path", default="data/exported_data.jsonl")

    # Brief summary before confirmation
    from_row = offset + 1
    to_row = offset + limit

    # Connect first to validate range
    conn = get_connection(host, port, user, password)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM YS_DB.TB_MODBUS_DEV_POINT")
    total = cur.fetchone()[0]
    conn.close()

    if offset >= total:
        click.echo(f"\n  [FAIL] Offset ({offset}) exceeds total records ({total}). Nothing to export.")
        click.echo("  Please re-run with correct parameters.")
        raise SystemExit(1)
    if offset + limit > total:
        click.echo(f"\n  [FAIL] Range {from_row} ~ {to_row} exceeds total records ({total}).")
        click.echo(f"  Max limit with offset={offset} is {total - offset}. Please re-run.")
        raise SystemExit(1)

    click.echo()
    click.echo("=" * 50)
    click.echo("  Export Summary")
    click.echo("=" * 50)
    click.echo(f"  Mode:     {'Append' if append else 'Overwrite'}")
    click.echo(f"  Output:   {output}")
    click.echo(f"  Range:    record {from_row} ~ {to_row} ({limit} records, total {total})")
    click.echo(f"  Batch:    {batch_size} records/batch, {sleep}s interval")
    click.confirm("  Proceed?", default=True, abort=True)
    click.echo()

    # Reconnect for actual export
    conn = get_connection(host, port, user, password)
    cur = conn.cursor()

    # Locate starting ROWID
    cur.execute("SELECT MIN(ROWID) FROM YS_DB.TB_MODBUS_DEV_POINT")
    min_rowid = cur.fetchone()[0]
    current_rowid = min_rowid + offset - 1

    remaining = total - offset
    actual_limit = min(limit, remaining)
    if actual_limit <= 0:
        click.echo("No more data to export.")
        conn.close()
        return

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    exported = 0
    total_batches = (actual_limit + batch_size - 1) // batch_size
    sql = f"SELECT {FIELD_CSV} FROM YS_DB.TB_MODBUS_DEV_POINT WHERE ROWID > ? ORDER BY ROWID LIMIT ?"

    write_mode = "a" if append else "w"
    with open(output, write_mode, encoding="utf-8") as f:
        for batch_no in range(total_batches):
            this_batch = min(batch_size, actual_limit - exported)
            if this_batch <= 0:
                break

            cur.execute(sql, (current_rowid, this_batch))
            rows = cur.fetchall()
            if not rows:
                click.echo(f"  [WARN] Batch {batch_no + 1} returned 0 rows, stopping early")
                break

            col_names = [desc[0] for desc in cur.description]
            for row in rows:
                record = dict(zip(col_names, row))
                if record.get("SYSTEM_TIME"):
                    record["SYSTEM_TIME"] = str(record["SYSTEM_TIME"])
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            exported += len(rows)
            current_rowid += this_batch
            click.echo(f"  Batch {batch_no + 1}/{total_batches}: {exported}/{actual_limit} exported")
            time.sleep(sleep)

    conn.close()
    click.echo(f"\n  [OK] Export complete: {output} ({exported} records)")


if __name__ == "__main__":
    main()
