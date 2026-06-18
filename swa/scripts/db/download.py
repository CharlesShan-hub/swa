"""
从 Dameng DM8 数据库导出波形数据到 JSONL 文件。

使用 ROWID 游标分页高效导出。

用法:
    uv run python scripts/db/download.py                          # 交互模式
    uv run python scripts/db/download.py --limit 5000 --password pwd
    uv run python scripts/db/download.py --help
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json, time
import click
from lib.db import get_connection, DEFAULT_HOST, DEFAULT_PORT, DEFAULT_USER
from lib.utils import dataset_dir


FIELDS = [
    "TEST_CASE_CODE", "SYSTEM_TIME", "RTU_REGS_SLAVE_ID",
    "RTU_REGS_P00_ROTOR_RPM", "RTU_REGS_P00_ENV_TEMP",
    "RTU_REGS_P00_ENV_HUMIDITY", "ACTUAL_VOLTAGE",
    "RTU_REGS_P00_WAVE_DATA",
]
FIELD_CSV = ", ".join(FIELDS)


@click.command(help="从 DM8 导出波形数据到 JSONL。")
@click.option("--host", default=None, help=f"数据库地址 (默认: {DEFAULT_HOST})")
@click.option("--port", type=int, default=None, help=f"数据库端口 (默认: {DEFAULT_PORT})")
@click.option("--user", default=None, help=f"数据库用户 (默认: {DEFAULT_USER})")
@click.option("--password", default=None, help="密码（不传则弹窗输入）")
@click.option("--limit", type=int, default=None, help="导出条数（不传则交互输入）")
@click.option("--batch", "batch_size", type=int, default=None, help="每批条数")
@click.option("--sleep", type=float, default=None, help="批次间隔秒数")
@click.option("--offset", type=int, default=None, help="跳过前 N 条")
@click.option("--append", is_flag=True, default=False, help="追加到已有文件")
@click.option("--output", default=None, help="输出路径（不传则交互输入）")
def main(host, port, user, password, limit, batch_size, sleep, offset, append, output):
    if password is None:
        import getpass
        password = getpass.getpass("请输入密码: ")
    if limit is None:
        limit = click.prompt("导出条数", type=int, default=38000)
    if batch_size is None:
        batch_size = click.prompt("每批条数", type=int, default=500)
    if sleep is None:
        sleep = click.prompt("批次间隔秒数", type=float, default=0.5)
    if offset is None:
        offset = click.prompt("跳过前 N 条", type=int, default=0)
    if output is None:
        default_dir = dataset_dir("default")
        default_out = os.path.join(default_dir, "default.jsonl")
        output = click.prompt("输出路径", default=default_out)

    # 先连接获取总数
    conn = get_connection(host, port, user, password)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM YS_DB.TB_MODBUS_DEV_POINT")
    total = cur.fetchone()[0]
    conn.close()

    if offset >= total:
        click.echo(f"\n  [FAIL] 偏移量 ({offset}) 超过总记录数 ({total})。")
        raise SystemExit(1)
    if offset + limit > total:
        click.echo(f"\n  [FAIL] 范围 {offset+1} ~ {offset+limit} 超过总记录数 ({total})。")
        click.echo(f"  最大 limit={offset} 时为 {total - offset}。")
        raise SystemExit(1)

    click.echo()
    click.echo("=" * 50)
    click.echo("  导出摘要")
    click.echo("=" * 50)
    click.echo(f"  模式:     {'追加' if append else '覆盖'}")
    click.echo(f"  输出:     {output}")
    click.echo(f"  范围:     第 {offset+1} ~ {offset+limit} 条 ({limit} 条, 共 {total})")
    click.echo(f"  每批:     {batch_size} 条, 间隔 {sleep}s")
    click.confirm("  确认?", default=True, abort=True)
    click.echo()

    conn = get_connection(host, port, user, password)
    cur = conn.cursor()

    cur.execute("SELECT MIN(ROWID) FROM YS_DB.TB_MODBUS_DEV_POINT")
    min_rowid = cur.fetchone()[0]
    current_rowid = min_rowid + offset - 1

    remaining = total - offset
    actual_limit = min(limit, remaining)
    if actual_limit <= 0:
        click.echo("没有更多数据可导出。")
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
                click.echo(f"  [WARN] 第 {batch_no + 1} 批返回 0 条，提前结束")
                break

            col_names = [desc[0] for desc in cur.description]
            for row in rows:
                record = dict(zip(col_names, row))
                if record.get("SYSTEM_TIME"):
                    record["SYSTEM_TIME"] = str(record["SYSTEM_TIME"])
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            exported += len(rows)
            current_rowid += this_batch
            click.echo(f"  第 {batch_no + 1}/{total_batches} 批: {exported}/{actual_limit} 条已导出")
            time.sleep(sleep)

    conn.close()
    click.echo(f"\n  [OK] 导出完成: {output} ({exported} 条)")


if __name__ == "__main__":
    main()
