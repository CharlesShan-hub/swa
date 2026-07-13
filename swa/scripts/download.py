"""
从 DM8 下载波形数据到 JSONL 文件。

用法:
    pixi run download --password pwd
    pixi run download --password pwd --output data/raw/weekly.jsonl
"""

import sys, os, json, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click
from swa.db import get_connection, fetch_total, fetch_page


@click.command()
@click.option("--host", default=None, help="数据库地址")
@click.option("--port", type=int, default=None, help="数据库端口")
@click.option("--user", default=None, help="数据库用户")
@click.option("--password", default=None, help="密码")
@click.option("--limit", type=int, default=10000, help="导出条数")
@click.option("--offset", type=int, default=0, help="跳过前 N 条")
@click.option("--batch", type=int, default=200, help="每批条数")
@click.option("--sleep", type=float, default=0.5, help="批次间隔秒数")
@click.option("--output", default="data/raw/downloaded.jsonl", help="输出路径")
def main(host, port, user, password, limit, offset, batch, sleep, output):
    if password is None:
        import getpass
        password = getpass.getpass("请输入密码: ")

    conn = get_connection(host, port, user, password)
    total = fetch_total(conn)
    click.echo(f"总记录数: {total}")

    actual_limit = min(limit, total - offset)
    if actual_limit <= 0:
        click.echo("无数据可导出")
        return

    exported = 0
    total_batches = (actual_limit + batch - 1) // batch

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for batch_no in range(total_batches):
            this_batch = min(batch, actual_limit - exported)
            if this_batch <= 0:
                break

            col_names, rows = fetch_page(conn, offset + exported, this_batch)
            for row in rows:
                record = dict(zip(col_names, row))
                if record.get("SYSTEM_TIME"):
                    record["SYSTEM_TIME"] = str(record["SYSTEM_TIME"])
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            exported += len(rows)
            click.echo(f"  第 {batch_no + 1}/{total_batches} 批: {exported}/{actual_limit}")
            time.sleep(sleep)

    conn.close()
    click.echo(f"完成: {output} ({exported} 条)")


if __name__ == "__main__":
    main()
