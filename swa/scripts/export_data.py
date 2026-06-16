"""
分批从达梦数据库导出波形数据到本地 JSON 文件。

原理：用 ROWID 游标分页，WHERE ROWID > 上一批最大值，不走 OFFSET。
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.swa.db.connection import get_connection

# 只导出需要的字段
FIELDS = [
    "TEST_CASE_CODE", "SYSTEM_TIME", "RTU_REGS_SLAVE_ID",
    "RTU_REGS_P00_ROTOR_RPM", "RTU_REGS_P00_ENV_TEMP",
    "RTU_REGS_P00_ENV_HUMIDITY", "ACTUAL_VOLTAGE",
    "RTU_REGS_P00_WAVE_DATA",
]
FIELD_CSV = ", ".join(FIELDS)


def export_data(output_path: str, limit: int, batch_size: int = 500,
                sleep_sec: float = 0.5, offset: int = 0, append: bool = False):
    conn = get_connection()
    cur = conn.cursor()

    # 总记录数
    cur.execute("SELECT COUNT(*) FROM YS_DB.TB_MODBUS_DEV_POINT")
    total = cur.fetchone()[0]

    # 定位起始 ROWID
    cur.execute(f"SELECT MIN(ROWID) FROM YS_DB.TB_MODBUS_DEV_POINT")
    min_rowid = cur.fetchone()[0]
    current_rowid = min_rowid + offset - 1  # 从第 offset 条的前一条开始

    remaining = total - offset
    actual_limit = min(limit, remaining)
    if actual_limit <= 0:
        print(f"已经没有更多数据了")
        conn.close()
        return

    print(f"数据库共 {total} 条，从 ROWID {current_rowid + 1} 开始，导出 {actual_limit} 条，分批 {batch_size} 条/批")

    exported = 0
    total_batches = (actual_limit + batch_size - 1) // batch_size
    sql = f"SELECT {FIELD_CSV} FROM YS_DB.TB_MODBUS_DEV_POINT WHERE ROWID > ? ORDER BY ROWID LIMIT ?"

    write_mode = "a" if append else "w"
    with open(output_path, write_mode, encoding="utf-8") as f:
        for batch_no in range(total_batches):
            this_batch = min(batch_size, actual_limit - exported)
            if this_batch <= 0:
                break

            cur.execute(sql, (current_rowid, this_batch))
            rows = cur.fetchall()
            if not rows:
                print(f"  ⚠ 第 {batch_no + 1} 批返回 0 条，提前结束")
                break

            col_names = [desc[0] for desc in cur.description]
            for row in rows:
                record = dict(zip(col_names, row))
                if record.get("SYSTEM_TIME"):
                    record["SYSTEM_TIME"] = str(record["SYSTEM_TIME"])
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            exported += len(rows)
            current_rowid += this_batch  # ROWID 连续递增，直接加

            print(f"  批次 {batch_no + 1}/{total_batches}: 已导出 {exported}/{actual_limit} 条")
            time.sleep(sleep_sec)

    conn.close()
    print(f"\n导出完成: {output_path} ({exported} 条)")


def main():
    parser = argparse.ArgumentParser(description="从达梦分批导出波形数据（ROWID 游标）")
    parser.add_argument("--limit", type=int, default=1000, help="导出条数 (默认 1000)")
    parser.add_argument("--batch", type=int, default=500, help="每批条数 (默认 500)")
    parser.add_argument("--sleep", type=float, default=0.5, help="每批间隔秒数 (默认 0.5)")
    parser.add_argument("--offset", type=int, default=0, help="跳过前 N 条 (默认 0)")
    parser.add_argument("--append", action="store_true", help="追加到已有文件")
    parser.add_argument("--output", default="data/exported_data.jsonl",
                        help="输出路径 (默认 data/exported_data.jsonl)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    export_data(args.output, limit=args.limit, batch_size=args.batch,
                sleep_sec=args.sleep, offset=args.offset, append=args.append)


if __name__ == "__main__":
    main()
