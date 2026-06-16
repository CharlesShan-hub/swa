"""
swa: Sine Wave Analysis
非侵入式出口硬压板状态检测 — 主入口

数据源由 src/swa/config/settings.py 中的 data_source.mode 决定：
    mode = "db"     → 从达梦数据库读取
    mode = "local"  → 从本地 JSONL 文件读取

用法：
    uv run python main.py
    uv run python main.py --limit 100
    uv run python main.py --slave-id 1 --since "2026-06-01"
"""

import argparse

from src.swa.config.settings import config
from src.swa.estimation.predictor import predict_from_record


def classify_voltage(v: float) -> str:
    """根据预测电压绝对值判断压板状态"""
    threshold = config.estimation.threshold_abs
    if abs(v) > threshold:
        return "投"
    else:
        return "退"


def show(records: list[dict], limit: int):
    """批量预测并输出"""
    for rec in records[:limit]:
        pred = predict_from_record(rec)
        code = str(rec.get("TEST_CASE_CODE", ""))[:8]
        sid = rec.get("RTU_REGS_SLAVE_ID", "?")
        t = str(rec.get("SYSTEM_TIME", ""))[:19]
        print(f"  {code:>8}  slave={sid}  {t}  电压={pred:>+7.2f}V  {classify_voltage(pred)}")


def main():
    parser = argparse.ArgumentParser(description="swa — 硬压板状态检测")
    parser.add_argument("--slave-id", type=int, default=None, help="按从机地址筛选")
    parser.add_argument("--since", type=str, default=None, help="起始时间")
    parser.add_argument("--limit", type=int, default=20, help="最大记录数")
    args = parser.parse_args()

    mode = config.data_source.mode

    if mode == "local":
        from src.swa.signal_process.loader import load_jsonl
        records = load_jsonl(config.data_source.local_path)
        if args.slave_id is not None:
            records = [r for r in records if r.get("RTU_REGS_SLAVE_ID") == args.slave_id]
        if args.since:
            records = [r for r in records if str(r.get("SYSTEM_TIME", "")) >= args.since]
        print(f"读取本地数据: {len(records)} 条")
        show(records, args.limit)

    elif mode == "db":
        from src.swa.db.connection import get_connection, fetch_records, extract_wave
        conn = get_connection()
        records = fetch_records(conn, slave_id=args.slave_id, limit=args.limit, since=args.since)
        conn.close()
        print(f"数据库: {len(records)} 条")
        show(records, args.limit)

    else:
        print(f"未知 mode: '{mode}'")


if __name__ == "__main__":
    main()
