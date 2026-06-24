"""
用 SavGol 对每段做平滑，输出平滑后的数据集。
每段内的 A1 / 温度 / 湿度 都做 SavGol 滤波。

用法:
    uv run python scripts/data/savgol_smooth.py
    uv run python scripts/data/savgol_smooth.py --name default --window 51 --min-seg 100
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import numpy as np
from datetime import datetime
from scipy.signal import savgol_filter
from collections import defaultdict
from lib.data import load_raw
from lib.metrics import extend
from lib.utils import dataset_dir

MAX_SEG = 500
MIN_SEG = 50
SAVGOL_WINDOW = 51
SAVGOL_ORDER = 3


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SavGol 平滑生成新数据集")
    parser.add_argument("--name", default="default", help="源数据集名称")
    parser.add_argument("--window", type=int, default=SAVGOL_WINDOW, help="SavGol 窗口大小")
    parser.add_argument("--min-seg", type=int, default=MIN_SEG, help="最短段长度")
    parser.add_argument("--max-seg", type=int, default=MAX_SEG, help="最长段长度")
    args = parser.parse_args()

    records = load_raw(args.name)
    records = extend(records, ["a1", "temp", "humid"])
    print(f"输入: {len(records)} 条")

    # 按时间排序，取需要的数据
    all_data = []
    for rec in records:
        v = rec["ACTUAL_VOLTAGE"]
        a1 = rec["_a1"]
        t = rec.get("RTU_REGS_P00_ENV_TEMP")
        h = rec.get("RTU_REGS_P00_ENV_HUMIDITY")
        when = rec.get("SYSTEM_TIME", "")
        if not isinstance(v, (int, float)) or a1 is None or t is None or h is None:
            continue
        try:
            dt = datetime.strptime(when[:19], "%Y-%m-%d %H:%M:%S")
            all_data.append((dt, round(v), a1, float(t), float(h), rec))
        except:
            continue

    all_data.sort(key=lambda x: x[0])

    # 分段
    segments = []
    cur_v = None
    current = []

    for d in all_data:
        dt, v, a1, t, h, rec = d
        if v != cur_v or len(current) >= args.max_seg:
            if current and len(current) >= args.min_seg:
                segments.append([cur_v, current])
            elif current and len(current) < args.min_seg and segments:
                segments[-1][1].extend(current)
            current = []
            cur_v = v
        current.append((dt, a1, t, h, rec))

    if current:
        if len(current) >= args.min_seg:
            segments.append([cur_v, current])
        elif segments:
            segments[-1][1].extend(current)

    print(f"分段: {len(segments)}")

    # 对每段做 SavGol 平滑，输出平滑后的记录
    out_records = []
    discarded = 0

    for seg in segments:
        v, data = seg[0], seg[1]
        a1s = np.array([d[1] for d in data])
        temps = np.array([d[2] for d in data])
        hums = np.array([d[3] for d in data])
        n = len(a1s)

        window = min(args.window, n if n % 2 == 1 else n - 1)
        if window < 5 or window % 2 == 0:
            discarded += n
            continue

        sg_a1 = savgol_filter(a1s, window, SAVGOL_ORDER)
        sg_temp = savgol_filter(temps, window, SAVGOL_ORDER)
        sg_hum = savgol_filter(hums, window, SAVGOL_ORDER)

        for i, (dt, _, _, _, rec) in enumerate(data):
            new_rec = dict(rec)
            new_rec["_a1"] = float(sg_a1[i])
            new_rec["_a1_raw"] = float(a1s[i])
            new_rec["_temp"] = float(sg_temp[i])
            new_rec["_humid"] = float(sg_hum[i])
            new_rec["_temp_raw"] = float(temps[i])
            new_rec["_humid_raw"] = float(hums[i])
            out_records.append(new_rec)

    print(f"输出: {len(out_records)} 条 (丢弃 {discarded} 条)")

    # 保存为 jsonl
    out_name = f"savgol_{args.name}"
    out_dir = os.path.join(dataset_dir(args.name), "..", out_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{out_name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"已保存: {out_path}")

    # 统计各电压
    groups = defaultdict(list)
    for rec in out_records:
        v = rec.get("ACTUAL_VOLTAGE")
        if isinstance(v, (int, float)):
            groups[round(v)].append(rec)
    print(f"\n各电压数:")
    for v in sorted(groups):
        print(f"  {v:>4}V: {len(groups[v])} 条")


if __name__ == "__main__":
    main()
