"""
分析原始数据集的电压、温度、湿度分布及交叉分布

用法:
    uv run python scripts/analyze_voltage_distribution.py
    uv run python scripts/analyze_voltage_distribution.py --data data/u1.jsonl
"""

import sys
import os
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.utils.loader import load_jsonl


def bucket_v(v):
    return round(abs(v) / 10) * 10

def bucket_t(t):
    return round(t / 5) * 5

def bucket_h(h):
    return round(h / 5) * 5


def print_cross_table(title, rows, col_keys, row_label="", col_label=""):
    """Print cross-distribution table with fixed-width columns for perfect alignment."""
    if not col_keys or not rows:
        return
    ncols = len(col_keys)
    # Each data column: " | " (3) + content (12) = 15 chars per column
    COL_W = 15
    DATA_W = COL_W - 3  # 12 chars of actual content per column
    # Row header: row_label(8) + " | " (3) + count(6) = 17 chars
    ROW_W = 17
    total_w = ROW_W + ncols * COL_W

    # Build header
    header = f"{row_label:>8} | {'count':>6}"
    for ck in col_keys:
        col_str = f"{ck}{col_label}"  # e.g., "25C" or "25%"
        header += f" | {col_str:>{DATA_W}}"
    sep = "-" * total_w

    print(f"\n{'=' * total_w}")
    print(f"{title:^{total_w}}")
    print(f"{'=' * total_w}")
    print(header)
    print(sep)
    for rk in sorted(rows):
        total = sum(rows[rk].values())
        line = f"{rk:>8} | {total:>6}"
        for ck in col_keys:
            c = rows[rk].get(ck, 0)
            if c == 0:
                line += f" | {'':>12}"
            else:
                pct = c / total * 100
                line += f" | {c:>4}({pct:>5.1f}%)"
        print(line)
    print(f"{'=' * total_w}")


def analyze_data(file_path):
    records = load_jsonl(file_path, extract_features=False)
    print(f"total samples: {len(records)}")

    # ── 提取数据 ──
    voltages = []
    temps = []
    hums = []
    vt = defaultdict(lambda: defaultdict(int))   # {v_bucket: {t_bucket: count}}
    vh = defaultdict(lambda: defaultdict(int))   # {v_bucket: {h_bucket: count}}

    for rec in records:
        v = rec.get("ACTUAL_VOLTAGE")
        t = rec.get("RTU_REGS_P00_ENV_TEMP")
        h = rec.get("RTU_REGS_P00_ENV_HUMIDITY")

        if v is not None:
            vf = abs(float(v))
            voltages.append(vf)
            vb = bucket_v(vf)

            if t is not None:
                tf = float(t)
                if len(temps) == 0 or not np.isnan(tf):
                    temps.append(tf)
                tb = bucket_t(tf)
                vt[vb][tb] += 1

            if h is not None:
                hf = float(h)
                if len(hums) == 0 or not np.isnan(hf):
                    hums.append(hf)
                hb = bucket_h(hf)
                vh[vb][hb] += 1

    # ── 1. Voltage distribution ──
    print(f"\n{'='*50}")
    print(f"{'1. Voltage distribution':^50}")
    print(f"{'='*50}")
    v_buckets = defaultdict(int)
    for v in voltages:
        v_buckets[bucket_v(v)] += 1
    total_v = len(voltages)
    for vb in sorted(v_buckets):
        n = v_buckets[vb]
        print(f"  {vb:>6}V: {n:>7} ({n/total_v*100:>5.1f}%)")
    print(f"  Range: {min(voltages):.1f} ~ {max(voltages):.1f} V")

    # ── 2. Temperature distribution ──
    print(f"\n{'='*50}")
    print(f"{'2. Temperature distribution':^50}")
    print(f"{'='*50}")
    t_buckets = defaultdict(int)
    for t in temps:
        t_buckets[bucket_t(t)] += 1
    total_t = len(temps)
    for tb in sorted(t_buckets):
        n = t_buckets[tb]
        print(f"  {tb:>5}C: {n:>7} ({n/total_t*100:>5.1f}%)")
    print(f"  Range: {min(temps):.1f} ~ {max(temps):.1f} °C")
    print(f"  Mean:  {np.mean(temps):.1f} °C")

    # ── 3. Humidity distribution ──
    print(f"\n{'='*50}")
    print(f"{'3. Humidity distribution':^50}")
    print(f"{'='*50}")
    h_buckets = defaultdict(int)
    for h in hums:
        h_buckets[bucket_h(h)] += 1
    total_h = len(hums)
    for hb in sorted(h_buckets):
        n = h_buckets[hb]
        print(f"  {hb:>4}%: {n:>7} ({n/total_h*100:>5.1f}%)")
    print(f"  Range: {min(hums):.1f} ~ {max(hums):.1f} %")
    print(f"  Mean:  {np.mean(hums):.1f} %")

    # ── 4. Voltage x Temperature ──
    all_tb = sorted(set(tb for vb in vt for tb in vt[vb]))
    print_cross_table("4. Voltage x Temperature", vt, all_tb, "voltage", "C")

    # ── 5. Voltage x Humidity ──
    all_hb = sorted(set(hb for vb in vh for hb in vh[vb]))
    print_cross_table("5. Voltage x Humidity", vh, all_hb, "voltage", "%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze voltage/temp/humid distribution in dataset")
    parser.add_argument("-d", "--data", default="data/exported_data.jsonl",
                        help="data file path (default: data/exported_data.jsonl)")
    args = parser.parse_args()
    print("=" * 50)
    print(f"  Analyze {args.data}")
    print("=" * 50)
    analyze_data(args.data)
