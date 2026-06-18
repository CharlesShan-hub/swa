"""
查看数据集的电压分布。

用法:
    uv run python scripts/traditional/test.py
    uv run python scripts/traditional/test.py --dataset default_4merge
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
from collections import Counter
from lib.data import load


def main():
    parser = argparse.ArgumentParser(description="查看数据集电压分布")
    parser.add_argument("--dataset", default="default", help="数据集名称")
    args = parser.parse_args()

    records = load(args.dataset)
    counter = Counter()
    for rec in records:
        v = rec.get("ACTUAL_VOLTAGE")
        if isinstance(v, str):
            counter["未知"] += 1
        elif isinstance(v, (int, float)):
            key = round(v)
            counter[key] += 1

    total = sum(counter.values())
    print(f"{args.dataset}: {total} 条")
    print(f"\n{'电压':>8} | {'数量':>6} | {'占比':>7}")
    print(f"{'-'*8}-+-{'-'*6}-+-{'-'*7}")
    for k in sorted(counter, key=lambda x: str(x)):
        n = counter[k]
        print(f"{str(k):>8} | {n:>6} | {n/total*100:>6.1f}%")


if __name__ == "__main__":
    main()
