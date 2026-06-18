"""
滑动窗口平均（基于预计算指标）。
在 metrics_xxx.jsonl 上按同电压-同温湿度分组做滑动平均。

用法:
    uv run python scripts/data/smooth.py --name default --window 32
    uv run python scripts/data/smooth.py --name default_4merge --window 16
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from collections import defaultdict
from lib.utils import dataset_dir

METRICS = ["a1", "a3", "a5", "vpp", "kurtosis", "temp", "humid", "rpm"]


def main():
    parser = argparse.ArgumentParser(description="滑动窗口平均（基于预计算指标）")
    parser.add_argument("--name", default="default", help="数据集名称")
    parser.add_argument("--window", type=int, default=32, help="窗口大小（帧数）")
    parser.add_argument("--step", type=int, default=1, help="滑动步长")
    args = parser.parse_args()

    # 加载预计算指标
    in_path = os.path.join(dataset_dir(args.name), f"metrics_{args.name}.jsonl")
    raw_records = []
    with open(in_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))
    print(f"输入: {len(raw_records)} 条")

    # 分桶（电压绝对值 + 温度 + 湿度都一致才滑动平均）
    buckets = defaultdict(list)
    for rec in raw_records:
        key = (rec["voltage_abs"], round(rec.get("temp", 0)), round(rec.get("humid", 0)))
        buckets[key].append(rec)

    # 每个桶先按时间排序，再滑动平均
    out_records = []
    discarded = 0
    for key, bucket in buckets.items():
        bucket.sort(key=lambda r: r.get("time", ""))
        n = len(bucket)
        if n < args.window:
            discarded += n
            continue

        for start in range(0, n - args.window + 1, args.step):
            window = bucket[start:start + args.window]
            mid = window[args.window // 2]
            new_rec = {"time": mid["time"], "voltage_abs": mid["voltage_abs"], "voltage_raw": mid.get("voltage_raw", mid["voltage_abs"])}
            for m in METRICS:
                vals = [r[m] for r in window if m in r]
                if vals:
                    new_rec[m] = round(float(np.mean(vals)), 6)
            out_records.append(new_rec)

    print(f"输出: {len(out_records)} 条（窗口={args.window}, 步长={args.step}）")
    print(f"丢弃: {discarded} 条（桶内不足 {args.window} 条）")

    # 保存
    out_name = f"smooth_{args.name}_w{args.window}"
    out_path = os.path.join(dataset_dir(args.name), f"metrics_{out_name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
