"""
预计算指标：读取原始 jsonl，计算 A1/A3/A5/Vpp/Kurtosis 等，存为轻量版。

用法:
    uv run python scripts/data/precompute.py --name default
    uv run python scripts/data/precompute.py --name default_4merge
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from lib.data import load_raw
from lib.metrics import extend
from lib.utils import dataset_dir

METRICS = ["a1", "a3", "a5", "vpp", "kurtosis", "temp", "humid", "rpm", "dc"]


def main():
    parser = argparse.ArgumentParser(description="预计算指标并生成轻量数据集")
    parser.add_argument("--name", default="default", help="数据集名称")
    args = parser.parse_args()

    records = load_raw(args.name)
    records = extend(records, METRICS)
    print(f"加载: {len(records)} 条")

    out_path = os.path.join(dataset_dir(args.name), f"metrics_{args.name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            v = rec["ACTUAL_VOLTAGE"]
            if not isinstance(v, (int, float)):
                continue

            out = {
                "time": rec.get("SYSTEM_TIME", ""),
                "voltage_abs": round(abs(v)),
                "voltage_raw": v,
            }
            for m in METRICS:
                val = rec.get(f"_{m}")
                if val is not None:
                    out[m] = round(float(val), 6)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
