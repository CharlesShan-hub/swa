"""
验证 extend 对 512 点和 2048 点波形都能正确处理，
并对比各指标在合并前后的方差变化。

用法:
    uv run python test/test_extend.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from collections import defaultdict
from lib.data import load
from lib.metrics import extend

# ── 需要对比的指标 ──
METRICS = ["a1", "a3", "a5", "vpp", "kurtosis", "skewness"]


def _stats(records, label):
    """按电压分组统计各指标的均值、标准差、变异系数"""
    groups = defaultdict(lambda: {k: [] for k in METRICS})
    for rec in records:
        v = rec["ACTUAL_VOLTAGE"]
        if not isinstance(v, (int, float)):
            continue
        vb = round(v)
        for k in METRICS:
            val = rec.get(f"_{k}")
            if val is not None:
                groups[vb][k].append(val)

    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    print(f"  {'电压':>6} | {'n':>6}", end="")
    for k in METRICS:
        print(f" | {k+'均值':>10} | {k+' CV':>8}", end="")
    print()
    print(f"  {'-'*6}-+-{'-'*6}", end="")
    for _ in METRICS:
        print(f"-+-{'-'*10}-+-{'-'*8}", end="")
    print()

    for vb in sorted(groups):
        d = groups[vb]
        n = len(d[METRICS[0]])
        if n < 5:
            continue
        print(f"  {vb:>6} | {n:>6}", end="")
        for k in METRICS:
            vals = np.array(d[k])
            mean_v = np.mean(vals)
            cv = np.std(vals) / mean_v * 100 if mean_v != 0 else 0
            print(f" | {mean_v:>10.4f} | {cv:>7.1f}%", end="")
        print()

    return groups


# ── 加载与扩展 ──
print("加载 default (512 点)...")
recs = load("default")
recs = extend(recs, METRICS)
g1 = _stats(recs, "default (单段 512 点)")

print("\n加载 default_4merge (2048 点)...")
recs4 = load("default_4merge")
recs4 = extend(recs4, METRICS)
g2 = _stats(recs4, "default_4merge (4 段拼接平均)")

# ── 方差改善对比 ──
print(f"\n{'='*80}")
print(f"  合并前后 CV 对比 (CV_merge / CV_single)")
print(f"{'='*80}")
print(f"  {'电压':>6}", end="")
for k in METRICS:
    print(f" | {k+' CV比':>10}", end="")
print()
print(f"  {'-'*6}", end="")
for _ in METRICS:
    print(f"-+-{'-'*10}", end="")
print()

common_v = sorted(set(g1.keys()) & set(g2.keys()))
for vb in common_v:
    n1 = len(g1[vb][METRICS[0]])
    n2 = len(g2[vb][METRICS[0]])
    if n1 < 5 or n2 < 5:
        continue
    print(f"  {vb:>6}", end="")
    for k in METRICS:
        vals1 = np.array(g1[vb][k])
        vals2 = np.array(g2[vb][k])
        cv1 = np.std(vals1) / np.mean(vals1) * 100 if np.mean(vals1) != 0 else 0
        cv2 = np.std(vals2) / np.mean(vals2) * 100 if np.mean(vals2) != 0 else 0
        ratio = cv2 / cv1 if cv1 != 0 else 0
        print(f" | {ratio:>10.2f}", end="")
    print()

# ── 理论值 ──
print(f"\n  理论改善: √4 = 2.0x (CV 应降为原来的 0.5 倍)")
print(f"  说明: 比值 < 1 表示合并后更稳定")
