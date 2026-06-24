"""
Leave-2-out 交叉验证：对所有电压，每次留 2 个做测试，看混合模型的泛化能力。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
from collections import defaultdict
from lib.data import load
from lib.traditional.hybrid import train_hybrid, PhysicalModel

# 加载数据
records = load("smooth_all_w32")
features = ["a1", "temp", "humid"]

# 所有可测试的电压（需有足够数据）
groups = defaultdict(list)
for rec in records:
    v = rec.get("ACTUAL_VOLTAGE")
    if isinstance(v, (int, float)):
        groups[round(v)].append(rec)

# 筛选 > 30 条的电压
all_voltages = sorted(v for v, d in groups.items() if len(d) > 30)
print(f"所有电压（>30条）: {all_voltages}")
print(f"共 {len(all_voltages)} 个电压，将进行 C({len(all_voltages)},2)={len(all_voltages)*(len(all_voltages)-1)//2} 组测试\n")

results = []

for i in range(len(all_voltages)):
    for j in range(i + 1, len(all_voltages)):
        test_v = [all_voltages[i], all_voltages[j]]
        train_v = [v for v in all_voltages if v not in test_v]

        result = train_hybrid(
            records, features,
            train_voltages=train_v,
            test_voltages=test_v,
            vol_limit=300,
        )

        results.append({
            "test_voltages": test_v,
            "test_mae_phys": result.get("test_mae_phys"),
            "test_mae": result.get("test_mae", -1),
            "k_global": result.get("k_global", 0),
        })

        print(f"  测试 {test_v}: 物理基线={result.get('test_mae_phys',-1):.1f}V → 修正后={result.get('test_mae',-1):.1f}V")

print(f"\n{'='*60}")
print(f"  Leave-2-out 结果汇总")
print(f"{'='*60}")

mae_phys_all = [r.get("test_mae_phys") for r in results if r.get("test_mae_phys") is not None]
mae_all = [r.get("test_mae") for r in results if r.get("test_mae") is not None]

if mae_phys_all:
    print(f"\n  物理基线 MAE: 平均={np.mean(mae_phys_all):.1f}V, 中位数={np.median(mae_phys_all):.1f}V")
    print(f"             最小={np.min(mae_phys_all):.1f}V, 最大={np.max(mae_phys_all):.1f}V")
if mae_all:
    print(f"  修正后 MAE:   平均={np.mean(mae_all):.1f}V, 中位数={np.median(mae_all):.1f}V")
    print(f"             最小={np.min(mae_all):.1f}V, 最大={np.max(mae_all):.1f}V")
    print(f"             平均改善率={((np.mean(mae_phys_all)-np.mean(mae_all))/np.mean(mae_phys_all)*100):.0f}%")

# 按测试集 MAE 排序
print(f"\n{'='*60}")
print(f"  按修正后 MAE 排序")
print(f"{'='*60}")
for r in sorted(results, key=lambda x: x.get("test_mae", 999)):
    phys = r.get("test_mae_phys", 0)
    final = r.get("test_mae", 0)
    impr = ((phys - final) / phys * 100) if phys > 0 else 0
    print(f"  {r['test_voltages']}: {phys:.1f}V → {final:.1f}V (改善 {impr:.0f}%)")
