"""
查看 u0/u1/u2 的 A1 分布，跟训练集对比。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from lib.data import load, group_by_voltage
from lib.metrics import extend

# 训练集各电压的 A1 均值
train_recs = load("default")
train_recs = extend(train_recs, ["a1"])
groups = group_by_voltage(train_recs)
print("训练集 A1 均值:")
for v in sorted(groups, key=lambda x: str(x)):
    vals = [r["_a1"] for r in groups[v] if r["_a1"] is not None]
    if len(vals) > 5:
        print(f"  {str(v):>5}V: A1_mean={np.mean(vals):.4f}, A1_std={np.std(vals):.4f}")

print()

# 测试集 A1 分布
for name, true_v in [("u0", 87), ("u1", 43), ("u2", 36)]:
    recs = load(name)
    recs = extend(recs, ["a1"])
    a1s = [r["_a1"] for r in recs if r["_a1"] is not None]
    a1s = np.array(a1s)
    print(f"{name} (真实 {true_v}V): A1 mean={a1s.mean():.4f}, std={a1s.std():.4f}, range=[{a1s.min():.4f}, {a1s.max():.4f}]")
