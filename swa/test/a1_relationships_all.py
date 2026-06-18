"""
对每个数据量 > 100 的电压，画 A1 / 湿度 / 温度之间的关系散点图。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.rcParams["font.family"] = "Microsoft YaHei"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from lib.data import load
from lib.metrics import extend

records = load("default")
records = extend(records, ["a1"])

# 按电压分组
groups = defaultdict(list)
for rec in records:
    v = rec["ACTUAL_VOLTAGE"]
    a1 = rec["_a1"]
    t = rec.get("RTU_REGS_P00_ENV_TEMP")
    h = rec.get("RTU_REGS_P00_ENV_HUMIDITY")
    if not isinstance(v, (int, float)) or a1 is None or t is None or h is None:
        continue
    try:
        groups[round(v)].append((a1, float(t), float(h)))
    except:
        continue

voltages = sorted(v for v, d in groups.items() if len(d) > 100)
n = len(voltages)

fig, axes = plt.subplots(n, 3, figsize=(14, 2.5 * n))
if n == 1:
    axes = axes.reshape(1, 3)

for row, v in enumerate(voltages):
    data = np.array(groups[v])
    a1s = data[:, 0]
    temps = data[:, 1]
    hums = data[:, 2]

    # 1. A1 vs 湿度
    ax = axes[row, 0]
    ax.scatter(hums, a1s, s=3, alpha=0.3, color="#2196F3")
    ax.set_xlabel("湿度 (%)", fontsize=8)
    ax.set_ylabel("A1", fontsize=8)
    ax.set_title(f"{v}V (n={len(data)})", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

    # 2. A1 vs 温度
    ax = axes[row, 1]
    ax.scatter(temps, a1s, s=3, alpha=0.3, color="#FF5722")
    ax.set_xlabel("温度 (°C)", fontsize=8)
    ax.set_ylabel("A1", fontsize=8)
    ax.set_title(f"{v}V (n={len(data)})", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

    # 3. 湿度 vs 温度
    ax = axes[row, 2]
    ax.scatter(temps, hums, s=3, alpha=0.3, color="#4CAF50")
    ax.set_xlabel("温度 (°C)", fontsize=8)
    ax.set_ylabel("湿度 (%)", fontsize=8)
    ax.set_title(f"{v}V (n={len(data)})", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

plt.suptitle("各电压 A1 / 湿度 / 温度 关系散点图", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("test/a1_relationships_all.png", dpi=150)
plt.close()
print(f"已保存: test/a1_relationships_all.png（{n} 个电压）")
