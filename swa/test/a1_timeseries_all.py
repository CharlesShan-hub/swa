"""
对每个数据量 > 100 的电压，画 A1 / 温度 / 湿度 随时间变化的散点图。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.rcParams["font.family"] = "Microsoft YaHei"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime
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
    when = rec.get("SYSTEM_TIME", "")
    if not isinstance(v, (int, float)) or a1 is None or t is None or h is None:
        continue
    try:
        dt = datetime.strptime(when[:19], "%Y-%m-%d %H:%M:%S")
        groups[round(v)].append((dt, a1, float(t), float(h)))
    except:
        continue

voltages = sorted(v for v, d in groups.items() if len(d) > 100)
n = len(voltages)

fig, axes = plt.subplots(n, 3, figsize=(16, 2.5 * n))
if n == 1:
    axes = axes.reshape(1, 3)

for row, v in enumerate(voltages):
    data = sorted(groups[v], key=lambda x: x[0])  # 按时间排序
    times = np.array([(d[0] - data[0][0]).total_seconds() / 3600 for d in data])
    a1s = np.array([d[1] for d in data])
    temps = np.array([d[2] for d in data])
    hums = np.array([d[3] for d in data])
    time_labels = [d[0].strftime("%m/%d\n%H:%M") for d in data]

    # A1 ~ 时间
    ax = axes[row, 0]
    ax.scatter(times, a1s, s=3, alpha=0.3, color="#2196F3")
    ax.set_ylabel("A1", fontsize=8)
    ax.set_title(f"{v}V (n={len(data)})", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

    # 温度 ~ 时间
    ax = axes[row, 1]
    ax.scatter(times, temps, s=3, alpha=0.3, color="#FF5722")
    ax.set_ylabel("温度 (°C)", fontsize=8)
    ax.set_title(f"{v}V (n={len(data)})", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

    # 湿度 ~ 时间
    ax = axes[row, 2]
    ax.scatter(times, hums, s=3, alpha=0.3, color="#4CAF50")
    ax.set_ylabel("湿度 (%)", fontsize=8)
    ax.set_title(f"{v}V (n={len(data)})", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

    # 底部行加时间标签
    if row == n - 1:
        step = max(1, len(times) // 8)
        for col in range(3):
            axes[row, col].set_xlabel("时间", fontsize=8)
            axes[row, col].set_xticks(times[::step])
            axes[row, col].set_xticklabels([time_labels[i] for i in range(0, len(times), step)], fontsize=6)
    else:
        for col in range(3):
            axes[row, col].set_xticks([])

plt.suptitle("各电压 A1 / 温度 / 湿度 随时间变化", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("test/a1_timeseries_all.png", dpi=150)
plt.close()
print(f"已保存: test/a1_timeseries_all.png（{n} 个电压）")
