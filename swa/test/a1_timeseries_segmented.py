"""
按时间顺序分段绘制各电压的 A1 / 温度 / 湿度。
每段 = 同一个电压连续采集，最长 500 点，最短 100 点。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.rcParams["font.family"] = "Microsoft YaHei"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from lib.data import load
from lib.metrics import extend

records = load("default")
records = extend(records, ["a1"])

# 按时间排序所有数据
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
        all_data.append((dt, round(v), a1, float(t), float(h)))
    except:
        continue

all_data.sort(key=lambda x: x[0])

# ── 分段：同电压连续 → 一段，最长 500 点，最短 100 点 ──
MAX_SEG = 500
MIN_SEG = 100

segments = []  # [(voltage, [data...]), ...]
cur_v = None
current = []

for d in all_data:
    dt, v, a1, t, h = d
    if v != cur_v or len(current) >= MAX_SEG:
        if current and len(current) >= MIN_SEG:
            segments.append([cur_v, current])
        elif current and len(current) < MIN_SEG and segments:
            segments[-1][1].extend(current)
        current = []
        cur_v = v
    current.append((dt, a1, t, h))

if current:
    if len(current) >= MIN_SEG:
        segments.append([cur_v, current])
    elif segments:
        segments[-1][1].extend(current)

print(f"总点数: {len(all_data)}")
print(f"分段数: {len(segments)}")
for seg in segments:
    v = seg[0]; data = seg[1]
    print(f"  {v:>4}V: {len(data)} 点  {data[0][0].strftime('%m/%d %H:%M')} ~ {data[-1][0].strftime('%m/%d %H:%M')}")

# ── 绘图 ──
n = len(segments)
fig, axes = plt.subplots(n, 3, figsize=(16, 1.8 * n))
if n == 1:
    axes = axes.reshape(1, 3)

for row, seg in enumerate(segments):
    v = seg[0]; data = seg[1]
    times = np.array([(d[0] - data[0][0]).total_seconds() / 60 for d in data])
    a1s = np.array([d[1] for d in data])
    temps = np.array([d[2] for d in data])
    hums = np.array([d[3] for d in data])
    dt_start = data[0][0].strftime("%m/%d %H:%M")
    dt_end = data[-1][0].strftime("%m/%d %H:%M")

    # A1 ~ 时间
    ax = axes[row, 0]
    ax.plot(times, a1s, "-", linewidth=0.8, color="#2196F3", alpha=0.7)
    ax.set_ylabel("A1", fontsize=8)
    ax.set_title(f"第{row+1}段: {v}V ({len(data)}点, {dt_start}~{dt_end})", fontsize=9, pad=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

    # 温度 ~ 时间
    ax = axes[row, 1]
    ax.plot(times, temps, "-", linewidth=0.8, color="#FF5722", alpha=0.7)
    ax.set_ylabel("温度 (°C)", fontsize=8)
    ax.set_title(f"第{row+1}段: 温度 ({len(data)}点)", fontsize=9, pad=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

    # 湿度 ~ 时间
    ax = axes[row, 2]
    ax.plot(times, hums, "-", linewidth=0.8, color="#4CAF50", alpha=0.7)
    ax.set_ylabel("湿度 (%)", fontsize=8)
    ax.set_title(f"第{row+1}段: 湿度 ({len(data)}点)", fontsize=9, pad=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

    # 时间标签
    for col in range(3):
        ax = axes[row, col]
        if len(times) > 1:
            step = max(1, len(times) // 4)
            ax.set_xticks(times[::step])
            ax.set_xticklabels([f"{times[i]:.0f}min" for i in range(0, len(times), step)], fontsize=5)
        else:
            ax.set_xticks([])

fig.suptitle("各电压按时间分段（连续采集段，每段最长500点）", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("test/a1_timeseries_segmented.png", dpi=150)
plt.close()
print(f"\n已保存: test/a1_timeseries_segmented.png（{n} 段）")
