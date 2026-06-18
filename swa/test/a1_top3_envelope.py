"""
对 -55V / -40V / 97V，画 A1 / 温度 / 湿度 随时间变化的散点 + 包络中值拟合曲线。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.rcParams["font.family"] = "Microsoft YaHei"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.interpolate import UnivariateSpline
from lib.data import load
from lib.metrics import extend

records = load("default")
records = extend(records, ["a1"])

N_BINS = 200

def mid_fit(x, y, s=0.001):
    """分箱取上下包络中值 + Spline，s 控制平滑度"""
    bins = np.linspace(x.min(), x.max(), N_BINS + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    mids = []
    used = []
    for i in range(N_BINS):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if mask.sum() < 10:
            continue
        used.append(centers[i])
        upper = np.percentile(y[mask], 90)
        lower = np.percentile(y[mask], 10)
        mids.append((upper + lower) / 2)
    used = np.array(used)
    mids = np.array(mids)
    spl = UnivariateSpline(used, mids, s=s)
    xsmooth = np.linspace(x.min(), x.max(), 1000)
    return xsmooth, spl(xsmooth)

targets = [-55, -40, 97]
colors = ["#2196F3", "#FF5722", "#4CAF50"]
ylabels = ["A1", "温度 (°C)", "湿度 (%)"]
ylabels_short = ["A1", "temp", "humid"]

fig, axes = plt.subplots(len(targets), 3, figsize=(16, 3 * len(targets)))

for row, target_v in enumerate(targets):
    # 收集数据
    data = []
    for rec in records:
        v = rec["ACTUAL_VOLTAGE"]
        if not isinstance(v, (int, float)) or round(v) != target_v:
            continue
        a1 = rec["_a1"]
        t = rec.get("RTU_REGS_P00_ENV_TEMP")
        h = rec.get("RTU_REGS_P00_ENV_HUMIDITY")
        when = rec.get("SYSTEM_TIME", "")
        if a1 is None or t is None or h is None:
            continue
        try:
            dt = datetime.strptime(when[:19], "%Y-%m-%d %H:%M:%S")
            data.append((dt, a1, float(t), float(h)))
        except:
            continue

    data.sort(key=lambda x: x[0])
    times = np.array([(d[0] - data[0][0]).total_seconds() / 3600 for d in data])
    a1s = np.array([d[1] for d in data])
    temps = np.array([d[2] for d in data])
    hums = np.array([d[3] for d in data])
    time_labels = [d[0].strftime("%m/%d\n%H:%M") for d in data]

    values_list = [a1s, temps, hums]
    s_list = [0.001, 0.1, 0.001]  # A1/湿度用小s，温度用大s更平滑

    for col, (vals, color, ylabel) in enumerate(zip(values_list, colors, ylabels)):
        ax = axes[row, col]
        ax.scatter(times, vals, s=3, alpha=0.3, color=color, label="原始数据")
        xs, ys = mid_fit(times, vals, s=s_list[col])
        ax.plot(xs, ys, "-", linewidth=2.5, color="#D32F2F", label="中值拟合")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{target_v}V → {ylabel} (n={len(data)})", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if col == 2:
            ax.legend(fontsize=8, loc="upper right")

    # 时间标签
    step = max(1, len(times) // 8)
    for col in range(3):
        ax = axes[row, col]
        if row == len(targets) - 1:
            ax.set_xlabel("时间", fontsize=8)
            ax.set_xticks(times[::step])
            ax.set_xticklabels([time_labels[i] for i in range(0, len(times), step)], fontsize=6)
        else:
            ax.set_xticks([])

plt.suptitle("-55V / -40V / 97V 的 A1 / 温度 / 湿度 随时间变化（包络中值拟合）", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("test/a1_top3_envelope.png", dpi=150)
plt.close()
print(f"已保存: test/a1_top3_envelope.png（电压: {targets}）")
