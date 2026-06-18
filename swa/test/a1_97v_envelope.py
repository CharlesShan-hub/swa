"""
97V 的 A1 / 湿度 / 温度：上下包络 + 中值拟合。
先对密集点做分箱，取上下边界和均值，再用 Spline 拟合。
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

data = []
for rec in records:
    v = rec["ACTUAL_VOLTAGE"]
    if not isinstance(v, (int, float)) or round(v) != 97:
        continue
    a1 = rec["_a1"]
    t = rec.get("RTU_REGS_P00_ENV_TEMP")
    h = rec.get("RTU_REGS_P00_ENV_HUMIDITY")
    when = rec.get("SYSTEM_TIME", "")
    if a1 is None or t is None or h is None:
        continue
    try:
        t, h = float(t), float(h)
        dt = datetime.strptime(when[:19], "%Y-%m-%d %H:%M:%S")
    except:
        continue
    data.append((dt, a1, t, h))

data.sort(key=lambda x: x[0])
times = np.array([(d[0] - data[0][0]).total_seconds() / 3600 for d in data])
a1s = np.array([d[1] for d in data])
temps = np.array([d[2] for d in data])
hums = np.array([d[3] for d in data])
time_labels = [d[0].strftime("%m/%d\n%H:%M") for d in data]

# ── 分箱取上下包络 + 均值 ──
N_BINS = 200  # 时间分箱数

def envelope_fit(x, y, n_bins=N_BINS, s_smooth=0.001):
    """分箱 -> 取上/下/中值 -> Spline 拟合"""
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    upper_vals = []
    lower_vals = []
    mean_vals = []
    center_used = []

    for i in range(n_bins):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if mask.sum() < 10:
            continue
        center_used.append(bin_centers[i])
        vals = y[mask]
        upper_vals.append(np.percentile(vals, 90))   # 上包络（90%分位）
        lower_vals.append(np.percentile(vals, 10))   # 下包络（10%分位）
        mean_vals.append(np.mean(vals))

    center_used = np.array(center_used)
    upper_vals = np.array(upper_vals)
    lower_vals = np.array(lower_vals)
    mean_vals = np.array(mean_vals)

    # 取中间值 = (上包络 + 下包络) / 2
    mid_vals = (upper_vals + lower_vals) / 2

    xsmooth = np.linspace(x.min(), x.max(), 1000)

    # 拟合三条线
    spl_upper = UnivariateSpline(center_used, upper_vals, s=s_smooth)
    spl_lower = UnivariateSpline(center_used, lower_vals, s=s_smooth)
    spl_mid = UnivariateSpline(center_used, mid_vals, s=s_smooth)
    spl_mean = UnivariateSpline(center_used, mean_vals, s=s_smooth)

    return xsmooth, {
        "upper": spl_upper(xsmooth),
        "lower": spl_lower(xsmooth),
        "mid": spl_mid(xsmooth),
        "mean": spl_mean(xsmooth),
        "upper_raw": (center_used, upper_vals),
        "lower_raw": (center_used, lower_vals),
        "mid_raw": (center_used, mid_vals),
    }

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

for ax, values, ylabel, color, title in [
    (ax1, a1s, "A1", "#2196F3", "A1"),
    (ax2, hums, "湿度 (%)", "#4CAF50", "湿度"),
    (ax3, temps, "温度 (°C)", "#FF5722", "温度"),
]:
    # 原始散点
    ax.scatter(times, values, s=2, alpha=0.3, color=color, label="原始数据")

    # 分箱包络拟合
    xsmooth, curves = envelope_fit(times, values)

    # 画中值拟合线（粗红线）
    ax.plot(xsmooth, curves["mid"], "-", linewidth=3, color="#D32F2F", label="中值拟合")

    ax.set_ylabel(ylabel)
    ax.set_title(f"97V 的 {title} 随时间变化（包络+中值）", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

step = max(1, len(times) // 10)
ax3.set_xticks(times[::step])
ax3.set_xticklabels([time_labels[i] for i in range(0, len(times), step)], fontsize=7)
ax3.set_xlabel("时间")

plt.tight_layout()
plt.savefig("test/a1_97v_envelope.png", dpi=150)
plt.close()
print(f"已保存: test/a1_97v_envelope.png（{N_BINS} 分箱，{len(data)} 条）")
