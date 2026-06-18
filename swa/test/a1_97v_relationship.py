"""
从 97V 的三条 Spline 拟合曲线上取点，查看 A1 / 湿度 / 温度三者之间的关系。
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

# ── 分箱包络 → 中值拟合 ──
N_BINS = 200
S_SMOOTH = 0.001

def get_mid_fit(x, y):
    bins = np.linspace(x.min(), x.max(), N_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    mid_vals = []
    centers_used = []
    for i in range(N_BINS):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if mask.sum() < 10:
            continue
        centers_used.append(bin_centers[i])
        upper = np.percentile(y[mask], 90)
        lower = np.percentile(y[mask], 10)
        mid_vals.append((upper + lower) / 2)
    centers_used = np.array(centers_used)
    mid_vals = np.array(mid_vals)
    spl = UnivariateSpline(centers_used, mid_vals, s=S_SMOOTH)
    return spl

# 三条 Spline
spl_a1 = get_mid_fit(times, a1s)
spl_hum = get_mid_fit(times, hums)
spl_temp = get_mid_fit(times, temps)

# ── 在时间轴上均匀取点 ──
N_POINTS = 100
t_smooth = np.linspace(times.min(), times.max(), N_POINTS)
a1_fit = spl_a1(t_smooth)
hum_fit = spl_hum(t_smooth)
temp_fit = spl_temp(t_smooth)

# ── 画三个散点关系图 ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. A1 vs 湿度
ax = axes[0]
ax.scatter(hum_fit, a1_fit, s=30, c=t_smooth, cmap="viridis", alpha=0.8, edgecolors="black", linewidth=0.5)
ax.set_xlabel("湿度 (%)")
ax.set_ylabel("A1")
ax.set_title("A1 vs 湿度（97V 拟合曲线）")
ax.grid(True, alpha=0.3)

# 2. A1 vs 温度
ax = axes[1]
ax.scatter(temp_fit, a1_fit, s=30, c=t_smooth, cmap="viridis", alpha=0.8, edgecolors="black", linewidth=0.5)
ax.set_xlabel("温度 (°C)")
ax.set_ylabel("A1")
ax.set_title("A1 vs 温度（97V 拟合曲线）")
ax.grid(True, alpha=0.3)

# 3. 湿度 vs 温度
ax = axes[2]
ax.scatter(temp_fit, hum_fit, s=30, c=t_smooth, cmap="viridis", alpha=0.8, edgecolors="black", linewidth=0.5)
ax.set_xlabel("温度 (°C)")
ax.set_ylabel("湿度 (%)")
ax.set_title("湿度 vs 温度（97V 拟合曲线）")
ax.grid(True, alpha=0.3)

# 颜色条
cbar = fig.colorbar(ax.collections[0], ax=axes, orientation="horizontal", pad=0.08, aspect=40)
cbar.set_label("时间 →")

plt.suptitle("97V 拟合曲线关系（颜色 = 时间先后）", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("test/a1_97v_relationships.png", dpi=150)
plt.close()
print(f"已保存: test/a1_97v_relationships.png（{N_POINTS} 个采样点）")
