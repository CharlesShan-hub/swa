"""
97V 的 A1 / 湿度 / 温度 随时间变化散点 + Spline 拟合曲线。
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

# Spline 拟合：s 控制平滑度，越大越平滑
SMOOTH = 30

def spline_fit(x, y, s=SMOOTH):
    """x 必须严格递增"""
    spl = UnivariateSpline(x, y, s=s)
    xsmooth = np.linspace(x.min(), x.max(), 1000)
    ysmooth = spl(xsmooth)
    return xsmooth, ysmooth, spl

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

for ax, values, ylabel, color, title in [
    (ax1, a1s, "A1", "#2196F3", "A1"),
    (ax2, hums, "湿度 (%)", "#4CAF50", "湿度"),
    (ax3, temps, "温度 (°C)", "#FF5722", "温度"),
]:
    ax.scatter(times, values, s=3, alpha=0.3, color=color, label="原始数据")
    xs, ys, spl = spline_fit(times, values)
    ax.plot(xs, ys, "-", linewidth=2.5, color="#D32F2F", label=f"Spline (s={SMOOTH})")
    ax.set_ylabel(ylabel)
    ax.set_title(f"97V 的 {title} 随时间变化", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

step = max(1, len(times) // 10)
ax3.set_xticks(times[::step])
ax3.set_xticklabels([time_labels[i] for i in range(0, len(times), step)], fontsize=7)
ax3.set_xlabel("时间")

plt.tight_layout()
plt.savefig("test/a1_97v_spline.png", dpi=150)
plt.close()
print(f"已保存: test/a1_97v_spline.png（s={SMOOTH}，{len(data)} 条）")
