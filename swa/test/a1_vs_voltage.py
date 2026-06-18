"""
97V 的 A1 / 湿度 / 温度 随时间变化的散点 + 拟合曲线。
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

# 收集 97V 数据
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
times = np.array([(d[0] - data[0][0]).total_seconds() / 3600 for d in data])  # 小时
a1s = np.array([d[1] for d in data])
temps = np.array([d[2] for d in data])
hums = np.array([d[3] for d in data])
time_labels = [d[0].strftime("%m/%d\n%H:%M") for d in data]

# 多项式拟合辅助函数
def poly_fit(x, y, deg=5):
    coeffs = np.polyfit(x, y, deg)
    poly = np.poly1d(coeffs)
    xsmooth = np.linspace(x.min(), x.max(), 500)
    ysmooth = poly(xsmooth)
    return xsmooth, ysmooth, poly

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

# 1. A1
ax1.scatter(times, a1s, s=3, alpha=0.4, color="#2196F3", label="原始 A1")
xs, ys, _ = poly_fit(times, a1s, deg=6)
ax1.plot(xs, ys, "-", linewidth=2.5, color="#D32F2F", label="拟合曲线")
ax1.set_ylabel("A1")
ax1.set_title("97V 的 A1 随时间变化", fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. 湿度
ax2.scatter(times, hums, s=3, alpha=0.4, color="#4CAF50", label="原始湿度")
xs, ys, _ = poly_fit(times, hums, deg=6)
ax2.plot(xs, ys, "-", linewidth=2.5, color="#D32F2F", label="拟合曲线")
ax2.set_ylabel("湿度 (%)")
ax2.set_title("97V 的湿度随时间变化", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. 温度
ax3.scatter(times, temps, s=3, alpha=0.4, color="#FF5722", label="原始温度")
xs, ys, _ = poly_fit(times, temps, deg=6)
ax3.plot(xs, ys, "-", linewidth=2.5, color="#D32F2F", label="拟合曲线")
ax3.set_ylabel("温度 (°C)")
ax3.set_xlabel("时间 (小时)")
ax3.set_title("97V 的温度随时间变化", fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 每 500 个点标一个时间戳
step = max(1, len(times) // 10)
tick_positions = times[::step]
tick_labels_short = [time_labels[i] for i in range(0, len(times), step)]
ax3.set_xticks(tick_positions)
ax3.set_xticklabels(tick_labels_short, fontsize=7)

plt.tight_layout()
plt.savefig("test/a1_97v_fitted.png", dpi=150)
plt.close()
print(f"已保存: test/a1_97v_fitted.png（{len(data)} 条）")
