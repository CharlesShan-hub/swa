"""
对 -55V / -40V / 97V，画 A1, A3, A5, Vpp, Kurtosis, Skewness, Temp, Humid
随时间变化的散点 + 包络中值拟合曲线。
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
records = extend(records, ["a1", "a3", "a5", "vpp", "kurtosis", "temp", "humid"])

N_BINS = 200

def mid_fit(x, y, s=0.001):
    bins = np.linspace(x.min(), x.max(), N_BINS + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    mids, used = [], []
    for i in range(N_BINS):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if mask.sum() < 10:
            continue
        used.append(centers[i])
        upper = np.percentile(y[mask], 90)
        lower = np.percentile(y[mask], 10)
        mids.append((upper + lower) / 2)
    used = np.array(used); mids = np.array(mids)
    spl = UnivariateSpline(used, mids, s=s)
    xs = np.linspace(x.min(), x.max(), 1000)
    return xs, spl(xs)

targets = [-40, -55, 97]
# 所有要画的指标（字段名, 颜色, 显示名）
metric_fields = [
    ("_a1", "#2196F3", "A1"),
    ("_a3", "#FF5722", "A3"),
    ("_a5", "#9C27B0", "A5"),
    ("_vpp", "#4CAF50", "Vpp"),
    ("_kurtosis", "#FF9800", "Kurtosis"),
    ("_temp", "#E91E63", "温度 (°C)"),
    ("_humid", "#795548", "湿度 (%)"),
]

n_metrics = len(metric_fields)
fig, axes = plt.subplots(len(targets), n_metrics, figsize=(4 * n_metrics, 3 * len(targets)))

for row, target_v in enumerate(targets):
    data = []
    for rec in records:
        v = rec["ACTUAL_VOLTAGE"]
        if not isinstance(v, (int, float)) or round(v) != target_v:
            continue
        vals = []
        ok = True
        for field, _, _ in metric_fields:
            val = rec.get(field)
            if val is None:
                ok = False
                break
            try:
                vals.append(float(val))
            except:
                ok = False
                break
        if not ok:
            continue
        when = rec.get("SYSTEM_TIME", "")
        try:
            dt = datetime.strptime(when[:19], "%Y-%m-%d %H:%M:%S")
        except:
            continue
        data.append((dt, *vals))

    data.sort(key=lambda x: x[0])
    times = np.array([(d[0] - data[0][0]).total_seconds() / 3600 for d in data])
    all_vals = [np.array([d[i+1] for d in data]) for i in range(n_metrics)]
    time_labels = [d[0].strftime("%m/%d\n%H:%M") for d in data]

    for col, (vals, (field, color, label)) in enumerate(zip(all_vals, metric_fields)):
        ax = axes[row, col]
        ax.scatter(times, vals, s=3, alpha=0.3, color=color, label="原始数据")
        xs, ys = mid_fit(times, vals, s=0.001)
        ax.plot(xs, ys, "-", linewidth=2.5, color="#D32F2F", label="中值拟合")
        ax.set_title(f"{target_v}V n={len(data)} → {label}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # 时间标签只做最后一个子图列的
    step = max(1, len(times) // 6)
    for col in range(n_metrics):
        ax = axes[row, col]
        if row == len(targets) - 1:
            ax.set_xlabel("时间", fontsize=8)
            ax.set_xticks(times[::step])
            ax.set_xticklabels([time_labels[i] for i in range(0, len(times), step)], fontsize=5)
        else:
            ax.set_xticks([])

plt.suptitle("-40V / -55V / 97V 各指标随时间变化", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("test/metrics_top3_envelope.png", dpi=150)
plt.close()
print(f"已保存: test/metrics_top3_envelope.png（{len(targets)} 个电压 × {n_metrics} 个指标）")
