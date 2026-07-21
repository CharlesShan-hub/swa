"""
按湿度分层，画各设备的 A1 vs 电压曲线。
每个设备一张图，每层湿度一条线。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("QtAgg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

from swa.data.manager import DataManager

dm = DataManager()
dm.load_project("new")
df = pd.read_sql("""
    SELECT r.device_id, r.actual_voltage, r.harm_a1, r.humidity
    FROM records r
    WHERE r.enabled = 1 AND r.harm_a1 IS NOT NULL AND r.device_id IS NOT NULL
    ORDER BY r.device_id, r.actual_voltage, r.humidity
""", dm._conn)
dm.close()

df = df[df["actual_voltage"] >= 0].copy()

# 湿度分 4 层（按总体百分位数）
hum_bins = pd.qcut(df["humidity"], q=4, labels=["低湿度", "中低", "中高", "高湿度"])
df["hum_layer"] = hum_bins

# 各层湿度范围
for label, sub in df.groupby("hum_layer", observed=True):
    print(f"  {label}: {sub['humidity'].min():.1f}%~{sub['humidity'].max():.1f}%")

devices = sorted(df["device_id"].unique())
colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
layers = sorted(df["hum_layer"].cat.categories.tolist())

# ── 打印数据表 ──
print()
for dev in devices:
    dev_short = dev[:16]
    print(f"{'=' * 70}")
    print(f"设备 {dev_short}")
    print(f"{'=' * 70}")
    sub = df[df["device_id"] == dev]
    # 表头
    print(f"{'电压':>5s}  ", end="")
    for layer in layers:
        print(f"  {str(layer):>20s}", end="")
    print()
    for v in sorted(sub["actual_voltage"].unique()):
        print(f"{v:>5.0f}  ", end="")
        for layer in layers:
            mask = (sub["actual_voltage"] == v) & (sub["hum_layer"] == layer)
            vals = sub.loc[mask, "harm_a1"]
            if len(vals) > 0:
                print(f"  {np.mean(vals):>8.1f} ({len(vals):>3d}条)", end="")
            else:
                print(f"  {'':>20s}", end="")
        print()

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

for idx, dev in enumerate(devices):
    ax = axes[idx]
    sub = df[df["device_id"] == dev]

    # 每个湿度层一条线
    for li, layer in enumerate(layers):
        layer_sub = sub[sub["hum_layer"] == layer]
        grp = layer_sub.groupby("actual_voltage", observed=True)["harm_a1"].mean().reset_index()
        ax.plot(grp["actual_voltage"], grp["harm_a1"], "-o", color=colors[li % len(colors)],
                label=f"{layer}", markersize=3, linewidth=1.2)

    ax.set_title(f"设备 {dev[:16]}...", fontsize=11)
    ax.set_xlabel("电压 (V)")
    ax.set_ylabel("A1 (基波幅值)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.show()
