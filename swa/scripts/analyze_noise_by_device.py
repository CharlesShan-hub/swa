"""
按设备、电压、湿度分层看 noise_pct 分布
每个设备一个图，横轴=电压，纵轴=noise_pct，不同湿度不同颜色线
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import sqlite3
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

DB_PATH = r"d:\project\work\swa\swa\src\data\projects\new\data.db"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("""
    SELECT r.id, r.actual_voltage, r.device_id, r.harm_a1,
           r.harm_noise_pct, r.temperature, r.humidity, r.rpm
    FROM records r
    WHERE r.enabled = 1 AND r.actual_voltage >= 0
      AND r.device_id IS NOT NULL AND r.humidity IS NOT NULL
    ORDER BY r.id
""", conn)
conn.close()

df = df[df["actual_voltage"] >= 0].copy()
devices = sorted(df["device_id"].dropna().unique())
print(f"总记录: {len(df)}, 设备数: {len(devices)}")

# 湿度分三层
def hum_label(h):
    if h < 35:
        return "低湿度 (<35%)"
    elif h < 45:
        return "中湿度 (35-45%)"
    else:
        return "高湿度 (>45%)"

df["hum_group"] = df["humidity"].apply(hum_label)
hum_order = ["低湿度 (<35%)", "中湿度 (35-45%)", "高湿度 (>45%)"]
colors = {"低湿度 (<35%)": "royalblue", "中湿度 (35-45%)": "orange", "高湿度 (>45%)": "crimson"}

# ── 打印文字 ─────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print("各设备 × 电压 × 湿度  的 noise_pct 均值")
print("=" * 80)

for dev in devices:
    short_dev = dev[-4:]
    sub = df[df["device_id"] == dev]
    print(f"\n  设备 {short_dev}:")
    print(f"  {'电压':>6s}  ", end="")
    for hg in hum_order:
        print(f"  {hg:16s}", end="")
    print()
    
    for v in sorted(sub["actual_voltage"].unique()):
        print(f"  {v:+.0f}V  ", end="")
        for hg in hum_order:
            mask = (sub["actual_voltage"] == v) & (sub["hum_group"] == hg)
            vals = sub.loc[mask, "harm_noise_pct"].dropna()
            if len(vals) > 0:
                print(f"  {vals.mean():.3f} (n={len(vals):>4d})", end="")
            else:
                print(f"  {'N/A':>16s}", end="")
        print()

# ── 打印 A1 均值对比 ─────────────────────────────────────────
print(f"\n{'=' * 80}")
print("各设备 × 电压 × 湿度  的 A1 均值 (以及噪声校正后)")
print("=" * 80)

for dev in devices:
    short_dev = dev[-4:]
    sub = df[df["device_id"] == dev].copy()
    sub["a1_clean"] = sub["harm_a1"] * (1 - sub["harm_noise_pct"].fillna(0))
    
    print(f"\n  设备 {short_dev}:")
    print(f"  {'电压':>6s}  {'湿度':>12s}  {'A1原始':>8s}  {'A1校正':>8s}  {'noise':>6s}  {'n':>6s}")
    
    for v in sorted(sub["actual_voltage"].unique()):
        for hg in hum_order:
            mask = (sub["actual_voltage"] == v) & (sub["hum_group"] == hg)
            item = sub[mask]
            if len(item) > 0:
                a1_m = item["harm_a1"].mean()
                a1c_m = item["a1_clean"].mean()
                np_m = item["harm_noise_pct"].mean()
                print(f"  {v:+.0f}V  {hg:12s}  {a1_m:8.1f}  {a1c_m:8.1f}  {np_m:.3f}  {len(item):6d}")

# ── 画图 ─────────────────────────────────────────────────────
n_dev = len(devices)
fig, axes = plt.subplots(1, n_dev, figsize=(7 * n_dev, 5))
if n_dev == 1:
    axes = [axes]

for ax, dev in zip(axes, devices):
    sub = df[df["device_id"] == dev]
    short_dev = dev[-4:]
    
    for hg in hum_order:
        hsub = sub[sub["hum_group"] == hg]
        # 按电压取均值
        grp = hsub.groupby("actual_voltage")["harm_noise_pct"].agg(["mean", "std", "count"])
        grp = grp[grp["count"] >= 5]  # 至少5条
        ax.errorbar(grp.index, grp["mean"], yerr=grp["std"]/np.sqrt(grp["count"]),
                    label=hg, color=colors[hg], marker="o", markersize=5, linewidth=1.5, capsize=3)
        
        # 打印一些关键值
        for v in grp.index:
            row = grp.loc[v]
            print(f"  设备{short_dev} {hg} V={v:.0f}: noise_pct={row['mean']:.3f}±{row['std']:.3f} n={int(row['count'])}")
    
    ax.set_xlabel("电压 (V)", fontsize=11)
    ax.set_ylabel("harm_noise_pct", fontsize=11)
    ax.set_title(f"设备 {short_dev}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "noise_by_device_humidity.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n已保存: {out_path}")
plt.close(fig)

# ── 额外分析：校正前后 A1 ~ V 曲线对比 ───────────────────────
print(f"\n\n{'=' * 80}")
print("噪声校正前后 A1 均值对比 (不分湿度, 分设备)")
print("=" * 80)

for dev in devices:
    short_dev = dev[-4:]
    sub = df[df["device_id"] == dev].copy()
    sub["a1_clean"] = sub["harm_a1"] * (1 - sub["harm_noise_pct"].fillna(0))
    
    print(f"\n  设备 {short_dev}:")
    for v in sorted(sub["actual_voltage"].unique()):
        item = sub[sub["actual_voltage"] == v]
        a1_raw = item["harm_a1"].mean()
        a1_cln = item["a1_clean"].mean()
        print(f"    V={v:+.0f}  A1={a1_raw:.1f} → A1_cln={a1_cln:.1f}  (↓{a1_raw-a1_cln:.1f}, {100*(a1_raw-a1_cln)/a1_raw:.1f}%)")
