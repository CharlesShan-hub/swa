"""
多指标对比图 — 一行一个指标，三列各设备，按湿度分层。

指标:
  1. harm_noise_pct   (噪声占比)
  2. harm_a1          (FFT基波幅值)
  3. harm_a1_eff      (矫正A1：有矫正用矫正值，无矫正用原始值)
  4. harm_a2          (FFT二次谐波幅值)
  5. harm_a2_div_a1   (A2/A1 谐波比例)
  6. error_div_a1     (相对拟合误差)
  7. temperature      (环境温度)
  8. humidity         (环境湿度)
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
    SELECT r.id, r.actual_voltage, r.device_id,
           r.harm_a1, r.harm_a1_corrected, r.harm_a2, r.harm_noise_pct,
           r.harm_error, r.harm_cycles,
           r.temperature, r.humidity, r.rpm
    FROM records r
    WHERE r.enabled = 1 AND r.actual_voltage >= 0
      AND r.device_id IS NOT NULL AND r.humidity IS NOT NULL
      AND r.temperature IS NOT NULL
    ORDER BY r.id
""", conn)
conn.close()

df = df[df["actual_voltage"] >= 0].copy()
df["harm_a2_div_a1"] = np.where(df["harm_a1"] > 0, df["harm_a2"] / df["harm_a1"], 0.0)
df["error_div_a1"] = np.where(df["harm_a1"] > 1e-6, df["harm_error"] / df["harm_a1"], 0.0)
# 矫正A1：有矫正值则用它，否则用原始值
df["harm_a1_eff"] = np.where(
    df["harm_a1_corrected"].notna() & (df["harm_a1_corrected"] != df["harm_a1"]),
    df["harm_a1_corrected"], df["harm_a1"]
)

devices = sorted(df["device_id"].dropna().unique())
print(f"总记录: {len(df)}, 设备数: {len(devices)}, 设备: {[d[-4:] for d in devices]}")

# ── 湿度分层 ──
groups = [
    ("低湿度 (<35%)", "royalblue", lambda h: h < 35),
    ("中湿度 (35-45%)", "orange", lambda h: 35 <= h < 45),
    ("高湿度 (>45%)", "crimson", lambda h: h >= 45),
]
group_order = [g[0] for g in groups]
color_map = {g[0]: g[1] for g in groups}

col_name = "group_humidity"
df[col_name] = "其他"
for gname, gcolor, gcond in groups:
    mask = df.apply(lambda r: gcond(r["humidity"]), axis=1)
    df.loc[mask, col_name] = gname

# ── 指标列表（A1 行用实线/虚线叠加显示原始与矫正的对比） ──
metrics = [
    ("harm_noise_pct",   "噪声占比 (noise_pct)",      "harm_noise_pct",    None),
    ("harm_a1",          "基波幅值 (A1)",              "harm_a1",           "harm_a1_eff"),
    ("harm_a2",          "二次谐波幅值 (A2)",          "harm_a2",           None),
    ("harm_a2_div_a1",   "谐波比例 (A2/A1)",           "harm_a2_div_a1",    None),
    ("error_div_a1",     "相对拟合误差 (error/A1)",     "error_div_a1",      None),
    ("temperature",      "环境温度",                   "temperature",       None),
    ("humidity",         "环境湿度",                   "humidity",          None),
]

n_metrics = len(metrics)
n_dev = len(devices)

fig, axes = plt.subplots(n_metrics, n_dev, figsize=(6 * n_dev, 4 * n_metrics))

for row_idx, (key, title, col, col_overlay) in enumerate(metrics):
    for col_idx, dev in enumerate(devices):
        ax = axes[row_idx][col_idx]
        sub = df[df["device_id"] == dev]

        # 主数据（实线）
        for gn in group_order:
            gsub = sub[sub[col_name] == gn]
            if len(gsub) < 5:
                continue
            grp = gsub.groupby("actual_voltage")[col].agg(["mean", "std", "count"])
            grp = grp[grp["count"] >= 3]
            if len(grp) == 0:
                continue
            ax.errorbar(
                grp.index, grp["mean"],
                yerr=grp["std"] / np.sqrt(grp["count"]),
                label=gn, color=color_map[gn],
                marker="o", markersize=4, linewidth=1.2, capsize=2,
            )

        # 叠加数据（虚线，A1 矫正值）
        if col_overlay is not None:
            for gn in group_order:
                gsub = sub[sub[col_name] == gn]
                if len(gsub) < 5:
                    continue
                grp = gsub.groupby("actual_voltage")[col_overlay].agg(["mean", "std", "count"])
                grp = grp[grp["count"] >= 3]
                if len(grp) == 0:
                    continue
                ax.errorbar(
                    grp.index, grp["mean"],
                    yerr=grp["std"] / np.sqrt(grp["count"]),
                    color=color_map[gn],
                    marker="x", markersize=4, linewidth=1.2, capsize=2,
                    linestyle="--", alpha=0.7,
                )
            # 在图例中加入矫正标识
            ax.plot([], [], color="gray", linewidth=1, linestyle="-", label="原始")
            ax.plot([], [], color="gray", linewidth=1, linestyle="--", label="矫正后")

        ax.set_xlabel("电压 (V)", fontsize=9)
        if col_idx == 0:
            ax.set_ylabel(title, fontsize=9)
        ax.set_title(f"设备 {dev[-4:]}" if row_idx == 0 else "", fontsize=10, fontweight="bold")
        if row_idx == 0:
            ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "multi_metrics_by_humidity.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"已保存: {out_path}")
plt.close(fig)

# ── 文字输出 ──
print(f"\n{'=' * 100}")
print(f"各指标 × 设备 × 电压  均值一览（按湿度分层）")
print("=" * 100)

for key, title, col, col_overlay in metrics:
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")
    print(f"  {'设备':>8s}  {'电压':>5s}  ", end="")
    for gn in group_order:
        print(f"  {gn:>16s}", end="")
    print()

    for dev in devices:
        sub = df[df["device_id"] == dev]
        for v in sorted(sub["actual_voltage"].unique()):
            short_dev = dev[-4:]
            print(f"  {short_dev:>8s}  {v:+.0f}V  ", end="")
            for gn in group_order:
                mask = (sub["actual_voltage"] == v) & (sub[col_name] == gn)
                vals = sub.loc[mask, col].dropna()
                if len(vals) > 0:
                    print(f"  {vals.mean():>10.3f} (n={len(vals):>4d})", end="")
                else:
                    print(f"  {'N/A':>19s}", end="")
            print()

print(f"\n全部完成！")
