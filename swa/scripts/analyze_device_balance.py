"""
设备平衡算法效果对比 — 6列图。
左3列 = 原始（当前），右3列 = 设备映射后。
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
from matplotlib.gridspec import GridSpec

DB_PATH = r"d:\project\work\swa\swa\src\data\projects\new\data.db"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("""
    SELECT r.id, r.actual_voltage, r.device_id,
           r.harm_a1, r.harm_a2, r.harm_noise_pct,
           r.harm_error, r.harm_cycles,
           r.temperature, r.humidity, r.rpm
    FROM records r
    WHERE r.enabled = 1 AND r.actual_voltage >= 0
      AND r.device_id IS NOT NULL AND r.humidity IS NOT NULL
    ORDER BY r.id
""", conn)
conn.close()

df = df[df["actual_voltage"] >= 0].copy()
df["harm_a2_div_a1"] = np.where(df["harm_a1"] > 0, df["harm_a2"] / df["harm_a1"], 0.0)
df["error_div_a1"] = np.where(df["harm_a1"] > 1e-6, df["harm_error"] / df["harm_a1"], 0.0)

devices = sorted(df["device_id"].dropna().unique())
dev_b = [d for d in devices if "B" in d][0]
print(f"设备: {[d[-4:] for d in devices]}, 基准: {dev_b[-4:]}")

# ── 计算设备映射 ratio = a×v + b ──
mappers = {}
for target_dev in devices:
    if target_dev == dev_b:
        continue
    ref_avg = df[df["device_id"] == dev_b].groupby("actual_voltage")["harm_a1"].mean()
    tgt_avg = df[df["device_id"] == target_dev].groupby("actual_voltage")["harm_a1"].mean()
    common_v = sorted(set(ref_avg.index) & set(tgt_avg.index))
    if len(common_v) < 3:
        continue
    ratios = [tgt_avg[v] / ref_avg[v] for v in common_v]
    a, b = np.polyfit(common_v, ratios, 1)
    mappers[target_dev] = (a, b)
    print(f"  映射 {target_dev[-4:]}: ratio = {a:.6f}×V + {b:.4f}")

# 应用映射
df["harm_a1_mapped"] = np.nan
for target_dev, (a, b) in mappers.items():
    mask = df["device_id"] == target_dev
    v = df.loc[mask, "actual_voltage"].abs().values
    ratios = a * v + b
    ratios = np.maximum(ratios, 0.01)
    df.loc[mask, "harm_a1_mapped"] = df.loc[mask, "harm_a1"].values / ratios
df.loc[df["device_id"] == dev_b, "harm_a1_mapped"] = df.loc[df["device_id"] == dev_b, "harm_a1"]

# 湿度分层
hum_order = ["低湿度 (<35%)", "中湿度 (35-45%)", "高湿度 (>45%)"]
def hum_label(h):
    if h < 35: return hum_order[0]
    elif h < 45: return hum_order[1]
    else: return hum_order[2]
df["hum_group"] = df["humidity"].apply(hum_label)
hcolors = {hum_order[0]: "royalblue", hum_order[1]: "orange", hum_order[2]: "crimson"}

# ── 指标定义: (key, title, raw_col, mapped_col) ──
# 只有 A1 会因设备映射改变，其他指标不变但重复列用于布局
metrics = [
    ("noise_pct",    "噪声占比",     "harm_noise_pct",  "harm_noise_pct"),
    ("a1",           "A1 (原始)",    "harm_a1",         "harm_a1_mapped"),
    ("a2",           "A2",           "harm_a2",         "harm_a2"),
    ("a2_div_a1",    "A2/A1",        "harm_a2_div_a1",  "harm_a2_div_a1"),
    ("error_div_a1", "误差/A1",      "error_div_a1",    "error_div_a1"),
]

n_metrics = len(metrics)
n_dev = len(devices)

fig = plt.figure(figsize=(11, 3.5 * n_metrics))
gs = GridSpec(n_metrics, n_dev * 2, figure=fig, hspace=0.35, wspace=0.35)

def plot_col(ax, sub, col, title_suffix):
    for hg in hum_order:
        hsub = sub[sub["hum_group"] == hg]
        if len(hsub) < 5: continue
        grp = hsub.groupby("actual_voltage")[col].agg(["mean", "std", "count"])
        grp = grp[grp["count"] >= 3]
        if len(grp) == 0: continue
        ax.errorbar(grp.index, grp["mean"],
                    yerr=grp["std"] / np.sqrt(grp["count"]),
                    label=hg, color=hcolors[hg],
                    marker="o", markersize=4, linewidth=1.2, capsize=2)
    ax.set_xlabel("电压 (V)", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

for row_idx, (key, title, raw_col, map_col) in enumerate(metrics):
    for col_idx, dev in enumerate(devices):
        # 左半: 原始
        ax_raw = fig.add_subplot(gs[row_idx, col_idx])
        sub = df[df["device_id"] == dev]
        plot_col(ax_raw, sub, raw_col, "")
        if col_idx == 0:
            ax_raw.set_ylabel(title, fontsize=9)
        ax_raw.set_title(f"原始 {dev[-4:]}" if row_idx == 0 else "", fontsize=9, fontweight="bold")
        if row_idx == 0:
            ax_raw.legend(fontsize=6, loc="best")

        # 右半: 映射后
        ax_map = fig.add_subplot(gs[row_idx, col_idx + n_dev])
        plot_col(ax_map, sub, map_col, "")
        ax_map.set_title(f"映射 {dev[-4:]}" if row_idx == 0 else "", fontsize=9, fontweight="bold")

fig.suptitle("左: 原始数据  |  右: 设备映射校准后", fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "device_balance_comparison.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n已保存: {out_path}")
plt.close(fig)

# ── 文字对比 ──
print(f"\n{'='*120}")
print("设备映射前后 A1 均值对比 (按电压)")
print("="*120)
print(f"  {'设备':>8s}  {'电压':>5s}  {'A1原始':>10s}  {'A1映射':>10s}  {'变化%':>8s}  {'n':>6s}")
for dev in devices:
    sub = df[df["device_id"] == dev]
    for v in sorted(sub["actual_voltage"].unique()):
        item = sub[sub["actual_voltage"] == v]
        a1_raw = item["harm_a1"].mean()
        a1_map = item["harm_a1_mapped"].mean()
        pct = (a1_map - a1_raw) / a1_raw * 100 if a1_raw > 0 else 0
        print(f"  {dev[-4:]:>8s}  {v:+.0f}V  {a1_raw:>10.1f}  {a1_map:>10.1f}  {pct:>+7.1f}%  {len(item):>6d}")
