"""
多指标对比图 — 一行一个指标，三列各设备，分别按湿度分层和温度分层生成两套图。

指标:
  1. harm_noise_pct   (噪声占比)
  2. harm_a1          (FFT基波幅值)
  3. harm_a2          (FFT二次谐波幅值)
  4. harm_a2_div_a1   (A2/A1 谐波比例)
  5. error_div_a1     (相对拟合误差)
  6. temperature      (环境温度)
  7. humidity         (环境湿度)
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
           r.harm_a1, r.harm_a2, r.harm_noise_pct,
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

devices = sorted(df["device_id"].dropna().unique())
print(f"总记录: {len(df)}, 设备数: {len(devices)}, 设备: {[d[-4:] for d in devices]}")

# ── 两种分层方式 ──

groups_config = {
    "humidity": {
        "label": "湿度",
        "out_suffix": "_by_humidity",
        "groups": [
            ("低湿度 (<35%)", "royalblue", lambda h: h < 35),
            ("中湿度 (35-45%)", "orange", lambda h: 35 <= h < 45),
            ("高湿度 (>45%)", "crimson", lambda h: h >= 45),
        ],
        "val_col": "humidity",
    },
    "temperature": {
        "label": "温度",
        "out_suffix": "_by_temperature",
        "groups": [
            ("低温 (<15°C)", "royalblue", lambda t: t < 15),
            ("中温 (15-25°C)", "orange", lambda t: 15 <= t < 25),
            ("高温 (>25°C)", "crimson", lambda t: t >= 25),
        ],
        "val_col": "temperature",
    },
}

# ── 指标列表 ──
metrics = [
    ("harm_noise_pct",   "噪声占比 (noise_pct)",      "harm_noise_pct"),
    ("harm_a1",          "基波幅值 (A1)",              "harm_a1"),
    ("harm_a2",          "二次谐波幅值 (A2)",          "harm_a2"),
    ("harm_a2_div_a1",   "谐波比例 (A2/A1)",           "harm_a2_div_a1"),
    ("error_div_a1",     "相对拟合误差 (error/A1)",     "error_div_a1"),
    ("temperature",      "环境温度",                   "temperature"),
    ("humidity",         "环境湿度",                   "humidity"),
]

n_metrics = len(metrics)
n_dev = len(devices)


# ── 画图函数 ──
def plot_grouped(cfg):
    """按 cfg 中定义的分层方式绘图并保存。"""
    label = cfg["label"]
    groups = cfg["groups"]
    out_suffix = cfg["out_suffix"]

    # 标记分组
    col_name = f"group_{out_suffix}"
    df[col_name] = "其他"
    group_order = []
    for gname, gcolor, gcond in groups:
        mask = df.apply(lambda r: gcond(r[cfg["val_col"]]), axis=1)
        df.loc[mask, col_name] = gname
        group_order.append(gname)

    color_map = {g[0]: g[1] for g in groups}

    fig, axes = plt.subplots(n_metrics, n_dev, figsize=(6 * n_dev, 4 * n_metrics))
    if n_metrics == 1:
        axes = [axes]
    if n_dev == 1:
        axes = [[ax] for ax in axes]

    for row_idx, (key, title, col) in enumerate(metrics):
        for col_idx, dev in enumerate(devices):
            ax = axes[row_idx][col_idx]
            sub = df[df["device_id"] == dev]

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

            ax.set_xlabel("电压 (V)", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(title, fontsize=9)
            ax.set_title(f"设备 {dev[-4:]}" if row_idx == 0 else "", fontsize=10, fontweight="bold")
            if row_idx == 0:
                ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), f"multi_metrics{out_suffix}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"已保存: {out_path}")
    plt.close(fig)

    # ── 文字输出 ──
    print(f"\n{'=' * 100}")
    print(f"各指标 × 设备 × 电压  均值一览（按{label}分层）")
    print("=" * 100)

    for key, title, col in metrics:
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


# ── 生成两套图 ──
for cfg_key in ["humidity", "temperature"]:
    plot_grouped(groups_config[cfg_key])

print(f"\n全部完成！")
