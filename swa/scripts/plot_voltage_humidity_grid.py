"""
热力图：横轴=电压, 纵轴=湿度区间, 颜色=A1 幅值
一组图一个指标，三个设备各一子图
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 配置 ──────────────────────────────────────────────────────
DB_PATH = r"d:\project\work\swa\swa\src\data\projects\new\data.db"

# 湿度分箱（数据范围 27~63%，用 5% 步长）
HUM_BINS = list(range(25, 70, 5))  # [25,30,35,40,45,50,55,60,65]
HUM_LABELS = [f"{b}-{b+5}" for b in HUM_BINS[:-1]]

METRICS = ["harm_a1"]


def load_data(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT r.id, r.actual_voltage, r.temperature, r.humidity, r.rpm,
               r.device_id, r.enabled, r.harm_a1
        FROM records r
        WHERE r.enabled = 1 AND r.actual_voltage IS NOT NULL
          AND r.device_id IS NOT NULL AND r.humidity IS NOT NULL
        ORDER BY r.id
    """, conn)
    conn.close()
    # 只保留正电压
    df = df[df["actual_voltage"] >= 0].copy()
    return df


def plot_metric_grid(df: pd.DataFrame, metric: str):
    """为每个设备画该指标的热力图。"""
    devices = sorted(df["device_id"].dropna().unique())
    print(f"\n设备列表 ({len(devices)} 个):")
    for d in devices:
        cnt = len(df[df["device_id"] == d])
        print(f"  {d[:24]:24s}  n={cnt}")

    n_dev = len(devices)
    fig, axes = plt.subplots(1, n_dev, figsize=(5 * n_dev, 4.5))
    if n_dev == 1:
        axes = [axes]

    fig.suptitle(f"指标: {metric.upper()}  —  电压 × 湿度 热力图", fontsize=13, fontweight="bold")

    # 计算全局颜色范围
    all_vals = df[metric].dropna()
    vmin, vmax = all_vals.quantile(0.01), all_vals.quantile(0.99)

    for ax, dev in zip(axes, devices):
        sub = df[df["device_id"] == dev].copy()
        if len(sub) == 0:
            ax.set_visible(False)
            continue

        # 湿度分箱
        sub["hum_bin"] = pd.cut(sub["humidity"], bins=HUM_BINS, labels=HUM_LABELS, right=False)

        # 电压四舍五入到整数
        sub["v_int"] = sub["actual_voltage"].round(0).astype(int)

        # 透视表：均值
        pivot = sub.pivot_table(
            values=metric,
            index="hum_bin",
            columns="v_int",
            aggfunc="mean",
        )
        # 样本数
        count_pivot = sub.pivot_table(
            values="id", index="hum_bin", columns="v_int", aggfunc="count"
        )

        # ── 打印文字 ──
        short_dev = dev[:20]
        print(f"\n{'=' * 60}")
        print(f"设备: {short_dev} | 指标: {metric}")
        print("=" * 60)
        print(f"  A1 均值 (行=湿度% , 列=电压):")
        print(pivot.to_string(float_format="%.1f"))
        print(f"\n  样本数:")
        print(count_pivot.to_string(float_format="%.0f"))

        # 打印每电压 A1 均值（不分湿度）
        print(f"\n  每电压 A1 均值:")
        for v in sorted(sub["v_int"].unique()):
            v_sub = sub[sub["v_int"] == v]
            print(f"    V={v:+.0f}  A1={v_sub[metric].mean():.1f}  n={len(v_sub)}")

        # ── 画热力图 ──
        plot_data = pivot.T  # 转置使电压在横轴
        plot_data = plot_data.sort_index()  # 按电压排序

        # 归一化 vmin/vmax 到 0~1 映射到颜色
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(plot_data.values, aspect="auto", cmap="viridis", norm=norm)

        # 标注数字 + 样本数
        for i in range(plot_data.shape[0]):
            for j in range(plot_data.shape[1]):
                val = plot_data.values[i, j]
                if not np.isnan(val):
                    txt_color = "white" if val > (vmin + vmax) * 0.6 else "black"
                    ax.text(i, j, f"{val:.0f}", ha="center", va="center",
                            fontsize=7, color=txt_color, fontweight="bold")
                    # 小字标注样本数
                    v = plot_data.index[i]
                    h = plot_data.columns[j]
                    cnt = count_pivot.loc[h, v] if h in count_pivot.index and v in count_pivot.columns else 0
                    ax.text(i, j + 0.3, f"n={int(cnt)}", ha="center", va="top",
                            fontsize=4.5, color=txt_color, alpha=0.7)

        ax.set_xticks(range(len(plot_data.index)))
        ax.set_xticklabels(plot_data.index, fontsize=8)
        ax.set_yticks(range(len(plot_data.columns)))
        ax.set_yticklabels(plot_data.columns, fontsize=8)
        ax.set_xlabel("电压 (V)", fontsize=9)
        ax.set_ylabel("湿度区间 (%)", fontsize=9)
        short_dev = dev[:16] if len(dev) > 16 else dev
        ax.set_title(f"设备 {short_dev}", fontsize=10)

    fig.colorbar(im, ax=axes, shrink=0.6, label=metric.upper())
    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), f"grid_{metric}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n  已保存: {out_path}")
    plt.close(fig)


def main():
    print("加载数据...")
    df = load_data(DB_PATH)
    print(f"总记录: {len(df)}")

    for metric in METRICS:
        print(f"\n\n{'#' * 70}")
        print(f"# 指标: {metric}")
        print(f"{'#' * 70}")
        plot_metric_grid(df, metric)


if __name__ == "__main__":
    main()
