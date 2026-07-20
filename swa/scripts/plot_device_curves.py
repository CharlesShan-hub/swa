"""
把三个设备的 A1 画成三条曲线，横轴按实验条件排序。

思路：
  每个实验条件（TEST_CASE_CODE）下取 A1 均值
  按 (电压绝对值, 温度, 湿度) 排序 → 横轴是自然递增的实验条件
  三条曲线 = 三个设备 → 应该形状相似、幅度不同

用法: pixi run python scripts/plot_device_curves.py --project new
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import click
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("QtAgg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

from swa.data.manager import DataManager


@click.command()
@click.option("--project", "-p", required=True, help="项目名称")
def main(project):
    dm = DataManager()
    dm.load_project(project)
    conn = dm._conn

    df = pd.read_sql_query("""
        SELECT r.id, r.device_id, r.actual_voltage, r.harm_a1,
               r.temperature, r.humidity, r.test_case_code
        FROM records r
        WHERE r.enabled = 1 AND r.harm_a1 IS NOT NULL AND r.device_id IS NOT NULL
        ORDER BY r.id
    """, conn)

    devices = sorted(df["device_id"].dropna().unique())
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    print(f"设备: {devices}")
    print(f"总样本: {len(df)} 条")

    # ── 1. 每个实验条件聚合 ──────────────────────────────────────
    # 按 (device_id, actual_voltage, test_case_code) 分组取均值
    grp = df.groupby(["device_id", "actual_voltage", "test_case_code"], as_index=False).agg(
        a1_mean=("harm_a1", "mean"),
        temp_mean=("temperature", "mean"),
        hum_mean=("humidity", "mean"),
        count=("id", "count"),
    )

    # 提取实验编号用于排序
    import re
    def exp_order(code):
        m = re.search(r"E(\d+)", str(code))
        return int(m.group(1)) if m else 999

    grp["exp_num"] = grp["test_case_code"].apply(exp_order)

    # 按 (实验编号, 电压绝对值) 排序 → 这就是自然递增的实验进展
    grp["abs_v"] = grp["actual_voltage"].abs()

    # 创建统一的条件序号（所有设备共享同一个横轴）
    conditions = grp[["exp_num", "abs_v"]].drop_duplicates().sort_values(["exp_num", "abs_v"]).reset_index(drop=True)
    conditions["condition_idx"] = range(len(conditions))
    grp = grp.merge(conditions, on=["exp_num", "abs_v"], how="left")

    print(f"\n聚合后 {len(grp)} 个实验条件组，共 {len(conditions)} 个唯一条件")

    # ── 2. 画图 ─────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # 图1: A1 曲线
    ax = axes[0]
    for dev, color in zip(devices, colors):
        sub = grp[grp["device_id"] == dev].sort_values("condition_idx")
        ax.plot(sub["condition_idx"], sub["a1_mean"], "-o", color=color,
                label=f"设备 {dev[:12]}...", markersize=3, linewidth=1.2)
    ax.set_ylabel("A1 (基波幅值)")
    ax.set_title("三个设备 A1 随实验条件的演变曲线")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 在每个数据点旁标电压值
    sub = grp[grp["device_id"] == devices[0]].sort_values("condition_idx")
    prev_v = None
    for _, row in sub.iterrows():
        v = row["abs_v"]
        if v != prev_v:
            ax.annotate(f"{v:.0f}V", (row["condition_idx"], row["a1_mean"]),
                       fontsize=6, alpha=0.6, color="gray")
            prev_v = v

    # 图2: 各设备 A1 随电压的变化（叠加显示，便于看形状差异）
    ax = axes[1]
    for dev, color in zip(devices, colors):
        sub = grp[grp["device_id"] == dev].groupby("abs_v")["a1_mean"].mean().reset_index()
        ax.plot(sub["abs_v"], sub["a1_mean"], "-o", color=color,
                label=f"设备 {dev[:12]}", markersize=4, linewidth=1.5)
    ax.set_xlabel("电压绝对值 (V)")
    ax.set_ylabel("A1 (基波幅值)")
    ax.set_title("各设备 A1 vs 电压（同电压 A1 均值）")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 图3: 温度和湿度变化
    ax = axes[2]
    sub = grp[grp["device_id"] == devices[0]].sort_values("condition_idx")
    ax.plot(sub["condition_idx"], sub["temp_mean"], "-", color="#E91E63", alpha=0.7, label="温度")
    ax_twin = ax.twinx()
    ax_twin.plot(sub["condition_idx"], sub["hum_mean"], "-", color="#9C27B0", alpha=0.7, label="湿度")
    ax.set_xlabel("实验条件序号 (按实验编号→电压排序)")
    ax.set_ylabel("温度 (°C)", color="#E91E63")
    ax_twin.set_ylabel("湿度 (%)", color="#9C27B0")
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
