"""
比较三个设备的温度和湿度测量值是否一致。
三个设备放在同一个箱子里，温湿度应该相同，但传感器可能有偏差。
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
        SELECT r.id, r.device_id, r.actual_voltage,
               r.temperature, r.humidity, r.harm_a1
        FROM records r
        WHERE r.enabled = 1 AND r.device_id IS NOT NULL
        ORDER BY r.id
    """, conn)

    devices = sorted(df["device_id"].dropna().unique())
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    # ── 1. 各设备温湿度均值对比 ──────────────────────────────
    print(f"{'='*60}")
    print(f"各设备 温度/湿度 均值对比")
    print(f"{'='*60}")
    for dev, color in zip(devices, colors):
        sub = df[df["device_id"] == dev]
        print(f"\n  设备 {dev[:12]}... ({len(sub)}条):")
        print(f"    温度: {sub['temperature'].mean():.2f} ± {sub['temperature'].std():.2f} °C")
        print(f"    湿度: {sub['humidity'].mean():.2f} ± {sub['humidity'].std():.2f} %")

    # ── 2. 同时间窗下对比（按 id 窗口对齐） ─────────────────
    print(f"\n{'='*60}")
    print(f"同时间段（每1000条窗口）各设备温湿度均值")
    print(f"{'='*60}")

    df["window"] = df["id"] // 1000
    pivot_t = df.pivot_table(index="window", columns="device_id", values="temperature", aggfunc="mean")
    pivot_h = df.pivot_table(index="window", columns="device_id", values="humidity", aggfunc="mean")

    print(f"\n  温度对比（部分行）:")
    print(f"  {'窗口':>6s}  ", end="")
    for d in devices:
        print(f"  {d[:12]:>12s}", end="")
    print()
    for w in sorted(pivot_t.index)[::5]:  # 每隔5个窗口显示
        print(f"  {w*1000:>6d}  ", end="")
        for d in devices:
            v = pivot_t.loc[w, d] if d in pivot_t.columns else None
            if pd.notna(v):
                print(f"  {v:>10.2f}°C", end="")
            else:
                print(f"  {'N/A':>12s}", end="")
        print()

    # ── 3. 设备间温湿度差值分布 ──────────────────────────────
    if len(devices) >= 2:
        print(f"\n{'='*60}")
        print(f"设备间温湿度差值（设备X - 设备1）")
        print(f"{'='*60}")
        ref_dev = devices[0]
        # 用窗口对齐来估计同时刻的差值
        for target_dev in devices[1:]:
            common_w = sorted(set(pivot_t.index) & set(pivot_t.columns) & set(pivot_t.columns))
            if not common_w:
                continue
            dT = pivot_t[target_dev] - pivot_t[ref_dev]
            dH = pivot_h[target_dev] - pivot_h[ref_dev]
            dT = dT.dropna()
            dH = dH.dropna()
            if len(dT) > 0:
                print(f"\n  设备{target_dev[:12]} - 设备{ref_dev[:12]}:")
                print(f"    Δ温度: {dT.mean():+.3f} ± {dT.std():.3f} °C")
                print(f"    Δ湿度: {dH.mean():+.3f} ± {dH.std():.3f} %")

    # ── 4. 画图：三设备的温度曲线 + 湿度曲线 ────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # 温度曲线
    ax = axes[0]
    for dev, color in zip(devices, colors):
        sub = df[df["device_id"] == dev].copy()
        # 每500条取均值画点（否则点太多看不清）
        sub["batch"] = sub["id"] // 500
        grp = sub.groupby("batch")["temperature"].mean()
        ax.plot(grp.index * 500, grp.values, "-", color=color, label=f"设备 {dev[:12]}", linewidth=0.8, alpha=0.8)
    ax.set_ylabel("温度 (°C)")
    ax.set_title("三个设备测得的温度对比")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 湿度曲线
    ax = axes[1]
    for dev, color in zip(devices, colors):
        sub = df[df["device_id"] == dev].copy()
        sub["batch"] = sub["id"] // 500
        grp = sub.groupby("batch")["humidity"].mean()
        ax.plot(grp.index * 500, grp.values, "-", color=color, label=f"设备 {dev[:12]}", linewidth=0.8, alpha=0.8)
    ax.set_xlabel("记录 ID")
    ax.set_ylabel("湿度 (%)")
    ax.set_title("三个设备测得的湿度对比")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
