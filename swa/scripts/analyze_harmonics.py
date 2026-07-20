"""
对比各设备的谐波参数分布（A1, A2, THD, 噪声, 周期数等）。
直接从 SQLite 读已算好的字段，不需要重新跑算法。

用法: pixi run python scripts/analyze_harmonics.py --project 项目名
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import click
import numpy as np
import pandas as pd
from swa.data.manager import DataManager


@click.command()
@click.option("--project", "-p", required=True, help="项目名称")
@click.option("--voltage", "-v", default=None, type=float,
              help="只看某个电压（不指定则看全部）")
def main(project, voltage):
    dm = DataManager()
    dm.load_project(project)
    conn = dm._conn

    # 读取所有启用数据
    where = "r.enabled = 1"
    params = []
    if voltage is not None:
        where += " AND r.actual_voltage = ?"
        params.append(voltage)

    df = pd.read_sql_query(f"""
        SELECT r.id, r.actual_voltage, r.temperature, r.humidity,
               r.harm_a1, r.harm_a2, r.harm_error, r.harm_cycles,
               r.harm_thd, r.harm_noise_pct, r.device_id, r.test_case_code
        FROM records r
        WHERE {where}
        ORDER BY r.device_id, r.actual_voltage
    """, conn, params=params)

    devices = sorted(df["device_id"].dropna().unique())
    print(f"设备数: {len(devices)}")
    print(f"总样本: {len(df)} 条")
    if voltage:
        print(f"仅电压: {voltage:+.0f}V")

    # ── 1. 各设备的数据量 ─────────────────────────────────
    print(f"\n{'='*70}")
    print(f"1. 各设备数据量")
    print(f"{'='*70}")
    cnt = df.groupby("device_id").size()
    for dev in devices:
        n = cnt.get(dev, 0)
        pct = n / len(df) * 100
        print(f"  设备 {dev[:16]}...: {n:>6}条 ({pct:.1f}%)")

    # ── 2. 谐波参数对比（全部电压） ──────────────────────
    harm_fields = ["harm_a1", "harm_a2", "harm_thd", "harm_noise_pct", "harm_cycles", "harm_error"]
    print(f"\n{'='*70}")
    print(f"2. 各设备谐波参数均值对比（全部电压）")
    print(f"{'='*70}")
    header = f"{'设备':<20s}" + "".join(f"{f:>14s}" for f in harm_fields)
    print(header)
    print("-" * len(header))
    for dev in devices:
        sub = df[df["device_id"] == dev]
        vals = "  ".join(f"{sub[f].mean():>10.2f}" if sub[f].notna().any() else "  " for f in harm_fields)
        print(f"  {dev[:16]:<18s}  {vals}")

    # ── 3. 各电压、各设备的 A1 对比 ──────────────────────
    print(f"\n{'='*70}")
    print(f"3. 各电压下各设备的 A1 (基波幅值) 均值")
    print(f"{'='*70}")
    pivot = df.pivot_table(
        index="actual_voltage", columns="device_id",
        values="harm_a1", aggfunc="mean"
    )
    # 按电压排序
    pivot = pivot.sort_index()
    for v, row in pivot.iterrows():
        parts = [f"{v:+.0f}V".rjust(6)]
        for dev in devices:
            val = row.get(dev, None)
            if pd.notna(val):
                parts.append(f"{val:>8.0f}")
            else:
                parts.append(f"{'':>8s}")
        print("  ".join(parts))

    # ── 4. 各电压下各设备的 THD 对比 ─────────────────────
    print(f"\n{'='*70}")
    print(f"4. 各电压下各设备的 THD 均值")
    print(f"{'='*70}")
    pivot2 = df.pivot_table(
        index="actual_voltage", columns="device_id",
        values="harm_thd", aggfunc="mean"
    )
    pivot2 = pivot2.sort_index()
    for v, row in pivot2.iterrows():
        parts = [f"{v:+.0f}V".rjust(6)]
        for dev in devices:
            val = row.get(dev, None)
            if pd.notna(val):
                parts.append(f"{val:>8.4f}")
            else:
                parts.append(f"{'':>8s}")
        print("  ".join(parts))

    # ── 5. 各设备噪声分布 ────────────────────────────────
    print(f"\n{'='*70}")
    print(f"5. 各设备 噪声>30% 的比例")
    print(f"{'='*70}")
    for dev in devices:
        sub = df[df["device_id"] == dev]
        total = len(sub)
        noisy = sub["harm_noise_pct"].dropna()
        bad = (noisy > 0.30).sum()
        print(f"  设备 {dev[:16]:<18s}: {bad:>5d}/{total:<6d} ({bad/total*100:5.1f}%) 噪声超标")

    # ── 6. 电压与 A1 的线性关系（每组设备） ──────────────
    print(f"\n{'='*70}")
    print(f"6. 各设备 电压 vs A1 线性关系")
    print(f"{'='*70}")
    for dev in devices:
        sub = df[df["device_id"] == dev].dropna(subset=["harm_a1", "actual_voltage"])
        if len(sub) > 10:
            x = sub["actual_voltage"].abs().values
            y = sub["harm_a1"].values
            slope, intercept = np.polyfit(x, y, 1)
            r2 = 1 - np.sum((y - (slope * x + intercept)) ** 2) / np.sum((y - np.mean(y)) ** 2)
            print(f"  设备 {dev[:16]:<18s}: A1 = {slope:.2f}×V + {intercept:.0f}  (R²={r2:.3f})")

    # ── 7. 设备间的 A1 偏差（以第一个设备为基准） ────────
    if len(devices) >= 2:
        print(f"\n{'='*70}")
        print(f"7. 各设备 A1 相对基准的偏差")
        print(f"{'='*70}")
        ref_dev = devices[0]
        ref = df[df["device_id"] == ref_dev].groupby("actual_voltage")["harm_a1"].mean()
        for dev in devices[1:]:
            sub = df[df["device_id"] == dev].groupby("actual_voltage")["harm_a1"].mean()
            common_v = sorted(set(ref.index) & set(sub.index))
            if common_v:
                diffs = [sub[v] - ref[v] for v in common_v]
                mean_diff = np.mean(diffs)
                print(f"  设备{dev[:12]} vs 基准: A1 平均偏差 = {mean_diff:+.1f}")
                for v in common_v:
                    print(f"    {v:+.0f}V: 基准A1={ref[v]:.0f}  设备A1={sub[v]:.0f}  差={sub[v]-ref[v]:+.0f}")


if __name__ == "__main__":
    main()
