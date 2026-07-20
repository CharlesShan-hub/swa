"""
分析设备间 A1 的映射关系：设备A的A1 × k + b = 设备B的A1？
如果成立，就可以把旧设备的模型通过数学变换适配到新设备。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import click
import numpy as np
import pandas as pd
from swa.data.manager import DataManager


@click.command()
@click.option("--project", "-p", required=True, help="项目名称")
def main(project):
    dm = DataManager()
    dm.load_project(project)
    conn = dm._conn

    df = pd.read_sql_query("""
        SELECT r.device_id, r.actual_voltage, r.harm_a1, r.harm_a2,
               r.harm_cycles, r.harm_thd, r.temperature, r.humidity
        FROM records r
        WHERE r.enabled = 1 AND r.harm_a1 IS NOT NULL
        ORDER BY r.device_id, r.actual_voltage
    """, conn)

    devices = sorted(df["device_id"].dropna().unique())
    print(f"设备: {devices}\n")

    # 1. 各电压下 A1 的比值
    print(f"{'='*70}")
    print(f"1. 各电压下 设备/设备1 的 A1 比值")
    print(f"{'='*70}")
    pivot = df.pivot_table(index="actual_voltage", columns="device_id", values="harm_a1", aggfunc="mean")
    pivot = pivot.sort_index()
    ref_dev = devices[0]
    header = f"{'电压':>8s}  {'A1_'+ref_dev[:8]:>10s}"
    for d in devices[1:]:
        header += f"  {'A1_'+d[:8]:>10s}  {'比值':>6s}"
    print(header)
    print("-" * len(header))
    for v, row in pivot.iterrows():
        ref_a1 = row[ref_dev]
        line = f"{v:+.0f}V".rjust(8) + f"  {ref_a1:>10.0f}"
        for d in devices[1:]:
            a1 = row.get(d, np.nan)
            if pd.notna(a1) and ref_a1 > 0:
                ratio = a1 / ref_a1
                line += f"  {a1:>10.0f}  {ratio:>6.3f}"
            else:
                line += f"  {'N/A':>10s}  {'N/A':>6s}"
        print(line)

    # 2. A1 线性映射 (设备1→设备X)
    print(f"\n{'='*70}")
    print(f"2. A1 线性映射：设备1_A1 × k + b = 设备X_A1")
    print(f"{'='*70}")
    ref = df[df["device_id"] == ref_dev].groupby("actual_voltage")["harm_a1"].mean()
    for dev in devices[1:]:
        sub = df[df["device_id"] == dev].groupby("actual_voltage")["harm_a1"].mean()
        common_v = sorted(set(ref.index) & set(sub.index))
        x = ref.loc[common_v].values  # 设备1的A1
        y = sub.loc[common_v].values  # 设备X的A1

        # 线性拟合: y = k*x + b
        k, b = np.polyfit(x, y, 1)
        y_pred = k * x + b
        residuals = y - y_pred
        mae = np.mean(np.abs(residuals))

        print(f"\n  设备{ref_dev[:12]} → 设备{dev[:12]}:")
        print(f"    A1_X = {k:.4f} × A1_1 + {b:.2f}")
        print(f"    MAE = {mae:.1f} (即映射后A1平均差{mae:.0f})")

        # 看各电压下的映射效果
        print(f"    {'电压':>8s}  {'A1_1':>8s}  {'A1_X_实际':>10s}  {'A1_X_预测':>10s}  {'误差':>8s}")
        for v, a1_ref, a1_actual, a1_pred in zip(common_v, x, y, y_pred):
            print(f"    {v:+.0f}V".rjust(8) + f"  {a1_ref:>8.0f}  {a1_actual:>10.0f}  {a1_pred:>10.1f}  {(a1_actual-a1_pred):>+8.1f}")

    # 3. 幅值归一化：如果只用 A1/A1_mean 做相对特征，设备间是不是更统一？
    print(f"\n{'='*70}")
    print(f"3. 如果用「相对幅值」(A1/该设备A1均值) 代替绝对A1")
    print(f"{'='*70}")
    for dev in devices:
        sub = df[df["device_id"] == dev]
        mean_a1 = sub["harm_a1"].mean()
        sub = sub.copy()
        sub["a1_rel"] = sub["harm_a1"] / mean_a1
        grp = sub.groupby("actual_voltage")["a1_rel"].mean()
        print(f"\n  设备{dev[:12]} (A1均值={mean_a1:.0f}):")
        for v in sorted(grp.index):
            print(f"    {v:+.0f}V: 相对A1={grp[v]:.3f}")


if __name__ == "__main__":
    main()
