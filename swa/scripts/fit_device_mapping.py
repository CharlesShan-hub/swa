"""
拟合电压相关的设备映射函数: A1_X = ratio(v) × A1_1
ratio(v) = a×v + b 或 a×v² + b×v + c

用法: pixi run python scripts/fit_device_mapping.py --project new
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import click
import numpy as np
import pandas as pd
from swa.data.manager import DataManager


@click.command()
@click.option("--project", "-p", required=True, help="项目名称")
@click.option("--ref-device", default=None, help="基准设备ID（不指定则用第一个）")
def main(project, ref_device):
    dm = DataManager()
    dm.load_project(project)
    conn = dm._conn

    df = pd.read_sql_query("""
        SELECT r.device_id, r.actual_voltage, r.harm_a1
        FROM records r
        WHERE r.enabled = 1 AND r.harm_a1 IS NOT NULL
        ORDER BY r.device_id, r.actual_voltage
    """, conn)

    devices = sorted(df["device_id"].dropna().unique())
    if ref_device and ref_device in devices:
        ref_dev = ref_device
    else:
        ref_dev = devices[0]

    print(f"基准设备: {ref_dev[:16]}...")
    print(f"{'='*70}")

    ref = df[df["device_id"] == ref_dev].groupby("actual_voltage")["harm_a1"].mean()

    for target_dev in devices:
        if target_dev == ref_dev:
            continue

        sub = df[df["device_id"] == target_dev].groupby("actual_voltage")["harm_a1"].mean()
        common_v = sorted(set(ref.index) & set(sub.index))

        voltages = np.array(common_v)
        ratios = np.array([sub[v] / ref[v] for v in common_v])

        print(f"\n  目标设备: {target_dev[:16]}...")

        # ── 线性拟合: ratio = a×v + b ──
        coeffs_linear = np.polyfit(voltages, ratios, 1)
        pred_linear = np.polyval(coeffs_linear, voltages)
        mae_linear = np.mean(np.abs(ratios - pred_linear))
        max_err_linear = np.max(np.abs(ratios - pred_linear))

        # ── 二次拟合: ratio = a×v² + b×v + c ──
        coeffs_quad = np.polyfit(voltages, ratios, 2)
        pred_quad = np.polyval(coeffs_quad, voltages)
        mae_quad = np.mean(np.abs(ratios - pred_quad))
        max_err_quad = np.max(np.abs(ratios - pred_quad))

        # 输出映射参数
        print(f"  线性映射: ratio = {coeffs_linear[0]:.6f}×V + {coeffs_linear[1]:.4f}")
        print(f"    MAE={mae_linear:.4f} 最大误差={max_err_linear:.4f}")
        print(f"  二次映射: ratio = {coeffs_quad[0]:.8f}×V² + {coeffs_quad[1]:.6f}×V + {coeffs_quad[2]:.4f}")
        print(f"    MAE={mae_quad:.4f} 最大误差={max_err_quad:.4f}")

        # 输出各电压的比值和预测值
        print(f"\n  {'电压':>6s}  {'比值':>6s}  {'线性预测':>8s}  {'二次预测':>8s}")
        for v, r, pl, pq in zip(voltages, ratios, pred_linear, pred_quad):
            print(f"  {v:+.0f}V".rjust(6) + f"  {r:>6.3f}  {pl:>8.4f}  {pq:>8.4f}")

        # ── 用映射后的A1预测电压的误差 ──
        # 用基准设备的 A1-电压关系来验证
        print(f"\n  ── 用映射还原A1后的误差 ──")
        # A1_target_corrected = A1_target / ratio_predicted → 应该接近 A1_ref
        # 然后用 A1_ref 的电压回归来预测
        # 简单验证：校正后的A1 vs 实际基准A1
        for method_name, coeffs, pred_ratio in [("线性", coeffs_linear, pred_linear), ("二次", coeffs_quad, pred_quad)]:
            a1_target = np.array([sub[v] for v in common_v])
            a1_corrected = a1_target / pred_ratio  # 映射回基准设备
            a1_ref_vals = np.array([ref[v] for v in common_v])
            err = a1_corrected - a1_ref_vals
            mae_a1 = np.mean(np.abs(err))
            print(f"    {method_name}校正: A1平均误差 = {mae_a1:.1f}")


if __name__ == "__main__":
    main()
