"""
分析各设备的预测偏差与温度、湿度、电压的关系。
用来判断设备间偏差是常数偏移还是与条件相关。

用法: pixi run python scripts/analyze_devices.py --project 项目名
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import click
import numpy as np
import pandas as pd
from swa.data.manager import DataManager, PROJECTS_DIR
from swa.detection.least_squares import _extract_features, _FEATURE_NAMES, _normalize


@click.command()
@click.option("--project", "-p", required=True, help="项目名称")
def main(project):
    # ── 加载项目 ──────────────────────────────────────────────
    dm = DataManager()
    dm.load_project(project)
    project_dir = os.path.join(PROJECTS_DIR, project)
    conn = dm._conn

    # ── 读取全部启用数据 ──────────────────────────────────────
    rows = conn.execute("""
        SELECT r.id, r.actual_voltage, r.temperature, r.humidity,
               r.rpm, r.device_id, w.wave_data
        FROM records r
        JOIN waveforms w ON w.record_id = r.id
        WHERE r.enabled = 1
        ORDER BY r.device_id, r.id
    """).fetchall()

    # ── 提取特征 ──────────────────────────────────────────────
    records = []
    for row in rows:
        rid, voltage, temp, humid, rpm_val, dev_id, wave_str = row
        try:
            wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
        except Exception:
            continue
        if len(wave) < 20:
            continue
        feats = _extract_features(wave)
        feats["id"] = rid
        feats["actual_voltage"] = voltage
        feats["temperature"] = float(temp) if temp is not None else 0.0
        feats["humidity"] = float(humid) if humid is not None else 0.0
        feats["rpm"] = float(rpm_val) if rpm_val is not None else 0.0
        feats["device_id"] = str(dev_id) if dev_id is not None else "None"
        records.append(feats)

    df = pd.DataFrame(records)
    print(f"总样本: {len(df)} 条")

    # ── 按设备分组 ──────────────────────────────────────────────
    devices = sorted(df["device_id"].unique())
    print(f"设备列表: {devices}\n")

    # 对每个设备独立做回归分析
    from swa.detection.least_squares import _FEATURE_NAMES
    for dev in devices:
        sub = df[df["device_id"] == dev].copy()
        sub = sub.reset_index(drop=True)  # 重置索引，避免位置错位
        if len(sub) < 10:
            continue

        # z-score 归一化
        sub_norm = sub.copy()
        mu = {f: float(sub[f].mean()) for f in _FEATURE_NAMES}
        sigma = {f: float(sub[f].std()) for f in _FEATURE_NAMES}
        for f in _FEATURE_NAMES:
            sub_norm[f] = (sub[f] - mu[f]) / max(sigma[f], 1e-12)

        # 线性回归
        X = sub_norm[_FEATURE_NAMES].values
        y = np.abs(sub_norm["actual_voltage"].values)
        X_aug = np.column_stack([np.ones(len(X)), X])
        coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        pred = X_aug @ coeffs
        errors = y - pred

        mae = float(np.mean(np.abs(errors)))
        bias = float(np.mean(errors))

        print(f"{'='*50}")
        print(f"设备 {dev}  ({len(sub)} 条)")
        print(f"  MAE:     {mae:.3f}V")
        print(f"  平均偏差: {bias:+.3f}V  (预测偏{'高' if bias<0 else '低'} {abs(bias):.2f}V)")

        # ── 偏差 vs 电压 ──────────────────────────────────────
        sub["abs_v"] = sub["actual_voltage"].abs()
        print(f"\n  ── 各电压偏差 ──")
        for av in sorted(sub["abs_v"].unique()):
            mask = sub["abs_v"] == av
            e = errors[mask]
            t = sub.loc[mask, "temperature"].mean()
            h = sub.loc[mask, "humidity"].mean()
            print(f"    {av:+.0f}V: 偏差均值={np.mean(e):+.3f}V  MAE={np.mean(np.abs(e)):.3f}V  "
                  f"n={int(mask.sum())}  T={t:.1f}°C  RH={h:.1f}%")

        # ── 偏差 vs 温度 ──────────────────────────────────────
        print(f"\n  ── 偏差 vs 温度 ──")
        sub["temp_bin"] = pd.cut(sub["temperature"], bins=5)
        for label, grp in sorted(sub.groupby("temp_bin", observed=True)):
            e = errors[grp.index]
            if len(e) > 0:
                print(f"    T≈{label.mid:.1f}°C: 偏差={np.mean(e):+.3f}V  n={len(e)}")

        # ── 偏差 vs 湿度 ──────────────────────────────────────
        print(f"\n  ── 偏差 vs 湿度 ──")
        sub["hum_bin"] = pd.cut(sub["humidity"], bins=5)
        for label, grp in sorted(sub.groupby("hum_bin", observed=True)):
            e = errors[grp.index]
            if len(e) > 0:
                print(f"    RH≈{label.mid:.1f}%: 偏差={np.mean(e):+.3f}V  n={len(e)}")

    # ── 设备间比较：用设备3模型预测其他设备 ──────────────────
    print(f"\n{'='*60}")
    print(f"用「最佳设备」模型预测其他设备")
    print(f"{'='*60}")

    # 找出数据量最多的设备作为基准（或用户指定的设备3）
    best_dev = "3" if "3" in devices else devices[0]
    print(f"基准设备: {best_dev}")

    # 用基准设备训练模型
    ref = df[df["device_id"] == best_dev].copy()
    ref_norm = ref.copy()
    mu = {f: float(ref[f].mean()) for f in _FEATURE_NAMES}
    sigma = {f: float(ref[f].std()) for f in _FEATURE_NAMES}
    for f in _FEATURE_NAMES:
        ref_norm[f] = (ref[f] - mu[f]) / max(sigma[f], 1e-12)

    X_ref = ref_norm[_FEATURE_NAMES].values
    y_ref = np.abs(ref["actual_voltage"].values)
    X_ref_aug = np.column_stack([np.ones(len(X_ref)), X_ref])
    ref_coeffs, *_ = np.linalg.lstsq(X_ref_aug, y_ref, rcond=None)

    # 用基准模型预测所有设备
    for dev in devices:
        sub = df[df["device_id"] == dev].copy()
        if len(sub) < 5:
            continue
        X_sub = sub[_FEATURE_NAMES].values
        X_sub_norm = (X_sub - np.array([mu[f] for f in _FEATURE_NAMES])) / np.array([max(sigma[f], 1e-12) for f in _FEATURE_NAMES])
        X_sub_aug = np.column_stack([np.ones(len(X_sub_norm)), X_sub_norm])
        pred_sub = X_sub_aug @ ref_coeffs
        actual_abs = np.abs(sub["actual_voltage"].values)
        err_sub = actual_abs - pred_sub

        bias = float(np.mean(err_sub))
        scatter = float(np.std(err_sub))
        print(f"\n  基准模型 → 设备 {dev} ({len(sub)} 条):")
        print(f"    平均偏差: {bias:+.3f}V  (设备比基准{'高' if bias>0 else '低'} {abs(bias):.2f}V)")
        print(f"    偏差标准差: {scatter:.3f}V")

        # 看偏差是否随电压线性变化
        abs_v = sub["actual_voltage"].abs().values
        if len(abs_v) > 10:
            slope = np.polyfit(abs_v, err_sub, 1)[0]
            print(f"    偏差-电压斜率: {slope:.4f}  ({'偏差随电压增大' if abs(slope)>0.01 else '基本常数'})")


if __name__ == "__main__":
    main()
