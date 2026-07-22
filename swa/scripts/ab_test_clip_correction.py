"""
A/B Test: 原始 A1 vs 矫正 A1 — 直接从数据库读取两列值对比
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import sqlite3
from swa.core.scoring import compute_score
from swa.detection.least_squares import (_extract_features, _normalize,
                                         _apply_window, _FEATURE_NAMES)

PROJECT_DIR = r"d:\project\work\swa\swa\src\data\projects\new"
DB_PATH = os.path.join(PROJECT_DIR, "data.db")
TRAIN_V = [70, 90, 110, 130, 150]
TEST_V = [80, 100, 120, 140]

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 用窗口=8，设备映射到指定设备
REF_DEVICE = "360111581B50303557363130300A6A39"

cur.execute("""
    SELECT r.id, r.actual_voltage, r.device_id, r.temperature, r.humidity,
           r.harm_a1, r.harm_a1_corrected, w.wave_data
    FROM records r
    JOIN waveforms w ON w.record_id = r.id
    WHERE r.enabled=1 AND r.actual_voltage>=0
    ORDER BY r.id
""")
rows = cur.fetchall()
conn.close()

print(f"总记录: {len(rows)}")

N_POINTS = 512
WINDOW_SIZE = 8


def load_data(rows, use_corrected: bool):
    """加载数据，use_corrected=True 时用 harm_a1_corrected，否则用 harm_a1。"""
    records = []
    for rid, voltage, dev_id, temp, humid, a1_orig, a1_corrected, wave_str in rows:
        try:
            wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
        except (ValueError, TypeError):
            continue
        if len(wave) < 20:
            continue

        feats = _extract_features(wave)

        # 选择 A1
        if use_corrected and a1_corrected is not None:
            a1 = float(a1_corrected)
        else:
            a1 = float(a1_orig) if a1_orig is not None else 0.0

        feats["id"] = rid
        feats["actual_voltage"] = voltage
        feats["harm_a1"] = a1
        feats["temperature"] = float(temp) if temp is not None else 0.0
        feats["humidity"] = float(humid) if humid is not None else 0.0
        feats["device_id"] = str(dev_id)[-4:] if dev_id else None
        records.append(feats)

    df = pd.DataFrame(records)

    # 设备映射（选择指定参考设备）
    dev_col = "device_id"
    ref_suffix = REF_DEVICE[-4:]
    unique_devices = df[dev_col].dropna().unique()
    if ref_suffix in unique_devices:
        ref_data = df[df[dev_col] == ref_suffix][["harm_a1", "actual_voltage"]].dropna()
        if len(ref_data) >= 5:
            coeffs = np.polyfit(ref_data["actual_voltage"], ref_data["harm_a1"], 1)
            slope, intercept = coeffs
            for dev in unique_devices:
                if dev != ref_suffix:
                    mask = df[dev_col] == dev
                    df.loc[mask, "harm_a1"] = df.loc[mask, "harm_a1"] / (slope * df.loc[mask, "actual_voltage"] / df.loc[mask, "actual_voltage"].mean() + intercept - intercept)

            # 简化：直接按电压映射 ratio(v) = ref_a1 / dev_a1
            v_ref = ref_data.groupby("actual_voltage")["harm_a1"].mean()
            for dev in unique_devices:
                if dev == ref_suffix:
                    continue
                dev_data = df[df[dev_col] == dev][["harm_a1", "actual_voltage"]].dropna()
                for v in dev_data["actual_voltage"].unique():
                    if v in v_ref.index:
                        ratio = v_ref[v] / dev_data[dev_data["actual_voltage"] == v]["harm_a1"].mean()
                        mask = (df[dev_col] == dev) & (df["actual_voltage"] == v)
                        df.loc[mask, "harm_a1"] *= ratio

    # 滑动窗口
    df_windowed = _apply_window(df, window_size=WINDOW_SIZE, method="mean")
    return df_windowed


def run_regression(df, train_v, test_v):
    """在 df 上运行回归并返回指标。"""
    train_df = df[df["actual_voltage"].isin(train_v)].copy()
    test_df = df[df["actual_voltage"].isin(test_v)].copy()

    if len(train_df) < 5 or len(test_df) < 5:
        return None

    FEATURES = _FEATURE_NAMES  # ["score", "harm_a1", "temperature", "humidity"]

    # 归一化
    train_n, test_n, norm_params = _normalize(train_df, test_df, FEATURES)

    # 线性回归
    X_train = train_n[FEATURES].values
    y_train = np.abs(train_n["actual_voltage"].values)
    X_train_aug = np.column_stack([np.ones(len(X_train)), X_train])
    coeffs, *_ = np.linalg.lstsq(X_train_aug, y_train, rcond=None)

    # 预测
    X_test = test_n[FEATURES].values
    X_test_aug = np.column_stack([np.ones(len(X_test)), X_test])
    pred = X_test_aug @ coeffs
    actual = np.abs(test_df["actual_voltage"].values)

    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # 逐电压
    test_df2 = test_df.copy()
    test_df2["abs_v"] = np.abs(test_df2["actual_voltage"])
    per_voltage = {}
    for v in sorted(test_df2["abs_v"].unique()):
        mask = test_df2["abs_v"] == v
        v_act = np.abs(test_df2.loc[mask, "actual_voltage"].values)
        v_pred = pred[mask]
        per_voltage[f"{v:.0f}V"] = float(np.mean(np.abs(v_act - v_pred)))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "test_count": len(test_df),
        "train_count": len(train_df),
        "per_voltage": per_voltage,
        "coeffs": coeffs,
        "pred": pred,
        "actual": actual,
    }


# 按电压过滤
def match_voltages(df, target_vs):
    result = []
    for sv in target_vs:
        for av in sorted(df["actual_voltage"].unique()):
            if abs(av - sv) < 1e-6:
                result.append(av)
                break
    return result

# ── A: 原始 A1 ──
print("\nA: 原始 A1 (FFT)")
print("=" * 40)
df_a = load_data(rows, use_corrected=False)
train_v = match_voltages(df_a, TRAIN_V)
test_v = match_voltages(df_a, TEST_V)
r_a = run_regression(df_a, train_v, test_v)
if r_a:
    print(f"  MAE={r_a['mae']:.4f}  RMSE={r_a['rmse']:.4f}  R2={r_a['r2']:.4f}")
    print(f"  训练: {r_a['train_count']}  测试: {r_a['test_count']}")
    for v, m in r_a['per_voltage'].items():
        print(f"  {v}: MAE={m:.4f}")

# ── B: 矫正 A1 ──
print("\nB: 矫正 A1 (边缘区域锚点法)")
print("=" * 40)
df_b = load_data(rows, use_corrected=True)
r_b = run_regression(df_b, train_v, test_v)
if r_b:
    print(f"  MAE={r_b['mae']:.4f}  RMSE={r_b['rmse']:.4f}  R2={r_b['r2']:.4f}")
    print(f"  训练: {r_b['train_count']}  测试: {r_b['test_count']}")
    for v, m in r_b['per_voltage'].items():
        print(f"  {v}: MAE={m:.4f}")

# ── 对比 ──
if r_a and r_b:
    print(f"\n{'='*55}")
    print(f"A/B 对比: 原始 A1 vs 矫正 A1")
    print(f"{'='*55}")
    print(f"\n{'指标':>15s}  {'原始A1':>12s}  {'矫正A1':>12s}  {'差值':>12s}  {'改善'}")
    print("-" * 65)
    for key in ["mae", "rmse", "r2"]:
        va = r_a[key]
        vb = r_b[key]
        diff = va - vb
        if key == "r2":
            better = "YES" if diff < 0 else ("NO" if diff > 0 else "=")
        else:
            better = "YES" if diff > 0 else ("NO" if diff < 0 else "=")
        print(f"  {key:>13s}  {va:>10.4f}  {vb:>10.4f}  {diff:>+10.4f}  {better}")

    print(f"\n  每电压 MAE:")
    all_v = sorted(set(r_a["per_voltage"]) | set(r_b["per_voltage"]))
    for v in all_v:
        va = r_a["per_voltage"].get(v, 0)
        vb = r_b["per_voltage"].get(v, 0)
        diff = va - vb
        better = "YES" if diff > 0 else ("NO" if diff < 0 else "=")
        print(f"    {v:>6s}:  原始={va:.4f}  矫正={vb:.4f}  差={diff:+.4f}  {better}")

    mae_diff = r_a["mae"] - r_b["mae"]
    print(f"\n  结论: 矫正使 MAE {'改善' if mae_diff > 0 else '变差'} {abs(mae_diff):.4f} ({abs(mae_diff)/r_a['mae']*100:.1f}%)")
