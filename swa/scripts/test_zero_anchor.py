"""
验证"零信号锚点"思路：加入 (A1=0, V=0) 物理约束
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import sqlite3

from swa.data.manager import PROJECTS_DIR
from swa.detection.least_squares import _FEATURE_NAMES, _normalize, _extract_features

project_dir = os.path.join(PROJECTS_DIR, "new")
db_path = os.path.join(project_dir, "data.db")

# ── 加载数据 ──
conn = sqlite3.connect(db_path)
rows = conn.execute("""
    SELECT r.id, r.actual_voltage, r.temperature, r.humidity, r.rpm,
           r.device_id, r.harm_a1
    FROM records r
    WHERE r.enabled = 1 AND r.actual_voltage >= 0
      AND r.device_id IS NOT NULL AND r.humidity IS NOT NULL
    ORDER BY r.id
""").fetchall()
conn.close()

records = []
for row in rows:
    rid, voltage, temp, humid, rpm_val, dev_id, a1 = row
    records.append({
        "id": rid, "actual_voltage": voltage,
        "temperature": float(temp) if temp is not None else 0.0,
        "humidity": float(humid) if humid is not None else 0.0,
        "rpm": float(rpm_val) if rpm_val is not None else 0.0,
        "device_id": str(dev_id) if dev_id else None,
        "harm_a1": float(a1) if a1 is not None else 0.0,
    })

# 从 waveforms 计算 alpha_7 和 score
conn = sqlite3.connect(db_path)
wave_rows = conn.execute("""
    SELECT w.record_id, w.wave_data FROM waveforms w
    JOIN records r ON r.id = w.record_id WHERE r.enabled = 1
""").fetchall()
conn.close()

wave_feats = {}
for rec_id, wave_str in wave_rows:
    try:
        wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
        if len(wave) >= 20:
            feats = _extract_features(wave)
            wave_feats[rec_id] = (feats["alpha_7"], feats["score"])
    except Exception:
        continue

for rec in records:
    f = wave_feats.get(rec["id"], (0.0, 0.0))
    rec["alpha_7"], rec["score"] = f

df = pd.DataFrame(records)

# ── 设备映射（基准 B）──
devices = sorted(df["device_id"].dropna().unique())
dev_b = [d for d in devices if "B" in d][0]

def apply_mapping(df, ref_id):
    result = df.copy()
    ref_sub = df[df["device_id"] == ref_id]
    for dev in df["device_id"].unique():
        if dev == ref_id:
            continue
        tgt = df[df["device_id"] == dev]
        common_v = sorted(set(ref_sub["actual_voltage"]) & set(tgt["actual_voltage"]))
        if len(common_v) < 3:
            continue
        ratios = []
        for v in common_v:
            r_mean = ref_sub[ref_sub["actual_voltage"] == v]["harm_a1"].mean()
            t_mean = tgt[tgt["actual_voltage"] == v]["harm_a1"].mean()
            if r_mean > 0 and t_mean > 0:
                ratios.append((v, t_mean / r_mean))
        if len(ratios) < 3:
            continue
        vs = np.array([r[0] for r in ratios])
        rs = np.array([r[1] for r in ratios])
        coeffs = np.polyfit(vs, rs, 1)
        mask = result["device_id"] == dev
        result.loc[mask, "harm_a1"] = result.loc[mask].apply(
            lambda r: r["harm_a1"] / (coeffs[0] * abs(r["actual_voltage"]) + coeffs[1])
            if (coeffs[0] * abs(r["actual_voltage"]) + coeffs[1]) > 0.01 else r["harm_a1"],
            axis=1
        )
    return result

df = apply_mapping(df, dev_b)

# ── 滑动窗口8 ──
def apply_window(df, window_size=8):
    if window_size <= 1:
        return df
    windows = []
    for voltage, group in df.groupby("actual_voltage", sort=False):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < window_size:
            row = {"actual_voltage": voltage}
            for feat in _FEATURE_NAMES:
                row[feat] = float(group[feat].mean())
            windows.append(row)
            continue
        for start in range(n - window_size + 1):
            seg = group.iloc[start:start + window_size]
            row = {"actual_voltage": voltage}
            for feat in _FEATURE_NAMES:
                row[feat] = float(seg[feat].mean())
            windows.append(row)
    return pd.DataFrame(windows)

df = apply_window(df, 8)

# ── 拆分 ──
train_v = [70, 90, 110, 130, 150]
test_v = [80, 100, 120, 140]
train_df = df[df["actual_voltage"].isin(train_v)].copy()
test_df = df[df["actual_voltage"].isin(test_v)].copy()


# ── 回归 ──
def run(train, test, zero_n=0):
    """zero_n: 加入 N 个零信号锚点"""
    t = train.copy()
    if zero_n > 0:
        zeros = pd.DataFrame([{f: 0.0 for f in _FEATURE_NAMES} for _ in range(zero_n)])
        zeros["actual_voltage"] = 0.0
        t = pd.concat([t, zeros], ignore_index=True)

    train_n, test_n, _ = _normalize(t, test, _FEATURE_NAMES)

    X = train_n[_FEATURE_NAMES].values
    y = np.abs(train_n["actual_voltage"].values)
    X_aug = np.column_stack([np.ones(len(X)), X])
    coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    intercept = float(coeffs[0])

    X_test = test_n[_FEATURE_NAMES].values
    train_pred = X_aug @ coeffs
    test_pred = np.column_stack([np.ones(len(X_test)), X_test]) @ coeffs

    return train_pred, test_pred, intercept, train_n, test_n


for label, zn in [("无锚点", 0), ("锚点×1", 1), ("锚点×10", 10), ("锚点×100", 100)]:
    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print("=" * 55)

    tp, tsp, intercept, train_n, test_n = run(train_df, test_df, zn)

    train_mae = np.mean(np.abs(tp - np.abs(train_n["actual_voltage"].values)))
    test_mae = np.mean(np.abs(tsp - np.abs(test_n["actual_voltage"].values)))

    print(f"  截距: {intercept:.4f}")

    print(f"\n  训练:")
    by_v = {}
    for idx in range(len(train_n)):
        v = train_n.iloc[idx]["actual_voltage"]
        by_v.setdefault(v, []).append(tp[idx])
    for v in sorted(by_v):
        preds = by_v[v]
        if v == 0:
            print(f"    V=锚点  预测={np.mean(preds):.1f}  n={len(preds)}")
        else:
            print(f"    V={v:+.0f}  预测={np.mean(preds):.1f}  范围={min(preds):.1f}~{max(preds):.1f}")
    print(f"  训练MAE={train_mae:.3f}")

    print(f"\n  测试:")
    for v in sorted(test_v):
        mask = np.abs(test_n["actual_voltage"].values - v) < 1e-6
        preds = tsp[mask]
        print(f"    V={v:+.0f}  预测={np.mean(preds):.1f}  范围={min(preds):.1f}~{max(preds):.1f}  MAE={np.mean(np.abs(preds-v)):.3f}")
    print(f"  测试MAE={test_mae:.3f}")
