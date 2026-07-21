"""
验证只用电信号特征（去掉温湿度、转速）的插值效果。
训练 50/70/90/110 预测 60/80/100
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import sqlite3

from swa.data.manager import DataManager, PROJECTS_DIR
from swa.detection.least_squares import _FEATURE_NAMES
from swa.core.scoring import compute_score, compute_alpha7

project_dir = os.path.join(PROJECTS_DIR, "new")
db_path = os.path.join(project_dir, "data.db")

conn = sqlite3.connect(db_path)
rows = conn.execute("""
    SELECT r.id, r.actual_voltage, r.temperature, r.humidity, r.rpm, r.device_id, w.wave_data
    FROM records r JOIN waveforms w ON w.record_id = r.id
    WHERE r.enabled = 1
    ORDER BY r.id
""").fetchall()
conn.close()

records = []
for row in rows:
    rid, v, t, h, rpm, dev, ws = row
    try:
        wave = np.array([float(x) for x in ws.split(",")], dtype=np.float64)
    except Exception:
        continue
    if len(wave) < 20:
        continue
    y = wave - np.mean(wave)
    n = len(y)
    fft_vals = np.fft.rfft(y)
    mag = np.abs(fft_vals[1:])
    a1 = 0.0
    if len(mag) > 3:
        search_end = min(len(mag), n // 3)
        fund_idx = int(np.argmax(mag[:search_end]) + 1)
        a1 = float(mag[fund_idx - 1]) if mag[fund_idx - 1] > 0 else 0.0
    alpha7 = compute_alpha7(wave) or 0.0
    score = compute_score(wave)
    records.append({
        "id": rid,
        "actual_voltage": v,
        "harm_a1": a1,
        "alpha_7": alpha7,
        "score": score,
        "temperature": float(t or 0),
        "humidity": float(h or 0),
        "rpm": float(rpm or 0),
    })

import pandas as pd
df = pd.DataFrame(records)
df = df[df["actual_voltage"] >= 0].copy()

train_v = [50, 70, 90, 110]
test_v = [60, 80, 100]

# 三种特征组合
feature_sets = [
    ("全部特征 (含温湿度)", _FEATURE_NAMES),
    ("仅信号特征", ["alpha_7", "score", "harm_a1"]),
    ("仅 harm_a1", ["harm_a1"]),
]

for label, feats in feature_sets:
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print("=" * 60)

    train_df = df[df["actual_voltage"].isin(train_v)].copy()
    test_df = df[df["actual_voltage"].isin(test_v)].copy()

    # z-score
    mu, sigma = {}, {}
    for f in feats:
        vals = train_df[f].values
        mu[f] = float(np.mean(vals))
        sigma[f] = float(np.std(vals))
        if sigma[f] < 1e-12:
            sigma[f] = 1.0
        train_df[f] = (train_df[f] - mu[f]) / sigma[f]
        test_df[f] = (test_df[f] - mu[f]) / sigma[f]

    X_train = train_df[feats].values
    y_train = np.abs(train_df["actual_voltage"].values)
    X_aug = np.column_stack([np.ones(len(X_train)), X_train])
    coeffs, *_ = np.linalg.lstsq(X_aug, y_train, rcond=None)

    X_test = test_df[feats].values
    X_test_aug = np.column_stack([np.ones(len(X_test)), X_test])
    preds = X_test_aug @ coeffs

    for v in sorted(test_v):
        mask = test_df["actual_voltage"] == v
        p = preds[mask]
        print(f"  实际={v:.0f}V  预测均值={np.mean(p):.1f}V  范围={np.min(p):.1f}~{np.max(p):.1f}V")

    mae = float(np.mean(np.abs(preds - np.abs(test_df["actual_voltage"].values))))
    print(f"  测试MAE: {mae:.3f}V")
