"""
对比不同特征组合的效果：
1. 仅 harm_a1
2. harm_a1 + score
3. harm_a1 + score + harm_a2
4. harm_a1 + score + harm_a2_div_a1
5. 全部特征
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import sqlite3

from swa.data.manager import PROJECTS_DIR
from swa.detection.least_squares import _load_data, _normalize, _FEATURE_NAMES

project_dir = os.path.join(PROJECTS_DIR, "new")
db_path = os.path.join(project_dir, "data.db")

conn = sqlite3.connect(db_path)
devices = [r[0] for r in conn.execute(
    "SELECT device_id FROM records WHERE enabled=1 AND device_id IS NOT NULL GROUP BY device_id ORDER BY COUNT(*) DESC"
).fetchall()]
conn.close()
dev_b = [d for d in devices if "B" in d][0]

train_v = [70, 80, 90, 100, 110, 120, 130, 140, 150]
test_v = [70, 80, 90, 100, 110, 120, 130, 140, 150]

train_df, test_df = _load_data(
    project_dir, train_v, test_v,
    window_size=8,
    device_mapping=True, ref_device_id=dev_b,
    noise_correction=False,
)

feature_sets = [
    ("仅 harm_a1",                  ["harm_a1"]),
    ("harm_a1 + score",             ["harm_a1", "score"]),
    ("harm_a1 + score + harm_a2",   ["harm_a1", "score", "harm_a2"]),
    ("harm_a1 + score + A2/A1",     ["harm_a1", "score", "harm_a2_div_a1"]),
    ("全部特征",                     _FEATURE_NAMES),
]

for label, feats in feature_sets:
    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print("=" * 50)

    train_n, test_n, norm = _normalize(train_df.copy(), test_df.copy(), feats)

    X = train_n[feats].values
    y = np.abs(train_n["actual_voltage"].values)
    X_aug = np.column_stack([np.ones(len(X)), X])
    coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    intercept = float(coeffs[0])

    X_test = test_n[feats].values
    train_pred = X_aug @ coeffs
    test_pred = np.column_stack([np.ones(len(X_test)), X_test]) @ coeffs

    train_mae = np.mean(np.abs(train_pred - y))
    test_mae = np.mean(np.abs(test_pred - np.abs(test_n["actual_voltage"].values)))

    print(f"  训练: MAE={train_mae:.3f}  R²={1 - np.sum((y-train_pred)**2)/np.sum((y-np.mean(y))**2):.4f}")
    print(f"  测试: MAE={test_mae:.3f}  R²={1 - np.sum((np.abs(test_n['actual_voltage'].values)-test_pred)**2)/np.sum((np.abs(test_n['actual_voltage'].values)-np.mean(np.abs(test_n['actual_voltage'].values)))**2):.4f}")

    print(f"\n  各电压:")
    for v in sorted(train_v):
        mask = np.abs(train_n["actual_voltage"].values - v) < 1e-6
        preds = train_pred[mask]
        if len(preds) > 0:
            mae_v = np.mean(np.abs(preds - abs(v)))
            print(f"    V={v:+.0f}  预测={np.mean(preds):.1f}  MAE={mae_v:.3f}")

    print(f"\n  系数:")
    for name, w in zip(feats, coeffs[1:]):
        print(f"    {name}: {w:.4f}")
    print(f"    截距: {intercept:.4f}")
