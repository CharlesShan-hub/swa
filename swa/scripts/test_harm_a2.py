"""
快速测试：添加 harm_a2 和 harm_a2_div_a1 对预测效果的影响。
只对比：
1. 当前最佳特征 (score, harm_a1, temp, humidity, rpm)
2. + harm_a2
3. + harm_a2_div_a1
4. + 两者
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import sqlite3

from swa.data.manager import PROJECTS_DIR
from swa.detection.least_squares import _load_data, _normalize, _FEATURE_NAMES

project_dir = os.path.join(PROJECTS_DIR, "new")

conn = sqlite3.connect(os.path.join(project_dir, "data.db"))
devices = [r[0] for r in conn.execute(
    "SELECT device_id FROM records WHERE enabled=1 AND device_id IS NOT NULL GROUP BY device_id ORDER BY COUNT(*) DESC"
).fetchall()]
conn.close()
dev_b = [d for d in devices if "B" in d][0]

train_v = [70, 80, 90, 100, 110, 120, 130, 140, 150]
test_v = [70, 80, 90, 100, 110, 120, 130, 140, 150]

print("加载数据中...")
t0 = time.time()
train_df, test_df = _load_data(
    project_dir, train_v, test_v,
    window_size=8,
    device_mapping=True, ref_device_id=dev_b,
    noise_correction=False,
)
print(f"加载完成: {len(train_df)} 训练, {len(test_df)} 测试 ({time.time()-t0:.1f}s)\n")

# 特征组合
base = ["score", "harm_a1", "temperature", "humidity", "rpm"]
sets = [
    ("基线 (无A2)",     base),
    ("+ harm_a2",       base + ["harm_a2"]),
    ("+ harm_a2_div_a1", base + ["harm_a2_div_a1"]),
    ("+ harm_a2 + A2/A1", base + ["harm_a2", "harm_a2_div_a1"]),
]

for label, feats in sets:
    train_n, test_n, norm = _normalize(train_df.copy(), test_df.copy(), feats)

    X = train_n[feats].values
    y = np.abs(train_n["actual_voltage"].values)
    X_aug = np.column_stack([np.ones(len(X)), X])
    coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)

    train_pred = X_aug @ coeffs
    test_pred = np.column_stack([np.ones(len(test_n)), test_n[feats].values]) @ coeffs

    train_mae = np.mean(np.abs(train_pred - y))
    test_mae = np.mean(np.abs(test_pred - np.abs(test_n["actual_voltage"].values)))

    print(f"  {label:25s}  MAE={test_mae:.4f}  R²={1 - np.sum((np.abs(test_n['actual_voltage'].values)-test_pred)**2)/np.sum((np.abs(test_n['actual_voltage'].values)-np.mean(np.abs(test_n['actual_voltage'].values)))**2):.4f}")

# 查看各电压预测
print(f"\n各电压预测详情 (取最优组合):")
best_feats = base + ["harm_a2", "harm_a2_div_a1"]
train_n, test_n, _ = _normalize(train_df.copy(), test_df.copy(), best_feats)
X = train_n[best_feats].values
y = np.abs(train_n["actual_voltage"].values)
X_aug = np.column_stack([np.ones(len(X)), X])
c, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
test_pred = np.column_stack([np.ones(len(test_n)), test_n[best_feats].values]) @ c

for v in sorted(test_v):
    mask = np.abs(test_n["actual_voltage"].values - v) < 1e-6
    preds = test_pred[mask]
    if len(preds) > 0:
        print(f"  V={v:+.0f}  预测={np.mean(preds):.1f}  MAE={np.mean(np.abs(preds-abs(v))):.3f}")
