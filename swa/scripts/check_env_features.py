"""逐个检查环境特征贡献（轻量版：只加载一次数据）"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import sqlite3
import pandas as pd
from swa.data.manager import PROJECTS_DIR
from swa.detection.least_squares import _load_data, _normalize

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

print("加载数据（窗口8 + 映射B）...")
train_df, test_df = _load_data(project_dir, train_v, test_v,
    window_size=8, device_mapping=True, ref_device_id=dev_b)
print(f"  训练: {len(train_df)}条, 测试: {len(test_df)}条")

configs = [
    ("仅信号(s+A1)",       ["score", "harm_a1"]),
    ("+温度",              ["score", "harm_a1", "temperature"]),
    ("+湿度",              ["score", "harm_a1", "humidity"]),
    ("+RPM",               ["score", "harm_a1", "rpm"]),
    ("+温+湿",             ["score", "harm_a1", "temperature", "humidity"]),
    ("+温+RPM",            ["score", "harm_a1", "temperature", "rpm"]),
    ("+湿+RPM",            ["score", "harm_a1", "humidity", "rpm"]),
    ("全部(温+湿+RPM)",    ["score", "harm_a1", "temperature", "humidity", "rpm"]),
]

for label, feats in configs:
    tn, tst, _ = _normalize(train_df.copy(), test_df.copy(), feats)
    
    X = tn[feats].values
    y = np.abs(tn["actual_voltage"].values)
    X_aug = np.column_stack([np.ones(len(X)), X])
    coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    
    train_pred = X_aug @ coeffs
    mae = float(np.mean(np.abs(train_pred - y)))
    r2 = 1 - np.sum((y - train_pred)**2) / np.sum((y - np.mean(y))**2)
    
    mask70 = np.abs(tn["actual_voltage"].values - 70) < 1e-6
    preds70 = train_pred[mask70]
    mae70 = float(np.mean(np.abs(preds70 - 70))) if len(preds70) > 0 else 0
    
    print(f"  {label:20s}  MAE={mae:.3f}  R²={r2:.4f}  70V={np.mean(preds70):.1f}  MAE70={mae70:.3f}")
