"""
测试 3 个核心特征：A1, error/A1, humidity vs 全部 7 个特征。
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import sqlite3

from swa.data.manager import PROJECTS_DIR
from swa.detection.least_squares import _normalize

project_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "data", "projects", "new")
db_path = os.path.join(project_dir, "data.db")

conn = sqlite3.connect(db_path)
df = pd.read_sql_query("""
    SELECT id, actual_voltage, temperature, humidity, rpm, device_id,
           harm_a1, harm_a2, harm_error
    FROM records WHERE enabled=1
    ORDER BY id
""", conn)
conn.close()

print(f"记录: {len(df)}")

# 构造特征
df["error_div_a1"] = np.where(df["harm_a1"] > 1e-6, df["harm_error"] / df["harm_a1"], 0.0)
df["score"] = 0.0  # 占位（后续脚本模式没有score）

# 滑动窗口
def window_avg(df, w=8):
    dfs = []
    for v, grp in df.groupby("actual_voltage", sort=False):
        grp = grp.reset_index(drop=True)
        n = len(grp)
        cols = ["harm_a1","error_div_a1","humidity","temperature","rpm","harm_a2","score","actual_voltage"]
        if n < w:
            row = {c: grp[c].mean() for c in cols}
            dfs.append(row)
        else:
            for start in range(n - w + 1):
                seg = grp.iloc[start:start+w]
                row = {c: seg[c].mean() for c in cols}
                dfs.append(row)
    return pd.DataFrame(dfs)

print("滑动窗口...")
df_w = window_avg(df)
print(f"窗口后: {len(df_w)}")

train_v = [70, 80, 90, 100, 110, 120, 130, 140, 150]
test_v = [70, 80, 90, 100, 110, 120, 130, 140, 150]
train_df = df_w[df_w["actual_voltage"].isin(train_v)].copy()
test_df = df_w[df_w["actual_voltage"].isin(test_v)].copy()

# 特征组合
sets = [
    ("3 核心 (A1, err/A1, 湿度)",  ["harm_a1", "error_div_a1", "humidity"]),
    ("+ 温度 + RPM",              ["harm_a1", "error_div_a1", "humidity", "temperature", "rpm"]),
    ("+ 全部 7 个",               ["harm_a1", "error_div_a1", "humidity", "temperature", "rpm", "harm_a2"]),
]

print(f"\n{'='*60}")
print("特征组合对比 (全范围 70-150V)")
print(f"{'='*60}")

for label, feats in sets:
    train_n, test_n, norm = _normalize(train_df.copy(), test_df.copy(), feats)

    X = train_n[feats].values
    y = np.abs(train_n["actual_voltage"].values)
    X_aug = np.column_stack([np.ones(len(X)), X])
    coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    intercept, weights = coeffs[0], coeffs[1:]

    train_pred = X_aug @ coeffs
    test_pred = np.column_stack([np.ones(len(test_n)), test_n[feats].values]) @ coeffs

    train_mae = np.mean(np.abs(train_pred - y))
    test_mae = np.mean(np.abs(test_pred - np.abs(test_n["actual_voltage"].values)))
    test_r2 = 1 - np.sum((np.abs(test_n["actual_voltage"].values)-test_pred)**2) / max(np.sum((np.abs(test_n["actual_voltage"].values)-np.mean(np.abs(test_n["actual_voltage"].values)))**2), 1e-12)

    print(f"\n  {label}")
    print(f"  {'='*50}")
    print(f"  训练 MAE={train_mae:.4f}")
    print(f"  测试 MAE={test_mae:.4f}  R²={test_r2:.4f}")
    print(f"  系数:")
    for name, w in zip(feats, weights):
        print(f"    {name}: {w:.4f}")
    print(f"    截距: {intercept:.4f}")
    print(f"  各电压:")
    for v in sorted(test_v):
        mask = np.abs(test_n["actual_voltage"].values - v) < 1e-6
        preds = test_pred[mask]
        if len(preds) > 0:
            print(f"    V={v:+.0f}  预测={np.mean(preds):.1f}  MAE={np.mean(np.abs(preds-abs(v))):.3f}")

print(f"\n完成!")
