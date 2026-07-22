"""
快速测试 A2 谐波特征效果。
直接从数据库读取预计算的 harm_a1, harm_a2，不走波形加载。
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import sqlite3

from swa.detection.least_squares import _normalize

project_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "data", "projects", "new")
db_path = os.path.join(project_dir, "data.db")

# 加载数据（仅特征，不加载波形）
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("""
    SELECT id, actual_voltage, temperature, humidity, rpm, device_id,
           harm_a1, harm_a2
    FROM records WHERE enabled=1
    ORDER BY id
""", conn)
conn.close()

print(f"加载 {len(df)} 条记录")

# A2/A1 比例
df["harm_a2_div_a1"] = np.where(df["harm_a1"] > 0, df["harm_a2"] / df["harm_a1"], 0.0)

# 滑动窗口平均
def window_avg(df, w=8):
    dfs = []
    for v, grp in df.groupby("actual_voltage", sort=False):
        grp = grp.reset_index(drop=True)
        n = len(grp)
        if n < w:
            row = {c: grp[c].mean() for c in ["harm_a1","harm_a2","harm_a2_div_a1","temperature","humidity","rpm","actual_voltage"]}
            dfs.append(row)
        else:
            for start in range(n - w + 1):
                seg = grp.iloc[start:start+w]
                row = {c: seg[c].mean() for c in ["harm_a1","harm_a2","harm_a2_div_a1","temperature","humidity","rpm","actual_voltage"]}
                dfs.append(row)
    return pd.DataFrame(dfs)

print("滑动窗口...")
df_w = window_avg(df)
print(f"窗口后: {len(df_w)} 条")

# 按电压分组
train_v = [70, 80, 90, 100, 110, 120, 130, 140, 150]
test_v = [70, 80, 90, 100, 110, 120, 130, 140, 150]

train_df = df_w[df_w["actual_voltage"].isin(train_v)].copy()
test_df = df_w[df_w["actual_voltage"].isin(test_v)].copy()
print(f"训练: {len(train_df)}  测试: {len(test_df)}")

# 特征组合对比
base = ["harm_a1", "temperature", "humidity", "rpm"]
sets = [
    ("基线 (无A2)",        base),
    ("+ harm_a2",          base + ["harm_a2"]),
    ("+ harm_a2_div_a1",   base + ["harm_a2_div_a1"]),
    ("+ harm_a2 + A2/A1",  base + ["harm_a2", "harm_a2_div_a1"]),
]

print(f"\n{'='*60}")
print("特征组合对比 (仅 harm_a1 + 环境 vs +A2)")
print(f"{'='*60}")

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
    test_r2 = 1 - np.sum((np.abs(test_n["actual_voltage"].values)-test_pred)**2) / max(np.sum((np.abs(test_n["actual_voltage"].values)-np.mean(np.abs(test_n["actual_voltage"].values)))**2), 1e-12)

    print(f"\n  {label:25s}: 训练 MAE={train_mae:.4f}  测试 MAE={test_mae:.4f}  R²={test_r2:.4f}")
    for v in sorted(test_v):
        mask = np.abs(test_n["actual_voltage"].values - v) < 1e-6
        preds = test_pred[mask]
        if len(preds) > 0:
            print(f"    V={v:+.0f}  预测={np.mean(preds):.1f}  MAE={np.mean(np.abs(preds-abs(v))):.3f}")

print(f"\n完成!")
