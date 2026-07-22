"""
测试 4 个核心组件：设备映射 + A1 + error/A1 + 湿度
对比全 7 特征的效果。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import sqlite3

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
print(f"总记录: {len(df)}")

# ── 特征构造 ──
df["error_div_a1"] = np.where(df["harm_a1"] > 1e-6, df["harm_error"] / df["harm_a1"], 0.0)
df["score"] = 0.0  # 占位

# ── 设备映射 ──
devices = sorted(df["device_id"].dropna().unique())
dev_b = [d for d in devices if "B" in d][0]
print(f"设备: {[d[-4:] for d in devices]}, 基准: {dev_b[-4:]}")

for target_dev in devices:
    if target_dev == dev_b: continue
    mask = df["device_id"] == target_dev
    sub = df[mask]
    ref_sub = df[df["device_id"] == dev_b]
    v_ref = ref_sub.groupby("actual_voltage")["harm_a1"].mean()
    v_tgt = sub.groupby("actual_voltage")["harm_a1"].mean()
    common_v = sorted(set(v_ref.index) & set(v_tgt.index))
    if len(common_v) < 3: continue
    ratios = np.array([v_tgt[v] / v_ref[v] for v in common_v])
    a, b = np.polyfit(common_v, ratios, 1)
    def mapper(a1, v):
        r = a * abs(v) + b
        return a1 / r if r > 0.01 else a1
    df.loc[mask, "harm_a1"] = df.loc[mask].apply(lambda r: mapper(r["harm_a1"], r["actual_voltage"]), axis=1)
    # 温湿度偏移校正
    dt = sub["temperature"].mean() - ref_sub["temperature"].mean()
    dh = sub["humidity"].mean() - ref_sub["humidity"].mean()
    df.loc[mask, "temperature"] -= dt
    df.loc[mask, "humidity"] -= dh
print("设备映射完成")

# ── 滑动窗口 ──
def window_avg(df, w=8):
    dfs = []
    cols = ["harm_a1", "harm_a2", "error_div_a1", "humidity", "temperature", "rpm", "score", "actual_voltage"]
    for v, grp in df.groupby("actual_voltage", sort=False):
        grp = grp.reset_index(drop=True)
        n = len(grp)
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

# ── 特征组合 ──
train_v = [70, 80, 90, 100, 110, 120, 130, 140, 150]
test_v = [70, 80, 90, 100, 110, 120, 130, 140, 150]
train_df = df_w[df_w["actual_voltage"].isin(train_v)].copy()
test_df = df_w[df_w["actual_voltage"].isin(test_v)].copy()

from swa.detection.least_squares import _normalize

sets = [
    ("4 核心 (A1, err/A1, 湿度)",     ["harm_a1", "error_div_a1", "humidity"]),
    ("+ 温度",                        ["harm_a1", "error_div_a1", "humidity", "temperature"]),
    ("+ 温度 + RPM + A2",             ["harm_a1", "error_div_a1", "humidity", "temperature", "rpm", "harm_a2"]),
]

print(f"\n{'='*60}")
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

    print(f"\n  {label}")
    print(f"  {'='*50}")
    print(f"  训练 MAE={train_mae:.4f}")
    print(f"  测试 MAE={test_mae:.4f}  R²={test_r2:.4f}")
    print(f"  系数:")
    intercept, *weights = coeffs
    print(f"    截距: {intercept:.4f}")
    for name, w in zip(feats, weights):
        print(f"    {name}: {w:.4f}")
    for v in sorted(test_v):
        mask = np.abs(test_n["actual_voltage"].values - v) < 1e-6
        preds = test_pred[mask]
        if len(preds) > 0:
            print(f"    V={v:+.0f}  预测={np.mean(preds):.1f}  MAE={np.mean(np.abs(preds-abs(v))):.3f}")

print(f"\n完成!")
