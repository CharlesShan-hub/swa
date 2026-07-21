"""
诊断：为什么训练 50/70/90/110 预测 60/80/100 不插值？
看看各电压的均值特征，检查特征-电压关系是否线性。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from swa.data.manager import DataManager
from swa.detection.least_squares import _extract_features, _FEATURE_NAMES

dm = DataManager()
dm.load_project("new")
conn = dm._conn

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
    feats = _extract_features(wave)
    feats["id"] = rid
    feats["actual_voltage"] = v
    feats["temperature"] = float(t or 0)
    feats["humidity"] = float(h or 0)
    feats["rpm"] = float(rpm or 0)
    feats["device_id"] = str(dev or "")
    records.append(feats)

df = pd.DataFrame(records)

# 只看正电压
df = df[df["actual_voltage"] >= 0].copy()

# 各电压的特征均值
print(f"{'电压':>5s}  {'条数':>5s}", end="")
for feat in _FEATURE_NAMES:
    print(f"  {feat:>10s}", end="")
print()

for v in sorted(df["actual_voltage"].unique()):
    sub = df[df["actual_voltage"] == v]
    print(f"{v:>5.0f}  {len(sub):>5d}", end="")
    for feat in _FEATURE_NAMES:
        vals = sub[feat].values
        print(f"  {np.mean(vals):>10.2f}", end="")
    print()

# 看看特征 vs 电压的线性度
print("\n── 各特征与电压的相关系数 ──")
for feat in _FEATURE_NAMES:
    r = np.corrcoef(df["actual_voltage"], df[feat])[0, 1]
    print(f"  {feat:>12s} vs 电压: r = {r:.4f}")

# 只用两个电压训练，看另一个的预测值
print("\n── 仅用 {50, 110}V 训练，预测中间各电压 ──")
train_v = [50, 110]
test_vs = [60, 70, 80, 90, 100]

train_df = df[df["actual_voltage"].isin(train_v)].copy()
train_df_norm = train_df.copy()
mu, sigma = {}, {}
for feat in _FEATURE_NAMES:
    mu[feat] = float(train_df[feat].mean())
    sigma[feat] = float(train_df[feat].std())
    if sigma[feat] < 1e-12:
        sigma[feat] = 1.0
    train_df_norm[feat] = (train_df[feat] - mu[feat]) / sigma[feat]

X = train_df_norm[_FEATURE_NAMES].values
y = np.abs(train_df_norm["actual_voltage"].values)
X_aug = np.column_stack([np.ones(len(X)), X])
coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)

for tv in [50] + test_vs + [110]:
    sub = df[df["actual_voltage"] == tv].copy()
    if len(sub) < 3:
        continue
    sub_norm = sub.copy()
    for feat in _FEATURE_NAMES:
        sub_norm[feat] = (sub[feat] - mu[feat]) / sigma[feat]
    X_test = sub_norm[_FEATURE_NAMES].values
    X_test_aug = np.column_stack([np.ones(len(X_test)), X_test])
    preds = X_test_aug @ coeffs
    actual = np.abs(sub["actual_voltage"].values)
    mean_pred = np.mean(preds)
    mean_actual = np.mean(actual)
    mae = np.mean(np.abs(actual - preds))
    print(f"  {tv:>3.0f}V: 实际均值={mean_actual:.1f}V  预测均值={mean_pred:.1f}V  MAE={mae:.2f}V")
