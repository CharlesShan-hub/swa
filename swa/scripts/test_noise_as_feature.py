"""
测试：把 noise_pct 作为特征加入模型
看模型能否自己学到"噪声大→电压小"的关系
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import sqlite3

from swa.data.manager import PROJECTS_DIR
from swa.detection.least_squares import _load_data, _normalize

project_dir = os.path.join(PROJECTS_DIR, "new")
db_path = os.path.join(project_dir, "data.db")

# 获取基准设备
conn = sqlite3.connect(db_path)
devices = [r[0] for r in conn.execute(
    "SELECT device_id FROM records WHERE enabled=1 AND device_id IS NOT NULL GROUP BY device_id ORDER BY COUNT(*) DESC"
).fetchall()]
conn.close()
dev_b = [d for d in devices if "B" in d][0]

train_v = [70, 80, 90, 100, 110, 120, 130, 140, 150]
test_v = [70, 80, 90, 100, 110, 120, 130, 140, 150]

# 加载原始数据 + 滑动窗口8 + 设备映射
train_df, test_df = _load_data(
    project_dir, train_v, test_v,
    window_size=8,
    device_mapping=True, ref_device_id=dev_b,
    noise_correction=False,  # 不校正 A1，用原始 A1
)

# 额外加载 noise_pct
# _load_data 已经把 harm_noise_pct 存到了 DataFrame 中
# 但窗口平均时没有包含它，所以这里需要重新加载带 noise_pct 的数据

# 换个方式：直接从数据库加载带 noise_pct 的窗口平均数据
print("从数据库加载原始数据，手动计算各指标...")

def load_with_noise(db_path, project_dir, train_v, test_v, ref_id):
    """加载数据+noise_pct，用自定义方式保留noise_pct"""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT r.id, r.actual_voltage, r.temperature, r.humidity, r.rpm,
               r.device_id, r.harm_noise_pct, w.wave_data
        FROM records r
        JOIN waveforms w ON w.record_id = r.id
        WHERE r.enabled = 1 AND r.actual_voltage >= 0
          AND r.device_id IS NOT NULL AND r.humidity IS NOT NULL
        ORDER BY r.id
    """).fetchall()
    conn.close()

    from swa.detection.least_squares import _extract_features
    
    records = []
    for row in rows:
        rid, voltage, temp, humid, rpm_val, dev_id, npct, wave_str = row
        try:
            wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
        except Exception:
            continue
        if len(wave) < 20:
            continue
        feats = _extract_features(wave)
        feats["id"] = rid
        feats["actual_voltage"] = voltage
        feats["temperature"] = float(temp) if temp is not None else 0.0
        feats["humidity"] = float(humid) if humid is not None else 0.0
        feats["rpm"] = float(rpm_val) if rpm_val is not None else 0.0
        feats["device_id"] = str(dev_id) if dev_id else None
        feats["noise_pct"] = float(npct) if npct is not None else 0.0
        records.append(feats)
    
    df = pd.DataFrame(records)
    
    # 设备映射
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
        cm = np.polyfit(vs, rs, 1)
        mask = df["device_id"] == dev
        df.loc[mask, "harm_a1"] = df.loc[mask].apply(
            lambda r: r["harm_a1"] / (cm[0] * abs(r["actual_voltage"]) + cm[1])
            if (cm[0] * abs(r["actual_voltage"]) + cm[1]) > 0.01 else r["harm_a1"],
            axis=1
        )
    
    # 滑动窗口8（保留 noise_pct）
    windows = []
    for voltage, group in df.groupby("actual_voltage", sort=False):
        group = group.reset_index(drop=True)
        n = len(group)
        ws = 8
        if n < ws:
            row = {"actual_voltage": voltage}
            for feat in ["alpha_7", "score", "harm_a1", "temperature", "humidity", "rpm", "noise_pct"]:
                row[feat] = float(group[feat].mean())
            windows.append(row)
            continue
        for start in range(n - ws + 1):
            seg = group.iloc[start:start + ws]
            row = {"actual_voltage": voltage}
            for feat in ["alpha_7", "score", "harm_a1", "temperature", "humidity", "rpm", "noise_pct"]:
                row[feat] = float(seg[feat].mean())
            windows.append(row)
    
    result = pd.DataFrame(windows)
    train = result[result["actual_voltage"].isin(train_v)].copy()
    test = result[result["actual_voltage"].isin(test_v)].copy()
    return train, test


FEAT_WITHOUT_NOISE = ["alpha_7", "score", "harm_a1", "temperature", "humidity", "rpm"]
FEAT_WITH_NOISE = FEAT_WITHOUT_NOISE + ["noise_pct"]

train, test = load_with_noise(db_path, project_dir, train_v, test_v, dev_b)
print(f"训练: {len(train)}条, 测试: {len(test)}条")

for label, feats in [("不含 noise_pct", FEAT_WITHOUT_NOISE), ("含 noise_pct", FEAT_WITH_NOISE)]:
    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print("=" * 55)
    
    train_n, test_n, norm = _normalize(train.copy(), test.copy(), feats)
    
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
    
    print(f"  训练MAE: {train_mae:.3f}")
    print(f"  测试MAE: {test_mae:.3f}")
    
    print(f"\n  各电压:")
    for v in sorted(train_v):
        mask = np.abs(train_n["actual_voltage"].values - v) < 1e-6
        preds = train_pred[mask]
        if len(preds) > 0:
            mae = np.mean(np.abs(preds - abs(v)))
            print(f"    V={v:+.0f}  预测={np.mean(preds):.1f}  MAE={mae:.3f}")
    
    print(f"\n  系数:")
    for name, w in zip(feats, coeffs[1:]):
        print(f"    {name}: {w:.4f}")
    print(f"    截距: {intercept:.4f}")
