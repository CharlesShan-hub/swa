"""看每个设备的湿度校正系数大小。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import sqlite3
from swa.data.manager import PROJECTS_DIR

db_path = os.path.join(PROJECTS_DIR, "new", "data.db")
conn = sqlite3.connect(db_path)
rows = conn.execute("""
    SELECT r.device_id, r.actual_voltage, r.harm_a1, r.humidity
    FROM records r
    WHERE r.enabled = 1 AND r.harm_a1 IS NOT NULL AND r.device_id IS NOT NULL
    ORDER BY r.device_id
""").fetchall()
conn.close()

devices = {}
for r in rows:
    dev, v, a1, h = r
    devices.setdefault(dev, []).append((v, a1, h))

for dev, data in devices.items():
    X = np.column_stack([
        np.abs(np.array([d[0] for d in data])),
        np.array([d[2] for d in data]),
    ])
    y = np.array([d[1] for d in data])
    X_aug = np.column_stack([np.ones(len(X)), X])
    coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    alpha, gamma = coeffs[1], coeffs[2]
    print(f"设备 {dev[:16]}...: 电压系数={alpha:.4f}  湿度系数={gamma:.4f}")

# 校正前后的线性度对比
print("\n── 校正前后 A1 vs 电压的相关系数（不分层）──")
for dev, data in devices.items():
    vs = np.abs(np.array([d[0] for d in data]))
    a1 = np.array([d[1] for d in data])
    h = np.array([d[2] for d in data])
    r_before = np.corrcoef(vs, a1)[0, 1]
    # 校正
    X = np.column_stack([vs, h])
    y = a1
    X_aug = np.column_stack([np.ones(len(X)), X])
    coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    gamma = coeffs[2]
    a1_corrected = a1 - gamma * (h - 40.0)
    r_after = np.corrcoef(vs, a1_corrected)[0, 1]
    print(f"  设备 {dev[:16]}...: r(校正前)={r_before:.4f}  r(校正后)={r_after:.4f}  Δ={r_after - r_before:+.4f}")
