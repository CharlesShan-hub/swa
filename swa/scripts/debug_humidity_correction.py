"""验证湿度校正函数是否在工作。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
import sqlite3
from swa.detection.least_squares import _humidity_correct_a1
from swa.data.manager import PROJECTS_DIR

db_path = os.path.join(PROJECTS_DIR, "new", "data.db")
conn = sqlite3.connect(db_path)
rows = conn.execute("""
    SELECT r.id, r.actual_voltage, r.harm_a1, r.humidity, r.device_id
    FROM records r
    WHERE r.enabled = 1 AND r.harm_a1 IS NOT NULL AND r.device_id IS NOT NULL
    ORDER BY r.id
""").fetchall()
conn.close()

records = [{"id": r[0], "actual_voltage": r[1], "harm_a1": r[2], "humidity": r[3], "device_id": str(r[4])} for r in rows]
df = pd.DataFrame(records)

# 只看设备 6A39
dev = df[df["device_id"].str.contains("6A39")]
print("=== 校正前（设备6A39 50V/60V）===")
for v in [50, 60]:
    s = dev[dev["actual_voltage"] == v]
    print(f"  {v}V: A1均值={s['harm_a1'].mean():.1f}  湿度均值={s['humidity'].mean():.1f}%")

df_corr = _humidity_correct_a1(dev)
print("\n=== 校正后 ===")
for v in [50, 60]:
    s = df_corr[df_corr["actual_voltage"] == v]
    print(f"  {v}V: A1均值={s['harm_a1'].mean():.1f}")
