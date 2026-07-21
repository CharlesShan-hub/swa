"""检查各设备在各电压下的 A1 均值，看设备混用是否导致非线性。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from swa.data.manager import DataManager

dm = DataManager()
dm.load_project("new")
conn = dm._conn
rows = conn.execute("""
    SELECT r.actual_voltage, r.harm_a1, r.device_id
    FROM records r
    WHERE r.enabled = 1 AND r.device_id IS NOT NULL
    ORDER BY r.device_id, r.actual_voltage
""").fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["voltage", "a1", "device"])
df = df[df["voltage"] >= 0]

# 先看总数据（不区分设备）
print("=== 不区分设备（当前情况）===")
for v in sorted(df["voltage"].unique()):
    sub = df[df["voltage"] == v]
    if len(sub) < 3:
        continue
    print(f"  {v:>3.0f}V ({len(sub):>4d}条): A1均值={sub['a1'].mean():.1f}")

# 按设备区分
print("\n=== 按设备区分 ===")
for dev in sorted(df["device"].unique()):
    ddf = df[df["device"] == dev]
    print(f"\n设备 {dev}:")
    for v in sorted(ddf["voltage"].unique()):
        sub = ddf[ddf["voltage"] == v]
        if len(sub) < 3:
            continue
        print(f"  {v:>3.0f}V ({len(sub):>4d}条): A1均值={sub['a1'].mean():.1f}")

# 各设备在各电压的数据量
print("\n=== 数据量分布（设备 x 电压）===")
pivot = df.pivot_table(index="device", columns="voltage", aggfunc="size", fill_value=0)
print(pivot)
