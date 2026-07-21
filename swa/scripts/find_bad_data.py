"""找出哪些设备×电压的数据点异常，方便清理。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from swa.data.manager import DataManager

dm = DataManager()
dm.load_project("new")
df = pd.read_sql("""
    SELECT r.id, r.device_id, r.actual_voltage, r.harm_a1, r.harm_noise_pct, r.harm_thd, r.enabled
    FROM records r
    WHERE r.harm_a1 IS NOT NULL AND r.device_id IS NOT NULL
    ORDER BY r.device_id, r.actual_voltage, r.id
""", dm._conn)
dm.close()

df = df[df["actual_voltage"] >= 50].copy()

# 对每个(设备, 电压)算 A1 的统计，标出离群点
print("=== 各设备×电压的 A1 统计，标出异常 ===")
for dev in sorted(df["device_id"].unique()):
    sub = df[df["device_id"] == dev]
    for v in sorted(sub["actual_voltage"].unique()):
        grp = sub[sub["actual_voltage"] == v]
        a1_vals = grp["harm_a1"].values
        q1, q3 = np.percentile(a1_vals, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = grp[(grp["harm_a1"] < lower) | (grp["harm_a1"] > upper)]
        if len(outliers) > 0:
            print(f"\n设备 {dev[:16]}...  {v:.0f}V: 共{len(grp)}条  (Q1={q1:.0f} Q3={q3:.0f} IQR={iqr:.0f})")
            # 只列出少量异常
            for _, r in outliers.head(10).iterrows():
                print(f"  id={r['id']:>5d}  A1={r['harm_a1']:.1f}  noise={r['harm_noise_pct']:.2%}  thd={r['harm_thd']:.3f}  enabled={r['enabled']}")
            if len(outliers) > 10:
                print(f"  ... 还有 {len(outliers)-10} 条")

# 特别关注设备253D 的60V数据
print("\n\n=== 设备253D 的 50V vs 60V A1 对比 ===")
dev = sorted(df["device_id"].unique())[1]  # 253D 一般是第二个
sub = df[df["device_id"] == dev]
for v in [50, 60]:
    grp = sub[sub["actual_voltage"] == v]
    print(f"{v}V: {len(grp)}条  均值={grp['harm_a1'].mean():.1f}  中位数={grp['harm_a1'].median():.1f}  min={grp['harm_a1'].min():.1f}  max={grp['harm_a1'].max():.1f}")

# 检查60V中A1 < 130的数据
bad = sub[(sub["actual_voltage"] == 60) & (sub["harm_a1"] < 130)]
print(f"\n253D 的 60V 中 A1<130 的条数: {len(bad)}/{len(sub[sub['actual_voltage']==60])}")
print("这些数据的质量指标:")
print(bad[["id", "harm_a1", "harm_noise_pct", "harm_thd"]].describe())
