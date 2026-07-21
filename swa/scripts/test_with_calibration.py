"""
模拟用户设置：映射校准 + 交叉电压预测。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from swa.data.manager import DataManager, PROJECTS_DIR
from swa.detection.least_squares import run

project_dir = os.path.join(PROJECTS_DIR, "new")

# 获取设备列表
dm = DataManager()
dm.load_project("new")
devices = [r[0] for r in dm._conn.execute(
    "SELECT DISTINCT device_id FROM records WHERE enabled=1 AND device_id IS NOT NULL"
).fetchall()]
dm.close()
print(f"设备列表: {devices}")

train_v = [50, 70, 90, 110]
test_v = [60, 80, 100]

# ── 模式A: 用户当前设置（全部设备 + 映射校准）──
for ref in devices:
    print(f"\n{'=' * 60}")
    print(f"模式A: 全部设备 + 映射到 {ref[:12]}...")
    print("=" * 60)
    r = run(project_dir, train_v, test_v, device_id=None,
            device_mapping=True, ref_device_id=ref)
    if "error" not in r:
        for v in sorted(set(tr["actual"] for tr in r["test_results"])):
            preds = [tr["pred"] for tr in r["test_results"] if tr["actual"] == v]
            print(f"  实际={v:.0f}V  预测均值={np.mean(preds):.1f}V  范围={np.min(preds):.1f}~{np.max(preds):.1f}V")
        print(f"  测试MAE={r['metrics']['test']['mae']:.3f}V")

# ── 模式B: 选数据量最大的设备 ──
import pandas as pd
conn = dm._conn
counts = pd.read_sql("SELECT device_id, COUNT(*) as cnt FROM records WHERE enabled=1 AND device_id IS NOT NULL GROUP BY device_id", conn)
best_dev = counts.loc[counts["cnt"].idxmax(), "device_id"]
print(f"\n{'=' * 60}")
print(f"模式B: 仅设备 {best_dev[:12]}... (最多数据)")
print("=" * 60)
r = run(project_dir, train_v, test_v, device_id=best_dev)
if "error" not in r:
    for v in sorted(set(tr["actual"] for tr in r["test_results"])):
        preds = [tr["pred"] for tr in r["test_results"] if tr["actual"] == v]
        print(f"  实际={v:.0f}V  预测均值={np.mean(preds):.1f}V  范围={np.min(preds):.1f}~{np.max(preds):.1f}V")
    print(f"  测试MAE={r['metrics']['test']['mae']:.3f}V")

# ── 模式C: 选A1线性度最好的设备 ──
conn.execute("""
    SELECT r.device_id, r.actual_voltage, AVG(r.harm_a1)
    FROM records r
    WHERE r.enabled=1 AND r.harm_a1 IS NOT NULL AND r.device_id IS NOT NULL
    GROUP BY r.device_id, r.actual_voltage
    ORDER BY r.device_id, r.actual_voltage
""")
