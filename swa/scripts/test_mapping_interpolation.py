"""
验证启用映射校准后，插值预测是否改善。
训练 50/70/90/110 预测 60/80/100
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from swa.data.manager import DataManager, PROJECTS_DIR
from swa.detection.least_squares import run

project_dir = os.path.join(PROJECTS_DIR, "new")

train_v = [50, 70, 90, 110]
test_v = [60, 80, 100]

# 方式1: 不启用映射校准（全部设备）
print("=" * 60)
print("方式1: 全部设备 / 无映射校准")
print("=" * 60)
r1 = run(project_dir, train_v, test_v, device_id=None)
if "error" not in r1:
    for v in sorted(set(tr["actual"] for tr in r1["test_results"])):
        preds = [tr["pred"] for tr in r1["test_results"] if tr["actual"] == v]
        print(f"  实际={v:.0f}V  预测均值={np.mean(preds):.1f}V  范围={min(preds):.1f}~{max(preds):.1f}V")
    mae = r1["metrics"]["test"]["mae"]
    print(f"  测试MAE: {mae:.3f}V")

# 获取设备列表
import sqlite3
db_path = os.path.join(project_dir, "data.db")
conn = sqlite3.connect(db_path)
devices = [r[0] for r in conn.execute("SELECT DISTINCT device_id FROM records WHERE enabled=1 AND device_id IS NOT NULL").fetchall()]
conn.close()
print(f"\n可用设备: {devices}")

# 方式2: 全部设备 + 启用映射校准（以第一个设备为基准）
if len(devices) >= 2:
    ref = devices[0]
    print(f"\n{'=' * 60}")
    print(f"方式2: 全部设备 + 映射校准 (基准={ref[:12]}...)")
    print("=" * 60)
    r2 = run(project_dir, train_v, test_v, device_id=None,
             device_mapping=True, ref_device_id=ref)
    if "error" not in r2:
        for v in sorted(set(tr["actual"] for tr in r2["test_results"])):
            preds = [tr["pred"] for tr in r2["test_results"] if tr["actual"] == v]
            print(f"  实际={v:.0f}V  预测均值={np.mean(preds):.1f}V  范围={min(preds):.1f}~{max(preds):.1f}V")
        mae = r2["metrics"]["test"]["mae"]
        print(f"  测试MAE: {mae:.3f}V")

# 方式3: 仅用单个设备（选数据量最大的）
import pandas as pd
conn = sqlite3.connect(db_path)
counts = pd.read_sql("SELECT device_id, COUNT(*) as cnt FROM records WHERE enabled=1 AND device_id IS NOT NULL GROUP BY device_id", conn)
conn.close()
best_dev = counts.loc[counts["cnt"].idxmax(), "device_id"]
print(f"\n{'=' * 60}")
print(f"方式3: 仅设备 {best_dev[:12]}... (数据最多)")
print("=" * 60)
r3 = run(project_dir, train_v, test_v, device_id=best_dev)
if "error" not in r3:
    for v in sorted(set(tr["actual"] for tr in r3["test_results"])):
        preds = [tr["pred"] for tr in r3["test_results"] if tr["actual"] == v]
        print(f"  实际={v:.0f}V  预测均值={np.mean(preds):.1f}V  范围={min(preds):.1f}~{max(preds):.1f}V")
    mae = r3["metrics"]["test"]["mae"]
    print(f"  测试MAE: {mae:.3f}V")
