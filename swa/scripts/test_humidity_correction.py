"""对比湿度校正前后的插值效果。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from swa.data.manager import PROJECTS_DIR
from swa.detection.least_squares import run

project_dir = os.path.join(PROJECTS_DIR, "new")
train_v = [50, 70, 90, 110]
test_v = [60, 80, 100]

# 无校正
r_off = run(project_dir, train_v, test_v, device_id=None)
# 有校正
r_on = run(project_dir, train_v, test_v, device_id=None, humidity_correction=True)
# 校正 + 映射
devices = list({tr["device_id"] for tr in r_off.get("device_info", [])})
import sqlite3
conn = sqlite3.connect(os.path.join(project_dir, "data.db"))
devices = [r[0] for r in conn.execute("SELECT DISTINCT device_id FROM records WHERE enabled=1 AND device_id IS NOT NULL").fetchall()]
conn.close()
ref = devices[0] if devices else None

r_both = run(project_dir, train_v, test_v, device_id=None,
             humidity_correction=True, device_mapping=True, ref_device_id=ref)

for label, r in [("无校正", r_off), ("仅湿度校正", r_on), ("湿度校正+映射校准", r_both)]:
    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    if "error" in r:
        print(f"  错误: {r['error']}")
        continue
    for v in sorted(set(tr["actual"] for tr in r["test_results"])):
        preds = [tr["pred"] for tr in r["test_results"] if tr["actual"] == v]
        print(f"  实际={v:.0f}V  预测均值={np.mean(preds):.1f}V  范围={np.min(preds):.1f}~{np.max(preds):.1f}V")
    mae = r["metrics"]["test"]["mae"]
    print(f"  测试MAE: {mae:.3f}V")
