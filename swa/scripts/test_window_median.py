"""
对比滑动窗口中位数 vs 均值的效果。
直接用 run() 函数，传 window_method 参数。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import sqlite3

from swa.data.manager import PROJECTS_DIR
from swa.detection.least_squares import run

project_dir = os.path.join(PROJECTS_DIR, "new")
db_path = os.path.join(project_dir, "data.db")

conn = sqlite3.connect(db_path)
devices = [r[0] for r in conn.execute(
    "SELECT DISTINCT device_id FROM records WHERE enabled=1 AND device_id IS NOT NULL ORDER BY device_id"
).fetchall()]
conn.close()
dev_b = [d for d in devices if "B" in d][0]
print(f"设备: {[d[-4:] for d in devices]}, 基准: {dev_b[-4:]}")

train_v = [70, 80, 90, 100, 110, 120, 130, 140, 150]
test_v = [70, 80, 90, 100, 110, 120, 130, 140, 150]

configs = [
    ("均值 (窗口=8)",  dict(window_size=8, window_method="mean")),
    ("中位数 (窗口=8)", dict(window_size=8, window_method="median")),
    ("无窗口",         dict(window_size=1, window_method="mean")),
]

for label, kw in configs:
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")

    r = run(
        project_dir, train_v, test_v,
        device_mapping=True, ref_device_id=dev_b,
        noise_correction=False,
        **kw,
    )
    if "error" in r:
        print(f"  错误: {r['error']}")
        continue
    m = r["metrics"]
    print(f"  样本: {m['train_count']}(训练) / {m['test_count']}(测试)")
    print(f"  训练 MAE={m['train']['mae']:.3f}  R²={m['train']['r2']:.4f}")
    print(f"  测试 MAE={m['test']['mae']:.3f}  R²={m['test']['r2']:.4f}")
    for v_label, v_mae in sorted(r.get("voltage_mae", {}).items(), key=lambda x: float(x[0].rstrip("V"))):
        print(f"    {v_label}: MAE={v_mae:.3f}")
