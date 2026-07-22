"""
对比不同窗口大小的影响：1, 8, 16, 32, 64
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

for ws in [1, 8, 16, 32, 64]:
    print(f"\n{'='*50}")
    print(f"  窗口={ws}")
    print(f"{'='*50}")

    r = run(
        project_dir, train_v, test_v,
        window_size=ws,
        device_mapping=True, ref_device_id=dev_b,
        noise_correction=False,
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
