"""
对比 4 种配置对 70V 预测的影响：
1. 默认（无映射、无湿度校正）
2. 仅设备映射
3. 仅湿度校正
4. 设备映射 + 湿度校正
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import sqlite3

from swa.data.manager import PROJECTS_DIR
from swa.detection.least_squares import run

project_dir = os.path.join(PROJECTS_DIR, "new")
db_path = os.path.join(project_dir, "data.db")

# 查三个设备的完整 ID
conn = sqlite3.connect(db_path)
devices = [r[0] for r in conn.execute(
    "SELECT device_id, COUNT(*) as cnt FROM records WHERE enabled=1 AND device_id IS NOT NULL GROUP BY device_id ORDER BY cnt DESC"
).fetchall()]
for i, d in enumerate(devices):
    cnt = conn.execute("SELECT COUNT(*) FROM records WHERE enabled=1 AND device_id=?", (d,)).fetchone()[0]
    print(f"设备 {i+1}: {d}  n={cnt}")
conn.close()

ref_device = devices[0]  # 数据最多的那个做基准
print(f"\n基准设备: {ref_device[:28]}")

train_v = [70, 90, 110]
test_v = [80, 100]

configs = [
    ("① 默认",             dict(device_mapping=False, humidity_correction=False)),
    ("② 仅映射校准",        dict(device_mapping=True,  humidity_correction=False, ref_device_id=ref_device)),
    ("③ 仅湿度校正",        dict(device_mapping=False, humidity_correction=True)),
    ("④ 映射 + 湿度校正",   dict(device_mapping=True,  humidity_correction=True,  ref_device_id=ref_device)),
]

for label, kw in configs:
    print(f"\n\n{'=' * 60}")
    print(f"{label}")
    print("=" * 60)

    r = run(project_dir, train_v, test_v,
            window_size=1, max_samples_per_voltage=0,
            device_id=None,
            **kw)

    if "error" in r:
        print(f"  错误: {r['error']}")
        continue

    m = r["metrics"]
    print(f"  训练集: {m['train_count']} 条  MAE={m['train']['mae']:.3f}V  R²={m['train']['r2']:.4f}")
    print(f"  测试集: {m['test_count']} 条  MAE={m['test']['mae']:.3f}V  R²={m['test']['r2']:.4f}")

    print(f"\n  各训练电压:")
    train_by_v = {}
    for tr in r["train_results"]:
        train_by_v.setdefault(tr["actual"], []).append(tr["pred"])
    for v in sorted(train_by_v):
        preds = train_by_v[v]
        print(f"    V={v:+.0f}  预测均值={np.mean(preds):.1f} 范围={min(preds):.1f}~{max(preds):.1f}")

    print(f"\n  各测试电压:")
    test_by_v = {}
    for tr in r["test_results"]:
        test_by_v.setdefault(tr["actual"], []).append(tr["pred"])
    for v in sorted(test_by_v):
        preds = test_by_v[v]
        mae_v = np.mean(np.abs(np.array(preds) - v))
        print(f"    V={v:+.0f}  预测均值={np.mean(preds):.1f} 范围={min(preds):.1f}~{max(preds):.1f}  MAE={mae_v:.3f}")

    print(f"\n  回归系数:")
    for name, val in sorted(r.get("coefficients", {}).items()):
        print(f"    {name}: {val:.4f}")
    print(f"    截距: {r.get('intercept', 0):.4f}")
