"""
完整模式：70/90/110/130/150 训练 → 80/100/120/140 测试
对比基线 vs 各优化方案
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
    "SELECT device_id FROM records WHERE enabled=1 AND device_id IS NOT NULL GROUP BY device_id ORDER BY COUNT(*) DESC"
).fetchall()]
conn.close()

dev_a1, dev_b, dev_a2 = devices

train_v = [70, 90, 110, 130, 150]
test_v = [80, 100, 120, 140]

configs = [
    ("① 窗口1 + 默认",       dict(window_size=1, device_mapping=False)),
    ("② 窗口8 + 映射(B)",    dict(window_size=8, device_mapping=True, ref_device_id=dev_b)),
    ("③ 窗口8 + 映射(A1)",   dict(window_size=8, device_mapping=True, ref_device_id=dev_a1)),
    ("④ 窗口8 + 映射(B) + 湿度校正",
                              dict(window_size=8, device_mapping=True, ref_device_id=dev_b,
                                   humidity_correction=True)),
]

for label, kw in configs:
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print("=" * 65)

    r = run(project_dir, train_v, test_v, max_samples_per_voltage=0,
            device_id=None, **kw)

    if "error" in r:
        print(f"  错误: {r['error']}")
        continue

    m = r["metrics"]
    print(f"  训练: {m['train_count']}条  MAE={m['train']['mae']:.3f}  R²={m['train']['r2']:.4f}")
    print(f"  测试: {m['test_count']}条  MAE={m['test']['mae']:.3f}  R²={m['test']['r2']:.4f}")

    print(f"\n  ── 训练电压预测 ──")
    by_v = {}
    for tr in r["train_results"]:
        by_v.setdefault(tr["actual"], []).append(tr["pred"])
    for v in sorted(by_v):
        preds = by_v[v]
        print(f"    V={v:+.0f}  预测={np.mean(preds):.1f}  范围={min(preds):.1f}~{max(preds):.1f}")

    print(f"\n  ── 测试电压预测 ──")
    by_v = {}
    for tr in r["test_results"]:
        by_v.setdefault(tr["actual"], []).append(tr["pred"])
    for v in sorted(by_v):
        preds = by_v[v]
        mae_v = np.mean(np.abs(np.array(preds) - abs(v)))
        print(f"    V={v:+.0f}  预测={np.mean(preds):.1f}  范围={min(preds):.1f}~{max(preds):.1f}  MAE={mae_v:.3f}")

    print(f"\n  ── 回归系数 ──")
    for name, val in sorted(r.get("coefficients", {}).items()):
        print(f"    {name}: {val:.4f}")
    print(f"    截距: {r.get('intercept', 0):.4f}")
