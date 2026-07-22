"""
90-150V 训练，留 70/80 测试，看零信号预测
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import sqlite3

from swa.data.manager import PROJECTS_DIR
from swa.detection.least_squares import run, _FEATURE_NAMES

project_dir = os.path.join(PROJECTS_DIR, "new")
db_path = os.path.join(project_dir, "data.db")

conn = sqlite3.connect(db_path)
devices = [r[0] for r in conn.execute(
    "SELECT device_id FROM records WHERE enabled=1 AND device_id IS NOT NULL GROUP BY device_id ORDER BY COUNT(*) DESC"
).fetchall()]
conn.close()
dev_b = [d for d in devices if "B" in d][0]

# 对比三种训练范围
configs = [
    ("A: 70-150V 全范围", [70, 80, 90, 100, 110, 120, 130, 140, 150], [70, 80, 90, 100, 110, 120, 130, 140, 150]),
    ("B: 90-150V", [90, 100, 110, 120, 130, 140, 150], [70, 80]),
    ("C: 110-150V", [110, 120, 130, 140, 150], [70, 80, 90, 100]),
]

for label, train_v, test_v in configs:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  训练: {train_v}")
    print(f"  测试: {test_v}")
    print("=" * 60)

    r = run(project_dir, train_v, test_v,
            window_size=8,
            device_mapping=True, ref_device_id=dev_b,
            humidity_correction=False,
            max_samples_per_voltage=0)

    if "error" in r:
        print(f"  错误: {r['error']}")
        continue

    m = r["metrics"]
    print(f"  训练: {m['train_count']}条  MAE={m['train']['mae']:.3f}  R²={m['train']['r2']:.4f}")
    print(f"  测试: {m['test_count']}条  MAE={m['test']['mae']:.3f}  R²={m['test']['r2']:.4f}")

    norm_params = r.get("norm_params", {})
    coeffs = r.get("coefficients", {})
    intercept = r.get("intercept", 0)

    print(f"\n  截距: {intercept:.2f}V")

    # 零信号预测
    zero_raw = {
        "harm_a1": 0.0, "alpha_7": 0.0, "score": 0.0,
        "temperature": 25.0, "humidity": 40.0, "rpm": 0.0,
    }
    if norm_params:
        zero_pred = intercept
        for name in _FEATURE_NAMES:
            p = norm_params[name]
            z = (zero_raw[name] - p["mean"]) / p["std"]
            w = coeffs.get(name, 0)
            zero_pred += w * z
        print(f"  零信号预测(V): {zero_pred:.2f}")

    # 各电压预测
    print(f"\n  ── 训练电压 ──")
    by_v = {}
    for tr in r["train_results"]:
        by_v.setdefault(tr["actual"], []).append(tr["pred"])
    for v in sorted(by_v):
        preds = by_v[v]
        print(f"    V={v:+.0f}  预测={np.mean(preds):.1f}  MAE={np.mean(np.abs(np.array(preds)-abs(v))):.3f}")

    if r["test_results"]:
        print(f"  ── 测试电压 ──")
        by_v = {}
        for tr in r["test_results"]:
            by_v.setdefault(tr["actual"], []).append(tr["pred"])
        for v in sorted(by_v):
            preds = by_v[v]
            mae = np.mean(np.abs(np.array(preds) - abs(v)))
            print(f"    V={v:+.0f}  预测={np.mean(preds):.1f}  范围={min(preds):.1f}~{max(preds):.1f}  MAE={mae:.3f}")
