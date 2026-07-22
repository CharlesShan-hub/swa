"""
对比不同噪声过滤阈值的效果
基线：窗口8 + 映射B
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
dev_b = [d for d in devices if "B" in d][0]

train_v = [70, 80, 90, 100, 110, 120, 130, 140, 150]
test_v = [70, 80, 90, 100, 110, 120, 130, 140, 150]

# 先看各阈值的保留记录数
print("各噪声阈值的记录保留情况:")
conn = sqlite3.connect(db_path)
total = conn.execute("SELECT COUNT(*) FROM records WHERE enabled=1").fetchone()[0]
print(f"  总记录: {total}")
for threshold in [1.0, 0.25, 0.22, 0.20, 0.18, 0.16]:
    if threshold >= 1.0:
        kept = total
    else:
        kept = conn.execute(
            "SELECT COUNT(*) FROM records WHERE enabled=1 AND (harm_noise_pct IS NULL OR harm_noise_pct <= ?)",
            (threshold,)
        ).fetchone()[0]
    print(f"  noise_pct ≤ {threshold:.2f}: {kept} 条 (排除 {total - kept} 条, {100*(total-kept)/total:.1f}%)")
conn.close()

# 对比各阈值对预测效果的影响
print(f"\n\n{'=' * 70}")
print("噪声过滤对预测效果的影响（窗口8+映射B 全范围训练/测试）")
print("=" * 70)

configs = [
    ("不过滤", dict(max_noise_pct=1.0)),
    ("noise≤0.25", dict(max_noise_pct=0.25)),
    ("noise≤0.22", dict(max_noise_pct=0.22)),
    ("noise≤0.20", dict(max_noise_pct=0.20)),
    ("noise≤0.18", dict(max_noise_pct=0.18)),
]

for label, kw in configs:
    print(f"\n── {label} ──")

    r = run(project_dir, train_v, test_v,
            window_size=8,
            device_mapping=True, ref_device_id=dev_b,
            humidity_correction=False,
            **kw)

    if "error" in r:
        print(f"  错误: {r['error']}")
        continue

    m = r["metrics"]
    print(f"  样本: {m['train_count']}条")
    print(f"  训练: MAE={m['train']['mae']:.3f}  R²={m['train']['r2']:.4f}")
    print(f"  测试: MAE={m['test']['mae']:.3f}  R²={m['test']['r2']:.4f}")

    print(f"\n  各电压:")
    for v in sorted(train_v):
        preds = [tr["pred"] for tr in r["train_results"] if tr["actual"] == v]
        if preds:
            print(f"    V={v:+.0f}  预测={np.mean(preds):.1f}  MAE={np.mean(np.abs(np.array(preds)-abs(v))):.3f}")
