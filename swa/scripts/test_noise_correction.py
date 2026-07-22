"""
对比噪声校正效果：A1_clean = A1 × (1 - noise_pct)
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

# 噪声校正预览
print("噪声校正对各电压 A1 的影响:")
conn = sqlite3.connect(db_path)
for v in sorted(train_v):
    rows = conn.execute("""
        SELECT harm_a1, harm_noise_pct FROM records
        WHERE enabled=1 AND actual_voltage=? AND harm_a1 IS NOT NULL AND harm_noise_pct IS NOT NULL
    """, (float(v),)).fetchall()
    if not rows:
        continue
    a1 = np.array([r[0] for r in rows])
    npct = np.array([r[1] for r in rows])
    a1_clean = a1 * (1 - npct)
    delta = np.mean(a1) - np.mean(a1_clean)
    print(f"  V={v:+.0f}  n={len(a1)}  A1={np.mean(a1):.1f} → A1_cln={np.mean(a1_clean):.1f}  (↓{delta:.1f})")
conn.close()

# 效果对比
configs = [
    ("基线 (无校正)",    dict(noise_correction=False)),
    ("噪声校正",         dict(noise_correction=True)),
    ("噪声校正+湿度校正", dict(noise_correction=True, humidity_correction=True)),
]

for label, kw in configs:
    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print("=" * 55)

    r = run(project_dir, train_v, test_v,
            window_size=8,
            device_mapping=True, ref_device_id=dev_b,
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
            mae = np.mean(np.abs(np.array(preds) - abs(v)))
            print(f"    V={v:+.0f}  预测={np.mean(preds):.1f}  MAE={mae:.3f}")

    print(f"\n  系数:")
    for name in sorted(r.get("coefficients", {}).keys()):
        print(f"    {name}: {r['coefficients'][name]:.4f}")
    print(f"    截距: {r.get('intercept', 0):.4f}")
