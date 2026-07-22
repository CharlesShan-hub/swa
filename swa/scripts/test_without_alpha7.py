"""
去掉 alpha_7 后的完整测试
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import sqlite3
import numpy as np
from swa.data.manager import PROJECTS_DIR
from swa.detection.least_squares import run

conn = sqlite3.connect(os.path.join(PROJECTS_DIR, "new", "data.db"))
devices = [r[0] for r in conn.execute(
    "SELECT device_id FROM records WHERE enabled=1 AND device_id IS NOT NULL GROUP BY device_id ORDER BY COUNT(*) DESC"
).fetchall()]
conn.close()
dev_b = [d for d in devices if "B" in d][0]

train_v = [70, 80, 90, 100, 110, 120, 130, 140, 150]
test_v = [70, 80, 90, 100, 110, 120, 130, 140, 150]

r = run(os.path.join(PROJECTS_DIR, "new"), train_v, test_v,
        window_size=8, device_mapping=True, ref_device_id=dev_b)

m = r["metrics"]
print(f"特征: score + harm_a1 + temperature + humidity + rpm")
print(f"训练: {m['train_count']}条  MAE={m['train']['mae']:.3f}  R²={m['train']['r2']:.4f}")
print(f"测试: {m['test_count']}条  MAE={m['test']['mae']:.3f}  R²={m['test']['r2']:.4f}")
print()
for v in sorted(train_v):
    preds = [tr["pred"] for tr in r["train_results"] if tr["actual"] == v]
    if preds:
        mae = float(np.mean(np.abs(np.array(preds) - abs(v))))
        print(f"  V={v:+.0f}  预测={np.mean(preds):.1f}  MAE={mae:.3f}")
print()
for name in sorted(r.get("coefficients", {}).keys()):
    print(f"  {name}: {r['coefficients'][name]:.4f}")
print(f"  截距: {r.get('intercept', 0):.4f}")
