"""
复现之前的"最佳结果"配置，对比加/不加湿度校正。
- 滑动窗口=8
- 设备映射校准（基准=设备B 6A39）
- 训练 70/90/110 → 测试 80/100
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
    "SELECT device_id, COUNT(*) FROM records WHERE enabled=1 AND device_id IS NOT NULL GROUP BY device_id ORDER BY COUNT(*) DESC"
).fetchall()]
for d in devices:
    print(f"  {d}")
conn.close()

# 基准选设备 B（第二个，ID 包含 B）
ref_device = [d for d in devices if "B" in d][0]
print(f"\n基准设备 (B): {ref_device}")

train_v = [70, 90, 110]
test_v = [80, 100]

configs = [
    ("无校正 + 窗口8", dict(window_size=8, device_mapping=False, humidity_correction=False)),
    ("映射校准 + 窗口8", dict(window_size=8, device_mapping=True, humidity_correction=False, ref_device_id=ref_device)),
    ("映射+湿度 + 窗口8", dict(window_size=8, device_mapping=True, humidity_correction=True, ref_device_id=ref_device)),
]

for label, kw in configs:
    print(f"\n\n{'=' * 65}")
    print(f"  {label}")
    print("=" * 65)

    r = run(project_dir, train_v, test_v,
            max_samples_per_voltage=0, device_id=None, **kw)

    if "error" in r:
        print(f"  错误: {r['error']}")
        continue

    m = r["metrics"]
    print(f"  样本数: {m['train_count']}(训练) / {m['test_count']}(测试)")
    print(f"  训练集: MAE={m['train']['mae']:.3f}V  R²={m['train']['r2']:.4f}")
    print(f"  测试集: MAE={m['test']['mae']:.3f}V  R²={m['test']['r2']:.4f}")

    print(f"\n  ── 各训练电压 ──")
    train_by_v = {}
    for tr in r["train_results"]:
        train_by_v.setdefault(tr["actual"], []).append(tr["pred"])
    for v in sorted(train_by_v):
        preds = train_by_v[v]
        print(f"    V={v:+.0f}  预测均值={np.mean(preds):.1f}  范围={min(preds):.1f}~{max(preds):.1f}")

    print(f"\n  ── 各测试电压 ──")
    test_by_v = {}
    for tr in r["test_results"]:
        test_by_v.setdefault(tr["actual"], []).append(tr["pred"])
    for v in sorted(test_by_v):
        preds = test_by_v[v]
        mae_v = np.mean(np.abs(np.array(preds) - v))
        print(f"    V={v:+.0f}  预测均值={np.mean(preds):.1f}  范围={min(preds):.1f}~{max(preds):.1f}  MAE={mae_v:.3f}")

    print(f"\n  ── 回归系数 ──")
    for name, val in sorted(r.get("coefficients", {}).items()):
        print(f"    {name}: {val:.4f}")
    print(f"    截距: {r.get('intercept', 0):.4f}")

    norm = r.get("norm_params", {})
    if norm:
        print(f"\n  ── harm_a1 归一化参数 ──")
        print(f"    均值={norm['harm_a1']['mean']:.2f}  标准差={norm['harm_a1']['std']:.2f}")
