"""
探索后续优化方向（基线：窗口8 + 映射校准）

测试以下改进：
1. 排除设备 253D（数据最异常的）
2. 切换基准设备（A1 vs B）
3. 仅用单个设备训练
4. 增加波形质量特征（std）
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import sqlite3

from swa.data.manager import PROJECTS_DIR
from swa.detection.least_squares import run, _load_data, _normalize, _FEATURE_NAMES

project_dir = os.path.join(PROJECTS_DIR, "new")
db_path = os.path.join(project_dir, "data.db")

conn = sqlite3.connect(db_path)
all_devices = [r[0] for r in conn.execute(
    "SELECT device_id FROM records WHERE enabled=1 AND device_id IS NOT NULL GROUP BY device_id ORDER BY COUNT(*) DESC"
).fetchall()]
for d in all_devices:
    cnt = conn.execute("SELECT COUNT(*) FROM records WHERE enabled=1 AND device_id=?", (d,)).fetchone()[0]
    print(f"  设备 {d[-4:]:6s}  n={cnt}")
conn.close()

dev_a1 = all_devices[0]  # 2539
dev_b = all_devices[1]   # 6A39
dev_a2 = all_devices[2]  # 253D

print(f"\n设备 A1 (2539) = {dev_a1}")
print(f"设备 B  (6A39) = {dev_b}")
print(f"设备 A2 (253D) = {dev_a2}")

train_v = [70, 90, 110]
test_v = [80, 100]
WINDOW = 8

print(f"\n{'=' * 70}")
print("基线: 窗口8 + 映射校准 (基准=B)")
print("=" * 70)
r = run(project_dir, train_v, test_v, window_size=WINDOW,
        device_mapping=True, ref_device_id=dev_b,
        humidity_correction=False, max_samples_per_voltage=0)
if "error" not in r:
    for v in sorted(set(tr["actual"] for tr in r["test_results"])):
        preds = [tr["pred"] for tr in r["test_results"] if tr["actual"] == v]
        print(f"  V={v:+.0f}  预测={np.mean(preds):.1f}  MAE={np.mean(np.abs(np.array(preds)-v)):.3f}")
    print(f"  测试MAE={r['metrics']['test']['mae']:.3f}  R²={r['metrics']['test']['r2']:.4f}")


# ── 方案A: 排除设备253D ─────────────────────────────────────
print(f"\n{'=' * 70}")
print("方案A: 排除设备253D (异常数据)")
print("=" * 70)
# 模拟排除：只用一个设备ID过滤，但跑的是全部设备...需要修改run函数
# 简单方法：用 device_id=dev_a1 或 dev_b 过滤，但一次性只能选一个设备
# 实际上要排除253D，需要多个device_id过滤，当前run不支持
# 先放一放


# ── 方案B: 切换基准设备 (A1 做基准) ──────────────────────────
print(f"\n{'=' * 70}")
print("方案B: 切换基准到设备A1 (2539)")
print("=" * 70)
r = run(project_dir, train_v, test_v, window_size=WINDOW,
        device_mapping=True, ref_device_id=dev_a1,
        humidity_correction=False, max_samples_per_voltage=0)
if "error" not in r:
    for v in sorted(set(tr["actual"] for tr in r["test_results"])):
        preds = [tr["pred"] for tr in r["test_results"] if tr["actual"] == v]
        print(f"  V={v:+.0f}  预测={np.mean(preds):.1f}  MAE={np.mean(np.abs(np.array(preds)-v)):.3f}")
    print(f"  测试MAE={r['metrics']['test']['mae']:.3f}  R²={r['metrics']['test']['r2']:.4f}")


# ── 方案C: 单设备训练 ────────────────────────────────────────
for label, did in [("仅设备A1 (2539)", dev_a1), ("仅设备B (6A39)", dev_b), ("仅设备A2 (253D)", dev_a2)]:
    print(f"\n{'=' * 70}")
    print(f"方案C: {label}")
    print("=" * 70)
    r = run(project_dir, train_v, test_v, window_size=WINDOW,
            device_id=did, device_mapping=False,
            humidity_correction=False, max_samples_per_voltage=0)
    if "error" not in r:
        for v in sorted(set(tr["actual"] for tr in r["test_results"])):
            preds = [tr["pred"] for tr in r["test_results"] if tr["actual"] == v]
            mae_v = np.mean(np.abs(np.array(preds)-v))
            print(f"  V={v:+.0f}  预测={np.mean(preds):.1f}  MAE={mae_v:.3f}")
        print(f"  测试MAE={r['metrics']['test']['mae']:.3f}  R²={r['metrics']['test']['r2']:.4f}")


# ── 方案D: 查看训练集内部各设备的预测偏差 ──────────────────────
print(f"\n\n{'=' * 70}")
print("方案D: 训练集各设备预测偏差分析 (基线配置)")
print("=" * 70)
r = run(project_dir, train_v, test_v, window_size=WINDOW,
        device_mapping=True, ref_device_id=dev_b,
        humidity_correction=False, max_samples_per_voltage=0)
if "error" not in r:
    for v in sorted(set(tr["actual"] for tr in r["train_results"])):
        sub = [tr for tr in r["train_results"] if tr["actual"] == v]
        print(f"\n  V={v:+.0f} (n={len(sub)}):")
        # 按设备分组
        by_dev = {}
        for tr in sub:
            dev_short = (tr.get("device_id", "?") or "?")[-4:]
            by_dev.setdefault(dev_short, []).append(tr["pred"])
        for dev_name in sorted(by_dev.keys()):
            preds = by_dev[dev_name]
            print(f"    设备 {dev_name}  预测={np.mean(preds):.1f}  范围={min(preds):.1f}~{max(preds):.1f}  n={len(preds)}")
