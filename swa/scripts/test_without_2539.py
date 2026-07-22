"""
测试禁用设备 2539 后的效果：全设备跑一次，按 device_id 拆分看结果。
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

r = run(
    project_dir, train_v, test_v,
    window_size=8,
    device_mapping=True, ref_device_id=dev_b,
    noise_correction=False,
)

if "error" in r:
    print(f"错误: {r['error']}")
    sys.exit(1)

m = r["metrics"]
print(f"\n=== 全设备 (3个) ===")
print(f"样本: {m['train_count']}(训练) / {m['test_count']}(测试)")
print(f"测试 MAE={m['test']['mae']:.3f}  R²={m['test']['r2']:.4f}")

# 按设备拆分
test_by_dev = {}
for tr in r["test_results"]:
    dev = tr.get("device_id", "未知")
    test_by_dev.setdefault(dev, []).append(tr)

print(f"\n{'='*60}")
print("按设备拆分测试结果")
print("="*60)

all_no_2539 = []
for dev in devices:
    short = dev[-4:]
    trs = test_by_dev.get(dev, [])
    if not trs:
        continue
    actuals = np.array([t["actual"] for t in trs])
    preds = np.array([t["pred"] for t in trs])
    mae = float(np.mean(np.abs(actuals - preds)))
    r2 = float(1 - np.sum((actuals - preds)**2) / max(np.sum((actuals - np.mean(actuals))**2), 1e-12))
    print(f"\n  设备 {short} ({len(trs)}条):")
    print(f"    MAE={mae:.4f}  R²={r2:.4f}")
    for v in sorted(set(actuals)):
        mask = actuals == v
        print(f"    V={v:+.0f}  预测={np.mean(preds[mask]):.1f}  MAE={np.mean(np.abs(preds[mask]-v)):.3f}")
    if "2539" not in dev:
        all_no_2539 += trs

# 排除 2539 后的合并
print(f"\n{'='*60}")
print("排除 2539 后 (253D + 6A39)")
print("="*60)
if all_no_2539:
    actuals = np.array([t["actual"] for t in all_no_2539])
    preds = np.array([t["pred"] for t in all_no_2539])
    mae = float(np.mean(np.abs(actuals - preds)))
    r2 = float(1 - np.sum((actuals - preds)**2) / max(np.sum((actuals - np.mean(actuals))**2), 1e-12))
    print(f"  总样本: {len(all_no_2539)}")
    print(f"  测试 MAE={mae:.4f}  R²={r2:.4f}")
    for v in sorted(set(actuals)):
        mask = actuals == v
        if np.any(mask):
            print(f"    V={v:+.0f}  预测={np.mean(preds[mask]):.1f}  MAE={np.mean(np.abs(preds[mask]-v)):.3f}")

# 温度对 2539 的影响：固定电压看 A1 vs 温度
print(f"\n{'='*60}")
print("设备 2539 的 A1 vs 温度 (固定电压看漂移)")
print("="*60)
import sqlite3
conn = sqlite3.connect(db_path)
# 取 2539 在 100V 的数据（数据量最多）
dev_2539 = [d for d in devices if "2539" in d][0]
cur = conn.cursor()
cur.execute("""
    SELECT temperature, harm_a1, actual_voltage
    FROM records WHERE enabled=1 AND device_id=? AND actual_voltage=100
    ORDER BY temperature
""", (dev_2539,))
rows = cur.fetchall()
conn.close()
if rows:
    temps = [r[0] for r in rows if r[0] is not None]
    a1s = [r[1] for r in rows if r[1] is not None]
    if len(temps) > 5:
        slope = np.polyfit(temps, a1s, 1)[0]
        print(f"  100V 下 A1 vs 温度斜率: {slope:.3f} (每°C A1 变化)")
        temps_253D = rows  # for comparison later
