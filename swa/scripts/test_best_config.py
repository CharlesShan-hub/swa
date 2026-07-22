"""
测试当前最佳配置：设备映射 + 滑动窗口 + 5特征 (score, A1, temp, humidity, rpm)
"""
import sys, os, time
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

print("运行检测...")
t0 = time.time()
r = run(
    project_dir, train_v, test_v,
    window_size=8,
    device_mapping=True, ref_device_id=dev_b,
    noise_correction=False,
)
t1 = time.time()
print(f"耗时: {t1-t0:.1f}s")

if "error" in r:
    print(f"错误: {r['error']}")
    sys.exit(1)

m = r["metrics"]
print(f"\n{'='*55}")
print("最佳配置结果")
print("="*55)
print(f"  特征: score, harm_a1, temperature, humidity, rpm")
print(f"  滑动窗口: {r['window_size']}")
print(f"  设备映射: {r.get('device_mapping', False)} (基准={r.get('ref_device_id','?')[-4:]})")
print(f"  样本: {m['train_count']}(训练) / {m['test_count']}(测试)")
print(f"  训练 MAE={m['train']['mae']:.3f}  R²={m['train']['r2']:.4f}")
print(f"  测试 MAE={m['test']['mae']:.3f}  R²={m['test']['r2']:.4f}")

print(f"\n  各电压预测:")
v_pred_mean = r.get("voltage_pred_mean", {})
for v_label, v_mae in sorted(r.get("voltage_mae", {}).items(), key=lambda x: float(x[0].rstrip("V"))):
    mean_str = f"  预测均值={v_pred_mean.get(v_label, 0):.2f}V" if v_label in v_pred_mean else ""
    print(f"    {v_label:>6s}: MAE={v_mae:.3f}V{mean_str}")

print(f"\n  回归系数:")
print(f"    截距: {r['intercept']:.4f}")
for name, val in sorted(r["coefficients"].items()):
    print(f"    {name}: {val:.4f}")
