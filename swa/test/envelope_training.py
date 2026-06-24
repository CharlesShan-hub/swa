"""
对每个 (电压, 温度, 湿度) 桶内的大量数据 → 用包络中值合成 1 个干净点。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import json
from collections import defaultdict
from lib.data import load
from scipy.interpolate import UnivariateSpline

records = load("smooth_all_w32")
print(f"输入: {len(records)} 条")

# 按 (电压绝对值, round(温度), round(湿度)) 分组
buckets = defaultdict(list)
for rec in records:
    v = rec.get("ACTUAL_VOLTAGE")
    if not isinstance(v, (int, float)):
        continue
    a1 = rec.get("_a1")
    temp = rec.get("_temp")
    humid = rec.get("_humid")
    if a1 is None or temp is None or humid is None:
        continue
    key = (round(v), round(float(temp)), round(float(humid)))
    buckets[key].append(a1)

print(f"分桶数: {len(buckets)}")

# 对每个桶做包络中值
out_points = []
for (v, t, h), vals in buckets.items():
    if len(vals) < 5:
        continue  # 数据太少不用
    vals = np.sort(vals)
    n = len(vals)
    # 去掉上下 10%
    lower = int(n * 0.1)
    upper = int(n * 0.9)
    mid = np.mean(vals[lower:upper]) if upper > lower else np.mean(vals)
    out_points.append({
        "voltage": v,
        "a1": round(float(mid), 6),
        "temp": t,
        "humid": h,
    })

print(f"输出: {len(out_points)} 个干净点")

# 查看每个电压有多少个温湿度条件
voltage_counts = defaultdict(int)
for p in out_points:
    voltage_counts[p["voltage"]] += 1

print(f"\n每个电压的温湿度组合数:")
for v in sorted(voltage_counts):
    print(f"  {v:>4}V: {voltage_counts[v]} 个温湿度条件")

# 保存
out_path = os.path.join(os.path.dirname(__file__), "..", "data", "envelope_points.json")
with open(out_path, "w") as f:
    json.dump(out_points, f, indent=2)
print(f"\n已保存: {out_path}")

# 看看拟合效果
print(f"\n{'='*60}")
print(f"  用这些点做物理模型拟合")
print(f"{'='*60}")

from lib.traditional.quadratic_zero import train, predict

all_v = sorted(set(p["voltage"] for p in out_points))
print(f"可用电压: {all_v}")

# 留出验证：测 -55V 和 -36V
test_v = [-55, -36]
train_v = [v for v in all_v if v not in test_v]

train_data = [p for p in out_points if p["voltage"] in train_v]
test_data = [p for p in out_points if p["voltage"] in test_v]

X_train = np.array([[p["a1"], p["temp"], p["humid"], 1.0 if p["voltage"] >= 0 else -1.0] for p in train_data])
y_train = np.array([abs(p["voltage"]) for p in train_data])
X_test = np.array([[p["a1"], p["temp"], p["humid"], 1.0 if p["voltage"] >= 0 else -1.0] for p in test_data])
y_test = np.array([abs(p["voltage"]) for p in test_data])

model = train(X_train, y_train)
y_pred_train = predict(model, X_train)
y_pred_test = predict(model, X_test)

mae_train = np.mean(np.abs(y_pred_train - y_train))
mae_test = np.mean(np.abs(y_pred_test - y_test))

print(f"\n训练点: {len(train_data)} 个")
print(f"测试点: {len(test_data)} 个")
print(f"训练 MAE: {mae_train:.2f}V")
print(f"测试 MAE: {mae_test:.2f}V")
print(f"\n拟合公式: V = A1 × (bias + A1 + temp + humid + temp² + humid² + sign)")
print(f"系数: {[f'{c:.4f}' for c in model]}")
