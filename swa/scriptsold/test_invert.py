"""
测试模型到底有没有学会正负号 vs 只是背诵分布。
把波形取反（×-1），看看预测值会不会跟着取反。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scripts.utils.loader import load_jsonl
from scripts.evaluate_by_voltage import load_model, predict

# 加载模型
model_path = "data/exported_data/model_params_random_forest_model.joblib"
model_info = load_model(model_path)

# 加载数据，取 30V 的样本
records = load_jsonl("data/exported_data.jsonl", extract_features=True)

print("=" * 70)
print("波形取反测试 — 看看模型学没学会正负号")
print("=" * 70)

# 取一条 30V 的
for rec in records:
    v = rec["ACTUAL_VOLTAGE"]
    if v == 30:
        break

print(f"\n原始数据: 真值 = {v}V")

# 原始预测
pred_orig = predict(model_info, rec)
print(f"  原始预测: {pred_orig:.2f}V")

# 制作取反的记录
rec_inverted = dict(rec)
wave = np.array([float(x) for x in rec["RTU_REGS_P00_WAVE_DATA"].split(",")], dtype=np.float64)
wave_inverted = -wave  # 取反!
rec_inverted["RTU_REGS_P00_WAVE_DATA"] = ",".join(f"{x:.6f}" for x in wave_inverted)

# 取反后预测
pred_inv = predict(model_info, rec_inverted)
print(f"  取反后预测: {pred_inv:.2f}V")
print(f"  如果学会正负号，应该预测 ≈ {-pred_orig:.2f}V")

# 再测几条不同电压的
print(f"\n{'#'*70}")
print(f"多电压测试")
print(f"{'#'*70}")
print(f"{'真值':>6} | {'原始预测':>8} | {'取反预测':>8} | {'理论取反':>8} | {'正确?':>6}")
print("-" * 50)

test_voltages = [110, 70, 30, -40, -50, -60]
found = {v: None for v in test_voltages}
for rec in records:
    v = rec["ACTUAL_VOLTAGE"]
    if v in found and found[v] is None:
        found[v] = rec

for v in test_voltages:
    rec = found[v]
    if rec is None:
        continue
    pred_orig = predict(model_info, rec)
    
    wave = np.array([float(x) for x in rec["RTU_REGS_P00_WAVE_DATA"].split(",")], dtype=np.float64)
    rec_inv = dict(rec)
    rec_inv["RTU_REGS_P00_WAVE_DATA"] = ",".join(f"{x:.6f}" for x in (-wave))
    pred_inv = predict(model_info, rec_inv)
    
    expected = -pred_orig
    good = "✅" if abs(pred_inv - expected) < 5 else "❌"
    print(f"{v:>+6}V | {pred_orig:>+8.2f} | {pred_inv:>+8.2f} | {expected:>+8.2f} | {good}")
