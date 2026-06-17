"""
分析相位与电压正负号的关系。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import defaultdict
from scipy.fftpack import fft
from scripts.utils.loader import load_jsonl

records = load_jsonl("data/exported_data.jsonl", extract_features=True)

# 按电压分组收集 A1 相位
groups = defaultdict(list)
for rec in records:
    v = rec["ACTUAL_VOLTAGE"]
    if v is None: continue
    wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
    wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
    ac = wave - np.mean(wave)
    fft_r = fft(ac)
    p1 = np.angle(fft_r[1])  # A1 相位
    p2 = np.angle(fft_r[2]) if len(ac) > 2 else 0  # A2 相位
    groups[round(v / 10) * 10].append((v, p1, p2))

print(f"{'电压':>8} | {'数量':>6} | {'A1平均相位':>12} | {'A1标准差':>10} | {'A1度':>8} | {'A2平均相位':>12} | {'A2度':>8}")
print("-" * 80)
for v in sorted(groups):
    vs = np.array([g[0] for g in groups[v]])
    p1s = np.array([g[1] for g in groups[v]])
    p2s = np.array([g[2] for g in groups[v]])
    print(f"{v:>+7}V | {len(vs):>6} | {np.mean(p1s):>+12.4f} | {np.std(p1s):>10.4f} | {np.degrees(np.mean(p1s)):>+7.1f}° | {np.mean(p2s):>+12.4f} | {np.degrees(np.mean(p2s)):>+7.1f}°")

# 用相位判断正负
print(f"\n用 A1 相位判断正负（真实值 vs 预测值）:")
all_pos_p1 = np.concatenate([np.array([g[1] for g in groups[v]]) for v in groups if v > 0])
all_neg_p1 = np.concatenate([np.array([g[1] for g in groups[v]]) for v in groups if v < 0])

print(f"  正电压平均相位: {np.mean(all_pos_p1):+.4f} rad = {np.degrees(np.mean(all_pos_p1)):.1f}°")
print(f"  负电压平均相位: {np.mean(all_neg_p1):+.4f} rad = {np.degrees(np.mean(all_neg_p1)):.1f}°")
print(f"  相位差: {abs(np.mean(all_pos_p1) - np.mean(all_neg_p1)):.4f} rad = {np.degrees(abs(np.mean(all_pos_p1) - np.mean(all_neg_p1))):.1f}°")

# 用相位判断正负（最近邻）
correct = 0
total = 0
for rec in records:
    v = rec["ACTUAL_VOLTAGE"]
    if v is None or v == 0: continue
    wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
    wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
    ac = wave - np.mean(wave)
    fft_r = fft(ac)
    p = np.angle(fft_r[1])
    
    if abs(p - np.mean(all_pos_p1)) < abs(p - np.mean(all_neg_p1)):
        pred_sign = 1
    else:
        pred_sign = -1
    
    if np.sign(v) == pred_sign:
        correct += 1
    total += 1

print(f"\n  用 A1 相位判断正负准确率: {correct}/{total} = {correct/total*100:.2f}%")
