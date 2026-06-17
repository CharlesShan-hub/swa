"""
验证相位解析：看看正电压和负电压的 A1 相位差是不是 180°
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.fftpack import fft
from scripts.utils.loader import load_jsonl

records = load_jsonl("data/exported_data.jsonl", extract_features=False)

pos_phases = []
neg_phases = []

for rec in records[:200]:
    v = rec.get("ACTUAL_VOLTAGE")
    if v is None: continue

    wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
    wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]

    ac = wave - np.mean(wave)
    n = len(ac)
    fft_result = fft(ac)

    # A1 的相位
    a1_complex = fft_result[1]
    a1_phase = np.angle(a1_complex)  # -π ~ π

    if v > 0:
        pos_phases.append((v, a1_phase))
    else:
        neg_phases.append((v, a1_phase))

print(f"正电压样本: {len(pos_phases)}")
print(f"负电压样本: {len(neg_phases)}")

if pos_phases:
    avg_pos = np.mean([p[1] for p in pos_phases])
    print(f"\n正电压 A1 平均相位: {avg_pos:.4f} rad ({np.degrees(avg_pos):.1f}°)")
    print(f"  前5个样本:")
    for v, p in pos_phases[:5]:
        print(f"    {v}V  phase={p:.4f} rad ({np.degrees(p):.1f}°)")

if neg_phases:
    avg_neg = np.mean([p[1] for p in neg_phases])
    print(f"\n负电压 A1 平均相位: {avg_neg:.4f} rad ({np.degrees(avg_neg):.1f}°)")
    print(f"  前5个样本:")
    for v, p in neg_phases[:5]:
        print(f"    {v}V  phase={p:.4f} rad ({np.degrees(p):.1f}°)")

if pos_phases and neg_phases:
    diff = abs(avg_pos - avg_neg)
    print(f"\n正负相位差: {diff:.4f} rad ({np.degrees(diff):.1f}°)")
    print(f"理论 180°: {np.pi:.4f} rad (180.0°)")

    # 用相位直接判断正负
    correct = 0
    for rec in records:
        v = rec.get("ACTUAL_VOLTAGE")
        if v is None: continue
        if v == 0: continue

        wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
        wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]
        ac = wave - np.mean(wave)
        fft_result = fft(ac)
        phase = np.angle(fft_result[1])

        # 判断：相位靠近 avg_pos 就是正，靠近 avg_neg 就是负
        if abs(phase - avg_pos) < abs(phase - avg_neg):
            pred_sign = 1
        else:
            pred_sign = -1

        actual_sign = 1 if v > 0 else -1
        if pred_sign == actual_sign:
            correct += 1

    print(f"\n用相位判断正负的准确率: {correct}/{len(records)} = {correct/len(records)*100:.2f}%")
