"""
验证波形平均对 A1 稳定性的改善效果。

取 30°C/40%RH/40V 的 2197 条，分组平均波形后再算 A1。

预期: 平均 N 条 → 噪声降为 1/√N
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scripts.utils.loader import load_jsonl

FS = 15873

def fit_a1(wave, rpm=None):
    n = len(wave)
    if rpm is not None and rpm > 0:
        f0 = rpm / 60.0
    else:
        fft_mag = np.abs(np.fft.fft(wave - np.mean(wave)))[:n // 2]
        f0 = (1 + np.argmax(fft_mag[1:])) * FS / n
    t = np.arange(n) / FS
    wt = 2 * np.pi * f0 * t
    A = np.column_stack([np.ones(n), np.cos(wt), np.sin(wt)])
    coeffs, *_ = np.linalg.lstsq(A, wave, rcond=None)
    return np.sqrt(coeffs[1]**2 + coeffs[2]**2)

records = load_jsonl("data/exported_data.jsonl", extract_features=False, max_per_voltage=5000)

# 收集 30°C/40%RH/40V 的所有波形
waves = []
rpm = None
for rec in records:
    v = rec.get("ACTUAL_VOLTAGE")
    t = rec.get("RTU_REGS_P00_ENV_TEMP")
    h = rec.get("RTU_REGS_P00_ENV_HUMIDITY")
    if v is None or t is None or h is None:
        continue
    if abs(float(v)) != 40: continue
    if not (28 <= float(t) <= 32): continue
    if not (38 <= float(h) <= 42): continue

    wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
    try:
        wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]
    except:
        continue
    rpm_val = rec.get("RTU_REGS_P00_ROTOR_RPM")
    rpm = float(rpm_val) if rpm_val is not None else 0
    waves.append(wave)

waves = np.array(waves)
n_total = len(waves)
print(f"符合条件的波形: {n_total} 条")
print(f"每个波形 512 点")

# 1. 单条波形的 A1 统计
a1_single = np.array([fit_a1(w, rpm) for w in waves])
print(f"\n=== 单条波形 ===")
print(f"  A1 mean={a1_single.mean():.4f}, std={a1_single.std():.4f}, CV={a1_single.std()/a1_single.mean()*100:.1f}%")
print(f"  等效电压噪声: ±{a1_single.std()/a1_single.mean()*40:.1f}V")

# 2. 平均 2、4、8、16、32 条后的效果
for n_avg in [2, 4, 8, 16, 32, 64, 128]:
    n_groups = n_total // n_avg
    a1_avg = np.zeros(n_groups)
    for g in range(n_groups):
        avg_wave = waves[g*n_avg:(g+1)*n_avg].mean(axis=0)
        a1_avg[g] = fit_a1(avg_wave, rpm)
    print(f"\n=== 每 {n_avg} 条平均 (n_groups={n_groups}) ===")
    print(f"  A1 mean={a1_avg.mean():.4f}, std={a1_avg.std():.4f}, CV={a1_avg.std()/a1_avg.mean()*100:.1f}%")
    print(f"  等效电压噪声: ±{a1_avg.std()/a1_avg.mean()*40:.1f}V")
    print(f"  理论改善 √{n_avg}={np.sqrt(n_avg):.1f}x, 实际 std: {a1_single.std():.4f}→{a1_avg.std():.4f}")
