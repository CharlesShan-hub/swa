"""
检查同一电压下物理参数 A1/A3/A5 随温湿度的变化。

选 -40V（样本最多），看 A1 跟温度、湿度的关系。

用法:
    uv run python scripts/check_phys_vs_env.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scripts.utils.loader import load_jsonl

FS = 15873

def fit_waveform(wave, rpm=None):
    n = len(wave)
    if rpm is not None and rpm > 0:
        f0 = rpm / 60.0
    else:
        fft_mag = np.abs(np.fft.fft(wave - np.mean(wave)))[:n // 2]
        f0 = (1 + np.argmax(fft_mag[1:])) * FS / n
    t = np.arange(n) / FS
    wt = 2 * np.pi * f0 * t
    A = np.column_stack([np.ones(n), np.cos(wt), np.sin(wt),
                         np.cos(3*wt), np.sin(3*wt),
                         np.cos(5*wt), np.sin(5*wt)])
    coeffs, *_ = np.linalg.lstsq(A, wave, rcond=None)
    return (np.sqrt(coeffs[1]**2 + coeffs[2]**2),
            np.sqrt(coeffs[3]**2 + coeffs[4]**2),
            np.sqrt(coeffs[5]**2 + coeffs[6]**2))

records = load_jsonl("data/exported_data.jsonl", extract_features=False, max_per_voltage=5000)

# 只挑 40V（绝对值）
a1s, a3s, a5s, temps, hums = [], [], [], [], []
for rec in records:
    v = rec.get("ACTUAL_VOLTAGE")
    if v is None or abs(float(v)) != 40:
        continue
    t = rec.get("RTU_REGS_P00_ENV_TEMP")
    h = rec.get("RTU_REGS_P00_ENV_HUMIDITY")
    if t is None or h is None:
        continue

    wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
    try:
        wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]
    except:
        continue
    rpm_val = rec.get("RTU_REGS_P00_ROTOR_RPM")
    rpm = float(rpm_val) if rpm_val is not None else 0
    a1, a3, a5 = fit_waveform(wave, rpm)

    a1s.append(a1)
    a3s.append(a3)
    a5s.append(a5)
    temps.append(float(t))
    hums.append(float(h))

a1s = np.array(a1s)
a3s = np.array(a3s)
a5s = np.array(a5s)
temps = np.array(temps)
hums = np.array(hums)

n = len(a1s)
print(f"40V 样本数: {n}")
print(f"温度范围: {temps.min():.1f} ~ {temps.max():.1f} °C")
print(f"湿度范围: {hums.min():.1f} ~ {hums.max():.1f} %")
print(f"\nA1 统计: mean={a1s.mean():.4f}, std={a1s.std():.4f}, min={a1s.min():.4f}, max={a1s.max():.4f}")
print(f"  std/mean = {a1s.std()/a1s.mean()*100:.2f}%")
print(f"A3 统计: mean={a3s.mean():.4f}, std={a3s.std():.4f}")
print(f"A5 统计: mean={a5s.mean():.4f}, std={a5s.std():.4f}")

# 按温度分桶看 A1 均值
print(f"\n=== A1 vs 温度 (40V 固定) ===")
print(f"  {'温度':>6} | {'样本':>6} | {'A1均值':>8} | {'A1 std':>8}")
temp_buckets = sorted(set(round(t / 5) * 5 for t in temps))
for tb in temp_buckets:
    mask = (temps >= tb - 2) & (temps < tb + 3)
    n = mask.sum()
    if n < 5:
        continue
    print(f"  {tb:>5}C | {n:>6} | {a1s[mask].mean():>8.4f} | {a1s[mask].std():>8.4f}")

# 按湿度分桶看 A1 均值
print(f"\n=== A1 vs 湿度 (40V 固定) ===")
print(f"  {'湿度':>6} | {'样本':>6} | {'A1均值':>8} | {'A1 std':>8}")
hum_buckets = sorted(set(round(h / 5) * 5 for h in hums))
for hb in hum_buckets:
    mask = (hums >= hb - 2) & (hums < hb + 3)
    n = mask.sum()
    if n < 5:
        continue
    print(f"  {hb:>4}% | {n:>6} | {a1s[mask].mean():>8.4f} | {a1s[mask].std():>8.4f}")

# 相关性
from scipy.stats import pearsonr
r_t, p_t = pearsonr(temps, a1s)
r_h, p_h = pearsonr(hums, a1s)
print(f"\n=== A1 与温湿度的相关性 ===")
print(f"  A1 vs 温度:   r={r_t:.4f}, p={p_t:.6f} {'**相关**' if p_t < 0.01 else ''}")
print(f"  A1 vs 湿度:   r={r_h:.4f}, p={p_h:.6f} {'**相关**' if p_h < 0.01 else ''}")

# 看看 40V 中是否 RPM 也有变化
rpm_vals = []
for rec in records:
    v = rec.get("ACTUAL_VOLTAGE")
    if v is None or abs(float(v)) != 40:
        continue
    rpm_v = rec.get("RTU_REGS_P00_ROTOR_RPM")
    if rpm_v is not None:
        rpm_vals.append(float(rpm_v))
rpm_vals = np.array(rpm_vals)
print(f"\n=== RPM 统计 (40V) ===")
print(f"  mean={rpm_vals.mean():.1f}, std={rpm_vals.std():.2f}, min={rpm_vals.min():.1f}, max={rpm_vals.max():.1f}")
print(f"  std/mean = {rpm_vals.std()/rpm_vals.mean()*100:.2f}%")
