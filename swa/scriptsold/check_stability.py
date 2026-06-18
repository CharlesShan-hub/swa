"""
检查相同温湿度下 A1/A3/A5 的稳定性（以 40V 为例）。
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

# 只挑 40V
data = []  # [(t, h, a1, a3, a5)]
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
    data.append((round(float(t)/5)*5, round(float(h)/5)*5, a1, a3, a5))

# 按 (温度桶, 湿度桶) 分组统计
from collections import defaultdict
buckets = defaultdict(list)
for tb, hb, a1, a3, a5 in data:
    buckets[(tb, hb)].append((a1, a3, a5))

print(f"{'T':>5} × {'RH':>4} | {'n':>6} | {'A1均值':>8} | {'A1 std':>8} | {'A1 CV':>7} | {'A3均值':>8} | {'A5均值':>8}")
print(f"{'-'*5}-+-{'-'*4}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")

for (tb, hb) in sorted(buckets):
    vals = np.array(buckets[(tb, hb)])
    a1s = vals[:, 0]
    a3s = vals[:, 1]
    a5s = vals[:, 2]
    n = len(a1s)
    cv = a1s.std() / a1s.mean() * 100
    print(f"{tb:>4}C | {hb:>3}% | {n:>6} | {a1s.mean():>8.4f} | {a1s.std():>8.4f} | {cv:>6.1f}% | {a3s.mean():>8.4f} | {a5s.mean():>8.4f}")

# 最大桶单独输出详细统计
max_key = max(buckets, key=lambda k: len(buckets[k]))
vals = np.array(buckets[max_key])
a1s = vals[:, 0]
print(f"\n=== 最大桶 T={max_key[0]}°C, RH={max_key[1]}% (n={len(a1s)}) 的 A1 分布 ===")
print(f"  mean={a1s.mean():.4f}, std={a1s.std():.4f}, CV={a1s.std()/a1s.mean()*100:.1f}%")
print(f"  min={a1s.min():.4f}, max={a1s.max():.4f}")
# 如果样本多，看看 A1 的分布形状
if len(a1s) > 100:
    p5 = np.percentile(a1s, 5)
    p25 = np.percentile(a1s, 25)
    p50 = np.percentile(a1s, 50)
    p75 = np.percentile(a1s, 75)
    p95 = np.percentile(a1s, 95)
    print(f"  p5={p5:.4f}, p25={p25:.4f}, p50={p50:.4f}, p75={p75:.4f}, p95={p95:.4f}")
    print(f"  90% 区间: {p5:.4f} ~ {p95:.4f} (跨度 {p95-p5:.4f})")
