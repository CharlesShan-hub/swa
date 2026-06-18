"""
每个电压下，A1 随温度/湿度的变化。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import defaultdict
from scripts.utils.loader import load_jsonl
from scipy.stats import pearsonr

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
    A = np.column_stack([np.ones(n), np.cos(wt), np.sin(wt)])
    coeffs, *_ = np.linalg.lstsq(A, wave, rcond=None)
    return np.sqrt(coeffs[1]**2 + coeffs[2]**2)

records = load_jsonl("data/exported_data.jsonl", extract_features=False, max_per_voltage=5000)

# 收集各电压的 {v: {"a1":[], "t":[], "h":[]}}
data = defaultdict(lambda: {"a1": [], "t": [], "h": []})

for rec in records:
    v = rec.get("ACTUAL_VOLTAGE")
    t = rec.get("RTU_REGS_P00_ENV_TEMP")
    h = rec.get("RTU_REGS_P00_ENV_HUMIDITY")
    if v is None or t is None or h is None:
        continue
    vb = round(abs(float(v)) / 10) * 10
    if vb == 0: continue

    wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
    try:
        wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]
    except:
        continue
    rpm_val = rec.get("RTU_REGS_P00_ROTOR_RPM")
    rpm = float(rpm_val) if rpm_val is not None else 0
    a1 = fit_waveform(wave, rpm)

    data[vb]["a1"].append(a1)
    data[vb]["t"].append(float(t))
    data[vb]["h"].append(float(h))

print(f"  {'电压':>6} | {'样本':>6} | {'A1均值':>8} | {'A1 std':>8} | {'A1变异':>7} | {'r(T,A1)':>8} | {'r(RH,A1)':>8}")
print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")

for vb in sorted(data):
    d = data[vb]
    a1 = np.array(d["a1"])
    t = np.array(d["t"])
    h = np.array(d["h"])
    n = len(a1)
    if n < 10: continue

    r_t, _ = pearsonr(t, a1)
    r_h, _ = pearsonr(h, a1)
    cv = a1.std() / a1.mean() * 100  # 变异系数 %

    print(f"  {vb:>5}V | {n:>6} | {a1.mean():>8.4f} | {a1.std():>8.4f} | {cv:>6.1f}% | {r_t:>+8.4f} | {r_h:>+8.4f}")

# 如果 40V 样本多，单独按温度分桶显示 A1
print(f"\n=== 40V: A1 vs 温度×湿度 详情 ===")
d40 = data[40]
a1 = np.array(d40["a1"])
t = np.array(d40["t"])
h = np.array(d40["h"])

# 温度×湿度 交叉表
tb_set = sorted(set(round(x / 5) * 5 for x in t))
hb_set = sorted(set(round(x / 5) * 5 for x in h))
print(f"  {'温度':>5} |", end="")
for hb in hb_set:
    print(f" {hb:>3}%({hb:>4}) |", end="")
print()
for tb in tb_set:
    print(f"  {tb:>4}C |", end="")
    for hb in hb_set:
        mask = (t >= tb-2) & (t < tb+3) & (h >= hb-2) & (h < hb+3)
        n = mask.sum()
        if n < 5:
            print(f" {'':>10} |", end="")
        else:
            print(f" {a1[mask].mean():>5.3f}({n:>3}) |", end="")
    print()
