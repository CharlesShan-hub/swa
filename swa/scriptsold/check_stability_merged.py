"""
检查合并后数据的稳定性：相同温湿度电压下 A1/A3/A5 的变异系数。

对比: 原始 vs 4合并 vs 8合并
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import defaultdict
from scripts.utils.loader import load_jsonl

FS = 15873

def fit_a1a3a5(wave, rpm=None):
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

for label, path in [("原始 45000", "data/exported_data.jsonl"),
                    ("4合并 9665", "data/exported_data_4merge.jsonl"),
                    ("8合并 4277", "data/exported_data_8merge.jsonl")]:
    records = load_jsonl(path, extract_features=False, max_per_voltage=0)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # 按 (电压桶, 温度桶, 湿度桶) 分组
    groups = defaultdict(list)
    for rec in records:
        v = rec.get("ACTUAL_VOLTAGE")
        t = rec.get("RTU_REGS_P00_ENV_TEMP")
        h = rec.get("RTU_REGS_P00_ENV_HUMIDITY")
        if v is None or t is None or h is None: continue
        vb = round(abs(float(v))/10)*10
        tb = round(float(t)/5)*5
        hb = round(float(h)/5)*5
        wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
        try:
            wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]
        except:
            continue
        rpm_val = rec.get("RTU_REGS_P00_ROTOR_RPM")
        rpm = float(rpm_val) if rpm_val is not None else 0
        a1, a3, a5 = fit_a1a3a5(wave, rpm)
        groups[(vb, tb, hb)].append((a1, a3, a5))

    # 找出样本最多的组
    max_group = max(groups, key=lambda k: len(groups[k]))
    vals = np.array(groups[max_group])
    a1s = vals[:, 0]
    a3s = vals[:, 1]
    a5s = vals[:, 2]
    n = len(a1s)
    vb, tb, hb = max_group

    cv_a1 = a1s.std() / a1s.mean() * 100 if a1s.mean() > 0 else 0
    cv_a3 = a3s.std() / a3s.mean() * 100 if a3s.mean() > 0 else 0
    cv_a5 = a5s.std() / a5s.mean() * 100 if a5s.mean() > 0 else 0

    print(f"  最大组: {vb}V, {tb}°C, {hb}% (n={n})")
    print(f"  A1: mean={a1s.mean():.4f}, std={a1s.std():.4f}, CV={cv_a1:.1f}%")
    print(f"  A3: mean={a3s.mean():.4f}, std={a3s.std():.4f}, CV={cv_a3:.1f}%")
    print(f"  A5: mean={a5s.mean():.4f}, std={a5s.std():.4f}, CV={cv_a5:.1f}%")

    # 显示所有 >=50 条的组
    print(f"\n  所有大组 (n>=50):")
    print(f"  {'电压':>5} | {'温度':>4} | {'湿度':>4} | {'n':>5} | {'A1均值':>8} | {'A1 CV':>7} | {'A3 CV':>7} | {'A5 CV':>7}")
    for (vb, tb, hb) in sorted(groups):
        vals = np.array(groups[(vb, tb, hb)])
        n = len(vals)
        if n < 50: continue
        a1s = vals[:, 0]
        a3s = vals[:, 1]
        a5s = vals[:, 2]
        cv1 = a1s.std()/a1s.mean()*100 if a1s.mean()>0 else 0
        cv3 = a3s.std()/a3s.mean()*100 if a3s.mean()>0 else 0
        cv5 = a5s.std()/a5s.mean()*100 if a5s.mean()>0 else 0
        print(f"  {vb:>4}V | {tb:>3}C | {hb:>3}% | {n:>5} | {a1s.mean():>8.4f} | {cv1:>6.1f}% | {cv3:>6.1f}% | {cv5:>6.1f}%")
