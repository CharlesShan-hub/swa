import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from scipy.stats import kurtosis, skew
from scripts.utils.loader import load_jsonl

records = load_jsonl("data/exported_data.jsonl")
print(f"Loaded {len(records)} records")

feats_all = []
for i, r in enumerate(records):
    wave_str = r.get("RTU_REGS_P00_WAVE_DATA", "")
    wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]
    ac = wave - np.mean(wave)
    rms = np.sqrt(np.mean(np.square(ac)))
    vpp = np.max(ac) - np.min(ac)
    crest = np.max(np.abs(ac)) / (rms + 1e-9)
    kurt_val = kurtosis(ac, fisher=False)
    skew_val = skew(ac)
    a1 = np.abs(np.fft.fft(ac))[1] * 2 / len(ac)
    feats_all.append([a1, vpp, rms, crest, kurt_val, skew_val])

    if (i + 1) % 5000 == 0:
        print(f"  processed {i + 1}/{len(records)}")

feats = np.array(feats_all)
corr = np.corrcoef(feats.T)

print("\n特征相关性分析（全部 {} 条数据）".format(len(records)))
print("=" * 60)
header = "{:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
    "", "A1", "Vpp", "RMS", "Crest", "Kurt", "Skew")
print(header)
names = ["A1", "Vpp", "RMS", "Crest", "Kurt", "Skew"]
for i, name in enumerate(names):
    row = "{:>6}:".format(name)
    for j in range(6):
        row += " {:>8.3f}".format(corr[i][j])
    print(row)

print()
print("Crest 值域: [{:.4f}, {:.4f}]".format(np.min(feats[:,3]), np.max(feats[:,3])))
print("Kurtosis 值域: [{:.4f}, {:.4f}]".format(np.min(feats[:,4]), np.max(feats[:,4])))
print("Skewness 值域: [{:.4f}, {:.4f}]".format(np.min(feats[:,5]), np.max(feats[:,5])))

print()
print("Crest 标准差: {:.6f} (理想正弦波=1.414)".format(np.std(feats[:,3])))
print("Kurtosis 标准差: {:.6f} (正态分布=3)".format(np.std(feats[:,4])))

print()
print("=" * 60)
print("指标对照说明")
print("=" * 60)
print("{:<8} {:<10} {:<40}".format("指标", "全称", "大白话"))
print("-" * 60)
print("{:<8} {:<10} {:<40}".format("A1", "基波幅值", "传感器主频率信号强度，电压越高A1越大"))
print("{:<8} {:<10} {:<40}".format("Vpp", "峰峰值", "波形最高点-最低点的差值，看波形上蹿下跳多猛"))
print("{:<8} {:<10} {:<40}".format("RMS", "有效值", "波形去掉直流后的平均能量，反映持续能量"))
print("{:<8} {:<10} {:<40}".format("Crest", "波峰因子", "峰值÷有效值，标准正弦波=1.414，>1.414说明有尖刺畸变"))
print("{:<8} {:<10} {:<40}".format("Kurt", "峭度", "波形尖锐程度，正态分布=3，>3有毛刺或异常尖峰"))
print("{:<8} {:<10} {:<40}".format("Skew", "偏度", "波形上下不对称程度，0=对称，正=上半周大，负=下半周大"))
