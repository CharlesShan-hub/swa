import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.stats import kurtosis, skew
from src.swa.signal_process.loader import load_jsonl

records = load_jsonl("data/exported_data.jsonl")[:1000]

feats_all = []
for r in records:
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

feats = np.array(feats_all)
corr = np.corrcoef(feats.T)

print("特征相关性分析（对1000条数据）")
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
print("结论:")
print("  1. Vpp-RMS-A1 几乎完全相关（r≈1.0），是冗余特征")
print("  2. Crest≈1.414 是常数（标准正弦波），无信息量")
print("  3. Kurtosis 和 Skewness 是独立信息，但量级小")
print("  -> 只保留 Kurtosis + Skewness，去掉 Vpp/RMS/Crest")
