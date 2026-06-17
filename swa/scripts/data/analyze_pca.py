"""PCA analysis: effective dimensions of FFT harmonics."""
import sys, os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from sklearn.decomposition import PCA
from scripts.utils.loader import load_jsonl

parser = argparse.ArgumentParser(description="FFT 谐波的 PCA 分析")
parser.add_argument("-d", "--data", default="data/exported_data.jsonl",
                    help="数据文件路径 (默认: data/exported_data.jsonl)")
parser.add_argument("--n-harmonics", type=int, default=10,
                    help="FFT 谐波数量 (默认: 10)")
args = parser.parse_args()

records = load_jsonl(args.data)
print(f"Loaded {len(records)} records")

n_harmonics = args.n_harmonics

# Extract harmonics
harmonics_list = []
y_list = []
for i, r in enumerate(records):
    wave_str = r.get("RTU_REGS_P00_WAVE_DATA", "")
    wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
    ac = wave - np.mean(wave)
    n = len(ac)
    fft_mag = np.abs(np.fft.fft(ac))[:n // 2]
    h = np.zeros(n_harmonics)
    for j in range(n_harmonics):
        idx = j + 1
        if idx < len(fft_mag):
            h[j] = 2.0 * fft_mag[idx] / n
    harmonics_list.append(h)
    v = r.get("ACTUAL_VOLTAGE")
    try: y_list.append(float(v))
    except: y_list.append(0.0)

    if (i + 1) % 5000 == 0:
        print(f"  processed {i + 1}/{len(records)}")

H = np.array(harmonics_list)
y = np.array(y_list)

# PCA
pca = PCA()
pca.fit(H)

# Explained variance
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(f"\nPCA analysis ({n_harmonics} harmonics)")
print("=" * 60)
print(f"{'PC':>6}  {'VarRatio':>10}  {'Cumulative':>10}  {'Note'}")
for i in range(min(n_harmonics, len(cumsum))):
    note = ""
    if i == 0 and cumsum[i] > 0.5:
        note = "<-- main signal energy"
    elif i < n_harmonics - 1 and cumsum[i] > 0.95 and (i == 0 or cumsum[i-1] < 0.95):
        note = "<-- covers 95% variance"
    elif i >= n_harmonics - 2:
        note = "<-- noise level"
    print(f"   PC{i+1:>2}:  {pca.explained_variance_ratio_[i]:>10.4f}  {cumsum[i]:>10.4f}  {note}")

# Key thresholds
print()
n_95 = np.argmax(cumsum >= 0.95) + 1
n_99 = np.argmax(cumsum >= 0.99) + 1
print(f"Components to cover 95% variance: {n_95}")
print(f"Components to cover 99% variance: {n_99}")
