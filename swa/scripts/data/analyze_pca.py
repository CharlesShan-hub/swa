"""PCA analysis: effective dimensions of FFT harmonics."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from sklearn.decomposition import PCA
from scripts.utils.loader import load_jsonl
from src.swa.estimation.feature_extractor import extract_features

records = load_jsonl("data/exported_data.jsonl")
print(f"Loaded {len(records)} records")

# Extract A1~A14 harmonics
harmonics_list = []
y_list = []
for i, r in enumerate(records):
    wave_str = r.get("RTU_REGS_P00_WAVE_DATA", "")
    wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]
    ac = wave - np.mean(wave)
    n = len(ac)
    fft_mag = np.abs(np.fft.fft(ac))[:n // 2]
    h = np.zeros(14)
    for j in range(14):
        idx = j + 1
        if idx < len(fft_mag):
            h[j] = 2.0 * fft_mag[idx] / n
    harmonics_list.append(h)
    v_str = str(r.get("ACTUAL_VOLTAGE", "")).lower().replace("v", "").strip()
    try: y_list.append(float(v_str))
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
print("\nPCA analysis (14 harmonics)")
print("=" * 60)
print(f"{'PC':>6}  {'VarRatio':>10}  {'Cumulative':>10}  {'Note'}")
for i in range(14):
    note = ""
    if i == 0 and cumsum[i] > 0.5:
        note = "<-- main signal energy"
    elif i < 10 and cumsum[i] > 0.95 and cumsum[i-1] < 0.95:
        note = "<-- covers 95% variance"
    elif i >= 10:
        note = "<-- noise level"
    print(f"   PC{i+1:>2}:  {pca.explained_variance_ratio_[i]:>10.4f}  {cumsum[i]:>10.4f}  {note}")

# Key thresholds
print()
n_95 = np.argmax(cumsum >= 0.95) + 1
n_99 = np.argmax(cumsum >= 0.99) + 1
print(f"Components to cover 95% variance: {n_95}")
print(f"Components to cover 99% variance: {n_99}")
