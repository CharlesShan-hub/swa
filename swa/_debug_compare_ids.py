"""Compare waveforms: real clipping vs false positive vs clean"""
import sqlite3, numpy as np

conn = sqlite3.connect(r'd:\project\work\swa\swa\src\data\projects\new\data.db')
cur = conn.cursor()

def analyze(rid):
    cur.execute('SELECT wave_data FROM waveforms WHERE record_id=?', (rid,))
    wave = np.array([float(x) for x in cur.fetchone()[0].split(',')], dtype=np.float64)
    y = wave - np.mean(wave)
    n = len(y)
    dc = np.mean(wave)
    
    fft_vals = np.fft.rfft(y)
    mag = np.abs(fft_vals[1:])
    fund_idx = int(np.argmax(mag[:n//3]) + 1)
    phase = np.angle(fft_vals[fund_idx])
    a1_orig = float(mag[fund_idx - 1])
    omega = 2 * np.pi * fund_idx / n
    basis = np.cos(omega * np.arange(n) + phase)

    signs = np.sign(y)
    zero_idx = np.where(np.diff(signs) != 0)[0]
    
    # 半周期分析
    peaks_pos = []
    peaks_neg = []
    ratios = []
    for i in range(len(zero_idx) - 1):
        s, e = zero_idx[i], zero_idx[i+1]
        if e - s < 5: continue
        seg = y[s:e+1]
        peak_val = float(np.max(seg))
        trough_val = float(np.min(seg))
        peak_amp = float(np.max(np.abs(seg)))
        peaks_pos.append(peak_val)
        peaks_neg.append(trough_val)
        
        # 当前半周期内，边缘区域拟合 vs 整体拟合
        half_len = e - s
        e1 = s + int(half_len * 0.3)
        e2 = e - int(half_len * 0.3)
        edge_mask = np.zeros(n, dtype=bool)
        edge_mask[s:e1] = True
        edge_mask[e2:e] = True
        edge_mask = edge_mask & (np.arange(n) >= s) & (np.arange(n) < e)
        
        # 这个半周期的拟合
        seg_basis = basis[s:e+1]
        seg_y = y[s:e+1]
        a_full = np.sum(seg_y * seg_basis) / max(np.sum(seg_basis**2), 1e-10) * n / 2
        
        seg_edge = edge_mask[s:e+1]
        if np.sum(seg_edge) >= 3:
            a_edge = np.sum(seg_y[seg_edge] * seg_basis[seg_edge]) / max(np.sum(seg_basis[seg_edge]**2), 1e-10) * n / 2
        else:
            a_edge = a_full
        
        ratios.append(a_edge / max(a_full, 1e-10))
    
    # 总体边缘区域拟合
    clean_mask = np.zeros(n, dtype=bool)
    for i in range(len(zero_idx) - 1):
        s, e = zero_idx[i], zero_idx[i+1]
        half_len = e - s
        if half_len < 5: continue
        e1 = s + int(half_len * 0.3)
        if e1 > s: clean_mask[s:e1] = True
        e2 = e - int(half_len * 0.3)
        if e2 < e: clean_mask[e2:e] = True
    
    yc = y[clean_mask]
    bc = basis[clean_mask]
    a1_edge = np.sum(yc * bc) / max(np.sum(bc**2), 1e-10) * n / 2
    
    # 平坦检测
    diff = np.abs(np.diff(y))
    typical_slope = float(np.percentile(diff[diff > 0], 75))
    flat_th = typical_slope * 0.05
    is_flat_raw = diff < flat_th
    valid_flat = np.zeros(n - 1, dtype=bool)
    for i in range(len(zero_idx) - 1):
        s, e = zero_idx[i], zero_idx[i+1]
        if e - s < 5: continue
        margin = int((e - s) * 0.2)
        zs = max(s + margin, s)
        ze = min(e - margin, e)
        if zs < ze: valid_flat[zs:ze] = is_flat_raw[zs:ze]
    
    padded = np.concatenate([[False], valid_flat, [False]])
    rs = np.where(~padded[:-1] & padded[1:])[0]
    re = np.where(padded[:-1] & ~padded[1:])[0]
    
    y_scale = float(np.max(np.abs(y)))
    has_flat = False
    max_flat_len = 0
    for s, e in zip(rs, re):
        if e - s >= 3:
            seg = y[s:e+1]
            seg_diff = np.max(np.abs(np.diff(seg)))
            if seg_diff < 0.001 * y_scale and e - s >= 5:
                has_flat = True
            if e - s > max_flat_len:
                max_flat_len = e - s
    
    # 形状检测（中位峰值对比）
    all_peaks = np.array([float(np.max(np.abs(y[s:e+1]))) for s, e in 
                         [(zero_idx[i], zero_idx[i+1]) for i in range(len(zero_idx)-1) if zero_idx[i+1]-zero_idx[i] >= 5]])
    median_peak = float(np.median(all_peaks)) if len(all_peaks) > 0 else 0
    clipped_count = int(np.sum(all_peaks < median_peak * 0.88)) if len(all_peaks) > 0 else 0
    has_shape = clipped_count >= 2 and a1_orig * 2 / n > 0.3
    
    print(f'\n{"="*50}')
    print(f'ID={rid}:  A1_orig={a1_orig:.2f}  A1_edge={a1_edge:.2f}  delta={a1_edge-a1_orig:+.2f}')
    print(f'  dc={dc:.2f}  fund_idx={fund_idx}  y_range=[{y.min():.2f}, {y.max():.2f}]')
    print(f'  half_cycles={len(all_peaks)}  median_peak={median_peak:.3f}')
    print(f'  shape: clipped={clipped_count}  has_shape={has_shape}')
    print(f'  flat: has_flat={has_flat}  max_flat_len={max_flat_len}')
    print(f'  detected: {has_flat or has_shape}')
    print(f'  pos_peaks: {np.sort(peaks_pos)[-5:].round(3).tolist()}')
    print(f'  neg_peaks: {np.sort(peaks_neg)[:5].round(3).tolist()}')
    print(f'  edge/full ratios per half-cycle: {[round(r, 3) for r in ratios]}')

for rid in [30737, 30739, 26650, 33511]:
    analyze(rid)
