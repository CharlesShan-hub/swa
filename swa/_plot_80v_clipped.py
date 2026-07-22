"""Plot several 80V clipped waveforms"""
import sqlite3, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

conn = sqlite3.connect(r'd:\project\work\swa\swa\src\data\projects\new\data.db')
cur = conn.cursor()

ids = [20100, 20109, 27311, 27329, 27355, 27389]
fig, axes = plt.subplots(3, 2, figsize=(12, 9))
axes = axes.flatten()

for idx, rid in enumerate(ids):
    ax = axes[idx]
    cur.execute('SELECT harm_a1, harm_a1_corrected FROM records WHERE id=?', (rid,))
    r = cur.fetchone()
    a1_orig, a1_corr = r[0], r[1]

    cur.execute('SELECT wave_data FROM waveforms WHERE record_id=?', (rid,))
    wave = np.array([float(x) for x in cur.fetchone()[0].split(',')], dtype=np.float64)
    n = len(wave)
    dc = np.mean(wave)
    y = wave - dc

    fft_vals = np.fft.rfft(y)
    mag = np.abs(fft_vals[1:])
    fund_idx = int(np.argmax(mag[:n//3]) + 1)
    phase = np.angle(fft_vals[fund_idx])
    omega = 2 * np.pi * fund_idx / n
    t = np.arange(n)
    
    orig_fit = 2*a1_orig/n * np.cos(omega*t + phase) + dc
    corr_fit = 2*a1_corr/n * np.cos(omega*t + phase) + dc if a1_corr else None

    ax.plot(wave, linewidth=0.5, color='#999', alpha=0.7, label='原始')
    ax.plot(orig_fit, linewidth=1.2, color='#e74c3c', linestyle='--', label=f'A1={a1_orig:.0f}')
    if corr_fit is not None:
        ax.plot(corr_fit, linewidth=1.2, color='#2ecc71', linestyle='-', label=f'A1_c={a1_corr:.0f}')
    ax.axhline(dc, color='gray', linestyle=':', linewidth=0.5)
    ax.set_title(f'ID={rid}  80V  A1={a1_orig:.0f}->{a1_corr:.0f}')
    ax.grid(True, alpha=0.3)
    if idx == 0: ax.legend(fontsize=7)

fig.suptitle('80V 削波记录', fontsize=14)
fig.tight_layout()
fig.savefig(r'd:\project\work\swa\swa\_plot_80v_clipped.png', dpi=150)
print('OK')
