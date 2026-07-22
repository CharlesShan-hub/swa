"""Plot 42986 with edge-region anchors and corrected A1"""
import sqlite3, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

conn = sqlite3.connect(r'd:\project\work\swa\swa\src\data\projects\new\data.db')
cur = conn.cursor()
cur.execute('SELECT wave_data FROM waveforms WHERE record_id=42986')
wave_str = cur.fetchone()[0]
wave = np.array([float(x) for x in wave_str.split(',')], dtype=np.float64)

n = len(wave)
dc = np.mean(wave)
y = wave - dc

fft_vals = np.fft.rfft(y)
mag = np.abs(fft_vals[1:])
fund_idx = int(np.argmax(mag[:n//3]) + 1)
phase = np.angle(fft_vals[fund_idx])
a1_orig = float(mag[fund_idx - 1])
omega = 2 * np.pi * fund_idx / n

# 边缘区域锚点
signs = np.sign(y)
zero_idx = np.where(np.diff(signs) != 0)[0]

anchor_x = []  # 边缘区域的点
for i in range(len(zero_idx) - 1):
    s, e = zero_idx[i], zero_idx[i+1]
    half_len = e - s
    if half_len < 5: continue
    e1 = s + int(half_len * 0.30)
    if e1 > s:
        anchor_x.extend(range(s, e1))
    e2 = e - int(half_len * 0.30)
    if e2 < e:
        anchor_x.extend(range(e2, e))

# 矫正 A1
from swa.data.loader import _detect_and_correct_clipping
a1_corr, cr = _detect_and_correct_clipping(wave, a1_orig, fund_idx, phase, n)
if a1_corr == a1_orig:
    a1_corr = None

t = np.arange(n)
orig_fit = 2 * a1_orig / n * np.cos(omega * t + phase) + dc
corr_fit = 2 * a1_corr / n * np.cos(omega * t + phase) + dc if a1_corr else None

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

for ax, (start, end) in [(ax1, (0, n)), (ax2, (80, 280))]:
    xr = np.arange(start, end)
    ax.plot(xr, wave[start:end], linewidth=0.6, color='#999', alpha=0.7, label='原始数据')
    ax.axhline(dc, color='gray', linestyle=':', linewidth=1, alpha=0.7, label=f'DC={dc:.2f}')
    
    # 边缘区域锚点（蓝色高亮）
    ax.scatter([x for x in anchor_x if start <= x < end],
               [wave[x] for x in anchor_x if start <= x < end],
               s=3, color='#2196F3', alpha=0.5, label=f'锚点({len(anchor_x)}点)')
    
    ax.plot(xr, orig_fit[start:end], linewidth=1.5, color='#e74c3c', linestyle='--', label=f'A1原始={a1_orig:.0f}')
    if corr_fit is not None:
        ax.plot(xr, corr_fit[start:end], linewidth=1.5, color='#2ecc71', linestyle='-', label=f'A1矫正={a1_corr:.0f}')
    
    ax.set_xlabel('采样点')
    ax.set_ylabel('幅值')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle(f'ID=42986 锚点法(边缘30%+30%)  A1原始={a1_orig:.0f} -> A1矫正={a1_corr:.0f}  cr={cr:.1%}', fontsize=13)
fig.tight_layout()
fig.savefig(r'd:\project\work\swa\swa\_plot_42986.png', dpi=150)
print(f'锚点({len(anchor_x)}点) saved')
