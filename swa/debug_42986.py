"""
调试 ID=42986 — 锚点法边缘区域拟合
可调参数: EDGE_RATIO (默认 0.30)
"""
import sqlite3, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============ 可调参数 ============
EDGE_RATIO = 0.30   # 每个半周期两侧各取的比例
# ==================================

conn = sqlite3.connect(r'd:\project\work\swa\swa\src\data\projects\new\data.db')
cur = conn.cursor()
cur.execute('SELECT wave_data FROM waveforms WHERE record_id=42986')
wave_str = cur.fetchone()[0]
wave = np.array([float(x) for x in wave_str.split(',')], dtype=np.float64)

n = len(wave)
dc = np.mean(wave)
y = wave - dc

# FFT
fft_vals = np.fft.rfft(y)
mag = np.abs(fft_vals[1:])
fund_idx = int(np.argmax(mag[:n//3]) + 1)
phase = np.angle(fft_vals[fund_idx])
a1_orig = float(mag[fund_idx - 1])
omega = 2 * np.pi * fund_idx / n
basis = np.cos(omega * np.arange(n) + phase)

# 过零点
signs = np.sign(y)
zero_idx = np.where(np.diff(signs) != 0)[0]
print(f'过零点数: {len(zero_idx)}')

# 半周期信息
half_lens = []
for i in range(len(zero_idx) - 1):
    hl = zero_idx[i+1] - zero_idx[i]
    half_lens.append(hl)
print(f'半周期长度: min={min(half_lens)} max={max(half_lens)} mean={np.mean(half_lens):.1f}')

# 边缘区域锚点
anchor_x = []
for i in range(len(zero_idx) - 1):
    s, e = zero_idx[i], zero_idx[i+1]
    half_len = e - s
    if half_len < 5: continue
    e1 = s + int(half_len * EDGE_RATIO)
    if e1 > s: anchor_x.extend(range(s, e1))
    e2 = e - int(half_len * EDGE_RATIO)
    if e2 < e: anchor_x.extend(range(e2, e))

# 边缘区域 LS 拟合
clean_mask = np.zeros(n, dtype=bool)
clean_mask[anchor_x] = True
yc = y[clean_mask]
bc = basis[clean_mask]
a1_clean = np.sum(yc * bc) / np.sum(bc ** 2)
a1_corr = a1_clean * n / 2

# 完整 LS 对比
a1_full = np.sum(y * basis) / np.sum(basis ** 2) * n / 2

print(f'\nEDGE_RATIO = {EDGE_RATIO}')
print(f'FFT A1:         {a1_orig:.2f}')
print(f'完整LS A1:      {a1_full:.2f}')
print(f'边缘区域 A1:    {a1_corr:.2f}')
print(f'delta:          {a1_corr - a1_orig:+.2f}  ({((a1_corr/a1_orig - 1)*100):+.2f}%)')
print(f'锚点数:         {len(anchor_x)} ({len(anchor_x)/n*100:.1f}%)')

# 重建基波
t = np.arange(n)
orig_fit = 2 * a1_orig / n * np.cos(omega * t + phase) + dc
corr_fit = 2 * a1_corr / n * np.cos(omega * t + phase) + dc

# ============ 画图 ============
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
for ax, (start, end) in [(ax1, (0, n)), (ax2, (80, 280))]:
    xr = np.arange(start, end)
    ax.plot(xr, wave[start:end], linewidth=0.6, color='#999', alpha=0.7, label='原始数据')
    ax.axhline(dc, color='gray', linestyle=':', linewidth=1, alpha=0.7, label=f'DC={dc:.2f}')

    mask = [x for x in anchor_x if start <= x < end]
    ax.scatter(mask, [wave[x] for x in mask], s=3, color='#2196F3', alpha=0.5, label=f'锚点({len(anchor_x)})')

    ax.plot(xr, orig_fit[start:end], linewidth=1.5, color='#e74c3c', linestyle='--', label=f'A1原始={a1_orig:.0f}')
    ax.plot(xr, corr_fit[start:end], linewidth=1.5, color='#2ecc71', linestyle='-', label=f'A1矫正={a1_corr:.0f}')
    ax.set_xlabel('采样点')
    ax.set_ylabel('幅值')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle(f'ID=42986  edge_ratio={EDGE_RATIO}  A1: {a1_orig:.0f} -> {a1_corr:.0f}  ({((a1_corr/a1_orig-1)*100):+.1f}%)', fontsize=13)
fig.tight_layout()
fig.savefig(r'd:\project\work\swa\swa\_plot_42986.png', dpi=150)
print(f'\n图片已保存: _plot_42986.png')
print(f'修改 EDGE_RATIO 重跑即可调整')
