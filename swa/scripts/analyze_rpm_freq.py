"""
直接看 RPM 对实际 FFT 基频位置的影响
不依赖数据库里存的 harm_cycles（那个是 7.0 固定值）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import sqlite3
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from swa.data.loader import compute_harmonics
from swa.core.scoring import compute_score, compute_alpha7

DB_PATH = r"d:\project\work\swa\swa\src\data\projects\new\data.db"
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 取少量样本：每个设备 × 每个电压 × 不同 RPM 区间
cur.execute("""
    SELECT r.id, r.actual_voltage, r.device_id, r.rpm, r.temperature, r.humidity,
           w.wave_data
    FROM records r
    JOIN waveforms w ON w.record_id = r.id
    WHERE r.enabled=1 AND r.rpm IS NOT NULL AND r.actual_voltage>=0
    ORDER BY r.id
""")
rows = cur.fetchall()
conn.close()

print(f"总记录: {len(rows)}")

# 每隔 N 条取一条（加速）
sample = rows[::3]  # 1/3 采样
print(f"采样: {len(sample)} 条")

results = []
for rid, voltage, dev_id, rpm, temp, humid, wave_str in sample:
    wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
    if len(wave) < 20:
        continue

    # FFT 分析
    y = wave - np.mean(wave)
    n = len(y)
    fft_vals = np.fft.rfft(y)
    mag = np.abs(fft_vals[1:])

    # 基频索引
    search_end = min(len(mag), n // 3)
    fund_idx = int(np.argmax(mag[:search_end]) + 1)

    # 实际周期数 = 基频索引 × 2 × n_cycles / n_points
    # 对于 512 点: fund_idx 是 FFT bin 编号
    # 实际周期数 = fund_idx * 2 / 512 * N_points ≈ fund_idx * (n / n) 
    # 准确地说: 周期数 = fund_idx * n / n = fund_idx
    # 不对，fund_idx 是 FFT 的 bin 索引，实际周期数需要换算
    # FFT bin 索引 k 对应的频率为 k * fs / N
    # 周期数 = 频率 * 时间 = k * fs / N * (N/fs) = k
    # 所以周期数 = fund_idx (对于 rfft, bin 1 对应 1 个完整周期)
    actual_cycles = float(fund_idx)

    # score
    score = compute_score(wave)
    alpha7 = compute_alpha7(wave) or 0.0

    results.append({
        'id': rid,
        'voltage': voltage,
        'device_id': dev_id[-4:] if dev_id else '?',
        'rpm': rpm,
        'fund_bin': fund_idx,
        'cycles': actual_cycles,
        'score': score,
        'alpha7': alpha7,
    })

df = pd.DataFrame(results)
print(f"解析完成: {len(df)} 条")

print(f"\nFFT 基频 bin 范围: {df['fund_bin'].min()} ~ {df['fund_bin'].max()}")
print(f"实际周期数 范围: {df['cycles'].min():.1f} ~ {df['cycles'].max():.1f}, 均值={df['cycles'].mean():.3f}")

# RPM vs 实际周期数
corr = df['rpm'].corr(df['cycles'])
print(f"\nRPM vs 实际周期数 相关系数: {corr:.4f}")

# 画图
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

devices = sorted(df['device_id'].unique())
colors = {'2539': '#2196F3', '253D': '#FF5722', '6A39': '#4CAF50'}

# (a) RPM vs 实际周期数
ax = axes[0, 0]
for dev in devices:
    sub = df[df['device_id'] == dev]
    ax.scatter(sub['rpm'], sub['cycles'], c=colors.get(dev, 'gray'), s=5, alpha=0.4, label=f'设备{dev}')
coeffs = np.polyfit(df['rpm'], df['cycles'], 1)
xr = np.linspace(df['rpm'].min(), df['rpm'].max(), 100)
ax.plot(xr, np.polyval(coeffs, xr), 'r--', lw=1.5, label=f'斜率={coeffs[0]:.6f}')
ax.set_xlabel('RPM', fontsize=11)
ax.set_ylabel('实际 FFT 周期数', fontsize=11)
ax.set_title(f'RPM vs 实际周期数  (r={corr:.4f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, markerscale=4)
ax.grid(True, alpha=0.3)

# (b) RPM vs score
ax = axes[0, 1]
for dev in devices:
    sub = df[df['device_id'] == dev]
    ax.scatter(sub['rpm'], sub['score'], c=colors.get(dev, 'gray'), s=5, alpha=0.4, label=f'设备{dev}')
cs = np.polyfit(df['rpm'], df['score'], 1)
ax.plot(xr, np.polyval(cs, xr), 'r--', lw=1.5, label=f'斜率={cs[0]:.6f}')
ax.set_xlabel('RPM', fontsize=11)
ax.set_ylabel('score', fontsize=11)
ax.set_title('RPM vs Score', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, markerscale=4)
ax.grid(True, alpha=0.3)

# (c) RPM vs alpha7
ax = axes[0, 2]
for dev in devices:
    sub = df[df['device_id'] == dev]
    ax.scatter(sub['rpm'], sub['alpha7'], c=colors.get(dev, 'gray'), s=5, alpha=0.4, label=f'设备{dev}')
ca = np.polyfit(df['rpm'], df['alpha7'], 1)
ax.plot(xr, np.polyval(ca, xr), 'r--', lw=1.5, label=f'斜率={ca[0]:.6f}')
ax.set_xlabel('RPM', fontsize=11)
ax.set_ylabel('alpha_7', fontsize=11)
ax.set_title('RPM vs alpha_7', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, markerscale=4)
ax.grid(True, alpha=0.3)

# (d) 周期数 vs score
ax = axes[1, 0]
for dev in devices:
    sub = df[df['device_id'] == dev]
    ax.scatter(sub['cycles'], sub['score'], c=colors.get(dev, 'gray'), s=5, alpha=0.4, label=f'设备{dev}')
ax.set_xlabel('实际周期数', fontsize=11)
ax.set_ylabel('score', fontsize=11)
ax.set_title('周期数 vs Score', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, markerscale=4)
ax.grid(True, alpha=0.3)

# (e) 周期数的分布直方
ax = axes[1, 1]
for dev in devices:
    sub = df[df['device_id'] == dev]
    ax.hist(sub['cycles'], bins=30, alpha=0.4, label=f'设备{dev}', color=colors.get(dev))
ax.set_xlabel('实际周期数', fontsize=11)
ax.set_ylabel('频次', fontsize=11)
ax.set_title('周期数分布', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (f) RPM 分布直方
ax = axes[1, 2]
for dev in devices:
    sub = df[df['device_id'] == dev]
    ax.hist(sub['rpm'], bins=30, alpha=0.4, label=f'设备{dev}', color=colors.get(dev))
ax.set_xlabel('RPM', fontsize=11)
ax.set_ylabel('频次', fontsize=11)
ax.set_title('RPM 分布', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = r'd:\project\work\swa\swa\scripts\rpm_vs_freq.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\n已保存: {out}")
plt.close(fig)

# 每设备统计
print(f"\n=== 每设备详细 ===")
for dev in devices:
    sub = df[df['device_id'] == dev]
    print(f"\n  设备{dev}:")
    print(f"    RPM    : 均值={sub['rpm'].mean():.0f}, std={sub['rpm'].std():.1f}, 范围=[{sub['rpm'].min():.0f}, {sub['rpm'].max():.0f}]")
    print(f"    周期数 : 均值={sub['cycles'].mean():.4f}, std={sub['cycles'].std():.4f}, 范围=[{sub['cycles'].min():.1f}, {sub['cycles'].max():.1f}]")
    c = sub['rpm'].corr(sub['cycles'])
    print(f"    RPM-周期相关系数: {c:.4f}")
