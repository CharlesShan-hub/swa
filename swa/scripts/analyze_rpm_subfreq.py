"""
使用亚 bin 精度（抛物线插值）估计基波的真实频率
然后看 RPM 是否与这个精确频率相关
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

from swa.core.scoring import compute_score


def estimate_precise_freq(wave: np.ndarray) -> dict:
    """用抛物线插值精确估计基波频率（亚 bin 精度）。

    Returns:
        {
            "peak_bin": int,        # FFT 峰值 bin
            "fine_cycles": float,   # 抛物线插值后的精细周期数
            "fine_amp": float,      # 插值后的幅值
            "bin_minus": float,     # bin-1 幅值
            "bin_plus": float,      # bin+1 幅值
        }
    """
    y = wave - np.mean(wave)
    n = len(y)
    fft_vals = np.fft.rfft(y)
    mag = np.abs(fft_vals[1:])  # 去掉 DC

    # 找峰值 bin
    search_end = min(len(mag), n // 3)
    peak_bin = int(np.argmax(mag[:search_end]))  # 0-indexed, 对应 1-cycle

    # 抛物线插值（使用 peak_bin-1, peak_bin, peak_bin+1）
    if peak_bin == 0 or peak_bin >= len(mag) - 1:
        return {
            "peak_bin": peak_bin + 1,
            "fine_cycles": float(peak_bin + 1),
            "fine_amp": float(mag[peak_bin]),
        }

    y0, y1, y2 = np.log(mag[peak_bin - 1]), np.log(mag[peak_bin]), np.log(mag[peak_bin + 1])
    denom = 2 * (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-12:
        sub_bin_offset = 0.0
    else:
        sub_bin_offset = (y0 - y2) / denom

    fine_bin = (peak_bin + 1) + sub_bin_offset  # +1 转换为 1-indexed
    # 幅值插值
    fine_amp = mag[peak_bin] - 0.25 * (mag[peak_bin - 1] - mag[peak_bin + 1]) * sub_bin_offset

    return {
        "peak_bin": peak_bin + 1,
        "fine_cycles": float(fine_bin),
        "fine_amp": float(fine_amp),
        "mag_peak": float(mag[peak_bin]),
        "mag_left": float(mag[peak_bin - 1]),
        "mag_right": float(mag[peak_bin + 1]),
    }


DB_PATH = r"d:\project\work\swa\swa\src\data\projects\new\data.db"
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("""
    SELECT r.id, r.actual_voltage, r.device_id, r.rpm,
           r.harm_a1, r.harm_error, r.temperature, r.humidity,
           w.wave_data
    FROM records r
    JOIN waveforms w ON w.record_id = r.id
    WHERE r.enabled=1 AND r.rpm IS NOT NULL AND r.actual_voltage>=0
    ORDER BY r.id
""")
rows = cur.fetchall()
conn.close()

print(f"总记录: {len(rows)}")

# 每 2 条取一条
sample = rows[::2]
print(f"采样: {len(sample)} 条")

results = []
for rid, voltage, dev_id, rpm, a1, a1err, temp, humid, wave_str in sample:
    wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
    if len(wave) < 20:
        continue

    freq_info = estimate_precise_freq(wave)
    score = compute_score(wave)

    results.append({
        "id": rid,
        "voltage": voltage,
        "device_id": dev_id[-4:] if dev_id else "?",
        "rpm": rpm,
        "harm_a1": a1 if a1 else 0.0,
        "score": score,
        "temperature": temp if temp else 0.0,
        "humidity": humid if humid else 0.0,
        "peak_bin": freq_info["peak_bin"],
        "fine_cycles": freq_info["fine_cycles"],
        "fine_amp": freq_info["fine_amp"],
        "mag_peak": freq_info["mag_peak"],
        "mag_left": freq_info["mag_left"],
        "mag_right": freq_info["mag_right"],
    })

df = pd.DataFrame(results)
print(f"解析: {len(df)} 条")

# ── 精细周期数统计 ──
print(f"\n=== 精细周期数（亚 bin 精度）===")
print(f"  范围: {df['fine_cycles'].min():.6f} ~ {df['fine_cycles'].max():.6f}")
print(f"  均值: {df['fine_cycles'].mean():.6f}")
print(f"  std : {df['fine_cycles'].std():.6f}")
print(f"  理论 @ 13007RPM: 7.000 周期")

# RPM vs 精细周期数
r_fine = df['rpm'].corr(df['fine_cycles'])
print(f"\nRPM vs 精细周期数: r = {r_fine:.6f}")

# 理论上应该: fine_cycles ≈ 7.0 × (rpm / 13007)
# 看看实际斜率
coeffs = np.polyfit(df['rpm'], df['fine_cycles'], 1)
print(f"  拟合斜率: {coeffs[0]:.8f}")
print(f"  理论斜率: {7.0 / 13007:.8f}")
print(f"  拟合截距: {coeffs[1]:.6f}")

# 精细周期数 vs score
r_fs = df['fine_cycles'].corr(df['score'])
print(f"\n精细周期数 vs score: r = {r_fs:.6f}")

# 精细周期数 vs A1
r_fa = df['fine_cycles'].corr(df['harm_a1'])
print(f"精细周期数 vs A1:    r = {r_fa:.6f}")

# 精细周期数 vs 误差
df['error_div_a1'] = np.where(df['harm_a1'] > 1e-6, abs(df['harm_a1'] - df['fine_amp']) / df['harm_a1'], 0.0)
r_fe = df['fine_cycles'].corr(df['error_div_a1'])
print(f"精细周期数 vs 幅值误差: r = {r_fe:.6f}")

# 按设备看
print(f"\n=== 每设备 RPM vs 精细周期数 ===")
devices = sorted(df['device_id'].unique())
for dev in devices:
    sub = df[df['device_id'] == dev]
    c = sub['rpm'].corr(sub['fine_cycles'])
    print(f"  设备{dev}: n={len(sub):>5d}, RPM~fine_cycles r={c:.4f}, "
          f"fine_cycles 范围=[{sub['fine_cycles'].min():.4f}, {sub['fine_cycles'].max():.4f}]")

# ── 画图 ──
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
colors = {'2539': '#2196F3', '253D': '#FF5722', '6A39': '#4CAF50'}

# (a) RPM vs 精细周期数
ax = axes[0, 0]
for dev in devices:
    sub = df[df['device_id'] == dev]
    ax.scatter(sub['rpm'], sub['fine_cycles'], c=colors.get(dev, 'gray'),
               s=4, alpha=0.3, label=f'设备{dev}')
xr = np.linspace(df['rpm'].min(), df['rpm'].max(), 100)
ax.plot(xr, np.polyval(coeffs, xr), 'r--', lw=1.5, label=f'斜率={coeffs[0]:.6f}')
ax.set_xlabel('RPM', fontsize=11)
ax.set_ylabel('精细周期数（亚bin）', fontsize=11)
ax.set_title(f'RPM vs 精细周期数  (r={r_fine:.4f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, markerscale=4)
ax.grid(True, alpha=0.3)

# (b) 精细周期数分布
ax = axes[0, 1]
for dev in devices:
    sub = df[df['device_id'] == dev]
    ax.hist(sub['fine_cycles'], bins=50, alpha=0.4, label=f'设备{dev}',
            color=colors.get(dev))
ax.set_xlabel('精细周期数', fontsize=11)
ax.set_ylabel('频次', fontsize=11)
ax.set_title('精细周期数分布', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) 精细周期数 vs score
ax = axes[0, 2]
for dev in devices:
    sub = df[df['device_id'] == dev]
    ax.scatter(sub['fine_cycles'], sub['score'], c=colors.get(dev, 'gray'),
               s=4, alpha=0.3, label=f'设备{dev}')
ax.set_xlabel('精细周期数', fontsize=11)
ax.set_ylabel('score', fontsize=11)
ax.set_title(f'精细周期数 vs score (r={r_fs:.4f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, markerscale=4)
ax.grid(True, alpha=0.3)

# (d) 精细周期数 vs A1 (固定80V)
ax = axes[1, 0]
for dev in devices:
    sub = df[(df['device_id'] == dev) & (df['voltage'] == 80)]
    if len(sub) > 5:
        ax.scatter(sub['fine_cycles'], sub['harm_a1'], c=colors.get(dev, 'gray'),
                   s=10, alpha=0.5, label=f'设备{dev}')
ax.set_xlabel('精细周期数 (80V)', fontsize=11)
ax.set_ylabel('A1', fontsize=11)
ax.set_title('80V下 精细周期数 vs A1', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, markerscale=4)
ax.grid(True, alpha=0.3)

# (e) 精细周期数 vs 温度
ax = axes[1, 1]
for dev in devices:
    sub = df[df['device_id'] == dev]
    ax.scatter(sub['temperature'], sub['fine_cycles'], c=colors.get(dev, 'gray'),
               s=4, alpha=0.3, label=f'设备{dev}')
ax.set_xlabel('温度 (°C)', fontsize=11)
ax.set_ylabel('精细周期数', fontsize=11)
ax.set_title('温度 vs 精细周期数', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, markerscale=4)
ax.grid(True, alpha=0.3)

# (f) 精细周期数 vs 湿度
ax = axes[1, 2]
for dev in devices:
    sub = df[df['device_id'] == dev]
    ax.scatter(sub['humidity'], sub['fine_cycles'], c=colors.get(dev, 'gray'),
               s=4, alpha=0.3, label=f'设备{dev}')
ax.set_xlabel('湿度 (%)', fontsize=11)
ax.set_ylabel('精细周期数', fontsize=11)
ax.set_title('湿度 vs 精细周期数', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, markerscale=4)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = r'd:\project\work\swa\swa\scripts\rpm_vs_fine_freq.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\n已保存: {out}")
plt.close(fig)

# 物理关系验证
print(f"\n{'='*60}")
print(f"物理关系验证:")
print(f"  样本点 512 个, 标称 ~7 周期 @ 13007 RPM")
print(f"  每周期采样点 ≈ {512/7:.1f}")
print(f"  精细周期范围: {df['fine_cycles'].min():.4f} ~ {df['fine_cycles'].max():.4f}")
print(f"  对应的 RPM 范围: {df['fine_cycles'].min()/7*13007:.0f} ~ {df['fine_cycles'].max()/7*13007:.0f}")
print(f"  实际 RPM 范围:    {df['rpm'].min():.0f} ~ {df['rpm'].max():.0f}")
print(f"{'='*60}")
