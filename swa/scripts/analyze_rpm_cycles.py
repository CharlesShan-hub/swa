"""
RPM vs harm_cycles — 转速决定频率，频率决定周期数
"""
import sqlite3, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

DB_PATH = r"d:\project\work\swa\swa\src\data\projects\new\data.db"
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("""
    SELECT r.id, r.actual_voltage, r.device_id,
           r.rpm, r.harm_cycles, r.harm_a1
    FROM records r
    WHERE r.enabled=1 AND r.rpm IS NOT NULL AND r.harm_cycles IS NOT NULL
      AND r.actual_voltage>=0
""", conn)
conn.close()

print(f"总记录: {len(df)}")
print(f"harm_cycles 范围: {df['harm_cycles'].min():.3f} ~ {df['harm_cycles'].max():.3f}, 均值={df['harm_cycles'].mean():.3f}")
print(f"RPM 范围: {df['rpm'].min():.0f} ~ {df['rpm'].max():.0f}, 均值={df['rpm'].mean():.0f}")

corr = df['rpm'].corr(df['harm_cycles'])
print(f"\nRPM vs harm_cycles 相关系数: {corr:.6f}")

print(f"\n理论关系: 如果 512点 ≈ 7周期 @ 13000RPM")
print(f"  每 RPM 变化 = 7 / 13000 = {7/13000:.6f} 周期/RPM")

devices = sorted(df['device_id'].dropna().unique())
colors = {d: c for d, c in zip(devices, ['#2196F3', '#FF5722', '#4CAF50'])}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) RPM vs cycles 总散点
ax = axes[0]
for dev in devices:
    sub = df[df['device_id'] == dev]
    ax.scatter(sub['rpm'], sub['harm_cycles'], c=colors[dev], s=3, alpha=0.3, label=f'设备{dev[-4:]}')
coeffs = np.polyfit(df['rpm'], df['harm_cycles'], 1)
rpm_range = np.linspace(df['rpm'].min(), df['rpm'].max(), 100)
ax.plot(rpm_range, np.polyval(coeffs, rpm_range), 'r--', linewidth=1.5,
        label=f'斜率={coeffs[0]:.6f}')
ax.set_xlabel('RPM', fontsize=11)
ax.set_ylabel('harm_cycles', fontsize=11)
ax.set_title('RPM vs 周期数', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, markerscale=4)
ax.grid(True, alpha=0.3)

# (b) 按电压
ax = axes[1]
for v in sorted(df['actual_voltage'].unique()):
    sub = df[df['actual_voltage'] == v]
    ax.scatter(sub['rpm'], sub['harm_cycles'], s=3, alpha=0.3, label=f'{v:+.0f}V')
ax.set_xlabel('RPM', fontsize=11)
ax.set_ylabel('harm_cycles', fontsize=11)
ax.set_title('RPM vs 周期数 (按电压)', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, markerscale=3, ncol=2)
ax.grid(True, alpha=0.3)

# (c) RPM vs 电压
ax = axes[2]
for dev in devices:
    sub = df[df['device_id'] == dev]
    ax.scatter(sub['actual_voltage'], sub['rpm'], c=colors[dev], s=3, alpha=0.3, label=f'设备{dev[-4:]}')
ax.set_xlabel('电压 (V)', fontsize=11)
ax.set_ylabel('RPM', fontsize=11)
ax.set_title('RPM vs 电压（负载影响）', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, markerscale=4)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = r'd:\project\work\swa\swa\scripts\rpm_vs_cycles.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\n已保存: {out}")
plt.close(fig)

# 每设备拟合
print(f"\n=== 每设备 RPM vs harm_cycles ===")
for dev in devices:
    sub = df[df['device_id'] == dev]
    c = np.polyfit(sub['rpm'], sub['harm_cycles'], 1)
    r = sub['rpm'].corr(sub['harm_cycles'])
    print(f"  设备{dev[-4:]}: slope={c[0]:.6f}, intercept={c[1]:.4f}, r={r:.4f}")

# RPM 分箱
print(f"\n=== RPM 分箱 × 周期数 ===")
df['rpm_bin'] = pd.cut(df['rpm'], bins=[12300, 12700, 13000, 13300, 13600])
print(df.groupby('rpm_bin')['harm_cycles'].agg(['count', 'mean', 'std', 'min', 'max']).to_string())

# 关键：同一个 RPM 下周期数的离散度
print(f"\n=== 关键验证 ===")
print(f"  去掉 RPM 均值之后的残差周期: harm_cycles - rpm × (7/13000)")
df['cycles_from_rpm'] = df['rpm'] * (7 / 13000)
df['cycles_residual'] = df['harm_cycles'] - df['cycles_from_rpm']
print(f"  理论周期 均值={df['cycles_from_rpm'].mean():.4f}")
print(f"  实际周期 均值={df['harm_cycles'].mean():.4f}")
print(f"  残差 std={df['cycles_residual'].std():.4f}")
print(f"  残差 范围=[{df['cycles_residual'].min():.4f}, {df['cycles_residual'].max():.4f}]")
