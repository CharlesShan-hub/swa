"""
RPM 为什么有用？—— 看 RPM 对 score 和 alpha7 的影响

score 用固定 7.0/8.1 周期投影，RPM 变化导致实际周期偏离 7.0，
这个偏离会被 score 捕捉到，这就是 RPM 的信息来源。

但 RPM 可能还有别的作用：同电压下 RPM 波动 → A1 波动？
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

from swa.core.scoring import compute_score, compute_alpha7

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
sample = rows[::2]  # 1/2 采样加速
print(f"采样: {len(sample)} 条")

results = []
for rid, voltage, dev_id, rpm, a1, a1err, temp, humid, wave_str in sample:
    wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
    if len(wave) < 20:
        continue

    feats = {}

    # 重建 score 和 alpha7
    alpha7 = compute_alpha7(wave)
    feats["alpha7"] = alpha7 if alpha7 is not None else 0.0
    feats["score"] = compute_score(wave)

    # 看 score 的两个分量
    y = wave - np.mean(wave)
    n = len(y)
    t = np.arange(n)
    # 7.0 周期
    c7 = np.cos(2 * np.pi * t * 7.0 / n)
    s7 = np.sin(2 * np.pi * t * 7.0 / n)
    a_c7 = 2 * np.sum(y * c7) / n
    a_s7 = 2 * np.sum(y * s7) / n
    feats["proj_7"] = np.sqrt(a_c7**2 + a_s7**2)
    # 8.1 周期
    c81 = np.cos(2 * np.pi * t * 8.1 / n)
    s81 = np.sin(2 * np.pi * t * 8.1 / n)
    a_c81 = 2 * np.sum(y * c81) / n
    a_s81 = 2 * np.sum(y * s81) / n
    feats["proj_81"] = np.sqrt(a_c81**2 + a_s81**2)

    feats["id"] = rid
    feats["voltage"] = voltage
    feats["device_id"] = dev_id[-4:] if dev_id else "?"
    feats["rpm"] = rpm
    feats["harm_a1"] = a1 if a1 else 0.0
    feats["error"] = a1err if a1err else 0.0
    feats["temperature"] = temp if temp else 0.0
    feats["humidity"] = humid if humid else 0.0
    results.append(feats)

df = pd.DataFrame(results)
print(f"解析: {len(df)} 条")

# ── 核心分析 ──
# RPM vs score
r_s = df['rpm'].corr(df['score'])
print(f"\nRPM vs score:         r = {r_s:.6f}")

# RPM vs proj_7 (7.0周期投影幅值)
r_p7 = df['rpm'].corr(df['proj_7'])
print(f"RPM vs proj_7 (7.0):  r = {r_p7:.6f}")

# RPM vs proj_81 (8.1周期投影幅值)
r_p81 = df['rpm'].corr(df['proj_81'])
print(f"RPM vs proj_81 (8.1): r = {r_p81:.6f}")

# RPM vs harm_a1
r_a1 = df['rpm'].corr(df['harm_a1'])
print(f"RPM vs harm_a1:       r = {r_a1:.6f}")

# 重点：固定电压下，RPM 波动对 A1 有没有影响
print(f"\n=== 固定电压下 RPM 对 A1 的影响 ===")
devices = sorted(df['device_id'].unique())
for dev in devices:
    sub = df[df['device_id'] == dev]
    print(f"\n  设备{dev}:")
    for v in sorted(sub['voltage'].unique()):
        vs = sub[sub['voltage'] == v]
        if len(vs) < 20:
            continue
        c = vs['rpm'].corr(vs['harm_a1'])
        print(f"    {v:+.0f}V (n={len(vs):>4d}): RPM~A1 r={c:+.4f}")

# 画图
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
colors = {'2539': '#2196F3', '253D': '#FF5722', '6A39': '#4CAF50'}

# (a) RPM vs score
ax = axes[0, 0]
for dev in devices:
    sub = df[df['device_id'] == dev]
    ax.scatter(sub['rpm'], sub['score'], c=colors.get(dev, 'gray'), s=4, alpha=0.3, label=f'设备{dev}')
ax.set_xlabel('RPM', fontsize=11)
ax.set_ylabel('score', fontsize=11)
ax.set_title(f'RPM vs score (r={r_s:.4f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, markerscale=4)
ax.grid(True, alpha=0.3)

# (b) RPM vs proj_7
ax = axes[0, 1]
for dev in devices:
    sub = df[df['device_id'] == dev]
    ax.scatter(sub['rpm'], sub['proj_7'], c=colors.get(dev, 'gray'), s=4, alpha=0.3, label=f'设备{dev}')
ax.set_xlabel('RPM', fontsize=11)
ax.set_ylabel('7.0周期投影幅值', fontsize=11)
ax.set_title(f'RPM vs 7.0周期投影 (r={r_p7:.4f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, markerscale=4)
ax.grid(True, alpha=0.3)

# (c) RPM vs proj_81
ax = axes[0, 2]
for dev in devices:
    sub = df[df['device_id'] == dev]
    ax.scatter(sub['rpm'], sub['proj_81'], c=colors.get(dev, 'gray'), s=4, alpha=0.3, label=f'设备{dev}')
ax.set_xlabel('RPM', fontsize=11)
ax.set_ylabel('8.1周期投影幅值', fontsize=11)
ax.set_title(f'RPM vs 8.1周期投影 (r={r_p81:.4f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, markerscale=4)
ax.grid(True, alpha=0.3)

# (d) 固定电压：RPM vs A1 (取80V为例)
ax = axes[1, 0]
for dev in devices:
    sub = df[(df['device_id'] == dev) & (df['voltage'] == 80)]
    if len(sub) > 5:
        ax.scatter(sub['rpm'], sub['harm_a1'], c=colors.get(dev, 'gray'), s=10, alpha=0.5, label=f'设备{dev}')
ax.set_xlabel('RPM', fontsize=11)
ax.set_ylabel('A1 (80V)', fontsize=11)
ax.set_title('80V下 RPM vs A1', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, markerscale=4)
ax.grid(True, alpha=0.3)

# (e) RPM vs 温度
ax = axes[1, 1]
for dev in devices:
    sub = df[df['device_id'] == dev]
    ax.scatter(sub['temperature'], sub['rpm'], c=colors.get(dev, 'gray'), s=4, alpha=0.3, label=f'设备{dev}')
ax.set_xlabel('温度 (°C)', fontsize=11)
ax.set_ylabel('RPM', fontsize=11)
ax.set_title('温度 vs RPM', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, markerscale=4)
ax.grid(True, alpha=0.3)

# (f) RPM 分布
ax = axes[1, 2]
for dev in devices:
    sub = df[df['device_id'] == dev]
    ax.hist(sub['rpm'], bins=40, alpha=0.4, label=f'设备{dev}', color=colors.get(dev))
ax.set_xlabel('RPM', fontsize=11)
ax.set_ylabel('频次', fontsize=11)
ax.set_title('RPM 分布', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = r'd:\project\work\swa\swa\scripts\rpm_vs_score.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\n已保存: {out}")
plt.close(fig)

# 总结
print(f"\n{'='*60}")
print(f"RPM 总结:")
print(f"  - 与所有主变量（电压/A1/温湿度）几乎零相关")
print(f"  - 在固定电压下，与 A1 的局部相关性也很弱")
print(f"  - 但与 score/投影幅值有微弱关系")
print(f"  - RPM 对模型贡献可能来自捕捉设备个体差异")
print(f"{'='*60}")
