"""
探索 512 点波形的非频域特征

提取各类时域/统计特征，分析它们与电压、温湿度的关系。
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import sqlite3
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from scipy import stats

DB_PATH = r"d:\project\work\swa\swa\src\data\projects\new\data.db"
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("""
    SELECT r.id, r.actual_voltage, r.device_id, r.rpm,
           r.harm_a1, r.harm_error, r.temperature, r.humidity,
           w.wave_data
    FROM records r
    JOIN waveforms w ON w.record_id = r.id
    WHERE r.enabled=1 AND r.actual_voltage>=0
    ORDER BY r.id
""")
rows = cur.fetchall()
conn.close()

print(f"总记录: {len(rows)}")

# 每 5 条取一条
sample = rows[::5]
print(f"采样: {len(sample)} 条\n")

results = []
errors = 0

for rid, voltage, dev_id, rpm, a1, a1err, temp, humid, wave_str in sample:
    try:
        wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
    except (ValueError, TypeError):
        errors += 1
        continue
    if len(wave) < 50:
        continue

    y = wave - np.mean(wave)
    n = len(y)

    # ── 1. 对称性 ──
    pos_peak = np.max(y)
    neg_peak = np.min(y)
    sym_ratio = abs(pos_peak / neg_peak) if neg_peak != 0 else 1.0  # 正负半周峰值比，理想=1
    pos_area = np.sum(y[y > 0])
    neg_area = abs(np.sum(y[y < 0]))
    area_ratio = pos_area / neg_area if neg_area > 0 else 1.0  # 正负半周面积比

    # ── 2. 峰值因子 ──
    rms = np.sqrt(np.mean(y ** 2))
    crest = np.max(np.abs(y)) / rms if rms > 0 else 0.0

    # ── 3. 波形熵（Shannon 熵） ──
    # 归一化到 [0,1] 范围，计算概率分布
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)
    hist, _ = np.histogram(y_norm, bins=50, range=(0, 1))
    prob = hist / (hist.sum() + 1e-10)
    entropy = -np.sum(prob * np.log(prob + 1e-10))

    # ── 4. 峭度 ──
    kurt = stats.kurtosis(y)

    # ── 5. 偏度 ──
    skew = stats.skew(y)

    # ── 6. 过零率 ──
    zero_crossings = np.sum(np.diff(np.sign(y)) != 0)
    # 理论值：~14次（7周期 × 2）

    # ── 7. 逐周期分析（找过零点分段） ──
    signs = np.sign(y)
    zero_indices = np.where(np.diff(signs) != 0)[0]  # 过零位置
    # 每两个过零点之间为一个半周期
    if len(zero_indices) >= 4:
        # 每两个过零点的间隔（点数）
        half_cycle_lens = np.diff(zero_indices)
        # 周期长度一致性（CV）
        cycle_len_cv = np.std(half_cycle_lens) / (np.mean(half_cycle_lens) + 1e-10)

        # 每半个周期的峰值
        half_peaks = []
        for i in range(len(zero_indices) - 1):
            seg = y[zero_indices[i] : zero_indices[i + 1] + 1]
            if len(seg) > 0:
                half_peaks.append(np.max(np.abs(seg)))
        peak_cv = np.std(half_peaks) / (np.mean(half_peaks) + 1e-10) if half_peaks else 0.0

        # 过零抖动
        zc_jitter = np.std(half_cycle_lens) if len(half_cycle_lens) >= 2 else 0.0
    else:
        cycle_len_cv = 0.0
        peak_cv = 0.0
        zc_jitter = 0.0

    # ── 8. DC 偏移 ──
    dc_offset = float(np.mean(wave))  # 原始波形的均值（不去 DC）

    # ── 9. 波形的峰峰值 ──
    peak_to_peak = float(np.ptp(y))

    row = {
        "id": rid,
        "voltage": float(voltage),
        "device_id": dev_id[-4:] if dev_id else "?",
        "harm_a1": float(a1) if a1 else 0.0,
        "temperature": float(temp) if temp else 0.0,
        "humidity": float(humid) if humid else 0.0,
        # 新特征
        "sym_ratio": sym_ratio,
        "area_ratio": area_ratio,
        "crest": crest,
        "entropy": entropy,
        "kurtosis": kurt,
        "skewness": skew,
        "zero_crossings": zero_crossings,
        "cycle_len_cv": cycle_len_cv,
        "peak_cv": peak_cv,
        "zc_jitter": zc_jitter,
        "dc_offset": dc_offset,
        "peak_to_peak": peak_to_peak,
        "rms": float(rms),
    }
    results.append(row)

df = pd.DataFrame(results)
print(f"解析成功: {len(df)}, 失败: {errors}\n")

# ── 相关性分析 ──
new_feats = [
    "sym_ratio", "area_ratio", "crest", "entropy", "kurtosis",
    "skewness", "zero_crossings", "cycle_len_cv", "peak_cv",
    "zc_jitter", "dc_offset", "peak_to_peak", "rms",
]
targets = ["voltage", "harm_a1", "temperature", "humidity"]

print(f"{'特征':>16s}", end="")
for t in targets:
    print(f"  vs {t:>10s}", end="")
print()
print("-" * 68)

for feat in new_feats:
    print(f"{feat:>16s}", end="")
    for t in targets:
        c = df[feat].corr(df[t])
        print(f"  {c:>+10.4f}", end="")
    print()

# 突出显示 |r| > 0.1 的有用特征
print(f"\n{'='*68}")
print("潜在有用特征 (|r| > 0.1 with any target):")
print(f"{'='*68}")
for feat in new_feats:
    max_r = max(abs(df[feat].corr(df[t])) for t in targets)
    if max_r > 0.1:
        print(f"  {feat:>16s}: max |r| = {max_r:.4f}")

# ── 画图：按电压看各特征的分布 ──
fig, axes = plt.subplots(4, 4, figsize=(18, 14))
axes = axes.flatten()

plot_feats = new_feats + ["harm_a1"]  # 13+1=14, 够了
for idx, feat in enumerate(plot_feats):
    if idx >= len(axes):
        break
    ax = axes[idx]

    # 按设备分类
    for dev in sorted(df["device_id"].unique()):
        sub = df[df["device_id"] == dev]
        # 按电压取均值
        v_mean = sub.groupby("voltage")[feat].mean()
        ax.plot(v_mean.index, v_mean.values, "o-", markersize=4, linewidth=1.2,
                label=f"设备{dev}")

    ax.set_xlabel("电压 (V)", fontsize=9)
    ax.set_ylabel(feat, fontsize=9)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7)

plt.tight_layout()
out = r"d:\project\work\swa\swa\scripts\wave_features_vs_voltage.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n已保存: {out}")
plt.close(fig)

# ── 画图2：按温湿度看 ──
fig, axes = plt.subplots(4, 4, figsize=(18, 14))
axes = axes.flatten()

for idx, feat in enumerate(plot_feats):
    if idx >= len(axes):
        break
    ax = axes[idx]

    # 湿度分层
    humid_groups = [
        ("<35%", df["humidity"] < 35, "royalblue"),
        ("35-45%", (df["humidity"] >= 35) & (df["humidity"] < 45), "orange"),
        (">=45%", df["humidity"] >= 45, "crimson"),
    ]
    for label, mask, color in humid_groups:
        sub = df[mask]
        if len(sub) < 10:
            continue
        v_mean = sub.groupby("voltage")[feat].mean()
        ax.plot(v_mean.index, v_mean.values, "o-", markersize=3, linewidth=1,
                color=color, label=label, alpha=0.7)

    ax.set_xlabel("电压 (V)", fontsize=9)
    ax.set_ylabel(feat, fontsize=9)
    ax.set_title(f"{feat} (按湿度)", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7)

plt.tight_layout()
out2 = r"d:\project\work\swa\swa\scripts\wave_features_by_humidity.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
print(f"已保存: {out2}")
plt.close(fig)

# ── 数值统计 ──
print(f"\n{'='*68}")
print("各特征数值范围:")
print(f"{'='*68}")
for feat in new_feats:
    vals = df[feat]
    print(f"  {feat:>16s}: [{vals.min():.4f}, {vals.max():.4f}]  mean={vals.mean():.4f}  std={vals.std():.4f}")
