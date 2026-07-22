"""
探索噪声与电压的关系：
- 各电压等级的波形噪声（std、noise_pct）分布
- 能否通过噪声特征做映射校正
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

DB_PATH = r"d:\project\work\swa\swa\src\data\projects\new\data.db"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("""
    SELECT r.id, r.actual_voltage, r.device_id, r.harm_a1, 
           r.harm_noise_pct, r.harm_thd,
           r.temperature, r.humidity, r.rpm, r.enabled,
           w.wave_data
    FROM records r
    JOIN waveforms w ON w.record_id = r.id
    WHERE r.enabled = 1 AND r.actual_voltage >= 0
      AND r.device_id IS NOT NULL
    ORDER BY r.id
""", conn)
conn.close()

print(f"总记录: {len(df)}")

# ── 1. 从波形计算噪声指标 ─────────────────────────────────────
def compute_wave_stats(wave_str):
    try:
        wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
    except Exception:
        return None
    if len(wave) < 20:
        return None
    std = float(np.std(wave))
    pkpk = float(np.max(wave) - np.min(wave))
    mean = float(np.mean(wave))
    # 去均值后的RMS（信号功率）
    ac_rms = float(np.sqrt(np.mean((wave - mean)**2)))
    # 峭度 (kurtosis)
    kurt = float(np.mean((wave - mean)**4) / (np.std(wave)**4)) if np.std(wave) > 0 else 0
    return {"wave_std": std, "wave_pkpk": pkpk, "wave_mean": mean, "wave_kurtosis": kurt, "wave_ac_rms": ac_rms}

stats = []
for _, row in df.iterrows():
    s = compute_wave_stats(row["wave_data"])
    if s:
        s["id"] = row["id"]
        s["actual_voltage"] = row["actual_voltage"]
        s["device_id"] = row["device_id"][-4:]  # 取后4位
        s["harm_a1"] = row["harm_a1"]
        s["harm_noise_pct"] = row["harm_noise_pct"]
        s["harm_thd"] = row["harm_thd"]
        stats.append(s)

df_stats = pd.DataFrame(stats)
print(f"有波形统计的记录: {len(df_stats)}")

# ── 2. 各电压等级的噪声分布 ───────────────────────────────────
print(f"\n{'=' * 70}")
print("各电压等级的波形噪声分布")
print("=" * 70)
for v in sorted(df_stats["actual_voltage"].unique()):
    sub = df_stats[df_stats["actual_voltage"] == v]
    print(f"\n  V={v:+.0f}  n={len(sub)}")
    print(f"    wave_std:   均值={sub['wave_std'].mean():.4f}  中位数={sub['wave_std'].median():.4f}  P90={sub['wave_std'].quantile(0.9):.4f}")
    print(f"    harm_a1:    均值={sub['harm_a1'].mean():.1f}  中位数={sub['harm_a1'].median():.1f}")
    print(f"    noise_pct:  均值={sub['harm_noise_pct'].mean():.3f}  中位数={sub['harm_noise_pct'].median():.3f}" if sub['harm_noise_pct'].notna().any() else "    noise_pct:  N/A")
    print(f"    wave_kurt:  均值={sub['wave_kurtosis'].mean():.2f}  中位数={sub['wave_kurtosis'].median():.2f}")
    print(f"    A1/std比:   均值={(sub['harm_a1'] / sub['wave_std']).mean():.1f}")

# ── 3. 噪声与 A1 的关系（按电压分层） ──────────────────────────
print(f"\n\n{'=' * 70}")
print("噪声 vs A1 的相关性（按电压分层）")
print("=" * 70)
for v in sorted(df_stats["actual_voltage"].unique()):
    sub = df_stats[df_stats["actual_voltage"] == v]
    if len(sub) < 20:
        continue
    # A1与wave_std的相关系数
    corr = sub["harm_a1"].corr(sub["wave_std"])
    corr_noise = sub["harm_a1"].corr(sub["harm_noise_pct"]) if sub["harm_noise_pct"].notna().any() else 0
    # A1 / std 的变异系数
    ratio = (sub["harm_a1"] / sub["wave_std"]).values
    print(f"  V={v:+.0f}  corr(A1,std)={corr:.3f}  corr(A1,noise)={corr_noise:.3f}  A1/std均值={np.mean(ratio):.1f}±{np.std(ratio):.1f}")

# ── 4. 噪声校正想法：是否可以用 wave_std 校正 A1？ ────────────
print(f"\n\n{'=' * 70}")
print("噪声校正探索：是否可以用 std 修正 A1？")
print("对于每个电压，拟合 A1_corrected = A1 - k*(std - std_median)")
print("=" * 70)

for v in sorted(df_stats["actual_voltage"].unique()):
    sub = df_stats[df_stats["actual_voltage"] == v].copy()
    if len(sub) < 20:
        continue
    
    std_median = sub["wave_std"].median()
    a1_median = sub["harm_a1"].median()
    
    # 按std分两组：低噪声 vs 高噪声
    low_noise = sub[sub["wave_std"] <= std_median]
    high_noise = sub[sub["wave_std"] > std_median]
    
    print(f"\n  V={v:+.0f}")
    print(f"    A1中位数: {a1_median:.1f}")
    print(f"    低噪声(n={len(low_noise)}): A1均值={low_noise['harm_a1'].mean():.1f}  std均值={low_noise['wave_std'].mean():.4f}")
    print(f"    高噪声(n={len(high_noise)}): A1均值={high_noise['harm_a1'].mean():.1f}  std均值={high_noise['wave_std'].mean():.4f}")
    delta = high_noise['harm_a1'].mean() - low_noise['harm_a1'].mean()
    print(f"    A1差值(高噪声-低噪声): {delta:.1f}")
    
    # 线性拟合: A1 = a + b*std
    X = np.column_stack([np.ones(len(sub)), sub["wave_std"].values])
    y = sub["harm_a1"].values
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(f"    A1 = {coeffs[0]:.1f} + {coeffs[1]:.1f} × std")
    print(f"    校正公式: A1_corr = A1 - {coeffs[1]:.1f} × (std - {std_median:.4f})")

# ── 5. 画图 ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# 图1: 各电压的 std 分布（箱线图）
ax = axes[0, 0]
data_by_v = [df_stats[df_stats["actual_voltage"] == v]["wave_std"].values for v in sorted(df_stats["actual_voltage"].unique())]
ax.boxplot(data_by_v, labels=[f"{v:.0f}V" for v in sorted(df_stats["actual_voltage"].unique())])
ax.set_title("各电压波形标准差分布")
ax.set_ylabel("wave_std")
ax.tick_params(axis="x", rotation=45)

# 图2: A1 vs std 散点图（按电压着色）
ax = axes[0, 1]
colors = plt.cm.viridis(np.linspace(0, 1, len(df_stats["actual_voltage"].unique())))
for i, v in enumerate(sorted(df_stats["actual_voltage"].unique())):
    sub = df_stats[df_stats["actual_voltage"] == v]
    ax.scatter(sub["wave_std"], sub["harm_a1"], s=3, color=colors[i], label=f"{v:.0f}V", alpha=0.5)
ax.set_xlabel("wave_std")
ax.set_ylabel("harm_a1")
ax.set_title("A1 vs 波形标准差")
ax.legend(fontsize=6, ncol=2)

# 图3: A1/std 比值 vs 电压
ax = axes[0, 2]
for v in sorted(df_stats["actual_voltage"].unique()):
    sub = df_stats[df_stats["actual_voltage"] == v]
    ratio = (sub["harm_a1"] / sub["wave_std"]).values
    ax.scatter([v] * len(ratio), ratio, s=3, alpha=0.3, color="steelblue")
    ax.scatter(v, np.mean(ratio), s=50, color="red", marker="D")
ax.set_xlabel("电压 (V)")
ax.set_ylabel("A1 / std")
ax.set_title("信噪比 (A1/std) vs 电压")
ax.axhline(y=500, color="gray", linestyle="--", alpha=0.5)

# 图4: harm_noise_pct vs 电压
ax = axes[1, 0]
sub = df_stats[df_stats["harm_noise_pct"].notna()]
data_by_v2 = [sub[sub["actual_voltage"] == v]["harm_noise_pct"].values for v in sorted(sub["actual_voltage"].unique())]
ax.boxplot(data_by_v2, labels=[f"{v:.0f}V" for v in sorted(sub["actual_voltage"].unique())])
ax.set_title("harm_noise_pct 分布")
ax.tick_params(axis="x", rotation=45)

# 图5: 高噪声 vs 低噪声的 A1 差异
ax = axes[1, 1]
volts = []
deltas = []
for v in sorted(df_stats["actual_voltage"].unique()):
    sub = df_stats[df_stats["actual_voltage"] == v].copy()
    if len(sub) < 20:
        continue
    std_median = sub["wave_std"].median()
    low = sub[sub["wave_std"] <= std_median]["harm_a1"].mean()
    high = sub[sub["wave_std"] > std_median]["harm_a1"].mean()
    volts.append(v)
    deltas.append(high - low)
ax.plot(volts, deltas, "o-", color="crimson")
ax.axhline(y=0, color="gray", linestyle="--")
ax.set_xlabel("电压 (V)")
ax.set_ylabel("高噪声A1 - 低噪声A1")
ax.set_title("噪声导致的A1偏移 vs 电压")

# 图6: A1 vs std 整体趋势
ax = axes[1, 2]
for dev in sorted(df_stats["device_id"].unique()):
    sub = df_stats[df_stats["device_id"] == dev]
    ax.scatter(sub["wave_std"], sub["harm_a1"], s=3, label=f"设备{dev}", alpha=0.5)
ax.set_xlabel("wave_std")
ax.set_ylabel("harm_a1")
ax.set_title("按设备分: A1 vs std")
ax.legend(fontsize=8)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "noise_analysis.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n已保存: {out_path}")
plt.close(fig)
