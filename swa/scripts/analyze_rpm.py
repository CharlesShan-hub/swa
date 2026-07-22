"""
RPM 关系分析 — 转速到底影响了什么？

观察 RPM 是否随以下变量变化:
  1. 时间（时序漂移）
  2. 电压（负载相关）
  3. 温度（热胀冷缩）
  4. 湿度（环境）
  5. A1 / error / score（信号质量）
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
    SELECT r.id, r.actual_voltage, r.device_id,
           r.harm_a1, r.harm_error, r.harm_noise_pct, r.harm_cycles,
           r.temperature, r.humidity, r.rpm,
           r.system_time, r.enabled
    FROM records r
    WHERE r.enabled = 1 AND r.actual_voltage >= 0
      AND r.device_id IS NOT NULL AND r.rpm IS NOT NULL
    ORDER BY r.id
""", conn)
conn.close()

df["time_ord"] = np.arange(len(df))
df["error_div_a1"] = np.where(df["harm_a1"] > 1e-6, df["harm_error"] / df["harm_a1"], 0.0)
devices = sorted(df["device_id"].dropna().unique())

print(f"总记录: {len(df)}")
print(f"RPM 范围: {df['rpm'].min():.0f} ~ {df['rpm'].max():.0f}, 均值={df['rpm'].mean():.0f}")

# ── 相关性矩阵 ──
corr_cols = ["rpm", "actual_voltage", "harm_a1", "harm_error", "error_div_a1",
             "harm_noise_pct", "temperature", "humidity", "harm_cycles"]
corr = df[corr_cols].corr()
print("\n=== RPM 与其他变量的 Pearson 相关系数 ===")
for col in corr_cols:
    if col != "rpm":
        print(f"  rpm vs {col:20s}: {corr.loc['rpm', col]:+.6f}")

# ── 1. 四象限子图 ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = {d: c for d, c in zip(devices, ["#2196F3", "#FF5722", "#4CAF50"])}
markers = {d: m for d, m in zip(devices, ["o", "s", "^"])}
labels = {d: f"设备{d[-4:]}" for d in devices}

# (a) RPM vs 温度
ax = axes[0, 0]
for dev in devices:
    sub = df[df["device_id"] == dev]
    ax.scatter(sub["temperature"], sub["rpm"], c=colors[dev], marker=markers[dev],
               s=4, alpha=0.3, label=labels[dev])
ax.set_xlabel("温度 (°C)", fontsize=11)
ax.set_ylabel("RPM", fontsize=11)
ax.set_title("RPM vs 温度", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, markerscale=4)
ax.grid(True, alpha=0.3)
# 合并所有设备拟合
all_t = df["temperature"].values
all_rpm = df["rpm"].values
coeffs = np.polyfit(all_t, all_rpm, 1)
ax.plot(all_t, np.polyval(coeffs, all_t), "r--", linewidth=1,
        label=f"斜率={coeffs[0]:.2f} RPM/°C")
ax.legend(fontsize=9, markerscale=4)

# (b) RPM vs 湿度
ax = axes[0, 1]
for dev in devices:
    sub = df[df["device_id"] == dev]
    ax.scatter(sub["humidity"], sub["rpm"], c=colors[dev], marker=markers[dev],
               s=4, alpha=0.3, label=labels[dev])
ax.set_xlabel("湿度 (%)", fontsize=11)
ax.set_ylabel("RPM", fontsize=11)
ax.set_title("RPM vs 湿度", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
coeffs = np.polyfit(df["humidity"].values, all_rpm, 1)
ax.plot(df["humidity"].values, np.polyval(coeffs, df["humidity"].values), "r--", linewidth=1,
        label=f"斜率={coeffs[0]:.2f} RPM/%")
ax.legend(fontsize=9)

# (c) RPM vs 电压
ax = axes[1, 0]
for dev in devices:
    sub = df[df["device_id"] == dev]
    ax.scatter(sub["actual_voltage"], sub["rpm"], c=colors[dev], marker=markers[dev],
               s=4, alpha=0.3, label=labels[dev])
ax.set_xlabel("电压 (V)", fontsize=11)
ax.set_ylabel("RPM", fontsize=11)
ax.set_title("RPM vs 电压", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
coeffs = np.polyfit(df["actual_voltage"].values, all_rpm, 1)
ax.plot(df["actual_voltage"].values, np.polyval(coeffs, df["actual_voltage"].values), "r--", linewidth=1,
        label=f"斜率={coeffs[0]:.4f} RPM/V")
ax.legend(fontsize=9)

# (d) RPM 时序
ax = axes[1, 1]
for dev in devices:
    sub = df[df["device_id"] == dev]
    ax.scatter(sub.index, sub["rpm"], c=colors[dev], marker=markers[dev],
               s=3, alpha=0.3, label=labels[dev])
ax.set_xlabel("记录序号 (时间)", fontsize=11)
ax.set_ylabel("RPM", fontsize=11)
ax.set_title("RPM 时序变化", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "rpm_analysis.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n已保存: {out_path}")
plt.close(fig)

# ── 2. RPM 对预测的影响 ──
# 看看去掉 RPM 后跟加上 RPM 的差异
print("\n=== RPM 的分布离散度 ===")
print(f"  CV (变异系数) = {df['rpm'].std() / df['rpm'].mean() * 100:.2f}%")
print(f"  RPM 最大变化 = {df['rpm'].max() - df['rpm'].min():.0f}")
print(f"  相对于均值的变化幅度 = {(df['rpm'].max() - df['rpm'].min()) / df['rpm'].mean() * 100:.2f}%")

# 按设备看 RPM 均值和波动
print("\n=== 按设备 RPM 分布 ===")
for dev in devices:
    sub = df[df["device_id"] == dev]["rpm"]
    print(f"  设备{dev[-4:]}: 均值={sub.mean():.0f}, "
          f"std={sub.std():.1f}, "
          f"CV={sub.std()/sub.mean()*100:.2f}%, "
          f"范围=[{sub.min():.0f}, {sub.max():.0f}]")

# 按温湿度区间看 RPM
df["temp_bin"] = pd.cut(df["temperature"], bins=[-10, 10, 20, 30, 50])
df["humid_bin"] = pd.cut(df["humidity"], bins=[0, 30, 40, 50, 100])
print("\n=== RPM × 温度区间 ===")
print(df.groupby("temp_bin")["rpm"].agg(["count", "mean", "std"]).to_string())
print("\n=== RPM × 湿度区间 ===")
print(df.groupby("humid_bin")["rpm"].agg(["count", "mean", "std"]).to_string())
