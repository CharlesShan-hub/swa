"""
削波矫正算法 — 在计算谐波时检测并矫正削波对 A1 的影响

方法:
  1. FFT 找到基频位置和初始 A1
  2. 检测削波（连续平坦区域）
  3. 用 FFT 拟合的正弦波替换削波区域的采样值
  4. 重新计算 FFT 得到矫正后的 A1

返回 (a1_corrected, clip_ratio, correction_pct)
  - a1_corrected: 矫正后的 A1
  - clip_ratio: 削波占比（0=无削波）
  - correction_pct: A1 提升百分比
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import sqlite3
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt


def correct_clipping_a1(wave: np.ndarray) -> tuple[float, float, float]:
    """检测削波并矫正 A1。

    Args:
        wave: 原始 512 点波形

    Returns:
        (a1_corrected, clip_ratio, correction_pct)
        a1_corrected: 矫正后的 A1
        clip_ratio: 削波采样点占比
        correction_pct: (a1_corrected - a1_original) / a1_original * 100
    """
    y = wave - np.mean(wave)
    n = len(y)

    # 1. FFT 找基频
    fft_vals = np.fft.rfft(y)
    mag = np.abs(fft_vals)
    search_end = min(len(mag) - 1, n // 3)
    fund_bin = int(np.argmax(mag[1:search_end]) + 1)
    a1_original = mag[fund_bin]

    # 2. 重建拟合正弦波
    phase = np.angle(fft_vals[fund_bin])
    t = np.arange(n)
    fit = a1_original * np.cos(2 * np.pi * fund_bin * t / n + phase) / n * 2

    # 3. 检测削波：在峰值附近找连续平坦区域
    diff = np.abs(np.diff(y))
    typical_slope = np.percentile(diff[diff > 0], 75)
    flat_threshold = typical_slope * 0.02

    is_flat = np.concatenate([diff < flat_threshold, [False]])

    # 只保留在半周期峰值附近（排除过零点）
    signs = np.sign(y)
    zero_idx = np.where(np.diff(signs) != 0)[0]
    valid_flat = np.zeros(n, dtype=bool)
    for i in range(len(zero_idx) - 1):
        seg_start = zero_idx[i]
        seg_end = zero_idx[i + 1]
        if seg_end - seg_start < 5:
            continue
        # 半周期中间 60%
        margin = int((seg_end - seg_start) * 0.2)
        flat_zone_start = max(seg_start + margin, seg_start)
        flat_zone_end = min(seg_end - margin, seg_end)
        if flat_zone_start < flat_zone_end:
            valid_flat[flat_zone_start:flat_zone_end] = (
                is_flat[flat_zone_start:flat_zone_end]
            )

    # 检查是否有连续 3+ 平坦点
    padded = np.concatenate([[False], valid_flat, [False]])
    run_starts = np.where(~padded[:-1] & padded[1:])[0]
    run_ends = np.where(padded[:-1] & ~padded[1:])[0]

    significant_clip = any(end - start >= 3 for start, end in zip(run_starts, run_ends))

    if not significant_clip:
        return a1_original, 0.0, 0.0

    clip_ratio = float(np.sum(valid_flat)) / n

    # 4. 矫正：用 FFT 拟合的正弦替换削波区域
    corrected = y.copy()
    for start, end in zip(run_starts, run_ends):
        if end - start >= 3:
            corrected[start:end] = fit[start:end]

    # 5. 重新 FFT
    fft_corrected = np.fft.rfft(corrected)
    a1_corrected = np.abs(fft_corrected)[fund_bin]

    # 防止过度矫正（如果矫正后偏差太大，可能原 A1 更准）
    correction_pct = (a1_corrected - a1_original) / a1_original * 100

    return a1_corrected, clip_ratio, correction_pct


# ── 全量测试 ──
DB_PATH = r"d:\project\work\swa\swa\src\data\projects\new\data.db"
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("""
    SELECT r.id, r.actual_voltage, r.device_id, r.harm_a1,
           w.wave_data
    FROM records r
    JOIN waveforms w ON w.record_id = r.id
    WHERE r.enabled=1 AND r.actual_voltage>=0
    ORDER BY r.id
""")
rows = cur.fetchall()
conn.close()

print(f"总记录: {len(rows)}")

# 采样检测（每 3 条取 1 条）
results = []
for rid, voltage, dev_id, a1_db, wave_str in rows[::3]:
    try:
        wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
    except (ValueError, TypeError):
        continue
    if len(wave) < 50:
        continue

    a1_corr, clip_ratio, corr_pct = correct_clipping_a1(wave)

    results.append({
        "id": rid,
        "voltage": float(voltage),
        "device_id": dev_id[-4:] if dev_id else "?",
        "a1_orig": float(a1_db) if a1_db else 0.0,
        "a1_corr": a1_corr,
        "clip_ratio": clip_ratio,
        "corr_pct": corr_pct,
    })

import pandas as pd
df = pd.DataFrame(results)
print(f"\n解析: {len(df)} 条, 有削波: {(df['clip_ratio'] > 0).sum()} 条")

# ── 统计 ──
clipped = df[df["clip_ratio"] > 0]
if len(clipped) > 0:
    print(f"\n=== 削波矫正效果 ===")
    print(f"  平均矫正幅度: {clipped['corr_pct'].mean():+.4f}%")
    print(f"  最大矫正幅度: {clipped['corr_pct'].max():+.4f}%")
    print(f"  中位矫正幅度: {clipped['corr_pct'].median():+.4f}%")
    print(f"  矫正 >1% 的记录: {(clipped['corr_pct'].abs() > 1).sum()}")
    print(f"    >2%: {(clipped['corr_pct'].abs() > 2).sum()}")
    print(f"    >5%: {(clipped['corr_pct'].abs() > 5).sum()}")

print(f"\n=== 按设备 ===")
for dev in sorted(df["device_id"].unique()):
    sub = df[df["device_id"] == dev]
    n_clip = (sub["clip_ratio"] > 0).sum()
    avg_corr = sub[sub["clip_ratio"] > 0]["corr_pct"].mean() if n_clip > 0 else 0
    print(f"  设备{dev}: 总数={len(sub):>5d}, 削波={n_clip:>4d} ({n_clip/len(sub)*100:5.1f}%), "
          f"平均矫正={avg_corr:+.4f}%")

print(f"\n=== 按电压 ===")
for v in sorted(df["voltage"].unique()):
    sub = df[df["voltage"] == v]
    n_clip = (sub["clip_ratio"] > 0).sum()
    avg_corr = sub[sub["clip_ratio"] > 0]["corr_pct"].mean() if n_clip > 0 else 0
    print(f"  {v:+.0f}V: 总数={len(sub):>4d}, 削波={n_clip:>4d} ({n_clip/len(sub)*100:5.1f}%), "
          f"平均矫正={avg_corr:+.4f}%")

# 散点图：clip_ratio vs correction_pct
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.scatter(clipped["clip_ratio"] * 100, clipped["corr_pct"], s=4, alpha=0.4)
ax.set_xlabel("削波占比 (%)")
ax.set_ylabel("A1 矫正幅度 (%)")
ax.set_title("削波占比 vs A1 矫正幅度", fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[1]
for dev in sorted(df["device_id"].unique()):
    sub = df[(df["device_id"] == dev) & (df["clip_ratio"] > 0)]
    if len(sub) > 0:
        ax.scatter(sub["clip_ratio"] * 100, sub["corr_pct"],
                   s=4, alpha=0.4, label=f"设备{dev}")
ax.set_xlabel("削波占比 (%)")
ax.set_ylabel("A1 矫正幅度 (%)")
ax.set_title("按设备分", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = r"d:\project\work\swa\swa\scripts\clipping_correction.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n已保存: {out}")
plt.close(fig)

# ── 画几个矫正前后的波形对比 ──
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 取矫正幅度最大的 4 条
top_corr = clipped.nlargest(4, "corr_pct")
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
axes = axes.flatten()

for idx, (_, row) in enumerate(top_corr.iterrows()):
    rid = int(row["id"])
    cur.execute("SELECT wave_data FROM waveforms WHERE record_id = ?", (rid,))
    wrow = cur.fetchone()
    if wrow is None:
        continue
    wave = np.array([float(x) for x in wrow[0].split(",")], dtype=np.float64)

    ax = axes[idx]
    y = wave - np.mean(wave)
    n = len(y)

    # FFT 原始
    fft_vals = np.fft.rfft(y)
    mag = np.abs(fft_vals)
    fund_bin = int(np.argmax(mag[1:n//3]) + 1)
    phase = np.angle(fft_vals[fund_bin])
    t = np.arange(n)
    a1_orig = mag[fund_bin]
    fit_orig = a1_orig * np.cos(2 * np.pi * fund_bin * t / n + phase) / n * 2

    # 矫正
    a1_corr, clip_r, _ = correct_clipping_a1(wave)
    fit_corr = a1_corr * np.cos(2 * np.pi * fund_bin * t / n + phase) / n * 2

    # 标注削波区域
    diff = np.abs(np.diff(y))
    typical_slope = np.percentile(diff[diff > 0], 75)
    is_flat = np.concatenate([diff < typical_slope * 0.02, [False]])
    padded = np.concatenate([[False], is_flat, [False]])
    starts = np.where(~padded[:-1] & padded[1:])[0]
    ends = np.where(padded[:-1] & ~padded[1:])[0]

    ax.plot(wave, "b-", linewidth=1, label="原始")
    ax.plot(fit_orig + np.mean(wave), "r--", linewidth=1, alpha=0.5, label=f"原始A1={a1_orig:.0f}")
    ax.plot(fit_corr + np.mean(wave), "g:", linewidth=1.5, alpha=0.7, label=f"矫正A1={a1_corr:.0f}")
    for s, e in zip(starts, ends):
        if e - s >= 3:
            ax.axvspan(s, e, alpha=0.2, color="red")
    ax.set_title(f"ID={rid} {row['voltage']:+.0f}V 矫正{row['corr_pct']:+.2f}%",
                 fontsize=10)
    ax.set_xlabel("采样点", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

conn.close()
plt.tight_layout()
out2 = r"d:\project\work\swa\swa\scripts\clipping_correction_examples.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
print(f"已保存: {out2}")
plt.close(fig)
