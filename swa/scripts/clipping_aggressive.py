"""
激进削波检测与矫正 — 直接看波峰偏离 FFT 拟合的程度

方法:
  FFT 拟合出理想正弦波 → 每个半周期峰值附近对比原始 vs 拟合
  如果原始波形的峰值区域"塌陷"（原始 < 拟合），则说明有削波

  比之前更激进: 不要求"连续平坦", 只要有系统性被压低也算削波
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


def aggressive_clip_detect(wave: np.ndarray) -> tuple[float, float, float]:
    """激进削波检测 + 矫正

    1. FFT 拟合正弦波
    2. 每个半周期峰值附近 ±5点，计算原始/拟合的比值
    3. 如果原始系统性小于拟合（比值 < 0.98），判定为削波
    4. 用 FFT 拟合值替换削波区域，重新算 A1

    Returns:
        (a1_corrected, clip_ratio, correction_pct)
    """
    y = wave - np.mean(wave)
    n = len(y)

    # 1. FFT 拟合
    fft_vals = np.fft.rfft(y)
    mag = np.abs(fft_vals)
    fund_bin = int(np.argmax(mag[1:n//3]) + 1)
    a1_orig = float(mag[fund_bin])
    phase = np.angle(fft_vals[fund_bin])
    t = np.arange(n)
    fit = a1_orig * np.cos(2 * np.pi * fund_bin * t / n + phase) / n * 2

    # 2. 过零点分割半周期
    signs = np.sign(y)
    zero_idx = np.where(np.diff(signs) != 0)[0]
    if len(zero_idx) < 2:
        return a1_orig, 0.0, 0.0

    is_clipped = np.zeros(n, dtype=bool)
    clipped_halves = 0

    for i in range(len(zero_idx) - 1):
        seg_start = zero_idx[i]
        seg_end = zero_idx[i + 1]
        if seg_end - seg_start < 5:
            continue

        seg_orig = np.abs(y[seg_start:seg_end+1])
        seg_fit = np.abs(fit[seg_start:seg_end+1])

        # 找这个半周期的峰值位置
        peak_local = np.argmax(seg_orig)
        peak_pos = seg_start + peak_local

        # 以峰值为中心，取 ±5 点（或半周期长度的 1/3，取较小值）
        half_win = min(5, (seg_end - seg_start) // 3)
        win_start = max(seg_start, peak_pos - half_win)
        win_end = min(seg_end, peak_pos + half_win)

        if win_end - win_start < 3:
            continue

        # 峰值区域原始 vs 拟合的比值
        orig_peak = np.max(np.abs(y[win_start:win_end+1]))
        fit_peak = np.max(np.abs(fit[win_start:win_end+1]))

        if orig_peak > 0 and fit_peak > 0:
            ratio = orig_peak / fit_peak
        else:
            ratio = 1.0

        # 激进判定: 原始峰值 < 拟合峰值的 98%
        if ratio < 0.98:
            clipped_halves += 1
            # 标记这个半周期的峰值区域为削波
            is_clipped[win_start:win_end+1] = True

    clip_ratio = float(np.sum(is_clipped)) / n
    significant = clipped_halves >= 2  # 至少 2 个半周期有削波

    if not significant:
        return a1_orig, 0.0, 0.0

    # 3. 矫正
    corrected = y.copy()
    corrected[is_clipped] = fit[is_clipped]

    # 重新算 A1
    fft_corr = np.fft.rfft(corrected)
    a1_corr = float(np.abs(fft_corr)[fund_bin])
    corr_pct = (a1_corr - a1_orig) / a1_orig * 100

    return a1_corr, clip_ratio, corr_pct


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

import pandas as pd

results = []
for rid, voltage, dev_id, a1_db, wave_str in rows[::2]:
    try:
        wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
    except (ValueError, TypeError):
        continue
    if len(wave) < 50:
        continue

    a1_corr, clip_ratio, corr_pct = aggressive_clip_detect(wave)
    results.append({
        "id": rid,
        "voltage": float(voltage),
        "device_id": dev_id[-4:] if dev_id else "?",
        "a1_orig": float(a1_db) if a1_db else 0.0,
        "a1_corr": a1_corr,
        "clip_ratio": clip_ratio,
        "corr_pct": corr_pct,
    })

df = pd.DataFrame(results)
clipped = df[df["clip_ratio"] > 0]
print(f"解析: {len(df)} 条, 检测到削波: {len(clipped)} ({len(clipped)/len(df)*100:.1f}%)")

if len(clipped) > 0:
    print(f"\n=== 矫正效果 ===")
    print(f"  平均矫正: {clipped['corr_pct'].mean():+.4f}%")
    print(f"  最大矫正: {clipped['corr_pct'].max():+.4f}%")
    print(f"  矫正>1%: {(clipped['corr_pct'] > 1).sum()} 条")
    print(f"  >2%: {(clipped['corr_pct'] > 2).sum()} 条")
    print(f"  >3%: {(clipped['corr_pct'] > 3).sum()} 条")
    print(f"  >5%: {(clipped['corr_pct'] > 5).sum()} 条")

print(f"\n=== 按电压 ===")
for v in sorted(df["voltage"].unique()):
    sub = df[df["voltage"] == v]
    n_clip = (sub["clip_ratio"] > 0).sum()
    avg_c = sub[sub["clip_ratio"] > 0]["corr_pct"].mean() if n_clip > 0 else 0
    print(f"  {v:+.0f}V: n={len(sub):>4d}, 削波={n_clip:>4d} ({n_clip/len(sub)*100:5.1f}%), 平均矫正={avg_c:+.4f}%")

print(f"\n=== 按设备 ===")
for dev in sorted(df["device_id"].unique()):
    sub = df[df["device_id"] == dev]
    n_clip = (sub["clip_ratio"] > 0).sum()
    avg_c = sub[sub["clip_ratio"] > 0]["corr_pct"].mean() if n_clip > 0 else 0
    print(f"  设备{dev}: n={len(sub):>4d}, 削波={n_clip:>4d} ({n_clip/len(sub)*100:5.1f}%), 平均矫正={avg_c:+.4f}%")

# 看 42986 的具体数值
row_42986 = df[df["id"] == 42986]
if len(row_42986) > 0:
    r = row_42986.iloc[0]
    print(f"\n=== ID=42986 详情 ===")
    print(f"  原始A1={r['a1_orig']:.2f}  矫正A1={r['a1_corr']:.2f}  矫正={r['corr_pct']:+.4f}%")
    print(f"  削波占比={r['clip_ratio']*100:.2f}%")
