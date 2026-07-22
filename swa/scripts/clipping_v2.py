"""
激进削波 V2 — 组合检测法

1. 平坦检测（原始方法，降低阈值到 5%）
2. 形状检测（峰值区域原始 < 拟合，但只在较高 A1 时启用，避免低电压误检）
3. 两者任一触发即判定削波
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


def detect_and_correct(wave: np.ndarray) -> tuple[float, float, float]:
    """激进削波检测 V2

    检测方法（组合）:
    1. 平坦检测: 相邻点差值 < 典型斜率 × 5%，连续 3+ 点
    2. 形状检测: 峰值附近原始/拟合比值 < 0.97 (仅 A1 较高时启用)
    """
    y = wave - np.mean(wave)
    n = len(y)

    # FFT 拟合
    fft_vals = np.fft.rfft(y)
    mag = np.abs(fft_vals)
    fund_bin = int(np.argmax(mag[1:n//3]) + 1)
    a1_orig = float(mag[fund_bin])
    phase = np.angle(fft_vals[fund_bin])
    t = np.arange(n)
    fit = a1_orig * np.cos(2 * np.pi * fund_bin * t / n + phase) / n * 2

    signs = np.sign(y)
    zero_idx = np.where(np.diff(signs) != 0)[0]

    # 平坦检测（阈值 = 5% 典型斜率）
    diff = np.abs(np.diff(y))
    typical_slope = np.percentile(diff[diff > 0], 75)
    flat_th = typical_slope * 0.05
    is_flat_raw = diff < flat_th

    # 只考虑半周期中间 60%
    valid_flat = np.zeros(n-1, dtype=bool)
    for i in range(len(zero_idx) - 1):
        s, e = zero_idx[i], zero_idx[i+1]
        if e - s < 5: continue
        margin = int((e - s) * 0.2)
        zs = max(s + margin, s)
        ze = min(e - margin, e)
        if zs < ze:
            valid_flat[zs:ze] = is_flat_raw[zs:ze]

    is_flat = np.concatenate([valid_flat, [False]])
    padded = np.concatenate([[False], valid_flat, [False]])
    runs_s = np.where(~padded[:-1] & padded[1:])[0]
    runs_e = np.where(padded[:-1] & ~padded[1:])[0]
    has_flat = any(e - s >= 3 for s, e in zip(runs_s, runs_e))

    # 形状检测（仅 A1 较高时，避免低电压噪声误检）
    # A1 的 FFT bin 值 > 100 才启用（对应约 50V 以上）
    has_shape_clip = False
    a1_threshold = a1_orig * 2 / n  # 换算到时域幅值
    if a1_threshold > 0.5:  # 时域峰值 > 0.5
        clipped_half_count = 0
        for i in range(len(zero_idx) - 1):
            s, e = zero_idx[i], zero_idx[i+1]
            if e - s < 5: continue
            peak_local = np.argmax(np.abs(y[s:e+1]))
            peak_pos = s + peak_local

            half_win = min(4, (e - s) // 3)
            ws = max(s, peak_pos - half_win)
            we = min(e, peak_pos + half_win)
            if we - ws < 3: continue

            orig_max = np.max(np.abs(y[ws:we+1]))
            fit_max = np.max(np.abs(fit[ws:we+1]))
            if fit_max > 0 and orig_max / fit_max < 0.97:
                clipped_half_count += 1

        has_shape_clip = clipped_half_count >= 3  # 至少 3 个半周期

    clipped = has_flat or has_shape_clip
    if not clipped:
        return a1_orig, 0.0, 0.0

    # 标记要替换的采样点
    to_replace = np.zeros(n, dtype=bool)
    for s, e in zip(runs_s, runs_e):
        if e - s >= 3:
            to_replace[s:e] = True
    if has_shape_clip:
        for i in range(len(zero_idx) - 1):
            s, e = zero_idx[i], zero_idx[i+1]
            if e - s < 5: continue
            peak_local = np.argmax(np.abs(y[s:e+1]))
            peak_pos = s + peak_local
            half_win = min(4, (e - s) // 3)
            ws = max(s, peak_pos - half_win)
            we = min(e, peak_pos + half_win)
            if we - ws < 3: continue
            orig_max = np.max(np.abs(y[ws:we+1]))
            fit_max = np.max(np.abs(fit[ws:we+1]))
            if fit_max > 0 and orig_max / fit_max < 0.97:
                to_replace[ws:we+1] = True

    # 矫正
    corrected = y.copy()
    corrected[to_replace] = fit[to_replace]
    fft_corr = np.fft.rfft(corrected)
    a1_corr = float(np.abs(fft_corr)[fund_bin])
    corr_pct = (a1_corr - a1_orig) / a1_orig * 100
    clip_ratio = float(np.sum(to_replace)) / n

    return a1_corr, clip_ratio, corr_pct


# ── 测试 ──
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
    if len(wave) < 50: continue
    a1_c, clip_r, corr_p = detect_and_correct(wave)
    results.append({
        "id": rid, "voltage": float(voltage),
        "device_id": dev_id[-4:] if dev_id else "?",
        "a1_orig": float(a1_db) if a1_db else 0.0,
        "a1_corr": a1_c, "clip_ratio": clip_r, "corr_pct": corr_p,
    })

df = pd.DataFrame(results)
clip = df[df["clip_ratio"] > 0]
print(f"解析: {len(df)}, 削波: {len(clip)} ({len(clip)/len(df)*100:.1f}%)")
if len(clip) > 0:
    print(f"平均矫正: {clip['corr_pct'].mean():+.4f}%  最大: {clip['corr_pct'].max():+.4f}%")
    print(f">1%: {(clip['corr_pct']>1).sum()}  >2%: {(clip['corr_pct']>2).sum()}  >3%: {(clip['corr_pct']>3).sum()}")

print(f"\n按电压:")
for v in sorted(df["voltage"].unique()):
    s = df[df["voltage"] == v]
    nc = (s["clip_ratio"] > 0).sum()
    ac = s[s["clip_ratio"]>0]["corr_pct"].mean() if nc > 0 else 0
    print(f"  {v:+.0f}V: n={len(s):>4d}, 削波={nc:>4d} ({nc/len(s)*100:5.1f}%), 矫正={ac:+.4f}%")

print(f"\n按设备:")
for dev in sorted(df["device_id"].unique()):
    s = df[df["device_id"] == dev]
    nc = (s["clip_ratio"] > 0).sum()
    ac = s[s["clip_ratio"]>0]["corr_pct"].mean() if nc > 0 else 0
    print(f"  设备{dev}: n={len(s):>4d}, 削波={nc:>4d} ({nc/len(s)*100:5.1f}%), 矫正={ac:+.4f}%")

r = df[df["id"] == 42986].iloc[0]
print(f"\nID=42986: A1 {r['a1_orig']:.2f}→{r['a1_corr']:.2f} ({r['corr_pct']:+.4f}%)  clip={r['clip_ratio']*100:.2f}%")
