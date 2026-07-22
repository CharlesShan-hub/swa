"""
检测波形削波（clipping）— 波峰被削平的现象

真正的削波特征是：峰值附近有连续多个几乎完全相同的采样值（ADC 饱和）。
而不是"接近峰值"——正弦波本身就有很多采样点接近峰值。
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

from swa.data.loader import compute_harmonics

DB_PATH = r"d:\project\work\swa\swa\src\data\projects\new\data.db"
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("""
    SELECT r.id, r.actual_voltage, r.device_id,
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


def detect_clipping(wave: np.ndarray) -> dict:
    """检测真正的削波——峰值附近连续平坦的采样点。

    方法:
    1. 过零点分割半周期
    2. 在每个半周期的峰值附近，计算相邻采样点的差值
    3. 如果连续 3+ 个点的差值 < 满量程的 0.1%，判定为削波
    4. 削波的"平坦部分"占比越大越严重

    对于 512 点/7周期的正弦波，峰值附近自然斜率 = 2π×7/512 × A1 ≈ 0.086 × A1
    相邻点理论差值 ≈ A1 × sin(2π×7/512) ≈ A1 × 0.086
    如果差值 < A1 × 0.001，那就是异常平坦。
    """
    y = wave - np.mean(wave)
    n = len(y)
    peak = np.max(np.abs(y))

    if peak < 0.01:
        return {"clipped": False, "clip_ratio": 0.0, "clip_severity": 0.0}

    # 相邻点差值的绝对值
    diff = np.abs(np.diff(y))

    # 正弦波的理论最大斜率（从过零点斜率估算）
    # 找过零点附近的斜率
    signs = np.sign(y)
    zero_idx = np.where(np.diff(signs) != 0)[0]
    if len(zero_idx) < 2:
        return {"clipped": False, "clip_ratio": 0.0, "clip_severity": 0.0}

    # 估计正常斜率
    typical_slope = np.percentile(diff[diff > 0], 75)  # 正常点的典型斜率

    # 平坦阈值：斜率 < 典型斜率的 5%
    flat_threshold = typical_slope * 0.02
    if flat_threshold < 1e-6:
        return {"clipped": False, "clip_ratio": 0.0, "clip_severity": 0.0}

    # 标记平坦点
    is_flat = diff < flat_threshold

    # 只考虑在半周期峰值附近的平坦点（排除过零点附近的平坦区）
    valid_flat = np.zeros(n - 1, dtype=bool)
    for i in range(len(zero_idx) - 1):
        seg_start = zero_idx[i]
        seg_end = zero_idx[i + 1]
        if seg_end - seg_start < 5:
            continue

        seg = y[seg_start:seg_end+1]
        peak_pos = seg_start + np.argmax(np.abs(seg))
        # 半周期中间 60% 的区域（排除过零点附近）
        margin = int((seg_end - seg_start) * 0.2)
        flat_zone_start = max(seg_start + margin, seg_start)
        flat_zone_end = min(seg_end - margin, seg_end)

        if flat_zone_start < flat_zone_end:
            valid_flat[flat_zone_start:flat_zone_end] = is_flat[flat_zone_start:flat_zone_end]

    # 统计有效平坦段的连续长度
    flat_count = 0
    max_flat_run = 0
    current_run = 0
    for i in range(n - 1):
        if valid_flat[i]:
            current_run += 1
            flat_count += 1
        else:
            max_flat_run = max(max_flat_run, current_run)
            current_run = 0
    max_flat_run = max(max_flat_run, current_run)

    # 削波判定：有至少一段 3+ 点的连续平坦
    clipped = max_flat_run >= 3
    clip_ratio = flat_count / (n - 1)
    severity = flat_count / max(max_flat_run, 1)

    return {
        "clipped": clipped,
        "clip_ratio": clip_ratio,
        "clip_severity": severity,
        "max_flat_run": max_flat_run,
        "typical_slope": typical_slope,
    }


# ── 批量检测 ──
records = []
for rid, voltage, dev_id, a1, a1err, temp, humid, wave_str in rows:
    try:
        wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
    except (ValueError, TypeError):
        continue
    if len(wave) < 50:
        continue

    clip_info = detect_clipping(wave)

    records.append({
        "id": rid,
        "voltage": float(voltage),
        "device_id": dev_id[-4:] if dev_id else "?",
        "harm_a1": float(a1) if a1 else 0.0,
        "harm_error": float(a1err) if a1err else 0.0,
        "temperature": float(temp) if temp else 0.0,
        "humidity": float(humid) if humid else 0.0,
        **clip_info,
    })

df = pd.DataFrame(records)
df["error_div_a1"] = np.where(df["harm_a1"] > 1e-6, df["harm_error"] / df["harm_a1"], 0.0)
print(f"解析: {len(df)} 条")

# ── 削波统计 ──
clipped_df = df[df["clipped"]]
clean_df = df[~df["clipped"]]
print(f"\n=== 削波统计 ===")
print(f"  总记录: {len(df)}")
print(f"  有削波: {len(clipped_df)} ({len(clipped_df)/len(df)*100:.1f}%)")
print(f"  无削波: {len(clean_df)} ({len(clean_df)/len(df)*100:.1f}%)")

if len(clipped_df) > 0:
    print(f"\n  削波严重程度（削波样本中）:")
    print(f"    clip_ratio 均值: {clipped_df['clip_ratio'].mean():.6f}")
    print(f"    max_flat_run 均值: {clipped_df['max_flat_run'].mean():.2f}")
    print(f"    max_flat_run 最大: {clipped_df['max_flat_run'].max()}")

# ── 削波 vs 电压 ──
print(f"\n=== 削波按电压分布 ===")
for v in sorted(df["voltage"].unique()):
    sub = df[df["voltage"] == v]
    n_clip = sub["clipped"].sum()
    print(f"  {v:+.0f}V: {n_clip:>4d}/{len(sub):>4d} = {n_clip/len(sub)*100:5.1f}% 削波")

# ── 削波 vs 设备 ──
print(f"\n=== 削波按设备分布 ===")
for dev in sorted(df["device_id"].unique()):
    sub = df[df["device_id"] == dev]
    n_clip = sub["clipped"].sum()
    n_clip_sev = sub[sub["max_flat_run"] >= 5]["clipped"].sum()
    print(f"  设备{dev}: {n_clip:>4d}/{len(sub):>4d} = {n_clip/len(sub)*100:5.1f}% 削波")

# ── 削波与 A1 ──
print(f"\n=== 削波 vs 指标 ===")
print(f"{'指标':>20s}  {'有削波':>12s}  {'无削波':>12s}  {'差值':>12s}")
for col in ["harm_a1", "harm_error", "error_div_a1"]:
    if len(clipped_df) > 0:
        cm = clipped_df[col].mean()
    else:
        cm = 0
    clm = clean_df[col].mean()
    print(f"  {col:>18s}  {cm:>10.4f}  {clm:>10.4f}  {cm-clm:>+10.4f}")

# ── 画几个削波波形 ──
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 最严重的削波
worst = df[df["clipped"]].nlargest(5, "max_flat_run")
for idx, (_, wrow) in enumerate(worst.iterrows()):
    if idx >= 5:
        break
    rid = int(wrow["id"])
    cur.execute("SELECT wave_data FROM waveforms WHERE record_id = ?", (rid,))
    row = cur.fetchone()
    if row is None:
        continue
    wave = np.array([float(x) for x in row[0].split(",")], dtype=np.float64)

    ax = axes[idx]
    # 显示半周期内的细节
    ax.plot(wave, "b-", linewidth=1)
    # 标注平坦区域
    diff = np.abs(np.diff(wave - np.mean(wave)))
    # 高亮平坦区域
    y_peak = np.max(np.abs(wave - np.mean(wave)))
    flat_threshold = wrow["typical_slope"] * 0.02
    is_flat = np.concatenate([diff < flat_threshold, [False]])
    ax.fill_between(range(len(wave)), wave.min(), wave.max(),
                     where=is_flat, alpha=0.3, color="red", label="平坦区域")
    ax.set_title(f"ID={rid} {wrow['voltage']:+.0f}V  flat_run={wrow['max_flat_run']}",
                 fontsize=10)
    ax.set_xlabel("采样点", fontsize=8)
    ax.set_ylabel("幅值", fontsize=8)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=8)

# 最后一个画无削波对比
clean_sample = df[~df["clipped"]].sample(1).iloc[0]
cur.execute("SELECT wave_data FROM waveforms WHERE record_id = ?", (int(clean_sample["id"]),))
row = cur.fetchone()
if row is not None:
    wave = np.array([float(x) for x in row[0].split(",")], dtype=np.float64)
    ax = axes[5]
    ax.plot(wave, "b-", linewidth=1, label="无削波")
    diff = np.abs(np.diff(wave - np.mean(wave)))
    y_peak = np.max(np.abs(wave - np.mean(wave)))
    # 用同样方法检测
    # 无削波的典型斜率
    typical_slope = np.percentile(diff[diff > 0], 75)
    is_flat = np.concatenate([diff < typical_slope * 0.02, [False]])
    ax.fill_between(range(len(wave)), wave.min(), wave.max(),
                     where=is_flat, alpha=0.3, color="green", label="(无平坦)")
    ax.set_title(f"ID={int(clean_sample['id'])} {clean_sample['voltage']:+.0f}V (无削波)",
                 fontsize=10)
    ax.set_xlabel("采样点", fontsize=8)
    ax.set_ylabel("幅值", fontsize=8)
    ax.grid(True, alpha=0.3)

conn.close()

plt.tight_layout()
out = r"d:\project\work\swa\swa\scripts\clipping_examples.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"已保存: {out}")
plt.close(fig)

# 削波矫正试验
print(f"\n=== 削波矫正试验 ===")
clipped_worst = df[df["clipped"]].nlargest(20, "max_flat_run")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
for _, wrow in clipped_worst.iterrows():
    rid = int(wrow["id"])
    cur.execute("SELECT wave_data FROM waveforms WHERE record_id = ?", (rid,))
    row = cur.fetchone()
    if row is None:
        continue
    wave = np.array([float(x) for x in row[0].split(",")], dtype=np.float64)

    # 矫正前
    a1_orig_h, _, err_orig_h, _, _, _ = compute_harmonics(wave)
    if a1_orig_h is None:
        continue
    a1_orig = float(a1_orig_h)
    err_orig = float(err_orig_h) if err_orig_h else 0.0

    # 矫正：对平坦区域做线性插值
    y = wave - np.mean(wave)
    diff = np.abs(np.diff(y))
    typical_slope = np.percentile(diff[diff > 0], 75)
    is_flat = diff < typical_slope * 0.02
    is_flat = np.concatenate([is_flat, [False]])

    if np.sum(is_flat) >= 3:
        # 用周围点插值替换平坦区域
        corrected = wave.copy()
        # 找连续平坦段
        padded = np.concatenate([[False], is_flat, [False]])
        run_starts = np.where(~padded[:-1] & padded[1:])[0]
        run_ends = np.where(padded[:-1] & ~padded[1:])[0]
        for start, end in zip(run_starts, run_ends):
            if end - start >= 3:
                left = max(0, start - 2)
                right = min(len(wave) - 1, end + 2)
                x = np.array([left, right])
                y_vals = wave[x]
                corrected[start:end+1] = np.interp(
                    np.arange(start, end+1), x, y_vals
                )

        a1_corr, _, err_corr, _, _, _ = compute_harmonics(corrected)
        print(f"  ID={rid:>5d}  V={wrow['voltage']:+.0f}  "
              f"A1: {a1_orig:.2f}→{a1_corr:.2f}  "
              f"({a1_corr-a1_orig:+.2f})  "
              f"error: {err_orig:.4f}→{err_corr:.4f}")

conn.close()
print(f"\n{'='*60}")
print("总结: 削波发生率和影响程度如上。")
print(f"{'='*60}")
