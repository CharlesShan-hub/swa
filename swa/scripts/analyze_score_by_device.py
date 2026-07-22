"""
分析 score 特征是否有设备间基线差异（像 A1 一样）。
从每个设备取少量波形算 score，看是否不同设备 score 不同。
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from swa.core.scoring import compute_score

DB_PATH = r"d:\project\work\swa\swa\src\data\projects\new\data.db"
conn = sqlite3.connect(DB_PATH)

devices = [r[0] for r in conn.execute(
    "SELECT DISTINCT device_id FROM records WHERE enabled=1 AND device_id IS NOT NULL ORDER BY device_id"
).fetchall()]
print(f"设备: {[d[-4:] for d in devices]}")

# 取每个设备在固定电压（如 100V）的少量波形
SAMPLE_SIZE = 200
records = {}
for dev in devices:
    rows = conn.execute("""
        SELECT r.id, r.actual_voltage, w.wave_data
        FROM records r JOIN waveforms w ON w.record_id = r.id
        WHERE r.enabled=1 AND r.device_id=? AND r.actual_voltage=100
        ORDER BY RANDOM() LIMIT ?
    """, (dev, SAMPLE_SIZE)).fetchall()
    records[dev] = rows
    print(f"  设备 {dev[-4:]}: 取 {len(rows)} 条 100V 波形")

conn.close()

# 算 score 和 A1
results = []
for dev, rows in records.items():
    for rid, voltage, wave_str in rows:
        try:
            wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
        except:
            continue
        score = compute_score(wave) or 0.0
        
        # 也算 A1 来对比
        y = wave - np.mean(wave)
        n = len(y)
        fft_vals = np.fft.rfft(y)
        mag = np.abs(fft_vals[1:])
        search_end = min(len(mag), n // 3)
        fund_idx = int(np.argmax(mag[:search_end]) + 1)
        a1 = float(mag[fund_idx - 1]) if fund_idx <= len(mag) else 0.0
        
        results.append({"device": dev[-4:], "score": score, "harm_a1": a1})

print(f"\n{'='*70}")
print("各设备 100V 下 score 和 A1 统计")
print("="*70)
print(f"  {'设备':>6s}  {'n':>5s}  {'score均值':>10s}  {'score std':>10s}  {'A1均值':>10s}  {'score/A1':>10s}")
for dev in devices:
    sub = [r for r in results if r["device"] == dev[-4:]]
    scores = [r["score"] for r in sub]
    a1s = [r["harm_a1"] for r in sub]
    ratios = [s/a1 for s, a1 in zip(scores, a1s) if a1 > 0]
    print(f"  {dev[-4:]:>6s}  {len(scores):>5d}"
          f"  {np.mean(scores):>10.2f}  {np.std(scores):>10.2f}"
          f"  {np.mean(a1s):>10.1f}  {np.mean(ratios):>10.4f}")

# ── 画图 ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, dev in enumerate(devices):
    ax = axes[idx]
    sub = [r for r in results if r["device"] == dev[-4:]]
    scores = [r["score"] for r in sub]
    a1s = [r["harm_a1"] for r in sub]
    ax.scatter(a1s, scores, alpha=0.4, s=10)
    # 拟合线
    if len(a1s) > 5:
        coeffs = np.polyfit(a1s, scores, 1)
        x_range = np.linspace(min(a1s), max(a1s), 100)
        ax.plot(x_range, np.polyval(coeffs, x_range), "r--", linewidth=1)
        ax.set_title(f"设备 {dev[-4:]}  (score={coeffs[0]:.4f}×A1+{coeffs[1]:.1f})")
    else:
        ax.set_title(f"设备 {dev[-4:]}")
    ax.set_xlabel("A1")
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "score_by_device.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n已保存: {out_path}")
plt.close(fig)
