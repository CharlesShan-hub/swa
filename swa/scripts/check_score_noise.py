"""
验证 score 和 noise_pct 的关系
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import sqlite3
import pandas as pd
from swa.core.scoring import compute_score, compute_alpha7
from swa.detection.least_squares import _extract_features

DB_PATH = r"d:\project\work\swa\swa\src\data\projects\new\data.db"

conn = sqlite3.connect(DB_PATH)
rows = conn.execute("""
    SELECT r.id, r.actual_voltage, r.harm_a1, r.harm_noise_pct, r.device_id, w.wave_data
    FROM records r JOIN waveforms w ON w.record_id = r.id
    WHERE r.enabled=1 AND r.actual_voltage>=0
    ORDER BY r.id LIMIT 5000
""").fetchall()
conn.close()

records = []
for rid, v, a1, npct, dev, ws in rows:
    try:
        wave = np.array([float(x) for x in ws.split(",")], dtype=np.float64)
    except:
        continue
    if len(wave) < 20:
        continue
    feats = _extract_features(wave)
    records.append({
        "voltage": v, "harm_a1": a1, "noise_pct": npct, "device": str(dev)[-4:],
        "alpha_7": feats["alpha_7"], "score": feats["score"],
    })

df = pd.DataFrame(records)
print(f"总记录: {len(df)}")

# score/alpha_7 和 noise_pct 的关系
print(f"\n全局相关性:")
print(f"  corr(noise_pct, harm_a1) = {df['noise_pct'].corr(df['harm_a1']):.3f}")
print(f"  corr(noise_pct, alpha_7) = {df['noise_pct'].corr(df['alpha_7']):.3f}")
print(f"  corr(noise_pct, score)   = {df['noise_pct'].corr(df['score']):.3f}")

# score/harm_a1 比值作为噪声指标
df["s_a1_ratio"] = df["score"] / df["harm_a1"].clip(lower=0.1)
print(f"  corr(noise_pct, score/A1) = {df['noise_pct'].corr(df['s_a1_ratio']):.3f}")

# alpha_7/harm_a1 比值
df["a7_a1_ratio"] = df["alpha_7"] / df["harm_a1"].clip(lower=0.1)
print(f"  corr(noise_pct, alpha7/A1) = {df['noise_pct'].corr(df['a7_a1_ratio']):.3f}")

# 按电压分层
print(f"\n按电压分层:")
for v in sorted(df["voltage"].unique()):
    sub = df[df["voltage"] == v]
    c1 = sub["noise_pct"].corr(sub["harm_a1"])
    c2 = sub["noise_pct"].corr(sub["score"])
    c3 = sub["noise_pct"].corr(sub["s_a1_ratio"])
    print(f"  V={v:+.0f}  n={len(sub):4d}  corr(noise,A1)={c1:.3f}  corr(noise,score)={c2:.3f}  corr(noise,score/A1)={c3:.3f}")
