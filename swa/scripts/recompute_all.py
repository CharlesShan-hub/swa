"""
全量重算：对现有项目的所有记录，用最终方案（锚点法削波矫正）重新计算谐波。

更新字段: harm_a1(原始值), harm_a1_corrected(矫正值), harm_clip_ratio
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sqlite3
import numpy as np
from tqdm import tqdm
from swa.data.loader import compute_harmonics

PROJECT_DIR = r"d:\project\work\swa\swa\src\data\projects\new"
DB_PATH = os.path.join(PROJECT_DIR, "data.db")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 获取所有启用的记录
cur.execute("""
    SELECT r.id, w.wave_data
    FROM records r
    JOIN waveforms w ON w.record_id = r.id
    WHERE r.enabled = 1
    ORDER BY r.id
""")
rows = cur.fetchall()
print(f"总记录: {len(rows)}")

# 加入 clip/a1_corrected 相关列（已有项目）
try:
    cur.execute("ALTER TABLE records ADD COLUMN harm_clip_ratio REAL")
except Exception:
    pass
try:
    cur.execute("ALTER TABLE records ADD COLUMN harm_clip_corrected INTEGER DEFAULT 0")
except Exception:
    pass
try:
    cur.execute("ALTER TABLE records ADD COLUMN harm_a1_corrected REAL")
except Exception:
    pass
conn.commit()

updated = 0
clipped = 0
batch = []
batch_size = 500

for rid, wave_str in tqdm(rows, desc="重算谐波"):
    a1_orig, a1_corrected, a2, err, cycles, thd, noise_pct, clip_ratio = compute_harmonics(
        wave_str, clip_correction=True
    )

    if a1_orig is None:
        continue

    batch.append((a1_orig, a1_corrected, a2, err, cycles, noise_pct, clip_ratio, 1 if clip_ratio and clip_ratio > 0 else 0, rid))
    if clip_ratio > 0:
        clipped += 1
    updated += 1

    if len(batch) >= batch_size:
        cur.executemany(
            "UPDATE records SET harm_a1=?, harm_a1_corrected=?, harm_a2=?, harm_error=?, harm_cycles=?, harm_noise_pct=?, harm_clip_ratio=?, harm_clip_corrected=? WHERE id=?",
            batch,
        )
        conn.commit()
        batch = []

# 最后一批
if batch:
    cur.executemany(
        "UPDATE records SET harm_a1=?, harm_a1_corrected=?, harm_a2=?, harm_error=?, harm_cycles=?, harm_noise_pct=?, harm_clip_ratio=?, harm_clip_corrected=? WHERE id=?",
        batch,
    )
    conn.commit()

conn.close()

print(f"\n完成!")
print(f"  更新: {updated} 条")
print(f"  检测到削波: {clipped} 条 ({clipped/updated*100:.1f}%)")
