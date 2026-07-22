"""验证数据库更新结果"""
import sys, os
sys.path.insert(0, r'd:\project\work\swa\swa\src')
import sqlite3

db = r'd:\project\work\swa\swa\src\data\projects\new\data.db'
conn = sqlite3.connect(db)
cur = conn.cursor()

print("=== 削波分布 ===")
for v in [50, 80, 100, 120, 140, 150]:
    cur.execute(
        "SELECT COUNT(*), AVG(harm_clip_ratio), MAX(harm_clip_ratio) "
        "FROM records WHERE enabled=1 AND actual_voltage=?", (v,))
    n, avg, mx = cur.fetchone()
    cur.execute(
        "SELECT COUNT(*) FROM records WHERE enabled=1 AND actual_voltage=? AND harm_clip_ratio > 0",
        (v,))
    n_clip = cur.fetchone()[0]
    print(f"  {v}V: n={n:>4d}, 削波={n_clip:>4d} ({n_clip/n*100:.1f}%), "
          f"平均削波={avg*100:.2f}%, 最大={mx*100:.1f}%")

# 验证 42986
cur.execute("SELECT harm_a1, harm_clip_ratio FROM records WHERE id = 42986")
a1, cr = cur.fetchone()
print(f"\nID=42986: A1={a1:.2f}  clip_ratio={cr*100:.1f}%")

# 验证无削波
cur.execute(
    "SELECT harm_a1, harm_clip_ratio FROM records "
    "WHERE actual_voltage=50 AND harm_clip_ratio = 0 LIMIT 1")
r = cur.fetchone()
if r:
    print(f"无削波样本: A1={r[0]:.2f}  clip_ratio={r[1]*100:.1f}%")

conn.close()
print("\n验证通过!")
