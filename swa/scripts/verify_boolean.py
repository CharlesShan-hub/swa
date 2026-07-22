"""验证新字段和阈值"""
import sys, os
sys.path.insert(0, r'd:\project\work\swa\swa\src')
import sqlite3
from swa.data.loader import compute_harmonics

db = r'd:\project\work\swa\swa\src\data\projects\new\data.db'
conn = sqlite3.connect(db)
cur = conn.cursor()

# 验证 42986
cur.execute("SELECT harm_a1, harm_clip_ratio, harm_clip_corrected FROM records WHERE id = 42986")
a1, cr, cf = cur.fetchone()
print(f'ID=42986: A1={a1:.2f}  clip_ratio={cr*100:.1f}%  corrected={bool(cf)}')

# 验证 17201
cur.execute("SELECT harm_a1, harm_clip_ratio, harm_clip_corrected FROM records WHERE id = 17201")
a1, cr, cf = cur.fetchone()
print(f'ID=17201: A1={a1:.2f}  clip_ratio={cr*100:.1f}%  corrected={bool(cf)}')

# 验证一条无削波的
cur.execute("SELECT harm_a1, harm_clip_ratio, harm_clip_corrected FROM records WHERE id = 1000")
a1, cr, cf = cur.fetchone()
print(f'ID=1000:  A1={a1:.2f}  clip_ratio={cr*100:.1f}%  corrected={bool(cf)}')

# 统计
cur.execute("SELECT COUNT(*), SUM(harm_clip_corrected) FROM records WHERE enabled=1")
n, n_c = cur.fetchone()
print(f'\n总记录: {n}, 已矫正: {n_c} ({n_c/n*100:.1f}%)')

conn.close()
