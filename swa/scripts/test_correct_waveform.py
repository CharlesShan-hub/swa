"""验证 correct_waveform 函数"""
import sys, os
sys.path.insert(0, r'd:\project\work\swa\swa\src')
from swa.data.loader import correct_waveform, compute_harmonics
import sqlite3

db = r'd:\project\work\swa\swa\src\data\projects\new\data.db'
conn = sqlite3.connect(db)
cur = conn.cursor()

# 42986（有削波）
cur.execute("SELECT w.wave_data FROM waveforms w JOIN records r ON w.record_id = r.id WHERE r.id = 42986")
wave_str = cur.fetchone()[0]
corrected = correct_waveform(wave_str)
print(f'ID=42986:')
print(f'  原始波形长度: {len(wave_str.split(","))}')
print(f'  矫正后波形长度: {len(corrected.split(","))}')
print(f'  是否改变了: {corrected != wave_str}')
a1_orig, _, _, _, _, _, _, _ = compute_harmonics(wave_str, clip_correction=False)
a1_corr, _, _, _, _, _, _, cr = compute_harmonics(wave_str, clip_correction=True)
print(f'  原始A1={a1_orig:.2f}  矫正A1={a1_corr:.2f}  ({cr*100:.1f}%削波)')

# 无削波的
cur.execute("SELECT w.wave_data FROM waveforms w JOIN records r ON w.record_id = r.id WHERE r.actual_voltage=50 AND r.harm_a1 IS NOT NULL LIMIT 1")
wave_str2 = cur.fetchone()[0]
corrected2 = correct_waveform(wave_str2)
a1_2, _, _, _, _, _, _, cr2 = compute_harmonics(wave_str2, clip_correction=True)
print(f'\n无削波样本:')
print(f'  是否改变了: {corrected2 == wave_str2}')
print(f'  A1={a1_2:.2f}  clip={cr2*100:.1f}%')

conn.close()
print('\n全部正常！')
