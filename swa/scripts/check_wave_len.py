"""检查 jsonl 中 wave_data 的点数（前几条 + 实验数据中的几条）。"""
import json
import re

path = "data/local.jsonl"

with open(path, encoding="utf-8") as f:
    lines = f.readlines()

# 前5条
print("=== 前5条数据 ===")
for i in range(min(5, len(lines))):
    rec = json.loads(lines[i])
    wave = rec.get("RTU_REGS_P00_WAVE_DATA", "")
    n = len(wave.split(","))
    print(f"  第{i}行: {n}点, code={rec.get('TEST_CASE_CODE','?')}, voltage={rec.get('ACTUAL_VOLTAGE','?')}")

# 找实验数据
print("\n=== 实验数据 (M30-E...) 中的几条 ===")
found = 0
for i, line in enumerate(lines):
    rec = json.loads(line)
    code = rec.get("TEST_CASE_CODE", "")
    if re.search(r"M30-E\d", str(code)):
        wave = rec.get("RTU_REGS_P00_WAVE_DATA", "")
        n = len(wave.split(","))
        print(f"  第{i}行: {n}点, code={code}")
        found += 1
        if found >= 5:
            break

# 统计不同点数的分布
print("\n=== 点数分布统计（抽样5000条）===")
from collections import Counter
counter = Counter()
sample_size = min(5000, len(lines))
for i in range(sample_size):
    rec = json.loads(lines[i])
    wave = rec.get("RTU_REGS_P00_WAVE_DATA", "")
    n = len(wave.split(","))
    counter[n] += 1
for n, cnt in sorted(counter.items()):
    print(f"  {n}点: {cnt}条 ({cnt/sample_size*100:.1f}%)")
