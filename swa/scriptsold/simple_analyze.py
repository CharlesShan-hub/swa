
"""
简单分析一下数据
"""
import json

file_path = "data/5000.jsonl"
records = []

print("正在加载数据...")
with open(file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.strip()
        if line:
            rec = json.loads(line)
            records.append(rec)
        if i &gt;= 5:  # 只看前几条
            break

print(f"\n前 5 条数据:")
for i, rec in enumerate(records):
    print(f"\nRecord {i}:")
    print(f"  Keys: {list(rec.keys())}")
    print(f"  ACTUAL_VOLTAGE: {rec.get('ACTUAL_VOLTAGE')} (type: {type(rec.get('ACTUAL_VOLTAGE'))})")
    print(f"  SIGNAL length: {len(rec.get('SIGNAL', []))}")
