
import json

# 检查 5000.jsonl 的第一条记录
with open("data/5000.jsonl", 'r', encoding='utf-8') as f:
    rec = json.loads(f.readline().strip())

print("5000.jsonl 的 keys:")
for k, v in rec.items():
    if isinstance(v, str) and len(v) > 50:
        print(f"  {k}: (str, len={len(v)}) {v[:50]}...")
    elif isinstance(v, list) and len(v) > 10:
        print(f"  {k}: (list, len={len(v)}) first few: {v[:5]}")
    else:
        print(f"  {k}: {repr(v)} (type={type(v).__name__})")
