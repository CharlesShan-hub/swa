"""
批量修正数据集中未知电压的标签。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

REPLACE = {
    "未知": "-87V",
    "未知1": "-43V",
    "未知2": "-36V",
    "未知3": "72V",
}

path = "data/default/default.jsonl"
records = []
fixes = {k: 0 for k in REPLACE}

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line.strip())
        v = rec.get("ACTUAL_VOLTAGE")
        if v in REPLACE:
            fixes[v] += 1
            rec["ACTUAL_VOLTAGE"] = REPLACE[v]
        records.append(rec)

print("修正统计:")
for old, new in REPLACE.items():
    print(f"  {old} ({fixes[old]}条) → {new}")

with open(path, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"\n完成! 共修正 {sum(fixes.values())} 条")
