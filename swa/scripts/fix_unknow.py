import json

with open("data/unknow.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    rec = json.loads(line)
    if rec.get("ACTUAL_VOLTAGE") == "未知":
        rec["ACTUAL_VOLTAGE"] = "-87V"
    new_lines.append(json.dumps(rec, ensure_ascii=False))

with open("data/unknow.jsonl", "w", encoding="utf-8") as f:
    for l in new_lines:
        f.write(l + "\n")

print(f"已修改 {len(new_lines)} 条，将 ACTUAL_VOLTAGE='未知' 改为 '-87V'")
