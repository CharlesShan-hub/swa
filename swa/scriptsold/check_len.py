
import json

# 检查 5000.jsonl 波形长度分布
with open("data/5000.jsonl", 'r', encoding='utf-8') as f:
    lens = {}
    for i, line in enumerate(f):
        if i >= 1000:
            break
        rec = json.loads(line.strip())
        w = rec.get("RTU_REGS_P00_WAVE_DATA", "")
        pts = [float(x) for x in w.split(",")]
        l = len(pts)
        lens[l] = lens.get(l, 0) + 1
    
    print("5000.jsonl 波形长度分布（前1000条）:")
    for l in sorted(lens.keys()):
        print(f"  {l}点: {lens[l]}条")

# 对比 exported_data.jsonl
with open("data/exported_data.jsonl", 'r', encoding='utf-8') as f:
    lens = {}
    for i, line in enumerate(f):
        if i >= 1000:
            break
        rec = json.loads(line.strip())
        w = rec.get("RTU_REGS_P00_WAVE_DATA", "")
        pts = [float(x) for x in w.split(",")]
        l = len(pts)
        lens[l] = lens.get(l, 0) + 1
    
    print("\nexported_data.jsonl 波形长度分布（前1000条）:")
    for l in sorted(lens.keys()):
        print(f"  {l}点: {lens[l]}条")
