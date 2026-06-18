"""分析 u1, u2, u3 的温度湿度分布"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
from collections import Counter

for fname in ["u1.jsonl", "u2.jsonl", "unknow.jsonl"]:
    path = f"data/{fname}"
    if not os.path.exists(path):
        print(f"\n{fname}: 文件不存在")
        continue
    
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    
    temps = Counter()
    hums = Counter()
    for r in records:
        t = r.get("RTU_REGS_P00_ENV_TEMP")
        h = r.get("RTU_REGS_P00_ENV_HUMIDITY")
        if t: temps[float(t)] += 1
        if h: hums[float(h)] += 1
    
    print(f"\n=== {fname} ({len(records)}条) ===")
    print(f"ACTUAL_VOLTAGE: {r.get('ACTUAL_VOLTAGE')!r}")
    print(f"温度: {min(temps):.1f} ~ {max(temps):.1f} °C")
    print(f"  分布:")
    for t in sorted(temps):
        print(f"    {t:.1f}°C: {temps[t]}条")
    print(f"湿度: {min(hums):.1f} ~ {max(hums):.1f} %")
    print(f"  分布:")
    for h in sorted(hums):
        print(f"    {h:.1f}%: {hums[h]}条")
