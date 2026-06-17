"""分析温湿度分布和电压的关联"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import Counter, defaultdict
from scripts.utils.loader import load_jsonl

records = load_jsonl("data/exported_data.jsonl", extract_features=False)

# 1. 温度分布
temps = [float(r["RTU_REGS_P00_ENV_TEMP"]) for r in records if r.get("RTU_REGS_P00_ENV_TEMP")]
print("=== 温度分布 ===")
temp_buckets = Counter(round(t / 5) * 5 for t in temps)
for k in sorted(temp_buckets):
    print(f"  {k:>4}°C: {temp_buckets[k]:>6}条 ({temp_buckets[k]/len(temps)*100:.1f}%)")

# 2. 湿度分布
hums = [float(r["RTU_REGS_P00_ENV_HUMIDITY"]) for r in records if r.get("RTU_REGS_P00_ENV_HUMIDITY")]
print("\n=== 湿度分布 ===")
hum_buckets = Counter(round(h / 10) * 10 for h in hums)
for k in sorted(hum_buckets):
    print(f"  {k:>3}%: {hum_buckets[k]:>6}条 ({hum_buckets[k]/len(hums)*100:.1f}%)")

# 3. 温度×电压 交叉分布
print("\n=== 温度×电压交叉 ===")
tv = defaultdict(lambda: defaultdict(int))
for r in records:
    t = round(float(r["RTU_REGS_P00_ENV_TEMP"]) / 5) * 5
    v = round(r["ACTUAL_VOLTAGE"] / 10) * 10
    tv[t][v] += 1

voltage_buckets = sorted(set(v for t in tv for v in tv[t]))
print(f"{'温度':>6} |", end="")
for v in voltage_buckets:
    print(f" {v:>+6}V |", end="")
print()
print("-" * (8 + 10 * len(voltage_buckets)))
for t in sorted(tv):
    print(f"{t:>5}°C |", end="")
    for v in voltage_buckets:
        n = tv[t][v]
        print(f" {n:>6} |", end="")
    print()

# 4. -40V 在这个温度下占比多少
print("\n=== -40V 在各温度的占比 ===")
total_per_temp = {t: sum(tv[t].values()) for t in tv}
for t in sorted(tv):
    n40 = tv[t].get(-40, 0)
    total = total_per_temp[t]
    print(f"  {t:>4}°C: -40V 占 {n40/total*100:.1f}% ({n40}/{total})")
