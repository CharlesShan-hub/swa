"""检查实验数据的 wave_data 长度、slave_id 分布。"""
import json, re
from collections import Counter

path = "data/local.jsonl"
with open(path, encoding="utf-8") as f:
    lines = f.readlines()

# 1. M30-E05-T18H60
print("=== M30-E05-T18H60 的记录 ===")
for i, line in enumerate(lines):
    rec = json.loads(line)
    code = rec.get("TEST_CASE_CODE", "")
    if code == "M30-E05-T18H60":
        wave = rec.get("RTU_REGS_P00_WAVE_DATA", "")
        n = len(wave.split(","))
        print(f"  第{i}行: {n}点, slave_id={rec.get('RTU_REGS_SLAVE_ID','?')}, "
              f"rpm={rec.get('RTU_REGS_P00_ROTOR_RPM','?')}, "
              f"time={rec.get('SYSTEM_TIME','?')}")

# 2. 实验数据中每个 TEST_CASE_CODE 有多少个 slave_id，以及各有多少条
print("\n=== 每个实验码在各设备(slave_id)的条数 ===")
experiment_pattern = re.compile(r"^M30-E\d+-T\d+H\d+$")
exp_counter = {}  # code -> {slave_id: count}

for line in lines:
    rec = json.loads(line)
    code = rec.get("TEST_CASE_CODE", "")
    if experiment_pattern.match(str(code)):
        sid = rec.get("RTU_REGS_SLAVE_ID", "?")
        if code not in exp_counter:
            exp_counter[code] = Counter()
        exp_counter[code][sid] += 1

# 按实验编号排序显示
for code in sorted(exp_counter.keys(), key=lambda x: int(re.search(r'E(\d+)', x).group(1))):
    devices = exp_counter[code]
    total = sum(devices.values())
    parts = [f"  slave_{k}={v}条" for k, v in sorted(devices.items())]
    print(f"  {code} (共{total}条):")
    for p in parts:
        print(f"    {p}")

# 3. 统计所有 slave_id
all_slave = Counter()
experiment_pattern2 = re.compile(r"^M30-E")
for line in lines:
    rec = json.loads(line)
    if experiment_pattern2.search(str(rec.get("TEST_CASE_CODE", ""))):
        all_slave[rec.get("RTU_REGS_SLAVE_ID", "?")] += 1

print(f"\n=== 实验数据中 slave_id 总分布 ===")
for sid, cnt in sorted(all_slave.items()):
    print(f"  slave_id={sid}: {cnt}条")
