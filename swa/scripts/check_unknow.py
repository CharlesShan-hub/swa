import json

with open("data/unknow.jsonl") as f:
    records = [json.loads(line) for line in f if line.strip()]

print(f"总行数: {len(records)}")
print(f"\n第一个 keys: {list(records[0].keys())}")
print(f"第一个 ACTUAL_VOLTAGE: {records[0].get('ACTUAL_VOLTAGE')!r}")
print(f"第一个 WAVE 长度: {len(records[0].get('RTU_REGS_P00_WAVE_DATA', '').split(',')) if records[0].get('RTU_REGS_P00_WAVE_DATA') else 0}")
