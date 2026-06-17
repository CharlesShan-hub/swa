import json

with open("data/unknow.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 3: break
        rec = json.loads(line.strip())
        print(f"第{i+1}条:")
        print(f"  TEST_CASE_CODE: {rec.get('TEST_CASE_CODE')!r}")
        print(f"  ACTUAL_VOLTAGE: {rec.get('ACTUAL_VOLTAGE')!r}")
        print(f"  SYSTEM_TIME: {rec.get('SYSTEM_TIME')!r}")
        print()
