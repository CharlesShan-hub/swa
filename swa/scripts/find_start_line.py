"""
快速扫描 local.jsonl，找到第一个实验数据 (M30-E...) 的行号，
用来设置"从第N行开始"的值。
"""

import json
import os
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
JSONL_PATH = os.path.join(DATA_DIR, "local.jsonl")

if not os.path.exists(JSONL_PATH):
    print(f"文件不存在: {JSONL_PATH}")
    print("请先下载数据。")
    exit(1)

pattern = re.compile(r"M30-E\d", re.IGNORECASE)

total_lines = 0
first_match_line = None
first_match_code = None

with open(JSONL_PATH, encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        total_lines += 1

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        code = record.get("TEST_CASE_CODE", "")
        if code and pattern.search(str(code)):
            if first_match_line is None:
                first_match_line = i  # 0-indexed
                first_match_code = code
            # 打印最后几条匹配到的实验（帮助确认末尾）
            if i >= total_lines - 5 or total_lines - i <= 5:
                print(f"  [{i}] {code}")

print(f"\nJSONL 总行数: {total_lines}")

if first_match_line is not None:
    print(f"\n第一个实验数据: 第 {first_match_line} 行 (0-indexed)")
    print(f"  TEST_CASE_CODE = {first_match_code}")
    print(f"\n→ 建议将「从第N行开始」设为: {first_match_line}")
else:
    print("\n未找到实验数据 (M30-E... 模式)")
