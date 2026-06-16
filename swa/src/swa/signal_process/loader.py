"""
数据加载模块

从本地 JSONL 文件中加载波形数据。
"""

import json
from typing import Iterator, Any


def load_jsonl(filepath: str) -> list[dict]:
    """
    从 JSONL 文件加载所有记录。

    Args:
        filepath: JSONL 文件路径

    Returns:
        记录列表
    """
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_jsonl(filepath: str) -> Iterator[dict]:
    """逐行迭代 JSONL 文件，节省内存"""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
