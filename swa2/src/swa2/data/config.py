"""本地配置文件管理 — 保存数据库连接信息等"""

import json
import os

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "config.json"
)

_DEFAULT_CONFIG = {
    "host": "10.15.10.1",
    "port": "5256",
    "user": "SYSDBA",
    "password": "SYSDBA",
    "batch_size": "400",
    "sleep_sec": "1.0",
}


def load_config() -> dict:
    """读取本地配置文件，不存在则返回默认值。"""
    if not os.path.exists(CONFIG_PATH):
        return dict(_DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return {**_DEFAULT_CONFIG, **json.load(f)}
    except Exception:
        return dict(_DEFAULT_CONFIG)


def save_config(config: dict):
    """保存配置到本地文件。"""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
