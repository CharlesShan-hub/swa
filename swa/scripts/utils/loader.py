"""
数据加载工具

从本地 JSONL 文件中加载波形数据，并预提取时域统计特征。
"""

import json
import os
import random
from typing import Iterator, Any
import numpy as np
from scipy.stats import kurtosis, skew


def get_dataset_model_path(data_path: str, default_model_path: str = "data/model_params") -> str:
    """
    根据数据集路径自动生成对应的模型保存/加载路径。
    
    Args:
        data_path: 数据集路径（如 "data/5000.jsonl" 或 "data/exported_data.jsonl"）
        default_model_path: 默认模型路径（向后兼容）
        
    Returns:
        模型路径前缀（如 "data/5000/model_params" 或 "data/model_params"）
    """
    # 如果 data_path 是 "data/5000.jsonl"，我们提取 "5000" 作为子目录
    filename = os.path.basename(data_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # 检查是否有对应的数据集目录（如 "data/5000/"）
    data_dir = os.path.dirname(data_path) or "data"
    dataset_dir = os.path.join(data_dir, name_without_ext)
    
    if os.path.exists(dataset_dir):
        # 如果已经有对应的数据集目录，就把模型放到那里
        os.makedirs(dataset_dir, exist_ok=True)
        return os.path.join(dataset_dir, "model_params")
    else:
        # 否则使用默认路径
        return default_model_path


def extract_time_domain_features(wave_str: str) -> dict:
    """
    从波形字符串提取时域统计特征 + A1 相位。

    Args:
        wave_str: 波形数据字符串（逗号分隔的数值）

    Returns:
        包含以下特征的字典：
        - vpp: 峰峰值
        - kurtosis: 峭度
        - skewness: 偏度
        - phase: A1 的 FFT 相位角（弧度，-π ~ π）
    """
    try:
        vals = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
        ac = vals - np.mean(vals)
        vpp = float(np.max(ac) - np.min(ac))
        kurt = float(kurtosis(ac, fisher=False))
        skewn = float(skew(ac))

        # A1 相位
        n = len(ac)
        fft_result = np.fft.fft(ac)
        phase = float(np.angle(fft_result[1]))  # A1 的相位，-π ~ π

        return {
            "vpp": vpp,
            "kurtosis": kurt,
            "skewness": skewn,
            "phase": phase
        }
    except:
        print("Error! Fail to extract features!")
        return {
            "vpp": 0.0,
            "kurtosis": 0.0,
            "skewness": 0.0,
            "phase": 0.0
        }


def load_jsonl(filepath: str, extract_features: bool = True) -> list[dict]:
    """
    从 JSONL 文件加载所有记录，并自动清洗 ACTUAL_VOLTAGE 字段为 float。

    Args:
        filepath: JSONL 文件路径
        extract_features: 是否预提取时域统计特征（Vpp/Kurtosis/Skewness），
                          数据增强等场景可设为 False 加快速度。

    Returns:
        记录列表（每条记录的 ACTUAL_VOLTAGE 已转为 float）
    """
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                rec["ACTUAL_VOLTAGE"] = parse_voltage(rec.get("ACTUAL_VOLTAGE"))
                if extract_features:
                    wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
                    time_feats = extract_time_domain_features(wave_str)
                    rec.update(time_feats)
                records.append(rec)
    return records


def iter_jsonl(filepath: str, extract_features: bool = True) -> Iterator[dict]:
    """逐行迭代 JSONL 文件，自动清洗 ACTUAL_VOLTAGE 字段为 float（省内存）"""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                rec["ACTUAL_VOLTAGE"] = parse_voltage(rec.get("ACTUAL_VOLTAGE"))
                if extract_features:
                    wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
                    time_feats = extract_time_domain_features(wave_str)
                    rec.update(time_feats)
                yield rec


def parse_voltage(v) -> float | None:
    """将 ACTUAL_VOLTAGE 原始值转为 float。支持 '-40V'、'30 v'、110 等格式，清洗失败返回 None。"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower().replace("v", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def split_jsonl(records: list[dict],
                full_dataset: bool = True,
                limit: int = 0,
                train_ratio: float = 0.8,
                val_ratio: float = 0.1,
                test_ratio: float = 0.1,
                seed: int = 42):
    """
    划分数据集为训练/验证/测试集。

    固定随机种子 → 打乱 → 按 limit 截取 → 按比例拆分。

    Args:
        records: 全量记录列表
        full_dataset: 是否使用全部数据
        limit: 非全集时，取前多少条
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        (train_records, val_records, test_records)
    """
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)

    if not full_dataset and limit > 0:
        shuffled = shuffled[:limit]

    total = len(shuffled)
    train_n = int(total * train_ratio)
    val_n = int(total * val_ratio)
    test_n = total - train_n - val_n

    train = shuffled[:train_n]
    val = shuffled[train_n:train_n + val_n]
    test = shuffled[train_n + val_n:]

    return train, val, test
