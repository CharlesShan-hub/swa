"""
数据清洗 — 电压解析、标签替换、异常过滤
"""

import numpy as np
import pandas as pd
from typing import Optional

from swa.data.loader import parse_voltage


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    完整清洗流程。

    1. 电压列解析
    2. 去掉无波形数据
    3. 去掉电压为空的行
    4. 类型转换（温度、湿度、RPM）
    """
    df = df.copy()

    # 解析电压
    df["actual_voltage"] = df["actual_voltage"].apply(parse_voltage)

    # 去掉无效行
    df = df.dropna(subset=["actual_voltage"])

    # 确保波形字段存在
    wave_col = "wave_data"
    if wave_col not in df.columns:
        # 尝试大写
        for c in ["RTU_REGS_P00_WAVE_DATA", "WAVE_DATA"]:
            if c in df.columns:
                wave_col = c
                break

    if wave_col in df.columns:
        df = df[df[wave_col].notna() & (df[wave_col] != "")].copy()

    # 类型转换
    for col in ["temperature", "humidity", "rpm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def filter_good_waveform(
    wave: np.ndarray,
    min_amplitude: float = 0.001,
    max_nan_ratio: float = 0.1,
) -> bool:
    """
    波形质量检查。

    Args:
        wave: 波形数组
        min_amplitude: 最小峰峰值
        max_nan_ratio: 最大 NaN 比例

    Returns:
        True 表示波形可用
    """
    if wave is None or len(wave) == 0:
        return False
    nan_ratio = np.mean(np.isnan(wave))
    if nan_ratio > max_nan_ratio:
        return False
    amp = np.ptp(wave)
    if amp < min_amplitude:
        return False
    return True


def label_voltage_groups(df: pd.DataFrame) -> dict:
    """
    统计各电压等级的数据量。
    """
    return df["actual_voltage"].value_counts().sort_index().to_dict()
