"""
数据集管理 — 切分、过滤、统计
"""

import os
import numpy as np
import pandas as pd
from typing import Optional

from swa.data.loader import load_jsonl, parse_wave, clean_voltage_column


def load_cleaned(path: str, n_points: int = 512) -> pd.DataFrame:
    """
    加载并清洗数据集。

    Returns:
        DataFrame，额外包含 parsed_wave 列
    """
    df = load_jsonl(path)
    df = clean_voltage_column(df)
    df["parsed_wave"] = df["wave_data"].apply(
        lambda x: parse_wave(x, n_points) if isinstance(x, str) else None
    )
    df = df.dropna(subset=["parsed_wave"])
    return df


def filter_by_voltage(df: pd.DataFrame, voltages: list[float]) -> pd.DataFrame:
    """按电压值筛选。"""
    return df[df["actual_voltage"].isin(voltages)].copy()


def filter_by_range(
    df: pd.DataFrame,
    voltage_min: Optional[float] = None,
    voltage_max: Optional[float] = None,
    temp_min: Optional[float] = None,
    temp_max: Optional[float] = None,
    humid_min: Optional[float] = None,
    humid_max: Optional[float] = None,
) -> pd.DataFrame:
    """多条件范围筛选。"""
    mask = pd.Series(True, index=df.index)
    if voltage_min is not None:
        mask &= df["actual_voltage"] >= voltage_min
    if voltage_max is not None:
        mask &= df["actual_voltage"] <= voltage_max
    if temp_min is not None:
        t = pd.to_numeric(df.get("temperature", pd.Series(np.nan, index=df.index)), errors="coerce")
        mask &= t >= temp_min
    if temp_max is not None:
        t = pd.to_numeric(df.get("temperature", pd.Series(np.nan, index=df.index)), errors="coerce")
        mask &= t <= temp_max
    if humid_min is not None:
        h = pd.to_numeric(df.get("humidity", pd.Series(np.nan, index=df.index)), errors="coerce")
        mask &= h >= humid_min
    if humid_max is not None:
        h = pd.to_numeric(df.get("humidity", pd.Series(np.nan, index=df.index)), errors="coerce")
        mask &= h <= humid_max
    return df[mask].copy()


def voltage_distribution(df: pd.DataFrame) -> pd.Series:
    """电压分布统计。"""
    return df["actual_voltage"].value_counts().sort_index()


def dataset_summary(df: pd.DataFrame) -> dict:
    """数据集摘要。"""
    v = df["actual_voltage"]
    return {
        "total": len(df),
        "voltage_types": int(v.nunique()),
        "voltage_range": (float(v.min()), float(v.max())),
        "top_voltages": v.value_counts().head(10).to_dict(),
    }
