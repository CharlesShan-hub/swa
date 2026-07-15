"""
数据加载 — 从 JSONL 文件加载波形数据
"""

import json
import os
from typing import Optional
import numpy as np
import pandas as pd


def load_jsonl(path: str) -> pd.DataFrame:
    """
    从 JSONL 文件加载数据。

    Args:
        path: JSONL 文件路径

    Returns:
        DataFrame，包含字段:
          - actual_voltage: 实际电压值 (float)
          - wave_data: 原始波形字符串
          - temperature: 温度 (°C)
          - humidity: 湿度 (%)
          - rpm: 转速
          - system_time: 时间
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    # 统一列名（大写 → 小写）
    rename = {
        "ACTUAL_VOLTAGE": "actual_voltage",
        "SYSTEM_TIME": "system_time",
        "RTU_REGS_P00_WAVE_DATA": "wave_data",
        "RTU_REGS_P00_ENV_TEMP": "temperature",
        "RTU_REGS_P00_ENV_HUMIDITY": "humidity",
        "RTU_REGS_P00_ROTOR_RPM": "rpm",
        "TEST_CASE_CODE": "test_case_code",
        "RTU_REGS_SLAVE_ID": "slave_id",
        "ENABLED": "enabled",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    return df


def parse_wave(wave_str: str, n_points: int = 512) -> Optional[np.ndarray]:
    """
    将波形字符串解析为 numpy 数组。

    Args:
        wave_str: 逗号分隔的电压值字符串
        n_points: 期望的点数（默认 512）

    Returns:
        numpy 数组，解析失败返回 None
    """
    try:
        values = [float(x) for x in wave_str.split(",")]
        if len(values) >= n_points:
            return np.array(values[:n_points], dtype=np.float64)
        return None
    except (ValueError, TypeError):
        return None


def parse_voltage(val, label_map: Optional[dict[str, float]] = None) -> Optional[float]:
    """
    解析电压值，处理字符串格式。

    支持:
      - 数值: 20, -43, 110
      - 字符串: "110V", "20V", "-43V"
      - 自定义标签映射（如未知1 → -43）

    Args:
        val: 原始电压值
        label_map: 自定义标签映射字典 {标签: 电压值}，优先级高于内置默认值

    Returns:
        解析后的电压值 (float)，失败返回 None
    """
    if val is None:
        return None

    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip()

    # 内置默认标签（优先级低于自定义）
    if label_map is None:
        label_map = {}

    default_map = {
        "未知3": 72.0,
        "未知2": 36.0,
        "未知1": -43.0,
        "未知": -87.0,
    }

    # 合并：自定义覆盖默认
    full_map = {**default_map, **label_map}

    # 注意顺序：先匹配长的，避免"未知"误匹配到"未知1"
    for key, v in sorted(full_map.items(), key=lambda x: -len(x[0])):
        if s == key:
            return v

    # 去掉尾部单位字符
    s = s.rstrip("Vv")
    try:
        return float(s)
    except ValueError:
        return None


def clean_voltage_column(df: pd.DataFrame, col: str = "actual_voltage",
                         label_map: Optional[dict[str, float]] = None) -> pd.DataFrame:
    """
    清洗电压列。
    """
    df = df.copy()
    df[col] = df[col].apply(lambda v: parse_voltage(v, label_map))
    return df.dropna(subset=[col])


def compute_harmonics(wave_str: str) -> tuple:
    """
    计算波形的谐波参数。

    Returns:
        (A1, A2, error, cycles, thd, noise_pct)
        - A1: 基频幅值
        - A2: 二次谐波幅值
        - error: 基频拟合误差 sqrt(mean((w - fitted)^2))
        - cycles: 周期数（基频 FFT bin 索引）
        - thd: 总谐波失真 (sqrt(sum(A2..A10^2)) / A1)
        - noise_pct: 信号中不属于基频的能量占比
        均返回 None 表示无法计算
    """
    try:
        w = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
    except (ValueError, TypeError, AttributeError):
        return None, None, None, None, None, None

    if len(w) < 20:
        return None, None, None, None, None, None

    w = w - np.mean(w)
    n = len(w)
    fft_vals = np.fft.rfft(w)
    mag = np.abs(fft_vals[1:])  # 跳过 DC

    if len(mag) < 5:
        return None, None, None, None, None, None

    # 找基频（bin 索引 = 周期数）
    search_end = min(len(mag), n // 3)
    fund_idx = int(np.argmax(mag[:search_end]) + 1)
    cycles = float(fund_idx)

    A1 = float(mag[fund_idx - 1])
    if A1 < 1e-6:
        return None, None, None, None, None, None

    # 2~10 次谐波
    harm_count = min(10, len(mag) // fund_idx)
    harm_sq = 0.0
    A2 = 0.0
    for k in range(2, harm_count + 1):
        idx = k * fund_idx - 1
        if idx < len(mag):
            a = mag[idx]
            harm_sq += a * a
            if k == 2:
                A2 = float(a)

    thd = float(np.sqrt(harm_sq) / A1) if A1 > 0 else 0.0

    # 基频拟合误差
    phase = np.angle(fft_vals[fund_idx])
    t = np.arange(n)
    fitted = 2 * A1 / n * np.cos(2 * np.pi * fund_idx * t / n + phase)
    error = float(np.sqrt(np.mean((w - fitted) ** 2)))

    # 噪声能量占比：不属于基频+belt 的能量 / 总能量
    total_power = np.sum(mag * mag)
    a1_power = A1 * A1
    noise_pct = float(np.sqrt((total_power - a1_power) / total_power)) if total_power > 0 else 1.0

    return A1, A2, error, cycles, thd, noise_pct


def check_waveform_quality(wave_str: str) -> bool:
    """
    检测波形质量，基于谐波分析判断是否噪声过大。

    原理：对波形做 FFT，提取基频 (A1)、二次谐波 (A2)、三次谐波 (A3)
    幅值。如果高次谐波占比太大，说明波形失真严重/噪声过大。

    Args:
        wave_str: 逗号分隔的波形数字符串

    Returns:
        True = 波形正常, False = 波形噪声过大
    """
    try:
        w = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
    except (ValueError, TypeError, AttributeError):
        return False

    if len(w) < 20:
        return False

    # 去除直流分量
    w = w - np.mean(w)

    # FFT 提取前 3 次谐波幅值
    n = len(w)
    fft_vals = np.fft.rfft(w)
    mag = np.abs(fft_vals[1:])  # 跳过 DC
    if len(mag) < 3:
        return False

    # 总信号能量 — 检测死通道
    total_energy = np.sum(mag) / n
    if total_energy < 50:
        return False

    # 找基频：前 1/3 频谱中幅值最大的（避开过高次谐波）
    search_end = min(len(mag), n // 6)
    fundamental_idx = np.argmax(mag[:search_end]) + 1

    A1 = mag[fundamental_idx - 1] if fundamental_idx - 1 < len(mag) else 0
    A2 = mag[2 * fundamental_idx - 1] if 2 * fundamental_idx - 1 < len(mag) else 0
    A3 = mag[3 * fundamental_idx - 1] if 3 * fundamental_idx - 1 < len(mag) else 0

    # 防止除零
    if A1 < 1e-6:
        return False  # 无有效信号

    # 谐波占比
    h2_ratio = A2 / A1
    h3_ratio = A3 / A1

    # 总谐波失真 + 噪声 (THD+N) 估计
    total_harmonic = np.sqrt(A2**2 + A3**2) / A1

    # 干净正弦波: h2_ratio < 0.05, h3_ratio < 0.03, THD+N < 0.08
    # 坏数据: 谐波占比过大
    if h2_ratio > 0.25:
        return False
    if h3_ratio > 0.20:
        return False
    if total_harmonic > 0.30:
        return False

    return True
