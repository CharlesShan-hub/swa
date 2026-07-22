"""
数据加载 — 从 JSONL 文件加载波形数据
"""

import json
import os
from typing import Optional
import numpy as np
import pandas as pd


def load_jsonl(path: str, skip_lines: int = 0) -> pd.DataFrame:
    """
    从 JSONL 文件加载数据。

    Args:
        path: JSONL 文件路径
        skip_lines: 跳过前 N 行不读取（默认 0）

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
        for i, line in enumerate(f):
            if i < skip_lines:
                continue
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
        "DEVICE_ID": "device_id",
        "ENABLED": "enabled",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    return df


def _median_filter(wave: np.ndarray, window_size: int = 5) -> np.ndarray:
    """对波形应用中值滤波（边缘对称填充）。"""
    if window_size < 3 or window_size % 2 == 0:
        window_size = 5
    half = window_size // 2
    # 对称填充边缘
    padded = np.concatenate([wave[half:0:-1], wave, wave[-2:-half-2:-1]])
    filtered = np.zeros_like(wave)
    for i in range(len(wave)):
        filtered[i] = np.median(padded[i:i + window_size])
    return filtered


def parse_wave(wave_str: str, n_points: int = 512,
               median_window: int = 0) -> Optional[np.ndarray]:
    """
    将波形字符串解析为 numpy 数组，可选中值滤波预处理。

    Args:
        wave_str: 逗号分隔的电压值字符串
        n_points: 期望的点数（默认 512）
        median_window: 中值滤波窗口大小（0=禁用, 3或5推荐），扫两次

    Returns:
        numpy 数组，解析失败返回 None
    """
    try:
        values = [float(x) for x in wave_str.split(",")]
        if len(values) >= n_points:
            wave = np.array(values[:n_points], dtype=np.float64)
            # 中值滤波：扫两次
            if median_window >= 3:
                wave = _median_filter(wave, median_window)
                wave = _median_filter(wave, median_window)
            return wave
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


def _detect_and_correct_clipping(wave: np.ndarray, a1_orig: float,
                                 fund_bin: int, phase: float,
                                 n: int) -> tuple[float, float]:
    """检测削波并用低幅值区域（锚点法）矫正 A1。

    思路（用户提出）:
    1. 零点（y=0）是可靠锚点，位置固定
    2. 每个半周期中，幅值 < 2/3 峰值的部分未被削波影响
    3. 只用这些"干净"区域重新拟合幅值 A
    4. 从干净区域拟合的 A 就是矫正后的真实 A1

    数学:
      y(t) = A · cos(ωt + φ)   (已知 ω 和 φ)
      在干净的 t 上最小二乘: A = Σ y·cos(ωt+φ) / Σ cos²(ωt+φ)

    Returns:
        (a1_corrected, clip_ratio)
    """
    y = wave - np.mean(wave)
    t = np.arange(n)
    omega = 2 * np.pi * fund_bin / n
    basis = np.cos(omega * t + phase)

    # ── 平坦削波检测 ──
    # 真实削波（削顶或削底）会在峰值附近产生一段平坦区域，
    # 这些区域内连续采样点的数值几乎恒定（max_diff < 0.1% 幅值）。
    # 不规则波形（非削波）在峰值附近不会有这种完全恒定的平坦段。
    signs = np.sign(y)
    zero_idx = np.where(np.diff(signs) != 0)[0]
    if len(zero_idx) < 2:
        return a1_orig, 0.0

    # 收集所有半周期
    half_info = []  # [(s, e)]
    for i in range(len(zero_idx) - 1):
        s, e = zero_idx[i], zero_idx[i + 1]
        if e - s >= 5:
            half_info.append((s, e))

    if len(half_info) < 4:
        return a1_orig, 0.0

    # 在半周期核心区（去掉边上20%）检测平坦段
    diff = np.abs(np.diff(y))
    typical_slope = float(np.percentile(diff[diff > 0], 75))
    flat_th = typical_slope * 0.05
    is_flat_raw = diff < flat_th

    valid_flat = np.zeros(n - 1, dtype=bool)
    for s, e in half_info:
        if e - s < 5: continue
        margin = int((e - s) * 0.2)
        zs = max(s + margin, s)
        ze = min(e - margin, e)
        if zs < ze:
            valid_flat[zs:ze] = is_flat_raw[zs:ze]

    padded = np.concatenate([[False], valid_flat, [False]])
    runs_s = np.where(~padded[:-1] & padded[1:])[0]
    runs_e = np.where(padded[:-1] & ~padded[1:])[0]

    # 平坦段验证：只检测顶部削波（底部在 0 是传感器设计特征）
    # 顶部削波：平坦段在正半周期（y>0），与初始检测阈值一致
    has_clip = False
    for s, e in zip(runs_s, runs_e):
        if e - s >= 5:
            seg_diff = np.max(np.abs(np.diff(y[s:e + 1])))
            if seg_diff < flat_th:
                # 确认是顶部削波（y>0 的正半周期），忽略底部触底
                if np.mean(y[s:e + 1]) > 0:
                    has_clip = True
                    break

    if not has_clip:
        return a1_orig, 0.0

    # ── 用户锚点法：过零点定位 + 边缘区域拟合 ──
    # 用 FFT 相位（拟合误差最小），每个半周期只取两侧边缘 [0, 0.3] 和 [0.7, 1]
    # 这些区域靠近过零点，不受削波影响
    clean_mask = np.zeros(n, dtype=bool)
    for i in range(len(zero_idx) - 1):
        s, e = zero_idx[i], zero_idx[i + 1]
        half_len = e - s
        if half_len < 5:
            continue
        # 前 30%（靠近左边过零点）
        edge1_end = s + int(half_len * 0.30)
        if edge1_end > s:
            clean_mask[s:edge1_end] = True
        # 后 30%（靠近右边过零点）
        edge2_start = e - int(half_len * 0.30)
        if edge2_start < e:
            clean_mask[edge2_start:e] = True

    # 最小二乘: A = Σ y·basis / Σ basis²  (仅在边缘区域上)
    y_clean = y[clean_mask]
    basis_clean = basis[clean_mask]
    if len(y_clean) < 10:
        return a1_orig, 0.0

    a1_clean = np.sum(y_clean * basis_clean) / np.sum(basis_clean ** 2)
    a1_corrected = a1_clean * n / 2

    clip_ratio = 1.0 - float(np.sum(clean_mask)) / n

    return a1_corrected, clip_ratio


def compute_harmonics(wave_str: str, clip_correction: bool = False) -> tuple:
    """
    计算波形的谐波参数。

    Args:
        wave_str: 逗号分隔的波形数据字符串
        clip_correction: 是否启用削波矫正（默认 False，向后兼容）

    Returns:
        (A1_orig, A1_corrected, A2, error, cycles, thd, noise_pct, clip_ratio)
        - A1_orig: 原始基频幅值（FFT 直接算出，始终有值）
        - A1_corrected: 削波矫正后的基频幅值（clip_correction=True 且有削波时有值，否则 None）
        - A2: 二次谐波幅值
        - error: 基频拟合误差 sqrt(mean((w - fitted)^2))（用矫正后 A1 重建）
        - cycles: 周期数（基频 FFT bin 索引）
        - thd: 已废弃，始终返回 0.0
        - noise_pct: 信号中不属于基频的能量占比
        - clip_ratio: 削波占比（0=无削波）
        均返回 None 表示无法计算
    """
    try:
        w = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
    except (ValueError, TypeError, AttributeError):
        return None, None, None, None, None, None, None, None

    if len(w) < 20:
        return None, None, None, None, None, None, None, None

    w = w - np.mean(w)
    n = len(w)
    fft_vals = np.fft.rfft(w)
    mag = np.abs(fft_vals[1:])  # 跳过 DC

    if len(mag) < 5:
        return None, None, None, None, None, None, None, None

    # 找基频（bin 索引 = 周期数）
    search_end = min(len(mag), n // 3)
    fund_idx = int(np.argmax(mag[:search_end]) + 1)
    cycles = float(fund_idx)

    A1_orig = float(mag[fund_idx - 1])
    if A1_orig < 1e-6:
        return None, None, None, None, None, None, None, None

    # 削波矫正
    phase = np.angle(fft_vals[fund_idx])
    if clip_correction:
        A1_corrected, clip_ratio = _detect_and_correct_clipping(w, A1_orig, fund_idx, phase, n)
        # 如果检测到削波，A1_corrected != A1_orig；否则不变
        if A1_corrected == A1_orig or clip_ratio == 0.0:
            A1_corrected = None  # 无削波时矫正值为 None
            A1_eff = A1_orig
        else:
            A1_eff = A1_corrected
    else:
        A1_corrected = None
        clip_ratio = 0.0
        A1_eff = A1_orig

    # 2~10 次谐波（仅取 A2，THD 已废弃）
    harm_count = min(10, len(mag) // fund_idx)
    A2 = 0.0
    for k in range(2, harm_count + 1):
        idx = k * fund_idx - 1
        if idx < len(mag) and k == 2:
            A2 = float(mag[idx])

    thd = 0.0  # THD 已废弃（error/A1 取代）

    # 基频拟合误差（用有效 A1 重建）
    t = np.arange(n)
    fitted = 2 * A1_eff / n * np.cos(2 * np.pi * fund_idx * t / n + phase)
    error = float(np.sqrt(np.mean((w - fitted) ** 2)))

    # 噪声能量占比
    total_power = np.sum(mag * mag)
    a1_power = A1_orig * A1_orig  # 噪声计算用原始 A1 更合理
    noise_pct = float(np.sqrt((total_power - a1_power) / total_power)) if total_power > 0 else 1.0

    return A1_orig, A1_corrected, A2, error, cycles, thd, noise_pct, clip_ratio


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
    if h2_ratio > 0.20:
        return False
    if h3_ratio > 0.20:
        return False
    if total_harmonic > 0.25:
        return False

    # 基波拟合误差检测 — error/A1 > 0.0008 视为坏数据
    t = np.arange(n)
    phase = np.angle(fft_vals[fundamental_idx])
    fitted = 2 * A1 / n * np.cos(2 * np.pi * fundamental_idx * t / n + phase)
    error = float(np.sqrt(np.mean((w - fitted) ** 2)))
    if error / A1 > 0.0008:
        return False

    return True
