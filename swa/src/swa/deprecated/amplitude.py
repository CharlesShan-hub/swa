"""
幅值提取与置信度计算

从去直流后的交流信号中提取幅值特征量（RMS / FFT / 峰峰值），
以及计算正弦波置信度（主频能量占比）。

参考：
    - 论文公式 (2): i(t) ∝ E · dA(t)/dt
    - 同事已有的 calculate_sine_confidence() 实现
"""

import numpy as np
from scipy.fftpack import fft
from typing import Literal, Optional

from ..config.settings import config


# ============================================================
# 预处理
# ============================================================

def remove_dc(signal: np.ndarray) -> np.ndarray:
    """去除直流分量"""
    return signal - np.mean(signal)


# ============================================================
# 幅值提取
# ============================================================

def rms(signal: np.ndarray) -> float:
    """计算 RMS 有效值"""
    return float(np.sqrt(np.mean(np.square(signal))))


def fft_amplitude(
    signal: np.ndarray,
    fs: Optional[float] = None,
    target_freq: Optional[float] = None,
) -> float:
    """
    在目标频率附近提取 FFT 幅值。

    Args:
        signal: 输入信号（建议先去直流）
        fs: 采样率，默认使用 config 中的值
        target_freq: 目标频率，默认使用转子频率

    Returns:
        目标频率处的幅值
    """
    if fs is None:
        fs = config.signal.sample_rate
    if target_freq is None:
        target_freq = config.signal.bandpass_center

    n = len(signal)
    fft_result = fft(signal)
    freq_axis = np.fft.fftfreq(n, d=1.0 / fs)

    idx = np.argmin(np.abs(freq_axis - target_freq))
    return float(2.0 * np.abs(fft_result[idx]) / n)


def peak_to_peak(signal: np.ndarray) -> float:
    """计算峰峰值"""
    return float(np.max(signal) - np.min(signal))


def extract_amplitude(
    signal: np.ndarray,
    method: Optional[Literal["rms", "fft", "peak_to_peak"]] = None,
) -> float:
    """
    统一幅值提取入口（自动去直流）。

    Args:
        signal: 原始输入信号
        method: 提取方法，默认使用 config 中的设置

    Returns:
        幅值特征量
    """
    if method is None:
        method = config.signal.amplitude_method  # type: ignore

    ac = remove_dc(signal)

    if method == "rms":
        return rms(ac)
    elif method == "fft":
        return fft_amplitude(ac)
    elif method == "peak_to_peak":
        return peak_to_peak(ac)
    else:
        raise ValueError(f"未知的幅值提取方法: {method}")


# ============================================================
# 置信度计算（来自同事）
# ============================================================

def sine_confidence(
    fft_magnitude: np.ndarray,
    dominant_freq_index: int,
    fft_size: int,
) -> float:
    """
    计算正弦波置信度：主频率能量占总能量（排除直流）的比例。

    Args:
        fft_magnitude: FFT 完整幅度谱
        dominant_freq_index: 主频率在正频率区的索引
        fft_size: FFT 点数

    Returns:
        置信度，范围 [0, 1]
    """
    half = fft_size // 2
    fft_pos = fft_magnitude[:half]
    total = np.sum(np.square(fft_pos[1:]))  # 排除直流
    if total < 1e-9:
        return 0.0
    dominant = np.square(fft_pos[dominant_freq_index])
    return float(np.clip(dominant / total, 0.0, 1.0))
