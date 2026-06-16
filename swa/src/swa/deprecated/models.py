"""
正弦波数学模型

用于信号拟合的标准正弦波模型。
"""

import numpy as np


def sine_wave(t: np.ndarray, amplitude: float, frequency: float,
              phase: float, dc_offset: float) -> np.ndarray:
    """
    正弦波数学模型：y = A·sin(2πft + φ) + B

    Args:
        t: 时间数组 (s)
        amplitude: 振幅 A
        frequency: 频率 f (Hz)
        phase: 初始相位 φ (rad)
        dc_offset: 直流偏移 B

    Returns:
        正弦波计算值
    """
    return amplitude * np.sin(2 * np.pi * frequency * t + phase) + dc_offset
