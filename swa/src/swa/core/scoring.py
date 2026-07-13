"""
波形评分与特征提取 — 最小二乘周期投影

支持两种模式:
  - score: 7.0 + 8.1 周期加权评分（原始算法）
  - alpha7: 7.0 周期余弦分量（抗湿度和 RPM 干扰）
"""

from typing import Optional
import numpy as np


def compute_score(
    wave: np.ndarray,
    f1: float = 7.0,
    f2: float = 8.1,
    w: float = 0.25,
) -> float:
    """
    最小二乘周期投影加权评分。

    Args:
        wave: 512 点原始波形
        f1: 第一个周期数（默认 7.0）
        f2: 第二个周期数（默认 8.1）
        w: beta 权重（默认 0.25）

    Returns:
        评分值（与电压正相关）
    """
    y = wave - np.mean(wave)
    n = len(y)
    t = np.arange(n, dtype=np.float64)

    c1 = np.cos(2 * np.pi * f1 * t / n)
    s1 = np.sin(2 * np.pi * f1 * t / n)
    c2 = np.cos(2 * np.pi * f2 * t / n)
    s2 = np.sin(2 * np.pi * f2 * t / n)

    X = np.column_stack([c1, s1, c2, s2])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a1, b1, a2, b2 = coeffs

    alpha = np.hypot(a1, b1)
    beta = np.hypot(a2, b2)
    return float(alpha - w * beta)


def compute_alpha7(wave: np.ndarray) -> Optional[float]:
    """
    7.0 周期余弦分量（alpha_7）。

    alpha_7 对湿度和 RPM 变化不敏感，适合作为稳定特征。

    Args:
        wave: 512 点原始波形

    Returns:
        alpha_7 值（余弦分量幅度），波形异常时返回 None
    """
    y = wave - np.mean(wave)
    n = len(y)
    t = np.arange(n, dtype=np.float64)

    c7 = np.cos(2 * np.pi * 7.0 * t / n)
    s7 = np.sin(2 * np.pi * 7.0 * t / n)

    X = np.column_stack([c7, s7])
    try:
        a7, b7 = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(np.hypot(a7, b7))
    except np.linalg.LinAlgError:
        return None


def s20_smooth(values: list[float], window: int = 20) -> list[float]:
    """
    S20 滑动窗口平均。

    Args:
        values: 原始值序列
        window: 窗口大小（默认 20）

    Returns:
        平滑后的序列（前 window-1 个值为 None）
    """
    result: list[float] = []
    for i in range(len(values)):
        if i < window - 1:
            result.append(float("nan"))
        else:
            result.append(float(np.mean(values[i - window + 1 : i + 1])))
    return result
