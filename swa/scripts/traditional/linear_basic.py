"""
线性基波模型: V = b0 + b1 * A1

只用基波幅值拟合电压，最简单。
"""

import numpy as np

NAME = "线性基波: V = b0 + b1*A1"


def train(X: np.ndarray, y: np.ndarray):
    """
    最小二乘拟合 V = b0 + b1*A1。

    Args:
        X: shape=(n, 13)，取第 0 列 (A1)
        y: shape=(n,)，真实电压值

    Returns:
        (b0, b1)
    """
    A1 = X[:, 0]
    # 构造设计矩阵 [1, A1]
    A = np.column_stack([np.ones_like(A1), A1])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return tuple(coeffs.tolist())  # (b0, b1)


def predict(model, X: np.ndarray) -> np.ndarray:
    """预测电压"""
    b0, b1 = model
    return b0 + b1 * X[:, 0]
