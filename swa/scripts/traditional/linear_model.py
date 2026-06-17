"""
完整线性模型：使用所有 16 个特征
    [A1-A10, T, RH, RPM, Vpp, Kurtosis, Skewness]
"""

import numpy as np

NAME = "完整线性模型: 16维特征 (A1-A10 + 环境 + 时域统计)"


def train(X: np.ndarray, y: np.ndarray):
    """
    最小二乘拟合完整线性模型
    X: shape (n, 16)
    y: shape (n,)
    """
    # 构造设计矩阵 [1, X]
    A = np.column_stack([np.ones_like(X[:, 0]), X])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return tuple(coeffs.tolist())


def predict(model, X: np.ndarray) -> np.ndarray:
    """预测电压"""
    b0 = model[0]
    coeffs = model[1:]
    return b0 + np.dot(X, coeffs)
