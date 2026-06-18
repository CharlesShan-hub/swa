"""
线性环境模型: V = b0 + b1*A1 + b2*T + b3*RH

在基波基础上加入温湿度补偿。
"""

import numpy as np

NAME = "线性环境: V = b0 + b1*A1 + b2*T + b3*RH"


def train(X: np.ndarray, y: np.ndarray):
    """
    最小二乘拟合。

    Args:
        X: shape=(n, 13)，取 [A1, T, RH]
        y: shape=(n,)

    Returns:
        (b0, b1, b2, b3)
    """
    A = X[:, [0, 10, 11]]  # A1, T, RH
    A = np.column_stack([np.ones(len(A)), A])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return tuple(coeffs.tolist())


def predict(model, X: np.ndarray) -> np.ndarray:
    """预测电压"""
    b0, b1, b2, b3 = model
    return b0 + b1 * X[:, 0] + b2 * X[:, 10] + b3 * X[:, 11]
