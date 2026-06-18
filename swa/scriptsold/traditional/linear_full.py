"""
线性全参数模型: V = b0 + b1*A1 + b2*T + b3*RH + b4*RPM

加入转速归一化补偿。
"""

import numpy as np

NAME = "线性全参: V = b0 + b1*A1 + b2*T + b3*RH + b4*RPM"


def train(X: np.ndarray, y: np.ndarray):
    """最小二乘拟合"""
    # 取 A1, T, RH, RPM
    A = X[:, [0, 10, 11, 12]]
    A = np.column_stack([np.ones(len(A)), A])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return tuple(coeffs.tolist())


def predict(model, X: np.ndarray) -> np.ndarray:
    """预测电压"""
    b0, b1, b2, b3, b4 = model
    return b0 + b1 * X[:, 0] + b2 * X[:, 10] + b3 * X[:, 11] + b4 * X[:, 12]
