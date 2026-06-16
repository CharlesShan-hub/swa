"""
二阶工程模型（文档第9节推荐公式）

V = b0
   + b1*A1 + b2*A2 + b3*A3
   + b4*T + b5*RH + b6*RPM
   + b7*A1²
   + b8*T² + b9*RH²
   + b10*A1·T + b11*A1·RH + b12*A1·RPM

共 13 个参数，兼顾精度与复杂度。
"""

import numpy as np

NAME = "二阶工程模型（13参数）"


def _build_design_matrix(X: np.ndarray) -> np.ndarray:
    """
    从原始特征构建设计矩阵。

    X 列: [A1..A10, Vpp, RMS, H2, H3, H4, T, RH, RPM]

    构造的列:
        1, A1, A2, A3, T, RH, RPM, A1², T², RH², A1·T, A1·RH, A1·RPM
    """
    A1 = X[:, 0]
    A2 = X[:, 1]
    A3 = X[:, 2]
    T = X[:, 10]
    RH = X[:, 11]
    RPM = X[:, 12]

    return np.column_stack([
        np.ones(len(X)),
        A1, A2, A3,
        T, RH, RPM,
        A1 ** 2,
        T ** 2, RH ** 2,
        A1 * T, A1 * RH, A1 * RPM,
    ])


def train(X: np.ndarray, y: np.ndarray):
    """最小二乘拟合 13 参数"""
    A = _build_design_matrix(X)
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs.tolist()


def predict(model, X: np.ndarray) -> np.ndarray:
    """预测电压"""
    coeffs = np.asarray(model)
    A = _build_design_matrix(X)
    return A @ coeffs
