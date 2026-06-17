"""
纯信号模型：只使用 A1~A10 + Vpp + Kurtosis + Skewness
完全不依赖温度 / 湿度 / 转速等环境参数。
"""

import numpy as np
from sklearn.linear_model import LinearRegression


NAME = "纯信号线性模型（无温湿度）"


def _select_features(X):
    """
    X: (n, 16) — [A1..A10, T, RH, RPM, Vpp, Kurt, Skew]
    取索引 0~9 (A1~A10), 13~15 (Vpp, Kurt, Skew)
    共 13 维
    """
    return np.hstack([X[:, 0:10], X[:, 13:16]])


def train(X, y):
    X_sel = _select_features(X)
    model = LinearRegression()
    model.fit(X_sel, y)
    return {"model": model}


def predict(model_info, X):
    model = model_info["model"]
    X_sel = _select_features(X)
    return model.predict(X_sel)
