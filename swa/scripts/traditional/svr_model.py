"""
SVR 模型：支持向量回归（RBF 核）
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

NAME = "SVR：支持向量回归（RBF 核）"


def train(X: np.ndarray, y: np.ndarray):
    """
    训练 SVR 回归模型
    X: shape (n, 16)
    y: shape (n,)
    """
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = SVR(
        kernel='rbf',
        C=1.0,
        epsilon=0.1,
        verbose=True
    )
    model.fit(X_scaled, y)
    
    return {"model": model, "scaler": scaler}


def predict(model, X: np.ndarray) -> np.ndarray:
    """预测电压"""
    # 使用保存的 scaler 标准化特征
    X_scaled = model["scaler"].transform(X)
    return model["model"].predict(X_scaled)
