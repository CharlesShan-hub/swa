"""
混合模型：A1~A10 用一次项，T/RH/RPM/Vpp/Kurt/Skew 用二次项
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


NAME = "混合模型: A1~A10线性 + 环境/时域二次"


def _build_features(X):
    """
    X: (n, 16) — [A1..A10, T, RH, RPM, Vpp, Kurt, Skew]
    
    构造：
    - A1~A10     → 一次项 (10 个)
    - T/RH/RPM/Vpp/Kurt/Skew → 一次项 + 平方项 (12 个)
    
    总共: 10 + 12 = 22 个特征 + 截距
    """
    # A1~A10: 索引 0~9
    harm = X[:, 0:10]

    # T/RH/RPM/Vpp/Kurt/Skew: 索引 10~15
    env = X[:, 10:16]

    # 一次项
    env_lin = env

    # 平方项
    env_sq = env ** 2

    return np.hstack([harm, env_lin, env_sq])


def train(X, y):
    X_aug = _build_features(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_aug)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    return {"model": model, "scaler": scaler}


def predict(model_info, X):
    model = model_info["model"]
    scaler = model_info["scaler"]
    
    X_aug = _build_features(X)
    X_scaled = scaler.transform(X_aug)
    
    return model.predict(X_scaled)
