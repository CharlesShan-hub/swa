
"""
二次多项式回归模型
包含一次项、平方项和交互项
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression


NAME = "二次多项式回归"


def _create_poly_features(X):
    """
    构造二次特征，只包含：
    - 一次项 (A1, A2, A3, T, RH, RPM)
    - 平方项 (A1^2, A2^2, A3^2, T^2, RH^2, RPM^2)
    - 交互项 (A1*T, A1*RH, A1*RPM)
    """
    # 先取前6个特征：假设顺序是 [A1, A2, A3, T, RH, RPM, ...]
    # 从我们的特征中：A1是第0个，A2第1，A3第2，T第10，RH第11，RPM第12
    # 不对，让我们看一下 feature_extractor 里的顺序
    # 实际我们的16个特征是：A1~A10, T, RH, RPM, Vpp, Kurtosis, Skewness
    idx_A1 = 0
    idx_A2 = 1
    idx_A3 = 2
    idx_T = 10
    idx_RH = 11
    idx_RPM = 12
    
    selected = X[:, [idx_A1, idx_A2, idx_A3, idx_T, idx_RH, idx_RPM]]
    
    n_samples = selected.shape[0]
    n_orig = selected.shape[1]
    
    # 构造新特征矩阵
    # 列顺序：截距(后面加), A1, A2, A3, T, RH, RPM,
    #         A1^2, A2^2, A3^2, T^2, RH^2, RPM^2,
    #         A1*T, A1*RH, A1*RPM
    features = []
    
    # 先加原始项
    for i in range(n_orig):
        features.append(selected[:, i].reshape(-1, 1))
    
    # 平方项
    for i in range(n_orig):
        features.append((selected[:, i] ** 2).reshape(-1, 1))
    
    # A1的交互项：A1*T, A1*RH, A1*RPM
    a1 = selected[:, 0]
    t = selected[:, 3]
    rh = selected[:, 4]
    rpm = selected[:, 5]
    features.append((a1 * t).reshape(-1, 1))
    features.append((a1 * rh).reshape(-1, 1))
    features.append((a1 * rpm).reshape(-1, 1))
    
    # 拼接所有特征
    return np.hstack(features)


def train(X, y):
    """
    训练二次多项式回归模型
    """
    # 构造二次特征
    X_poly = _create_poly_features(X)
    
    # 标准化特征（多项式回归通常需要）
    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X_poly)
    
    # 训练线性回归
    model = LinearRegression()
    model.fit(X_poly_scaled, y)
    
    return {
        "model": model,
        "scaler": scaler
    }


def predict(model_info, X):
    """
    预测
    """
    model = model_info["model"]
    scaler = model_info["scaler"]
    
    X_poly = _create_poly_features(X)
    X_poly_scaled = scaler.transform(X_poly)
    
    return model.predict(X_poly_scaled)
