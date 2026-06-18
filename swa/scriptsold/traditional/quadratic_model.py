"""
二次多项式回归模型
使用完整 16 维特征 + 平方项 + 交互项
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression


NAME = "二次多项式回归"


def train(X, y):
    """
    训练二次多项式回归模型
    """
    # 构造二次特征：一次项 + 平方项 + 交互项
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X)
    
    # 标准化
    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X_poly)
    
    # 训练
    model = LinearRegression()
    model.fit(X_poly_scaled, y)
    
    return {
        "model": model,
        "scaler": scaler,
        "poly": poly
    }


def predict(model_info, X):
    """预测"""
    poly = model_info["poly"]
    scaler = model_info["scaler"]
    model = model_info["model"]
    
    X_poly = poly.transform(X)
    X_poly_scaled = scaler.transform(X_poly)
    
    return model.predict(X_poly_scaled)
