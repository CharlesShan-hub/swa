"""
CatBoost 模型：梯度提升树
"""

import numpy as np
from catboost import CatBoostRegressor

NAME = "CatBoost: 梯度提升树"


def train(X: np.ndarray, y: np.ndarray):
    """
    训练 CatBoost 回归模型
    X: shape (n, 16)
    y: shape (n,)
    """
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=100,
        eval_metric="RMSE",
        early_stopping_rounds=50
    )
    # 使用简单的验证划分（80%训练，20%验证）
    train_size = int(0.8 * len(X))
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:]
    y_val = y[train_size:]
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    return {"model": model}


def predict(model, X: np.ndarray) -> np.ndarray:
    """预测电压"""
    return model["model"].predict(X)
