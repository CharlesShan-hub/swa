"""
RandomForest 模型：随机森林回归
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

NAME = "RandomForest：随机森林回归"


def train(X: np.ndarray, y: np.ndarray):
    """
    训练 RandomForest 回归模型
    X: shape (n, 16)
    y: shape (n,)
    """
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X, y)
    return {"model": model}


def predict(model, X: np.ndarray) -> np.ndarray:
    """预测电压"""
    return model["model"].predict(X)
