"""
ExtraTrees 模型：极端随机树回归
"""

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

NAME = "ExtraTrees：极端随机树回归（抗噪）"


def train(X: np.ndarray, y: np.ndarray):
    """
    训练 ExtraTrees 回归模型
    X: shape (n, 16)
    y: shape (n,)
    """
    model = ExtraTreesRegressor(
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
