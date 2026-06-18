"""
LightGBM 模型：梯度提升树，比 XGBoost 更快，对特征尺度不敏感。

输入 13 维特征 [A1~A10, T, RH, RPM] 直接预测电压。
"""

import numpy as np
import lightgbm as lgb

from scripts.utils.device import get_lightgbm_device

NAME = "LightGBM"


def train(X: np.ndarray, y: np.ndarray, params: dict = None) -> dict:
    """
    训练 LightGBM 回归模型。

    Args:
        X: shape=(n, 18)，特征
        y: shape=(n,)，真实电压值
        params: LightGBM 超参数（可选）

    Returns:
        包含模型对象的字典
    """
    default_params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,      # L2 正则化
        "reg_alpha": 0.1,       # L1 正则化
        "random_state": 42,
        "verbose": -1,
        "device": get_lightgbm_device(),
    }
    if params:
        default_params.update(params)

    device_name = default_params["device"]
    print(f"  设备: {device_name}")

    model = lgb.LGBMRegressor(**default_params)
    model.fit(X, y)
    return {"model": model}


def predict(model_dict: dict, X: np.ndarray) -> np.ndarray:
    """预测电压"""
    return model_dict["model"].predict(X)
