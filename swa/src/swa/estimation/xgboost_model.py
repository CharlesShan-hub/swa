"""
XGBoost 模型：输入 [A1~A10, T, RH, RPM] 直接预测电压。

支持样本加权，缓解数据不平衡问题。
"""

import numpy as np
import xgboost as xgb

NAME = "XGBoost (推荐)"


def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    计算样本权重：按电压分组，少数类权重大。

    将电压四舍五入到最近的 10V 作为分组依据，
    每组权重 = 总样本数 / (组数 × 该组样本数)
    权重上限 5.0，下限 0.2，避免极端值。
    """
    groups = np.round(y / 10) * 10
    unique, counts = np.unique(groups, return_counts=True)
    total = len(y)
    n_groups = len(unique)
    weight_map = {u: total / (n_groups * c) for u, c in zip(unique, counts)}
    weights = np.array([weight_map[g] for g in groups])
    # 限幅，防止极端权重
    weights = np.clip(weights, 0.2, 5.0)
    return weights


def train(X: np.ndarray, y: np.ndarray, params: dict = None,
          sample_weight: bool = True) -> dict:
    """
    训练 XGBoost 回归模型。

    Args:
        X: shape=(n, 13)，特征
        y: shape=(n,)，真实电压值
        params: XGBoost 超参数（可选）
        sample_weight: 是否使用样本加权平衡数据

    Returns:
        包含模型对象的字典
    """
    default_params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.1,
        "lambda": 1.0,
        "alpha": 0.1,
        "gamma": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBRegressor(**default_params)

    if sample_weight:
        sw = _compute_sample_weights(y)
        model.fit(X, y, sample_weight=sw)
        # 打印权重信息
        unique_v = np.unique(np.round(y / 10) * 10)
        print(f"  样本加权: 已启用 (最少权重={sw.min():.2f}, 最多权重={sw.max():.2f})")
    else:
        model.fit(X, y)

    return {"model": model}


def predict(model_dict: dict, X: np.ndarray) -> np.ndarray:
    """预测电压"""
    model = model_dict["model"]
    if isinstance(model, xgb.Booster):
        return model.predict(xgb.DMatrix(X))
    return model.predict(X)
