"""
模型运行器

加载训练好的模型参数，对记录进行电压预测。
"""

import json
import importlib
import os
import numpy as np

from ..config.settings import config
from .feature_extractor import extract_from_record


def _load_algorithm():
    """动态加载配置中指定的算法模块"""
    algo_name = config.estimation.algorithm
    module = importlib.import_module(f"src.swa.estimation.{algo_name}")
    return module


def load_model():
    """
    加载训练好的模型。

    支持三种格式：
    - JSON（线性模型：参数列表）
    - .ubj（XGBoost 原生格式）
    - .pth（PyTorch 状态字典，leNet 用）
    """
    model_path = config.estimation.model_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    with open(model_path) as f:
        meta = json.load(f)

    algo = meta.get("algorithm", config.estimation.algorithm)
    model_file = meta.get("model_path")

    if model_file and model_file.endswith(".pth"):
        # PyTorch 模型
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module = importlib.import_module(f"src.swa.estimation.{algo}")
        wave_len = meta.get("wave_len", 512)
        if algo == "lenet_hybrid":
            n_fft = meta.get("n_fft", 10)
            model = module.HybridNet(n_fft=n_fft, wave_len=wave_len)
        else:
            model = module.LeNet1D()
        model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        result = {"model": model}
        # 加载归一化参数
        for k in ["wave_mean", "wave_std", "fft_mean", "fft_std", "env_mean", "env_std"]:
            if k in meta:
                result[k] = np.array(meta[k], dtype=np.float32)
        return result
    elif model_file and model_file.endswith(".ubj"):
        # XGBoost
        import xgboost as xgb
        bst = xgb.Booster()
        bst.load_model(model_file)
        return {"model": bst}
    elif model_file and model_file.endswith(".txt"):
        # LightGBM
        import lightgbm as lgb
        model = lgb.Booster(model_file=model_file)
        return {"model": model}
    else:
        # 普通线性模型
        return meta.get("params", meta)


def _extract_raw(record: dict) -> np.ndarray:
    """提取原始波形 + 环境参数 (515维)，供 LeNet 使用"""
    wave_str = record.get("RTU_REGS_P00_WAVE_DATA", "")
    wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]

    def _f(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    env = np.array([
        _f(record.get("RTU_REGS_P00_ENV_TEMP", 0)),
        _f(record.get("RTU_REGS_P00_ENV_HUMIDITY", 0)),
        _f(record.get("RTU_REGS_P00_ROTOR_RPM", 0)),
    ])
    return np.concatenate([wave, env])


def predict_from_record(record: dict) -> float:
    """对一条记录预测电压"""
    algo = config.estimation.algorithm
    module = _load_algorithm()
    model = load_model()

    if algo == "lenet":
        features = _extract_raw(record).reshape(1, -1)
        return float(module.predict(model, features)[0])

    if algo == "lenet_hybrid":
        return float(module.predict(model, record)[0])

    # 普通算法：FFT 特征
    features = extract_from_record(record).reshape(1, -1)
    return float(module.predict(model, features)[0])
