"""
模型运行器

加载训练好的模型参数，对记录进行电压预测。
"""

import json
import importlib
import os
import numpy as np

from scripts.utils.device import get_device
from scripts.utils.loader import get_dataset_model_path
from ..config.settings import config
from .feature_extractor import extract_from_record


def _get_model_path(algo_name=None):
    """根据算法名称和数据路径确定模型路径"""
    if algo_name is None:
        algo_name = config.estimation.algorithm
    
    # 首先尝试数据集特定的路径
    data_path = config.data_source.local_path
    model_base = get_dataset_model_path(data_path, config.estimation.model_path)
    
    # 根据算法类型确定文件扩展名
    if algo_name == "linear_model":
        ext = ".json"
    elif algo_name in ["random_forest_model", "extra_trees_model", "svr_model"]:
        ext = ".joblib"
    elif algo_name == "xgboost_model":
        ext = ".ubj"
    elif algo_name == "lightgbm_model":
        ext = ".txt"
    elif algo_name == "catboost_model":
        ext = ".cbm"
    elif algo_name in ["lenet", "lenet_hybrid", "lenet_bipath"]:
        ext = ".json"
    else:
        ext = ".json"
    
    candidate = f"{model_base}_{algo_name}{ext}"
    
    if os.path.exists(candidate):
        return candidate
    
    # 如果找不到，尝试默认路径（去掉 _algo 后缀）
    base, _ = os.path.splitext(config.estimation.model_path)
    default_candidate = f"{base}_{algo_name}{ext}"
    
    if os.path.exists(default_candidate):
        return default_candidate
    
    # 最后尝试原始默认路径
    if os.path.exists(config.estimation.model_path):
        return config.estimation.model_path
    
    raise FileNotFoundError(f"找不到模型文件。尝试的路径：\n1. {candidate}\n2. {default_candidate}\n3. {config.estimation.model_path}")


def _load_algorithm(algo_name=None):
    """动态加载配置中指定的算法模块。先尝试新路径 scripts.traditional，再回退 src.swa.estimation"""
    if algo_name is None:
        algo_name = config.estimation.algorithm
    for prefix in ("scripts.traditional", "src.swa.estimation"):
        try:
            return importlib.import_module(f"{prefix}.{algo_name}")
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(f"找不到算法模块: {algo_name}")


def load_model():
    """
    加载训练好的模型。

    支持三种格式：
    - JSON（线性模型：参数列表）
    - .ubj（XGBoost 原生格式）
    - .pth（PyTorch 状态字典，leNet 用）
    """
    model_path = _get_model_path()
    print(f"加载模型: {model_path}")

    with open(model_path) as f:
        meta = json.load(f)

    algo = meta.get("algorithm", config.estimation.algorithm)
    model_file = meta.get("model_path")

    # 如果 model_file 是相对路径，尝试转换为绝对路径（相对于 model_path 所在目录）
    if model_file and not os.path.isabs(model_file):
        model_dir = os.path.dirname(model_path)
        model_file = os.path.join(model_dir, os.path.basename(model_file))

    if model_file and model_file.endswith(".pth"):
        # PyTorch 模型
        import torch
        device = get_device()
        module = _load_algorithm(algo)
        wave_len = meta.get("wave_len", 512)
        if algo == "lenet_hybrid":
            n_fft = meta.get("n_fft", 10)
            model = module.HybridNet(n_fft=n_fft, wave_len=wave_len)
        elif algo == "lenet_bipath":
            n_fft = meta.get("n_fft", 10)
            model = module.BiPathNet(n_fft=n_fft, wave_len=wave_len)
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
    elif model_file and model_file.endswith(".cbm"):
        # CatBoost
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(model_file)
        return {"model": model}
    elif model_file and model_file.endswith(".joblib"):
        # sklearn 模型（RandomForest, ExtraTrees, SVR 等）
        import joblib
        model = joblib.load(model_file)
        return model
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
    module = _load_algorithm(algo)
    model = load_model()

    if algo == "lenet":
        features = _extract_raw(record).reshape(1, -1)
        return float(module.predict(model, features)[0])

    if algo == "lenet_hybrid" or algo == "lenet_bipath":
        return float(module.predict(model, record)[0])

    # 普通算法：FFT 特征
    features = extract_from_record(record).reshape(1, -1)
    return float(module.predict(model, features)[0])
