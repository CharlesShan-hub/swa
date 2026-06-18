"""
分电压评估脚本 - 全面评估版

加载训练好的模型，逐条预测，按真实电压分桶统计：
- 基础指标：MAE / RMSE
- 百分位误差：95% / 99% / 最大值
- 符号准确率
- 投/退判断准确率

用法：
    uv run python scripts/evaluate_by_voltage.py --model data/model_params_linear_model.json --data data/exported_data.jsonl
    uv run python scripts/evaluate_by_voltage.py --model data/model_params_random_forest_model.joblib --data data/exported_data.jsonl
    uv run python scripts/evaluate_by_voltage.py --model data/model_params_catboost_model.cbm --data data/exported_data.jsonl
    uv run python scripts/evaluate_by_voltage.py --model data/model_nfft11.json --data data/exported_data.jsonl
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from scipy.fftpack import fft
from scipy.stats import kurtosis, skew
from tqdm import tqdm

from src.swa.config.settings import config
from scripts.utils.loader import load_jsonl, split_jsonl, get_dataset_model_path
from src.swa.estimation.feature_extractor import extract_from_record


def parse_voltage(v) -> float:
    """解析 ACTUAL_VOLTAGE 字段为数值"""
    if v is None:
        return float("nan")
    s = str(v).strip().lower()
    s = s.replace("v", "").strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_model(model_path: str):
    """
    加载各种类型的模型
    
    支持的模型：
    - linear_model: .json
    - RandomForest/ExtraTrees/SVR: .joblib
    - XGBoost: .ubj
    - LightGBM: .txt
    - CatBoost: .cbm
    - 深度学习 (LeNet/LeNetHybrid): .json + .pth
    """
    ext = os.path.splitext(model_path)[1].lower()
    
    # 从文件名中推断算法名
    filename = os.path.basename(model_path)
    algorithm_name = None
    for algo in ["linear_model", "quadratic_model", "random_forest_model", "extra_trees_model", "svr_model", "xgboost_model", "lightgbm_model", "catboost_model"]:
        if algo in filename:
            algorithm_name = algo
            break
    
    if ext == ".json":
        # 可能是线性模型或深度学习模型的元文件
        with open(model_path) as f:
            meta = json.load(f)
        
        if "coef" in meta or "params" in meta:
            # 线性模型：参数是 [intercept, coef1, coef2, ..., coef16]
            coef_key = "coef" if "coef" in meta else "params"
            params = np.array(meta[coef_key])
            return {
                "type": "linear",
                "meta": meta,
                "coef": params[1:],
                "intercept": params[0],
                "algorithm": "linear_model"
            }
        elif "algorithm" in meta and meta.get("model_path", "").endswith(".pth"):
            # 深度学习模型
            return _load_dl_model(model_path, meta)
        else:
            raise ValueError("不认识的 JSON 模型格式")
    
    elif ext == ".joblib":
        # sklearn 模型 (RandomForest, ExtraTrees, SVR)
        import joblib
        loaded = joblib.load(model_path)
        model = loaded
        # 抑制 sklearn 的 verbose 输出
        if isinstance(loaded, dict) and "model" in loaded and hasattr(loaded["model"], "verbose"):
            loaded["model"].verbose = 0
        return {
            "type": "sklearn",
            "model": model,
            "algorithm": algorithm_name
        }
    
    elif ext == ".ubj":
        # XGBoost
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        return {
            "type": "xgboost",
            "model": {"model": model},
            "algorithm": "xgboost_model"
        }
    
    elif ext == ".txt":
        # LightGBM
        import lightgbm as lgb
        model = lgb.Booster(model_file=model_path)
        return {
            "type": "lightgbm",
            "model": {"model": model},
            "algorithm": "lightgbm_model"
        }
    
    elif ext == ".cbm":
        # CatBoost
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(model_path)
        return {
            "type": "catboost",
            "model": {"model": model},
            "algorithm": "catboost_model"
        }
    
    else:
        raise ValueError(f"不支持的模型格式: {ext}")


def _load_dl_model(meta_path: str, meta: dict):
    """加载深度学习模型 (LeNet/LeNetHybrid/LeNetBiPath)"""
    from scripts.utils.device import get_device
    device = get_device()
    
    algo = meta.get("algorithm")
    model_file = meta.get("model_path")
    n_fft = meta.get("n_fft", 10)
    
    if not model_file or not model_file.endswith(".pth"):
        raise ValueError(f"深度学习模型需要 .pth 文件，当前: {model_file}")
    
    # 补齐路径
    model_path = model_file if os.path.isabs(model_file) else os.path.join(
        os.path.dirname(os.path.abspath(meta_path)), os.path.basename(model_file)
    )
    
    if algo == "lenet_hybrid":
        from src.swa.estimation.lenet_hybrid import HybridNet
        model = HybridNet(n_fft=n_fft, n_env=6)
    elif algo == "lenet_bipath":
        from src.swa.estimation.lenet_bipath import BiPathNet
        model = BiPathNet(n_fft=n_fft, n_env=6)
    elif algo == "lenet":
        from src.swa.estimation.lenet import LeNet1D
        model = LeNet1D()
    else:
        raise ValueError(f"不支持的深度学习算法: {algo}")
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # 归一化参数
    norm_params = {}
    for k in ["wave_mean", "wave_std", "fft_mean", "fft_std", "env_mean", "env_std"]:
        if k in meta:
            norm_params[k] = np.array(meta[k], dtype=np.float32)
    
    return {
        "type": "dl",
        "model": model,
        "norm_params": norm_params,
        "algo": algo,
        "n_fft": n_fft,
        "device": device
    }


def predict(model_info, record):
    """统一预测接口"""
    model_type = model_info["type"]
    
    if model_type == "dl":
        return _predict_dl(model_info, record)
    
    # 对于传统 ML 模型，使用对应的算法模块的 predict 函数
    if "algorithm" in model_info and model_info["algorithm"] is not None:
        import importlib
        try:
            module = importlib.import_module(f"scripts.traditional.{model_info['algorithm']}")
            features = extract_from_record(record).reshape(1, -1)
            return float(module.predict(model_info["model"], features)[0])
        except Exception:
            pass
    
    # 备用方案（如果无法使用算法模块）
    if model_type == "linear":
        features = extract_from_record(record)
        return float(np.dot(features, model_info["coef"]) + model_info["intercept"])
    
    elif model_type in ["sklearn", "xgboost", "catboost"]:
        features = extract_from_record(record).reshape(1, -1)
        model = model_info["model"]
        if isinstance(model, dict) and "model" in model:
            return float(model["model"].predict(features)[0])
        else:
            return float(model.predict(features)[0])
    
    elif model_type == "lightgbm":
        features = extract_from_record(record).reshape(1, -1)
        model = model_info["model"]
        if isinstance(model, dict) and "model" in model:
            return float(model["model"].predict(features)[0])
        else:
            return float(model.predict(features)[0])
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def _predict_dl(model_info, record):
    """深度学习预测"""
    model = model_info["model"]
    norm_params = model_info["norm_params"]
    algo = model_info["algo"]
    n_fft = model_info["n_fft"]
    device = model_info["device"]
    
    wave_str = record.get("RTU_REGS_P00_WAVE_DATA", "")
    wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]
    
    def _f(v, d=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return d
    
    temp = _f(record.get("RTU_REGS_P00_ENV_TEMP"))
    humid = _f(record.get("RTU_REGS_P00_ENV_HUMIDITY"))
    rpm = _f(record.get("RTU_REGS_P00_ROTOR_RPM"))
    
    # FFT 谐波
    ac = wave - np.mean(wave)
    n = len(ac)
    fft_mag = np.abs(fft(ac))[:n // 2]
    harmonics = np.zeros(n_fft)
    for i in range(n_fft):
        idx = i + 1
        if idx < len(fft_mag):
            harmonics[i] = 2.0 * fft_mag[idx] / n
    
    # 时域特征（从 record 读取，或计算）
    vpp = record.get("vpp")
    kurt = record.get("kurtosis")
    skewness = record.get("skewness")
    if vpp is None or kurt is None or skewness is None:
        ac = wave - np.mean(wave)
        vpp = float(np.max(ac) - np.min(ac))
        kurt = float(kurtosis(ac, fisher=False))
        skewness = float(skew(ac))
    
    if algo in ["lenet_hybrid", "lenet_bipath"]:
        # 波形归一化（逐条）
        wm = np.mean(wave)
        ws = np.std(wave) + 1e-8
        wave_norm = ((wave - wm) / ws).reshape(1, -1)
        
        # FFT 和环境归一化（用训练集的参数）
        fft_norm = ((harmonics - norm_params["fft_mean"]) / norm_params["fft_std"]).reshape(1, -1)
        aux = np.array([temp, humid, rpm, vpp, kurt, skewness])
        env_norm = ((aux - norm_params["env_mean"]) / norm_params["env_std"]).reshape(1, -1)
        
        wave_t = torch.tensor(wave_norm, dtype=torch.float32).to(device)
        fft_t = torch.tensor(fft_norm, dtype=torch.float32).to(device)
        env_t = torch.tensor(env_norm, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            if algo == "lenet_hybrid":
                pred = model(wave_t, fft_t, env_t).cpu().numpy()[0]
            else:
                pred = model(wave_t, fft_t, env_t).cpu().numpy()[0]
        return float(pred)
    
    elif algo == "lenet":
        env = np.array([temp, humid, rpm])
        ws = wave.copy()
        wm = np.mean(ws)
        wstd = np.std(ws) + 1e-8
        wave_norm = ((ws - wm) / wstd).reshape(1, -1)
        env_norm = env.reshape(1, -1)
        full = np.concatenate([wave_norm, env_norm], axis=1)
        x = torch.tensor(full, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            wave_t = x[:, :512]
            env_t = x[:, 512:]
            pred = model(wave_t, env_t).cpu().numpy()[0]
        return float(pred)
    
    else:
        raise ValueError(f"不支持的深度学习算法: {algo}")


def calculate_metrics(y_true, y_pred, threshold_abs=30.0):
    """
    计算全面的评估指标（忽略正负号，只比较绝对值）
    
    Returns:
        dict: 包含所有指标的字典
    """
    y_true = np.abs(np.array(y_true))
    y_pred = np.abs(np.array(y_pred))
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    # 计算相对误差（对于0V用绝对误差）
    rel_err_mask = y_true != 0
    rel_errors = np.zeros_like(y_true)
    rel_errors[rel_err_mask] = abs_errors[rel_err_mask] / np.abs(y_true[rel_err_mask])
    # 对于0V，假设"在范围内"的标准是绝对值<1V（因为没有参考值）
    # 或者对于所有样本，同时支持绝对误差和相对误差两个标准，
    # 取更宽松的那个（即满足任一就算对）
    abs_error_5 = abs_errors < 5.0  # 绝对误差5V
    abs_error_10 = abs_errors < 10.0  # 绝对误差10V
    abs_error_15 = abs_errors < 15.0  # 绝对误差15V
    
    rel_error_5 = np.zeros_like(y_true, dtype=bool)
    rel_error_10 = np.zeros_like(y_true, dtype=bool)
    rel_error_15 = np.zeros_like(y_true, dtype=bool)
    
    rel_error_5[rel_err_mask] = rel_errors[rel_err_mask] < 0.05
    rel_error_10[rel_err_mask] = rel_errors[rel_err_mask] < 0.10
    rel_error_15[rel_err_mask] = rel_errors[rel_err_mask] < 0.15
    
    # 对于非0V，用相对误差；对于0V，用绝对误差
    acc_5 = np.mean(
        (rel_err_mask & rel_error_5) | (~rel_err_mask & abs_error_5)
    )
    acc_10 = np.mean(
        (rel_err_mask & rel_error_10) | (~rel_err_mask & abs_error_10)
    )
    acc_15 = np.mean(
        (rel_err_mask & rel_error_15) | (~rel_err_mask & abs_error_15)
    )
    
    # 基础指标
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    # 百分位误差
    p95 = float(np.percentile(abs_errors, 95))
    p99 = float(np.percentile(abs_errors, 99))
    max_err = float(np.max(abs_errors))
    
    # 符号准确率
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    sign_acc = float(np.mean(sign_true == sign_pred))
    
    # 投/退判断准确率（保留，但可能不是重点）
    status_true = np.abs(y_true) > threshold_abs
    status_pred = np.abs(y_pred) > threshold_abs
    status_acc = float(np.mean(status_true == status_pred))
    
    # 投/退的混淆矩阵
    tp = float(np.sum((status_true & status_pred)))
    tn = float(np.sum((~status_true & ~status_pred)))
    fp = float(np.sum((~status_true & status_pred)))
    fn = float(np.sum((status_true & ~status_pred)))
    
    # 精确率和召回率（针对"投"状态）
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        "n": len(y_true),
        "mae": mae,
        "rmse": rmse,
        "p95": p95,
        "p99": p99,
        "max_err": max_err,
        "sign_acc": sign_acc,
        "status_acc": status_acc,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "acc_5": float(acc_5),  # ±5% 误差内准确率
        "acc_10": float(acc_10), # ±10% 误差内准确率
        "acc_15": float(acc_15)  # ±15% 误差内准确率
    }


def main():
    parser = argparse.ArgumentParser(description="分电压评估模型 - 全面版")
    parser.add_argument("--model", 
                        help="模型文件路径 (支持 .json/.joblib/.ubj/.txt/.cbm)")
    parser.add_argument("--algorithm", 
                        choices=["linear_model", "quadratic_model", "random_forest_model", "extra_trees_model", "svr_model",
                                 "xgboost_model", "lightgbm_model", "catboost_model",
                                 "lenet", "lenet_hybrid", "lenet_bipath"],
                        help="算法名称，配合 --data 自动推断模型路径")
    parser.add_argument("--data", default=config.data_source.local_path,
                        help="数据文件路径")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制评估条数 (默认: 全部)")
    parser.add_argument("--threshold", type=float, default=30.0,
                        help="投/退判断阈值 (默认: 30V)")
    args = parser.parse_args()

    # 确定模型路径
    if args.model is None and args.algorithm is None:
        raise ValueError("必须指定 --model 或 --algorithm 中的一个")
    
    if args.model is None:
        # 自动推断模型路径
        model_base = get_dataset_model_path(args.data, "data/model_params")
        algorithm = args.algorithm
        
        # 根据算法名称确定扩展名
        if algorithm in ["linear_model"]:
            ext = ".json"
        elif algorithm in ["random_forest_model", "extra_trees_model", "svr_model", "quadratic_model"]:
            ext = ".joblib"
        elif algorithm in ["xgboost_model"]:
            ext = ".ubj"
        elif algorithm in ["lightgbm_model"]:
            ext = ".txt"
        elif algorithm in ["catboost_model"]:
            ext = ".cbm"
        elif algorithm in ["lenet", "lenet_hybrid", "lenet_bipath"]:
            ext = ".json"
        
        args.model = f"{model_base}_{algorithm}{ext}"
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model_info = load_model(args.model)
    print(f"  类型: {model_info['type']}")

    # 加载数据
    print(f"加载数据: {args.data}")
    records = load_jsonl(args.data)

    # 使用与训练完全相同的切分逻辑（跟 train_model.py 一致！）
    train_records, val_records, test_records = split_jsonl(
        records,
        full_dataset=True,
        limit=0,
        train_ratio=0.9,
        val_ratio=0.0,
        test_ratio=0.1,
        seed=42
    )
    # 注意：只取测试集进行评估，完全不会用到训练/验证数据！
    if args.limit:
        test_records = test_records[:args.limit]
    print(f"评估集 (纯测试集): {len(test_records)} 条\n")

    # 逐条预测
    results_by_voltage = {}  # {voltage: {"y_true": [...], "y_pred": [...], "temps": [...], "hums": [...]}}
    all_y_true = []
    all_y_pred = []
    skipped = 0
    
    for i, rec in enumerate(tqdm(test_records, desc="评估中", unit="条", file=sys.stderr)):
        voltage = parse_voltage(rec.get("ACTUAL_VOLTAGE"))
        if np.isnan(voltage):
            skipped += 1
            continue
        
        pred = predict(model_info, rec)
        
        # 解析温湿度
        temp = rec.get("RTU_REGS_P00_ENV_TEMP")
        humid = rec.get("RTU_REGS_P00_ENV_HUMIDITY")
        try:
            temp_f = float(temp) if temp is not None else float("nan")
        except (ValueError, TypeError):
            temp_f = float("nan")
        try:
            humid_f = float(humid) if humid is not None else float("nan")
        except (ValueError, TypeError):
            humid_f = float("nan")
        
        # 按 10V 分桶
        bucket = round(voltage / 10) * 10
        if bucket not in results_by_voltage:
            results_by_voltage[bucket] = {"y_true": [], "y_pred": [], "temps": [], "hums": []}
        results_by_voltage[bucket]["y_true"].append(voltage)
        results_by_voltage[bucket]["y_pred"].append(pred)
        results_by_voltage[bucket]["temps"].append(temp_f)
        results_by_voltage[bucket]["hums"].append(humid_f)
        
        all_y_true.append(voltage)
        all_y_pred.append(pred)

    # 计算整体指标
    overall_metrics = calculate_metrics(all_y_true, all_y_pred, args.threshold)
    
    # 打印整体结果
    print(f"\n{'='*80}")
    print(f"{'整体评估结果':^80}")
    print(f"{'='*80}")
    print(f"样本数: {overall_metrics['n']}  (跳过 {skipped} 条)")
    print(f"")
    print(f"误差指标:")
    print(f"  MAE:      {overall_metrics['mae']:>7.4f} V")
    print(f"  RMSE:     {overall_metrics['rmse']:>7.4f} V")
    print(f"  95%分位:  {overall_metrics['p95']:>7.4f} V")
    print(f"  99%分位:  {overall_metrics['p99']:>7.4f} V")
    print(f"  最大误差: {overall_metrics['max_err']:>7.4f} V")
    print(f"")
    print(f"电压测量精度:")
    print(f"  ±5% 误差内准确率:  {overall_metrics['acc_5']*100:>6.2f}%")
    print(f"  ±10% 误差内准确率: {overall_metrics['acc_10']*100:>6.2f}%")
    print(f"  ±15% 误差内准确率: {overall_metrics['acc_15']*100:>6.2f}%")
    print(f"")
    
    # 打印分电压结果
    print(f"\n{'='*160}")
    print(f"{'分电压评估结果':^160}")
    print(f"{'='*160}")
    print(f"{'电压':>8} | {'数量':>6} | {'占比':>6} | {'MAE':>8} | {'RMSE':>8} | {'P95':>8} | {'P99':>8} | {'±5%':>8} | {'±10%':>8} | {'±15%':>8} | {'平均预测':>8}")
    print(f"{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    
    for bucket in sorted(results_by_voltage.keys()):
        y_true = results_by_voltage[bucket]["y_true"]
        y_pred = results_by_voltage[bucket]["y_pred"]
        metrics = calculate_metrics(y_true, y_pred, args.threshold)
        pct = metrics["n"] / len(test_records) * 100
        mean_pred = float(np.mean(y_pred))
        
        print(f"{bucket:>7}V | {metrics['n']:>6} | {pct:>5.1f}% | "
              f"{metrics['mae']:>7.4f} | {metrics['rmse']:>7.4f} | {metrics['p95']:>7.4f} | {metrics['p99']:>7.4f} | "
              f"{metrics['acc_5']*100:>7.2f}% | {metrics['acc_10']*100:>7.2f}% | {metrics['acc_15']*100:>7.2f}% | "
              f"{mean_pred:>7.2f}")
    
    print(f"{'='*160}")

    # ── 电压×温度 交叉表 ──
    print(f"\n{'='*120}")
    print(f"{'电压×温度分布  (按5°C分桶)':^120}")
    print(f"{'='*120}")
    all_temp_buckets = set()
    vt = {}
    for bucket in results_by_voltage:
        vt[bucket] = {}
        for t in results_by_voltage[bucket]["temps"]:
            if not np.isnan(t):
                tb = round(t / 5) * 5
                vt[bucket][tb] = vt[bucket].get(tb, 0) + 1
                all_temp_buckets.add(tb)
    all_temp_buckets = sorted(all_temp_buckets)
    if all_temp_buckets:
        header = f"{'电压':>8} | {'数量':>6}"
        for tb in all_temp_buckets:
            header += f" | {tb:>4}°C"
        print(header)
        print(f"{'-'*8}-+-{'-'*6}" + "".join(f"-+-{'-'*6}" for _ in all_temp_buckets))
        for bucket in sorted(vt):
            n = len(results_by_voltage[bucket]["y_true"])
            line = f"{bucket:>7}V | {n:>6}"
            for tb in all_temp_buckets:
                c = vt[bucket].get(tb, 0)
                pct = c / n * 100 if n > 0 else 0
                line += f" | {c:>3}({pct:>4.1f}%)"
            print(line)
        print(f"{'='*120}")

    # ── 电压×湿度 交叉表 ──
    print(f"\n{'='*120}")
    print(f"{'电压×湿度分布  (按5%分桶)':^120}")
    print(f"{'='*120}")
    all_hum_buckets = set()
    vh = {}
    for bucket in results_by_voltage:
        vh[bucket] = {}
        for h in results_by_voltage[bucket]["hums"]:
            if not np.isnan(h):
                hb = round(h / 5) * 5
                vh[bucket][hb] = vh[bucket].get(hb, 0) + 1
                all_hum_buckets.add(hb)
    all_hum_buckets = sorted(all_hum_buckets)
    if all_hum_buckets:
        header = f"{'电压':>8} | {'数量':>6}"
        for hb in all_hum_buckets:
            header += f" | {hb:>3}%"
        print(header)
        print(f"{'-'*8}-+-{'-'*6}" + "".join(f"-+-{'-'*5}" for _ in all_hum_buckets))
        for bucket in sorted(vh):
            n = len(results_by_voltage[bucket]["y_true"])
            line = f"{bucket:>7}V | {n:>6}"
            for hb in all_hum_buckets:
                c = vh[bucket].get(hb, 0)
                pct = c / n * 100 if n > 0 else 0
                line += f" | {c:>3}({pct:>4.1f}%)"
            print(line)
        print(f"{'='*120}")


if __name__ == "__main__":
    main()
