"""
训练脚本：从本地 JSONL 数据训练电压估算模型。

训练/测试数量在 src/swa/config/settings.py 中配置：
    estimation.train_size = 25000
    estimation.test_size  = 5000

用法：
    uv run python scripts/train_model.py --algorithm linear_basic
    uv run python scripts/train_model.py --algorithm quadratic_model
    uv run python scripts/train_model.py --algorithm xgboost_model
"""

import argparse
import json
import sys
import os
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import random

# 固定随机种子，保证可复现
_SEED = 42
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)

from src.swa.config.settings import config
from src.swa.estimation.feature_extractor import extract_from_record
from src.swa.signal_process.loader import load_jsonl


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


def _extract_features(rec: dict, algorithm: str) -> np.ndarray:
    """根据算法类型提取特征"""
    if algorithm == "lenet":
        # LeNet 吃原始波形 (512,) + [T, RH, RPM]
        wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
        wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]

        def _f(v, default=0.0):
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        env = np.array([
            _f(rec.get("RTU_REGS_P00_ENV_TEMP", 0)),
            _f(rec.get("RTU_REGS_P00_ENV_HUMIDITY", 0)),
            _f(rec.get("RTU_REGS_P00_ROTOR_RPM", 0)),
        ])
        return np.concatenate([wave, env])  # shape=(515,)
    else:
        return extract_from_record(rec)  # shape=(13,)


def _print_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    max_err = np.max(np.abs(y_pred - y_true))
    print(f"\n评估结果:")
    print(f"  MAE:  {mae:.4f} V")
    print(f"  RMSE: {rmse:.4f} V")
    print(f"  最大误差: {max_err:.4f} V")


def _save_pytorch_model(model, algorithm, base):
    model_path = f"{base}.pth"
    torch.save(model["model"].state_dict(), model_path)
    save_data = {"algorithm": algorithm, "model_path": model_path}
    for k in ["wave_mean", "wave_std", "fft_mean", "fft_std", "env_mean", "env_std"]:
        if k in model:
            save_data[k] = model[k].tolist()
    with open(f"{base}.json", "w") as f:
        json.dump(save_data, f)
    print(f"\n模型已保存: {model_path}")


def _save_xgboost(model, base):
    model_path = f"{base}.ubj"
    model["model"].save_model(model_path)
    with open(f"{base}.json", "w") as f:
        json.dump({"algorithm": "xgboost_model", "model_path": model_path}, f)
    print(f"\nXGBoost 模型已保存: {model_path}")


def _save_linear(model, algorithm, base):
    with open(f"{base}.json", "w") as f:
        json.dump({"algorithm": algorithm, "params": model}, f, indent=2)
    print(f"\n模型已保存: {base}.json")


def main():
    parser = argparse.ArgumentParser(description="训练电压估算模型")
    parser.add_argument("--data", default=config.data_source.local_path,
                        help=f"训练数据 JSONL 路径 (默认: {config.data_source.local_path})")
    parser.add_argument("--algorithm",
                        choices=["linear_basic", "linear_with_env", "linear_full",
                                 "quadratic_model", "xgboost_model", "lightgbm_model",
                                 "lenet",
                                 "lenet_hybrid"],
                        default=config.estimation.algorithm, help="算法 (默认: settings.py 中的配置)")
    parser.add_argument("--limit", type=int, default=0, help="限制训练条数 (默认: 全部)")
    parser.add_argument("--output", default=config.estimation.model_path,
                        help=f"模型参数输出路径 (默认: {config.estimation.model_path})")
    parser.add_argument("--n_fft", type=int, default=10,
                        help="FFT 谐波数量，仅对 lenet_hybrid 有效 (默认: 10)")
    args = parser.parse_args()

    # 加载算法模块
    module = importlib.import_module(f"src.swa.estimation.{args.algorithm}")
    print(f"算法: {module.NAME}")

    # 加载数据
    print(f"加载数据: {args.data}")
    records = load_jsonl(args.data)
    if args.limit:
        records = records[:args.limit]
    print(f"共 {len(records)} 条")

    # 打乱数据，防止顺序偏倚
    random.shuffle(records)
    print("已打乱顺序")

    # 是否深度学习（需要验证集）
    is_dl = args.algorithm in ("lenet", "lenet_hybrid")

    cfg_est = config.estimation
    if is_dl:
        # 8:1:1 切分（训练/验证/测试）
        total_needed = cfg_est.train_size + cfg_est.val_size + cfg_est.test_size
        if len(records) < total_needed:
            ratio = len(records) / total_needed
            train_n = int(cfg_est.train_size * ratio)
            val_n = int(cfg_est.val_size * ratio)
            test_n = len(records) - train_n - val_n
        else:
            train_n = cfg_est.train_size
            val_n = cfg_est.val_size
            test_n = cfg_est.test_size
        train_records = records[:train_n]
        val_records = records[train_n:train_n + val_n]
        test_records = records[train_n + val_n:train_n + val_n + test_n]
        print(f"训练集: {train_n}, 验证集: {val_n}, 测试集: {test_n}")
    else:
        # 9:1 切分（训练/测试），验证集合并到训练集
        total_n = cfg_est.train_size + cfg_est.val_size + cfg_est.test_size
        test_n = cfg_est.test_size
        train_n = min(len(records) - test_n, total_n - test_n)
        if train_n < 1:
            train_n = int(len(records) * 0.9)
            test_n = len(records) - train_n
        train_records = records[:train_n]
        val_records = []
        test_records = records[train_n:train_n + test_n]
        print(f"训练集: {train_n}, 测试集: {test_n}")

    # 保存路径（所有分支共用）
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    base, ext = os.path.splitext(args.output)

    if args.algorithm == "lenet_hybrid":
        model = module.train(train_records, val_records=val_records, test_records=test_records, n_fft=args.n_fft)
        # 最终测试评估
        (wave_t, fft_t, env_t), y_test, _ = module._build_tensors(test_records, n_fft=args.n_fft)
        with torch.no_grad():
            y_pred = model["model"](wave_t, fft_t, env_t).numpy()
        _print_metrics(y_test, y_pred)
        _save_pytorch_model(model, args.algorithm, base)
        return

    # 非 hybrid 算法：提取特征
    X_list, y_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    skipped = 0

    # 训练集
    for rec in train_records:
        voltage = parse_voltage(rec.get("ACTUAL_VOLTAGE"))
        if np.isnan(voltage):
            skipped += 1
            continue
        features = _extract_features(rec, args.algorithm)
        X_list.append(features)
        y_list.append(voltage)

    # 验证集（仅神经网络需要）
    has_val = args.algorithm in ("lenet",)
    if has_val:
        for rec in val_records:
            voltage = parse_voltage(rec.get("ACTUAL_VOLTAGE"))
            if np.isnan(voltage): continue
            X_val_list.append(_extract_features(rec, args.algorithm))
            y_val_list.append(voltage)

    # 测试集
    for rec in test_records:
        voltage = parse_voltage(rec.get("ACTUAL_VOLTAGE"))
        if np.isnan(voltage): continue
        X_test_list.append(_extract_features(rec, args.algorithm))
        y_test_list.append(voltage)

    X_train = np.array(X_list)
    y_train = np.array(y_list)
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)
    print(f"训练: {len(y_train)}, 测试: {len(y_test)}, 跳过: {skipped}")

    # 训练
    if has_val:
        X_val = np.array(X_val_list)
        y_val = np.array(y_val_list)
        X_test_arr = np.array(X_test_list)
        y_test_arr = np.array(y_test_list)
        print(f"验证: {len(y_val)}")
        model = module.train(X_train, y_train, X_val=X_val, y_val=y_val,
                            X_test=X_test_arr, y_test=y_test_arr)
    else:
        model = module.train(X_train, y_train)

    # 评估
    y_pred = module.predict(model, X_test)
    mae = np.mean(np.abs(y_pred - y_test))
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    max_err = np.max(np.abs(y_pred - y_test))
    print(f"\n评估结果:")
    print(f"  MAE:  {mae:.4f} V")
    print(f"  RMSE: {rmse:.4f} V")
    print(f"  最大误差: {max_err:.4f} V")

    # 保存模型
    if args.algorithm == "xgboost_model":
        _save_xgboost(model, base)
    elif args.algorithm == "lightgbm_model":
        model_path = f"{base}.txt"
        model["model"].booster_.save_model(model_path)
        with open(f"{base}.json", "w") as f:
            json.dump({"algorithm": "lightgbm_model", "model_path": model_path}, f)
        print(f"\nLightGBM 模型已保存: {model_path}")
    elif args.algorithm in ("lenet", "lenet_hybrid"):
        _save_pytorch_model(model, args.algorithm, base)
    else:
        _save_linear(model, args.algorithm, base)


if __name__ == "__main__":
    main()
