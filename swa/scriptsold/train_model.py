"""
传统机器学习训练脚本：linear / random_forest / extra_trees / svr / xgboost / lightgbm / catboost

用法：
    # 训练单个模型（全量特征）
    uv run python scripts/train_model.py --algorithm linear_model
    uv run python scripts/train_model.py --algorithm random_forest_model
    
    # 不用谐波特征（仅 T,RH,RPM,Vpp,Kurt,Skew）
    uv run python scripts/train_model.py --algorithm linear_model --n-harmonics 0
    
    # 只用 A1~A2 + 环境 + 时域
    uv run python scripts/train_model.py --algorithm linear_model --n-harmonics 2
    
    # 训练所有模型（数据只加载一次）
    uv run python scripts/train_model.py --all
"""

import argparse
import json
import sys
import os
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

_SEED = 42
np.random.seed(_SEED)
torch.manual_seed(_SEED)
torch.cuda.manual_seed_all(_SEED)

from src.swa.config.settings import config
from src.swa.estimation.feature_extractor import extract_from_record
from scripts.utils.loader import load_jsonl, split_jsonl, get_dataset_model_path

FS = 15873  # 采样率 Hz


def _fit_waveform(wave, rpm=None):
    """傅里叶级数拟合: y = C + A₁sin(ωt+φ₁) + A₃sin(3ωt+φ₃) + A₅sin(5ωt+φ₅)"""
    n = len(wave)
    if rpm is not None and rpm > 0:
        f0 = rpm / 60.0
    else:
        fft_mag = np.abs(np.fft.fft(wave - np.mean(wave)))[:n // 2]
        f0 = (1 + np.argmax(fft_mag[1:])) * FS / n
    t = np.arange(n) / FS
    wt = 2 * np.pi * f0 * t
    A = np.column_stack([np.ones(n), np.cos(wt), np.sin(wt),
                         np.cos(3*wt), np.sin(3*wt),
                         np.cos(5*wt), np.sin(5*wt)])
    coeffs, *_ = np.linalg.lstsq(A, wave, rcond=None)
    amp1 = np.sqrt(coeffs[1]**2 + coeffs[2]**2)
    amp3 = np.sqrt(coeffs[3]**2 + coeffs[4]**2)
    amp5 = np.sqrt(coeffs[5]**2 + coeffs[6]**2)
    return amp1, amp3, amp5


def _extract_phys(rec):
    """从记录提取物理特征 [A1, A3, A5]"""
    wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
    wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]
    rpm_val = rec.get("RTU_REGS_P00_ROTOR_RPM")
    rpm = float(rpm_val) if rpm_val is not None else 0
    return np.array(_fit_waveform(wave, rpm))


def _print_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    max_err = np.max(np.abs(y_pred - y_true))
    print(f"\n评估结果:")
    print(f"  MAE:  {mae:.4f} V")
    print(f"  RMSE: {rmse:.4f} V")
    print(f"  最大误差: {max_err:.4f} V")


def _save_xgboost(model, base):
    model_path = f"{base}.ubj"
    model["model"].save_model(model_path)
    with open(f"{base}.json", "w") as f:
        json.dump({"algorithm": "xgboost_model", "model_path": model_path}, f)
    print(f"\nXGBoost 模型已保存: {model_path}")


def _save_lightgbm(model, base):
    model_path = f"{base}.txt"
    model["model"].booster_.save_model(model_path)
    with open(f"{base}.json", "w") as f:
        json.dump({"algorithm": "lightgbm_model", "model_path": model_path}, f)
    print(f"\nLightGBM 模型已保存: {model_path}")


def _save_catboost(model, base):
    model_path = f"{base}.cbm"
    model["model"].save_model(model_path)
    with open(f"{base}.json", "w") as f:
        json.dump({"algorithm": "catboost_model", "model_path": model_path}, f)
    print(f"\nCatBoost 模型已保存: {model_path}")


def _save_sklearn(model, algorithm, base):
    """保存 sklearn 模型（使用 joblib）"""
    import joblib
    model_path = f"{base}.joblib"
    joblib.dump(model, model_path)
    with open(f"{base}.json", "w") as f:
        json.dump({"algorithm": algorithm, "model_path": model_path}, f)
    print(f"\n模型已保存: {model_path}")


def _save_linear(model, algorithm, base):
    with open(f"{base}.json", "w") as f:
        json.dump({"algorithm": algorithm, "params": model}, f, indent=2)
    print(f"\n模型已保存: {base}.json")


def _select_features(X, n_harmonics):
    """
    根据 n_harmonics 从 16 维特征向量中选择列。
    
    原始特征排列: [A1~A10(10), T,RH,RPM(3), Vpp,Kurt,Skew(3)]
    
    n_harmonics=0: [T,RH,RPM, Vpp,Kurt,Skew]              (6 维)
    n_harmonics=1: [A1, T,RH,RPM, Vpp,Kurt,Skew]          (7 维)
    n_harmonics=10: [A1~A10, T,RH,RPM, Vpp,Kurt,Skew]     (16 维, 全量)
    """
    cols = []
    if n_harmonics > 0:
        cols.extend(range(min(n_harmonics, 10)))  # A1~An (最多 10)
    cols.extend(range(10, 16))  # T, RH, RPM, Vpp, Kurt, Skew
    return X[:, cols]


def train_single_algorithm(algorithm_name, X_train, y_train, X_test, y_test, output_base):
    """训练单个算法"""
    module = importlib.import_module(f"scripts.traditional.{algorithm_name}")
    print(f"\n{'='*60}")
    print(f"算法: {module.NAME}")
    print(f"{'='*60}")

    # 训练
    print(f"  [debug] X_train.shape={X_train.shape}, n_params={X_train.shape[1] + 1}")
    model = module.train(X_train, y_train)

    # 评估
    y_pred = module.predict(model, X_test)
    _print_metrics(y_test, y_pred)

    # 保存：直接在基础路径上加上算法名
    base = f"{output_base}_{algorithm_name}"
    if algorithm_name == "xgboost_model":
        _save_xgboost(model, base)
    elif algorithm_name == "lightgbm_model":
        _save_lightgbm(model, base)
    elif algorithm_name == "catboost_model":
        _save_catboost(model, base)
    elif algorithm_name == "linear_model":
        _save_linear(model, algorithm_name, base)
    else:
        _save_sklearn(model, algorithm_name, base)


def main():
    parser = argparse.ArgumentParser(description="训练传统 ML 电压估算模型")
    parser.add_argument("--data", default=config.data_source.local_path,
                        help=f"训练数据 JSONL 路径 (默认: {config.data_source.local_path})")
    parser.add_argument("--algorithm",
                        choices=["linear_model", "pure_signal_model", "hybrid_model", "random_forest_model", "extra_trees_model", "svr_model", 
                                 "xgboost_model", "lightgbm_model", "catboost_model", "quadratic_model"],
                        default=config.estimation.algorithm, help="算法 (默认: settings.py 中的配置)")
    parser.add_argument("--all", action="store_true", help="训练所有传统 ML 模型")
    parser.add_argument("--no-full-dataset", action="store_true", dest="full_dataset", default=True,
                        help="不使用全量数据，配合 --limit 使用")
    parser.add_argument("--limit", type=int, default=0,
                        help="非全量时限制条数 (默认: 全部)")
    parser.add_argument("--train-ratio", type=float, default=0.9,
                        help="训练集比例 (默认: 0.9)")
    parser.add_argument("--val-ratio", type=float, default=0.0,
                        help="验证集比例 (默认: 0.0，合并到训练集)")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="测试集比例 (默认: 0.1)")
    parser.add_argument("--output", default=config.estimation.model_path,
                        help=f"模型参数输出路径 (默认: {config.estimation.model_path})")
    parser.add_argument("--n-harmonics", type=int, default=0,
                        help="使用 A1~An 谐波特征 (0=不用谐波, 默认 10=全量)")
    parser.add_argument("--max-per-voltage", type=int, default=5000,
                        help="每个电压桶最多保留条数 (默认 5000，0=不限制)")
    parser.add_argument("--features", type=str, default="16dim",
                        choices=["16dim", "nodim", "phys", "phys1"],
                        help="特征模式: 16dim=全16维, nodim=无谐波(6维), phys=物理参数A1+A3+A5(3维), phys1=仅A1(1维)")

    args = parser.parse_args()

    # 所有传统 ML 模型列表
    all_algorithms = [
        "linear_model",
        "pure_signal_model",
        "hybrid_model",
        "random_forest_model", 
        "extra_trees_model", 
        "svr_model",
        "xgboost_model", 
        "lightgbm_model", 
        "catboost_model",
        "quadratic_model"
    ]

    # 加载 & 划分数据（只加载一次）
    print(f"加载数据: {args.data}")
    records = load_jsonl(args.data, max_per_voltage=args.max_per_voltage)
    print(f"共 {len(records)} 条（max_per_voltage={args.max_per_voltage}），使用{'全量' if args.full_dataset else f'前 {args.limit} 条'}，拆分比例 {args.train_ratio}:{args.val_ratio}:{args.test_ratio}")

    train_records, val_records, test_records = split_jsonl(
        records,
        full_dataset=args.full_dataset,
        limit=args.limit,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=_SEED
    )
    # 传统 ML 验证集合并到训练集
    all_train_records = train_records + val_records
    print(f"训练集: {len(all_train_records)}, 测试集: {len(test_records)}")

    # 确定输出路径：如果有对应的数据集目录，就用那个
    output_base = get_dataset_model_path(args.data, args.output)
    # 去掉扩展名，确保是基础路径
    base, ext = os.path.splitext(output_base)
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    print(f"模型将保存到: {base}_*.model")

    # 提取特征（只提取一次）
    X_list, y_list = [], []
    X_test_list, y_test_list = [], []
    skipped = 0

    use_phys = args.features == "phys"

    for rec in all_train_records:
        voltage = rec["ACTUAL_VOLTAGE"]
        if voltage is None:
            skipped += 1
            continue
        if use_phys:
            feats = _extract_phys(rec)
            if args.features == "phys1":
                feats = feats[:1]  # 只用 A1
            X_list.append(feats)
        else:
            X_list.append(extract_from_record(rec))
        y_list.append(voltage)

    for rec in test_records:
        voltage = rec["ACTUAL_VOLTAGE"]
        if voltage is None:
            continue
        if use_phys:
            feats = _extract_phys(rec)
            X_test_list.append(feats[:1] if args.features == "phys1" else feats)
        else:
            X_test_list.append(extract_from_record(rec))
        y_test_list.append(voltage)

    X_train = np.array(X_list)
    y_train = np.abs(np.array(y_list))
    X_test = np.array(X_test_list)
    y_test = np.abs(np.array(y_test_list))
    print(f"训练: {len(y_train)}, 测试: {len(y_test)}, 跳过: {skipped}")

    # 特征选择
    if args.features == "nodim":
        X_train = _select_features(X_train, 0)
        X_test = _select_features(X_test, 0)
        feat_dim = X_train.shape[1]
        print(f"特征模式: nodim (6维: T,RH,RPM,Vpp,Kurt,Skew)")
    elif args.features == "phys":
        feat_dim = X_train.shape[1]
        print(f"特征模式: phys (3维: A1,A3,A5)")
    elif args.features == "phys1":
        feat_dim = X_train.shape[1]
        print(f"特征模式: phys1 (1维: A1)")
    else:
        n_h = args.n_harmonics
        X_train = _select_features(X_train, n_h)
        X_test = _select_features(X_test, n_h)
        feat_dim = X_train.shape[1]
        print(f"谐波数: A1~A{n_h} ({'无谐波' if n_h == 0 else '全量' if n_h == 10 else f'A1~A{n_h}'}), 特征维度: {feat_dim}")

    if args.all:
        # 训练所有算法
        print(f"\n{'#'*60}")
        print(f"开始训练所有 {len(all_algorithms)} 个模型...")
        print(f"{'#'*60}")
        
        for algorithm in all_algorithms:
            try:
                train_single_algorithm(algorithm, X_train, y_train, X_test, y_test, base)
            except Exception as e:
                print(f"\n❌ 训练 {algorithm} 时出错: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'#'*60}")
        print(f"所有模型训练完成！")
        print(f"{'#'*60}")
    else:
        # 只训练单个算法
        train_single_algorithm(args.algorithm, X_train, y_train, X_test, y_test, base)


if __name__ == "__main__":
    main()
