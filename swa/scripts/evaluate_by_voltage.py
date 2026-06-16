"""
分电压评估脚本

加载训练好的模型，逐条预测，按真实电压分桶统计 MAE / RMSE / 最大误差。

用法：
    uv run python scripts/evaluate_by_voltage.py --model data/model_nfft11 --data data/exported_data.jsonl
    uv run python scripts/evaluate_by_voltage.py --model data/model_nfft7  --data data/exported_data.jsonl
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

from src.swa.config.settings import config
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


def load_model(meta_path: str, n_fft_override: int = None):
    """从 .json 元文件加载 PyTorch 模型"""
    with open(meta_path) as f:
        meta = json.load(f)

    algo = meta.get("algorithm")
    model_file = meta.get("model_path")
    n_fft = n_fft_override if n_fft_override is not None else meta.get("n_fft", 10)

    if not model_file or not model_file.endswith(".pth"):
        raise ValueError(f"仅支持 PyTorch 模型 (.pth)，当前: {model_file}")

    # 补齐路径：绝对路径直接用，相对路径基于 json 所在目录
    model_path = model_file if os.path.isabs(model_file) else os.path.join(
        os.path.dirname(os.path.abspath(meta_path)), os.path.basename(model_file)
    )

    from scripts.utils.device import get_device
    device = get_device()

    if algo == "lenet_hybrid":
        from src.swa.estimation.lenet_hybrid import HybridNet
        model = HybridNet(n_fft=n_fft, n_env=6)
    elif algo == "lenet":
        from src.swa.estimation.lenet import LeNet1D
        model = LeNet1D()
    else:
        raise ValueError(f"不支持的算法: {algo}")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 归一化参数
    norm_params = {}
    for k in ["wave_mean", "wave_std", "fft_mean", "fft_std", "env_mean", "env_std"]:
        if k in meta:
            norm_params[k] = np.array(meta[k], dtype=np.float32)

    return model, norm_params, algo, n_fft, device


def predict_lenet_hybrid(model, record, norm_params, n_fft, device):
    """对一条记录用 lenet_hybrid 推理"""
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

    # 波形归一化（逐条）
    wm = np.mean(wave)
    ws = np.std(wave) + 1e-8
    wave_norm = ((wave - wm) / ws).reshape(1, -1)

    # FFT 和环境归一化（用训练集的参数）
    fft_norm = ((harmonics - norm_params["fft_mean"]) / norm_params["fft_std"]).reshape(1, -1)

    # 时域特征
    ac = wave - np.mean(wave)
    vpp = float(np.max(ac) - np.min(ac))
    kurt = float(kurtosis(ac, fisher=False))
    skewness = float(skew(ac))
    aux = np.array([temp, humid, rpm, vpp, kurt, skewness])
    env_norm = ((aux - norm_params["env_mean"]) / norm_params["env_std"]).reshape(1, -1)

    wave_t = torch.tensor(wave_norm, dtype=torch.float32).to(device)
    fft_t = torch.tensor(fft_norm, dtype=torch.float32).to(device)
    env_t = torch.tensor(env_norm, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(wave_t, fft_t, env_t).cpu().numpy()[0]
    return float(pred)


def predict_lenet(model, record, device):
    """对一条记录用 lenet-1D 推理"""
    wave_str = record.get("RTU_REGS_P00_WAVE_DATA", "")
    wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]

    def _f(v, d=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return d

    env = np.array([
        _f(record.get("RTU_REGS_P00_ENV_TEMP")),
        _f(record.get("RTU_REGS_P00_ENV_HUMIDITY")),
        _f(record.get("RTU_REGS_P00_ROTOR_RPM")),
    ])

    # 波形归一化
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


def main():
    parser = argparse.ArgumentParser(description="分电压评估模型")
    parser.add_argument("--model", required=True,
                        help="模型元文件路径 (不含扩展名，如 data/model_nfft11)")
    parser.add_argument("--data", default=config.data_source.local_path,
                        help="数据文件路径")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制评估条数 (默认: 全部)")
    parser.add_argument("--n-fft", type=int, default=None,
                        help="覆盖 n_fft (旧模型未保存此参数时使用)")
    args = parser.parse_args()

    # 加载模型
    meta_path = args.model
    if not meta_path.endswith(".json"):
        meta_path = meta_path + ".json"
    print(f"加载模型: {meta_path}")
    model, norm_params, algo, n_fft, device = load_model(meta_path, n_fft_override=args.n_fft)
    print(f"  算法: {algo}, n_fft: {n_fft}, 设备: {device}")

    # 加载数据
    print(f"加载数据: {args.data}")
    records = load_jsonl(args.data)

    # 使用与训练完全相同的切分逻辑（先打乱再切分）
    import random
    random.seed(42)
    random.shuffle(records)

    cfg_est = config.estimation
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
    test_records = records[train_n + val_n:train_n + val_n + test_n]
    if args.limit:
        test_records = test_records[:args.limit]
    print(f"评估集: {len(test_records)} 条\n")

    # 逐条预测
    results = {}  # {voltage: {"y_true": [...], "y_pred": [...]}}
    skipped = 0
    for i, rec in enumerate(test_records):
        voltage = parse_voltage(rec.get("ACTUAL_VOLTAGE"))
        if np.isnan(voltage):
            skipped += 1
            continue

        if algo == "lenet_hybrid":
            pred = predict_lenet_hybrid(model, rec, norm_params, n_fft, device)
        elif algo == "lenet":
            pred = predict_lenet(model, rec, device)
        else:
            raise ValueError(f"不支持的算法: {algo}")

        # 按 10V 分桶
        bucket = round(voltage / 10) * 10
        if bucket not in results:
            results[bucket] = {"y_true": [], "y_pred": []}
        results[bucket]["y_true"].append(voltage)
        results[bucket]["y_pred"].append(pred)

        if (i + 1) % 1000 == 0:
            print(f"  已评估 {i + 1}/{len(test_records)} 条...")

    print(f"\n{'='*60}")
    print(f"{'分电压评估结果':^60}")
    print(f"{'='*60}")
    print(f"{'电压':>8} | {'数量':>6} | {'占比':>6} | {'MAE':>8} | {'RMSE':>8} | {'最大误差':>10} | {'平均预测':>8}")
    print(f"{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}")

    # 按电压排序
    total_mae_list = []
    total_count = 0
    for bucket in sorted(results.keys()):
        y_true = np.array(results[bucket]["y_true"])
        y_pred = np.array(results[bucket]["y_pred"])
        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        max_err = float(np.max(np.abs(y_pred - y_true)))
        mean_pred = float(np.mean(y_pred))
        n = len(y_true)
        pct = n / len(test_records) * 100
        total_mae_list.append(mae * n)
        total_count += n
        print(f"{bucket:>7}V | {n:>6} | {pct:>5.1f}% | {mae:>7.4f} | {rmse:>7.4f} | {max_err:>9.4f} | {mean_pred:>7.2f}")

    # 整体加权 MAE
    weighted_mae = sum(total_mae_list) / total_count if total_count else 0
    print(f"{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}")
    print(f"{'合计':>8} | {total_count:>6} | {'100%':>6} | {weighted_mae:>7.4f} | {'':>8} | {'':>10} | {'':>8}")
    print(f"{'='*60}")

    # 再算一个整体指标
    all_true, all_pred = [], []
    for bucket in results:
        all_true.extend(results[bucket]["y_true"])
        all_pred.extend(results[bucket]["y_pred"])
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    overall_mae = np.mean(np.abs(all_pred - all_true))
    overall_rmse = np.sqrt(np.mean((all_pred - all_true) ** 2))
    print(f"\n整体: MAE={overall_mae:.4f}V  RMSE={overall_rmse:.4f}V  (跳过 {skipped} 条)")


if __name__ == "__main__":
    main()
