"""
用所有传统 ML 模型预测电压，输出每个模型的预测结果。
不需要真值，只看模型预测了什么。

用法：
    uv run python scripts/predict_all.py -d data/some_data.jsonl
    uv run python scripts/predict_all.py -d data/some_data.jsonl --limit 10
"""

import sys
import os
import argparse
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from tqdm import tqdm
from scripts.utils.loader import load_jsonl, get_dataset_model_path
from src.swa.estimation.feature_extractor import extract_from_record

FS = 15873


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
    return np.array([np.sqrt(coeffs[1]**2 + coeffs[2]**2),
                     np.sqrt(coeffs[3]**2 + coeffs[4]**2),
                     np.sqrt(coeffs[5]**2 + coeffs[6]**2)])


def _extract_features(rec, features):
    """提取特征向量，根据 features 模式选择提取方式"""
    if features == "phys":
        wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
        wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]
        rpm_val = rec.get("RTU_REGS_P00_ROTOR_RPM")
        rpm = float(rpm_val) if rpm_val is not None else 0
        feat = _fit_waveform(wave, rpm)
    elif features == "phys1":
        wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
        wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]
        rpm_val = rec.get("RTU_REGS_P00_ROTOR_RPM")
        rpm = float(rpm_val) if rpm_val is not None else 0
        feat = _fit_waveform(wave, rpm)[:1]  # 只用 A1
    elif features == "nodim":
        return extract_from_record(rec)[10:16]
    else:  # 16dim
        return extract_from_record(rec)
    return feat

# 抑制 joblib/Parallel 噪音
os.environ["JOBLIB_VERBOSITY"] = "0"

ALGORITHMS = [
    "linear_model",
    "pure_signal_model",
    "hybrid_model",
    "quadratic_model",
    "svr_model",
    "extra_trees_model",
    "xgboost_model",
    "catboost_model",
    "lightgbm_model",
    "random_forest_model",
]

ALIAS = {
    "linear_model": "linear",
    "pure_signal_model": "pure_sig",
    "hybrid_model": "hybrid",
    "quadratic_model": "quadratic",
    "svr_model": "svr",
    "extra_trees_model": "extra_trees",
    "xgboost_model": "xgboost",
    "catboost_model": "catboost",
    "lightgbm_model": "lightgbm",
    "random_forest_model": "rf",
}


def _find_model(algo: str, data_path: str) -> str:
    """尝试找模型文件，先找数据集对应目录，再找默认路径"""
    model_base = get_dataset_model_path(data_path, "data/model_params")

    ext_map = {
        "linear_model": ".json",
        "pure_signal_model": ".joblib",
        "hybrid_model": ".joblib",
        "quadratic_model": ".joblib",
        "svr_model": ".joblib",
        "extra_trees_model": ".joblib",
        "random_forest_model": ".joblib",
        "xgboost_model": ".ubj",
        "lightgbm_model": ".txt",
        "catboost_model": ".cbm",
    }

    path = f"{model_base}_{algo}{ext_map[algo]}"
    if os.path.exists(path):
        return path

    # 回退到 data/model_params_*
    fallback = f"data/model_params_{algo}{ext_map[algo]}"
    if os.path.exists(fallback):
        return fallback

    # 再回退到 data/exported_data/
    fallback2 = f"data/exported_data/model_params_{algo}{ext_map[algo]}"
    if os.path.exists(fallback2):
        return fallback2

    return path  # 让调用方报错


def _load_model(algo: str, model_path: str):
    """加载模型"""
    module = importlib.import_module(f"scripts.traditional.{algo}")

    if algo == "linear_model":
        import json as j
        with open(model_path) as f:
            meta = j.load(f)
        return module, np.array(meta["params"])

    elif algo == "catboost_model":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(verbose=0)
        model.load_model(model_path)
        return module, {"model": model}

    elif algo == "xgboost_model":
        import xgboost as xgb
        bst = xgb.Booster()
        bst.load_model(model_path)
        return module, {"model": bst}

    elif algo == "lightgbm_model":
        import lightgbm as lgb
        bst = lgb.Booster(model_file=model_path)
        return module, {"model": bst}

    else:
        import joblib
        model = joblib.load(model_path)
        if hasattr(model, "verbose"):
            model.verbose = 0
        return module, model


def main():
    parser = argparse.ArgumentParser(description="多模型批量预测电压")
    parser.add_argument("-d", "--data", required=True, help="JSONL 数据文件路径")
    parser.add_argument("--limit", type=int, default=0, help="限制预测条数 (默认: 全部)")
    parser.add_argument("--features", type=str, default="nodim",
                        choices=["16dim", "nodim", "phys", "phys1"],
                        help="特征模式 (默认 nodim，需与训练时一致)")
    args = parser.parse_args()

    records = load_jsonl(args.data, extract_features=args.features not in ("phys", "phys1"))
    if args.limit:
        records = records[:args.limit]

    print(f"加载数据: {args.data}  ({len(records)} 条)\n")

    # 预加载所有模型
    models = {}
    for algo in ALGORITHMS:
        try:
            path = _find_model(algo, args.data)
            module, model = _load_model(algo, path)
            models[algo] = (module, model)
        except Exception as e:
            print(f"  [!] {algo}: 加载失败 ({e})")
            models[algo] = None

    # 逐条预测
    features = args.features
    all_preds = []
    for rec in tqdm(records, desc="预测", unit="条"):
        X = _extract_features(rec, features).reshape(1, -1)
        row = {
            "case": rec.get("TEST_CASE_CODE") or "",
            "time": rec.get("SYSTEM_TIME") or "",
        }
        for algo in ALGORITHMS:
            m = models[algo]
            if m is None:
                row[algo] = None
            else:
                try:
                    module, model = m
                    row[algo] = abs(float(module.predict(model, X)[0]))
                except Exception:
                    row[algo] = None
        all_preds.append(row)

    # 打印（动态算列宽）
    w_case = max(len(r["case"]) for r in all_preds + [{"case": "用例"}])
    w_time = max(len(r["time"]) for r in all_preds + [{"time": "时间"}])
    w_algo = max(len(v) for v in ALIAS.values())

    header = f"{'#':>4} | {'用例':>{w_case}} | {'时间':>{w_time}}"
    for a in ALGORITHMS:
        header += f" | {ALIAS[a]:>{w_algo}}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    # 平均值行（放最上面）
    avg_line = f"{'avg':>4} | {'':>{w_case}} | {'':>{w_time}}"
    for a in ALGORITHMS:
        vals = [r[a] for r in all_preds if r[a] is not None]
        avg = sum(vals) / len(vals) if vals else 0
        avg_line += f" | {avg:>{w_algo}.2f}"
    print(avg_line)
    print(sep)

    for i, row in enumerate(all_preds):
        line = f"{i+1:>4} | {row['case']:>{w_case}} | {row['time']:>{w_time}}"
        for a in ALGORITHMS:
            v = row[a]
            line += f" | {v:>{w_algo}.2f}" if v is not None else f" | {'--':>{w_algo}}"
        print(line)


if __name__ == "__main__":
    main()
