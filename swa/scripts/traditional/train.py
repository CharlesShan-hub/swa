"""
训练传统 ML 电压估算模型。

用法:
    uv run python scripts/traditional/train.py                            # 默认 default 数据集
    uv run python scripts/traditional/train.py --dataset default_4merge --features a1
    uv run python scripts/traditional/train.py --dataset default --model linear
"""

import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from lib.data import load
from lib.metrics import extend
from lib.utils import dataset_dir


def _print_metrics(y_true, y_pred, label="评估"):
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    max_err = float(np.max(np.abs(y_pred - y_true)))
    print(f"  {label}: MAE={mae:.4f}V, RMSE={rmse:.4f}V, max={max_err:.4f}V")
    return {"mae": mae, "rmse": rmse, "max_err": max_err}


def main():
    parser = argparse.ArgumentParser(description="训练传统 ML 电压估算模型")
    parser.add_argument("--dataset", default="default", help="数据集名称")
    parser.add_argument("--model", default="linear", choices=["linear", "xgboost", "quadratic"],
                        help="模型名称 (linear/xgboost/quadratic)")
    parser.add_argument("--features", nargs="+", default=["a1", "a3", "a5", "vpp", "kurtosis", "temp", "humid", "rpm"],
                        help="使用的特征，如 a1 a3 a5 vpp 等")
    parser.add_argument("--train-voltages", nargs="+", type=float,
                        default=[-40, 30, 97, 70, 72, -55, 110, -20, -50, 50, 80, 10, -10, -110],
                        help="训练集电压列表")
    parser.add_argument("--test-voltages", nargs="+", type=float,
                        default=[],
                        help="测试集电压列表（留出验证），留空则全部训练")
    parser.add_argument("--output", default=None,
                        help="模型保存路径，默认 data/{dataset}/model_params_*.json")
    parser.add_argument("--repeat", type=str,
                        default="10:50,-10:50,-110:50",
                        help="电压重复倍率，默认所有小样本 50 倍")
    args = parser.parse_args()

    # 解析 repeat 参数
    repeat = {}
    if args.repeat:
        for item in args.repeat.split(","):
            k, v = item.split(":")
            repeat[int(k)] = int(v)

    # 1. 加载
    print(f"加载数据集: {args.dataset}")
    is_metrics = args.dataset.startswith("smooth_") or args.dataset.startswith("metrics_")
    records = load(args.dataset, repeat=repeat or None)
    print(f"  共 {len(records)} 条")

    # 2. 扩展指标（原始数据需要从波形计算，smooth 数据已预计算）
    if not is_metrics:
        records = extend(records, args.features)
    print(f"  特征: {args.features}")

    # 3. 按电压划分
    if args.test_voltages:
        from lib.data import split_by_voltage
        train_recs, test_recs = split_by_voltage(records, args.train_voltages, args.test_voltages)
        print(f"  训练集: {len(train_recs)} 条 (电压: {args.train_voltages})")
        print(f"  测试集: {len(test_recs)} 条 (电压: {args.test_voltages})")
    else:
        # 无测试集时，全部用于训练
        train_recs = records
        test_recs = []
        print(f"  全部训练: {len(train_recs)} 条")
        print(f"  测试集: 无")

    # 4. 构造特征矩阵
    def to_Xy(recs):
        X, y = [], []
        for rec in recs:
            v = rec.get("ACTUAL_VOLTAGE")
            if not isinstance(v, (int, float)):
                continue
            feat = [rec.get(f"_{k}") for k in args.features]
            if any(f is None for f in feat):
                continue
            X.append(feat)
            y.append(abs(float(v)))
        return np.array(X), np.array(y)

    X_train, y_train = to_Xy(train_recs)
    X_test, y_test = to_Xy(test_recs)
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # 5. 训练
    if args.model == "linear":
        from lib.traditional.linear import train, predict, NAME
        print(f"\n模型: {NAME}")
        model = train(X_train, y_train)
        y_pred = predict(model, X_train)
        _print_metrics(y_train, y_pred, "训练集")
        if len(X_test):
            y_pred = predict(model, X_test)
            _print_metrics(y_test, y_pred, "测试集")

        print(f"\n  线性模型参数:")
        print(f"    bias = {model[0]:+.4f}")
        for i, c in enumerate(model[1:]):
            print(f"    w{i}({args.features[i]}) = {c:+.4f}")

        # 保存
        output_dir = dataset_dir(args.dataset)
        output_path = os.path.join(output_dir, f"model_linear.json")
        with open(output_path, "w") as f:
            json.dump({"algorithm": "linear", "features": args.features, "params": model}, f, indent=2)
        print(f"\n模型已保存: {output_path}")

    elif args.model == "xgboost":
        from lib.traditional.xgboost import train, predict, NAME
        print(f"\n模型: {NAME}")
        model = train(X_train, y_train)
        y_pred = predict(model, X_train)
        _print_metrics(y_train, y_pred, "训练集")
        if len(X_test):
            y_pred = predict(model, X_test)
            _print_metrics(y_test, y_pred, "测试集")

        output_dir = dataset_dir(args.dataset)
        output_path = os.path.join(output_dir, f"model_xgboost.ubj")
        model["model"].save_model(output_path)
        meta_path = os.path.join(output_dir, f"model_xgboost.json")
        with open(meta_path, "w") as f:
            json.dump({"algorithm": "xgboost", "features": args.features, "model_path": output_path}, f, indent=2)
        print(f"\nXGBoost 模型已保存: {output_path}")

    elif args.model == "quadratic":
        from lib.traditional.quadratic import train, predict, NAME
        print(f"\n模型: {NAME}（含平方项和交互项）")
        model = train(X_train, y_train)
        y_pred = predict(model, X_train)
        _print_metrics(y_train, y_pred, "训练集")
        if len(X_test):
            y_pred = predict(model, X_test)
            _print_metrics(y_test, y_pred, "测试集")

        output_dir = dataset_dir(args.dataset)
        output_path = os.path.join(output_dir, f"model_quadratic.json")
        with open(output_path, "w") as f:
            json.dump({"algorithm": "quadratic", "features": args.features, "params": model, "n_params": len(model)}, f, indent=2)
        print(f"\n二次模型已保存: {output_path}（{len(model)} 参数）")

    else:
        print(f"未知模型: {args.model}, 可用: linear, xgboost, quadratic")
        return


if __name__ == "__main__":
    main()
