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
    parser.add_argument("--model", default="linear", choices=["linear", "xgboost", "quadratic", "cubic", "quadratic_nox", "cubic_nox", "quadratic_zero", "hybrid", "hybrid_poly"],
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
    parser.add_argument("--vol-limit", type=int, default=None,
                        help="每个电压最多取 N 条（均衡样本）")
    parser.add_argument("--degree", type=int, default=1,
                        help="多项式修正次数（仅 hybrid_poly 使用）")
    args = parser.parse_args()

    # 解析 repeat 参数
    repeat = {}
    if args.repeat:
        for item in args.repeat.split(","):
            k, v = item.split(":")
            repeat[int(k)] = int(v)

    # 1. 加载
    print(f"加载数据集: {args.dataset}")
    is_metrics = args.dataset.startswith("smooth_") or args.dataset.startswith("metrics_") or args.dataset.startswith("savgol_")
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

    # 4. 均衡采样（每个电压最多 N 条）
    if args.vol_limit:
        from lib.data import group_by_voltage
        groups = group_by_voltage(train_recs)
        balanced = []
        for v in sorted(groups, key=lambda x: str(x)):
            group = groups[v]
            if len(group) > args.vol_limit:
                group = list(np.random.RandomState(42).choice(group, args.vol_limit, replace=False))
            balanced.extend(group)
            print(f"    电压 {v:>4}V: {len(groups[v])} → {len(group)} 条")
        train_recs = balanced
        print(f"  均衡后训练集: {len(train_recs)} 条")

    # 5. 构造特征矩阵
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
        from lib.traditional.xgboost_wrapper import train, predict, NAME
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

    elif args.model == "cubic":
        from lib.traditional.cubic import train, predict, NAME
        print(f"\n模型: {NAME}")
        model = train(X_train, y_train)
        y_pred = predict(model, X_train)
        _print_metrics(y_train, y_pred, "训练集")
        if len(X_test):
            y_pred = predict(model, X_test)
            _print_metrics(y_test, y_pred, "测试集")

        output_dir = dataset_dir(args.dataset)
        output_path = os.path.join(output_dir, f"model_cubic.json")
        with open(output_path, "w") as f:
            json.dump({"algorithm": "cubic", "features": args.features, "params": model, "n_params": len(model)}, f, indent=2)
        print(f"\n三次模型已保存: {output_path}（{len(model)} 参数）")

    elif args.model == "quadratic_nox":
        from lib.traditional.quadratic_nox import train, predict, NAME
        print(f"\n模型: {NAME}")
        model = train(X_train, y_train)
        y_pred = predict(model, X_train)
        _print_metrics(y_train, y_pred, "训练集")
        if len(X_test):
            y_pred = predict(model, X_test)
            _print_metrics(y_test, y_pred, "测试集")

        output_dir = dataset_dir(args.dataset)
        output_path = os.path.join(output_dir, f"model_quadratic_nox.json")
        with open(output_path, "w") as f:
            json.dump({"algorithm": "quadratic_nox", "features": args.features, "params": model, "n_params": len(model)}, f, indent=2)
        print(f"\n二次无交互模型已保存: {output_path}（{len(model)} 参数）")

    elif args.model == "cubic_nox":
        from lib.traditional.cubic_nox import train, predict, NAME
        print(f"\n模型: {NAME}")
        model = train(X_train, y_train)
        y_pred = predict(model, X_train)
        _print_metrics(y_train, y_pred, "训练集")
        if len(X_test):
            y_pred = predict(model, X_test)
            _print_metrics(y_test, y_pred, "测试集")

        output_dir = dataset_dir(args.dataset)
        output_path = os.path.join(output_dir, f"model_cubic_nox.json")
        with open(output_path, "w") as f:
            json.dump({"algorithm": "cubic_nox", "features": args.features, "params": model, "n_params": len(model)}, f, indent=2)
        print(f"\n三次无交互模型已保存: {output_path}（{len(model)} 参数）")

    elif args.model == "quadratic_zero":
        from lib.traditional.quadratic_zero import train, predict, NAME
        print(f"\n模型: {NAME}")
        model = train(X_train, y_train)
        y_pred = predict(model, X_train)
        _print_metrics(y_train, y_pred, "训练集")
        if len(X_test):
            y_pred = predict(model, X_test)
            _print_metrics(y_test, y_pred, "测试集")

        output_dir = dataset_dir(args.dataset)
        output_path = os.path.join(output_dir, f"model_quadratic_zero.json")
        with open(output_path, "w") as f:
            json.dump({"algorithm": "quadratic_zero", "features": args.features, "params": model, "n_params": len(model)}, f, indent=2)
        print(f"\n过零二次模型已保存: {output_path}（{len(model)} 参数）")

    elif args.model == "hybrid":
        from lib.traditional.hybrid import train_hybrid, PhysicalModel, ResidualModel, NAME
        print(f"\n模型: {NAME}")

        result = train_hybrid(
            records, args.features,
            train_voltages=args.train_voltages,
            test_voltages=args.test_voltages or None,
            vol_limit=args.vol_limit,
        )

        # 保存
        output_dir = dataset_dir(args.dataset)
        phys_path = os.path.join(output_dir, f"model_hybrid_phys.json")
        res_path = os.path.join(output_dir, f"model_hybrid_res.ubj")
        meta_path = os.path.join(output_dir, f"model_hybrid.json")

        with open(phys_path, "w") as f:
            json.dump({"k_global": result["k_global"]}, f)
        result["res"].model.save_model(res_path)
        with open(meta_path, "w") as f:
            json.dump({
                "algorithm": "hybrid",
                "features": args.features,
                "k_global": result["k_global"],
                "phys_path": phys_path,
                "res_path": res_path,
                "train_mae": result["train_mae"],
                "test_mae": result.get("test_mae"),
            }, f, indent=2)
        print(f"\n混合模型已保存: {meta_path}")

    elif args.model == "hybrid_poly":
        from lib.traditional.hybrid_poly import train, predict, NAME
        print(f"\n模型: {NAME} (degree={args.degree})")

        # 构造特征矩阵
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

        model = train(X_train, y_train, degree=args.degree)
        y_pred_train = predict(model, X_train)
        _print_metrics(y_train, y_pred_train, "训练集")
        if len(X_test):
            y_pred_test = predict(model, X_test)
            _print_metrics(y_test, y_pred_test, "测试集")

        phys_train = model["k_global"] * X_train[:, 0]
        phys_mae = np.mean(np.abs(np.abs(phys_train) - y_train))
        print(f"  物理基线 MAE: {phys_mae:.2f}V → 修正后: {np.mean(np.abs(y_pred_train - y_train)):.2f}V")

        output_dir = dataset_dir(args.dataset)
        output_path = os.path.join(output_dir, f"model_hybrid_poly.json")
        with open(output_path, "w") as f:
            json.dump({
                "algorithm": "hybrid_poly",
                "features": args.features,
                "k_global": model["k_global"],
                "coeffs": model["coeffs"],
                "degree": model["degree"],
                "n_params": len(model["coeffs"]),
            }, f, indent=2)
        print(f"\n混合多项式模型已保存: {output_path}")

    else:
        print(f"未知模型: {args.model}, 可用: linear, xgboost, quadratic, cubic, quadratic_nox, cubic_nox, quadratic_zero, hybrid, hybrid_poly")
        return


if __name__ == "__main__":
    main()
