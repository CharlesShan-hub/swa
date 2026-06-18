"""
比较不同特征集的线性模型性能（温度均衡采样，每温度最多 600 条）。

特征集:
  A: [Vpp, Kurt, Skew]                (3 特征, 4 参数)
  B: [A1~A10, Vpp, Kurt, Skew]        (13 特征, 14 参数)
  C: [A1~A10, T, RH, RPM, Vpp, ...]   (16 特征, 17 参数, 即完整 linear_model)

用法:
    uv run python scripts/compare_feature_sets.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from collections import defaultdict
from sklearn.linear_model import LinearRegression

from scripts.utils.loader import load_jsonl
from src.swa.estimation.feature_extractor import extract_from_record


MAX_PER_BUCKET = 600  # 每个温度桶最多 600 条


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.array(y) - np.array(p))**2)))


def _temp_bucket(t):
    return round(t / 5) * 5


def main():
    records = load_jsonl("data/exported_data.jsonl", extract_features=True)
    print(f"total: {len(records)}")

    # ── 提取 ──
    X, y, T = [], [], []
    for rec in records:
        v = rec.get("ACTUAL_VOLTAGE")
        t = rec.get("RTU_REGS_P00_ENV_TEMP")
        if v is None or t is None:
            continue
        try:
            x = extract_from_record(rec)
        except Exception:
            continue
        X.append(x)
        y.append(abs(float(v)))
        T.append(float(t))

    X = np.array(X)
    y = np.array(y)
    T = np.array(T)
    print(f"used: {len(y)}")

    # ── 按温度桶分层，每桶最多 MAX_PER_BUCKET 条 ──
    np.random.seed(42)
    buckets = defaultdict(list)  # {tb: [(idx, feat, label), ...]}
    for i in range(len(y)):
        tb = _temp_bucket(T[i])
        buckets[tb].append(i)

    train_idx = []
    test_idx = []
    for tb in sorted(buckets):
        idx_list = np.array(buckets[tb])
        np.random.shuffle(idx_list)
        if len(idx_list) <= MAX_PER_BUCKET:
            # 数据少 -> 80% 训练, 20% 测试
            n_train = int(len(idx_list) * 0.8)
        else:
            # 数据多 -> 取 MAX_PER_BUCKET 训练, 剩下的测试
            n_train = MAX_PER_BUCKET
        train_idx.extend(idx_list[:n_train])
        test_idx.extend(idx_list[n_train:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    print(f"train: {len(train_idx)}, test: {len(test_idx)}, "
          f"balanced: {all(len(buckets[tb]) >= MAX_PER_BUCKET or True for tb in buckets)}")

    # 打印采样后各温度桶的训练样本数
    print(f"\n  Training set per temperature:")
    train_T = T[train_idx]
    for tb in sorted(set(_temp_bucket(t) for t in train_T)):
        n = (train_T >= tb - 2) & (train_T < tb + 3)
        print(f"    {tb:>4}C: {n.sum()} samples")

    # ── 4 组特征 ──
    FEATURE_SETS = {
        "Vpp+Kurt+Skew (3)":         X[:, [13, 14, 15]],
        "A1+Vpp+Kurt+Skew (4)":      np.hstack([X[:, 0:1], X[:, 13:16]]),
        "A1~A10+Vpp+Kurt+Skew (13)": np.hstack([X[:, 0:10], X[:, 13:16]]),
        "Full 16-feat (16)":         X,
    }

    results = {}
    for name, X_feat in FEATURE_SETS.items():
        X_tr, X_te = X_feat[train_idx], X_feat[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = LinearRegression()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rmse = _rmse(y_te, pred)
        results[name] = {"rmse": rmse, "n_params": model.coef_.shape[0] + 1, "pred": pred}
        print(f"\n  {name}")
        print(f"  params: {model.coef_.shape[0] + 1}")
        print(f"  Test RMSE: {rmse:.4f} V")

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"  Summary (balanced: max {MAX_PER_BUCKET} per temp bucket)")
    print(f"{'='*60}")
    print(f"  {'Feature Set':^35} | {'Params':>7} | {'Test RMSE':>10}")
    print(f"  {'-'*35}-+-{'-'*7}-+-{'-'*10}")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["rmse"]):
        print(f"  {name:>35} | {r['n_params']:>7} | {r['rmse']:>10.4f}")

    # ── 按温度看差异 ──
    print(f"\n{'='*80}")
    print(f"  Detail: per-temperature RMSE")
    print(f"{'='*80}")
    y_te = y[test_idx]
    T_te = T[test_idx]
    names = list(results.keys())
    header = f"  {'temp':>6} | {'count':>6}"
    for n in names:
        header += f" | {n[:10]:>10}"
    print(header)
    print(f"  {'-'*6}-+-{'-'*6}" + "-+-" + "-" * 10 * len(names))
    temp_buckets = sorted(set(_temp_bucket(t) for t in T_te))
    for tb in temp_buckets:
        mask = (T_te >= tb - 2) & (T_te < tb + 3)
        n = mask.sum()
        if n == 0:
            continue
        line = f"  {tb:>5}C | {n:>6}"
        for name in names:
            rmse = _rmse(y_te[mask], results[name]["pred"][mask])
            line += f" | {rmse:>10.4f}"
        print(line)


if __name__ == "__main__":
    main()
