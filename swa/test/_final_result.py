"""Leave-3-out 最终评估：Error = pred - true，输出 均值±标准差"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import defaultdict
from lib.data import load
from lib.traditional.hybrid_poly import train, predict

records = load("savgol_default")
features = ["a1", "temp", "humid"]
groups = defaultdict(list)
for rec in records:
    v = rec.get("ACTUAL_VOLTAGE")
    if not isinstance(v, (int, float)):
        continue
    feat = [rec.get(f"_{f}") for f in features]
    if any(f is None for f in feat):
        continue
    groups[round(v)].append((feat, abs(float(v))))

VOL_LIMIT = 500  # 每个电压均匀取最多 500 条

def uniform_sample(data, limit):
    """均匀采样：从 data 中均匀取最多 limit 条"""
    if len(data) <= limit:
        return data
    step = len(data) / limit
    return [data[int(round(i * step))] for i in range(limit)]

all_v = sorted(v for v in groups if len(groups[v]) > 50)
test_sets = [
    [40, 50, 70], [40, 60, 78], [50, 70, 72],
    [40, 72, 78], [50, 60, 70], [50, 72, 78],
    [60, 70, 72], [40, 50, 60],
]

print(f"{'测试电压（不参与训练）':<18} {'整体误差':<16}");
print()

results = []
for test_v in test_sets:
    train_v = [v for v in all_v if v not in test_v]
    train_d = sum((uniform_sample(groups[v], VOL_LIMIT) for v in train_v if v in groups), [])
    test_d = sum((groups[v] for v in test_v if v in groups), [])
    Xtr = np.array([d[0] for d in train_d]); ytr = np.array([d[1] for d in train_d])
    Xte = np.array([d[0] for d in test_d]); yte = np.array([d[1] for d in test_d])
    model = train(Xtr, ytr, degree=1)
    yp = predict(model, Xte)
    err_all = yp - yte
    mu_all, std_all = float(np.mean(err_all)), float(np.std(err_all))
    results.append((std_all, test_v))

    # 每个测试电压的误差
    details = []
    for v in test_v:
        idx = [i for i, d in enumerate(test_d) if d[1] == abs(v)]
        if idx:
            e = yp[idx] - yte[idx]
            vm = float(np.mean(e))
            vs = float(np.std(e))
            details.append(f"{v}V: {vm:+.1f}±{vs:.1f}V")

    print(f"  {str(test_v):<18} {mu_all:>+5.1f}±{std_all:<4.1f}V   {'  '.join(details)}")

print(f"\n{'='*55}")
stds = [r[0] for r in results]
print(f"  平均 Error 标准差: {np.mean(stds):.1f}V")
print(f"  最差 Error 标准差: {np.max(stds):.1f}V")
print(f"  最好 Error 标准差: {np.min(stds):.1f}V")
