"""
查看混合模型对未见电压的残差（residual）具体是多少。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from collections import defaultdict
from lib.data import load
from lib.traditional.hybrid import train_hybrid

records = load("smooth_all_w32")
features = ["a1", "temp", "humid"]

# 留出 -87V(u0) 和 -36V(u2)
train_v = [-55, -50, -43, -40, -20, 30, 50, 70, 72, 80, 97, 110]
test_v = [-87, -36]

result = train_hybrid(records, features, train_voltages=train_v, test_voltages=test_v, vol_limit=300)
phys, res = result["phys"], result["res"]

for test_v_single in test_v:
    groups = defaultdict(list)
    for rec in records:
        v = rec.get("ACTUAL_VOLTAGE")
        if not isinstance(v, (int, float)):
            continue
        feat = [rec.get(f"_{f}") for f in features]
        if any(f is None for f in feat):
            continue
        if round(v) == test_v_single:
            groups[test_v_single].append((feat, abs(float(v))))

    data = groups[test_v_single]
    X = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])

    V_phys = phys.predict_base(X[:, 0])
    residuals = res.predict(X)
    V_final = V_phys + residuals

    # 按湿度排序，看湿度如何影响残差
    idx = np.argsort(X[:, 2])
    X_sorted = X[idx]
    res_sorted = residuals[idx]
    phys_sorted = V_phys[idx]
    final_sorted = V_final[idx]
    y_sorted = y[idx]

    n_show = min(10, len(X_sorted))
    print(f"\n{test_v_single}V (n={len(X_sorted)}) — 按湿度排序，每个区间采样:")
    print(f"  {'A1':>8} {'湿度':>6} {'温度':>6} | {'V_phys':>8} {'残差':>8} {'V_final':>8} {'真值':>6}")
    print(f"  {'-'*8} {'-'*6} {'-'*6} | {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    step = max(1, len(X_sorted) // n_show)
    for i in range(0, len(X_sorted), step):
        print(f"  {X_sorted[i,0]:>8.4f} {X_sorted[i,2]:>6.1f} {X_sorted[i,1]:>6.1f} | "
              f"{phys_sorted[i]:>8.1f} {res_sorted[i]:>+8.1f} {final_sorted[i]:>8.1f} {y_sorted[i]:>6.0f}")

    print(f"\n  汇总:")
    print(f"    物理基线: 均值={V_phys.mean():.1f}V")
    print(f"    残差:     均值={residuals.mean():+.1f}V, 范围=[{residuals.min():+.1f}, {residuals.max():+.1f}]")
    print(f"    修正后:   均值={V_final.mean():.1f}V")
    print(f"    真值:     均值={y.mean():.0f}V")

    # 对比训练集中最接近的电压的 A1
    print(f"\n  训练集在相近湿度(~53%)下的 A1-电压对照:")
    for train_v_single in [-43, -55, 97]:
        tgroups = []
        for rec in records:
            v = rec.get("ACTUAL_VOLTAGE")
            if not isinstance(v, (int, float)):
                continue
            feat = [rec.get(f"_{f}") for f in features]
            if any(f is None for f in feat):
                continue
            if round(v) == train_v_single and 52 < feat[2] < 54:
                tgroups.append(feat)
        if tgroups:
            ta1_mean = np.mean([t[0] for t in tgroups])
            print(f"    {train_v_single:>4}V: A1≈{ta1_mean:.3f} (在湿度53%附近, {len(tgroups)}条)")
