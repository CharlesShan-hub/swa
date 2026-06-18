"""
从 -55V / -40V / 97V 的拟合曲线上取点，计算 A1 与温湿度的定量关系。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from datetime import datetime
from scipy.interpolate import UnivariateSpline
from lib.data import load
from lib.metrics import extend

records = load("default")
records = extend(records, ["a1"])

N_BINS = 200

def mid_fit(x, y, s=0.001):
    """分箱取上下包络中值 + Spline"""
    bins = np.linspace(x.min(), x.max(), N_BINS + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    mids, used = [], []
    for i in range(N_BINS):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if mask.sum() < 10:
            continue
        used.append(centers[i])
        upper = np.percentile(y[mask], 90)
        lower = np.percentile(y[mask], 10)
        mids.append((upper + lower) / 2)
    used = np.array(used); mids = np.array(mids)
    spl = UnivariateSpline(used, mids, s=s)
    xs = np.linspace(x.min(), x.max(), 500)
    return xs, spl(xs), spl

results = {}

for target_v in [-55, -40, 97]:
    data = []
    for rec in records:
        v = rec["ACTUAL_VOLTAGE"]
        if not isinstance(v, (int, float)) or round(v) != target_v:
            continue
        a1 = rec["_a1"]
        t = rec.get("RTU_REGS_P00_ENV_TEMP")
        h = rec.get("RTU_REGS_P00_ENV_HUMIDITY")
        when = rec.get("SYSTEM_TIME", "")
        if a1 is None or t is None or h is None:
            continue
        try:
            dt = datetime.strptime(when[:19], "%Y-%m-%d %H:%M:%S")
            data.append((dt, a1, float(t), float(h)))
        except:
            continue
    data.sort(key=lambda x: x[0])
    times = np.array([(d[0] - data[0][0]).total_seconds() / 3600 for d in data])
    a1s = np.array([d[1] for d in data])
    temps = np.array([d[2] for d in data])
    hums = np.array([d[3] for d in data])

    # 三条拟合曲线（A1/温度/湿度 都用小 s 精拟合）
    xs, a1_smooth, _ = mid_fit(times, a1s, s=0.001)
    xs, t_smooth, _ = mid_fit(times, temps, s=0.001)
    xs, h_smooth, _ = mid_fit(times, hums, s=0.001)

    results[target_v] = {
        "a1": a1_smooth, "temp": t_smooth, "humid": h_smooth,
        "n": len(data)
    }

    print(f"\n{'='*60}")
    print(f"  {target_v}V (n={len(data)})")
    print(f"{'='*60}")

    # 1. A1 vs 湿度 的线性相关性
    corr_ah = np.corrcoef(h_smooth, a1_smooth)[0, 1]
    print(f"\n  A1 vs 湿度:  相关系数 r = {corr_ah:.4f}")

    # 线性回归斜率（A1 随湿度变化多少）
    A = np.column_stack([np.ones_like(h_smooth), h_smooth])
    coeffs, *_ = np.linalg.lstsq(A, a1_smooth, rcond=None)
    print(f"  线性回归:    A1 = {coeffs[1]:+.4f} × 湿度 {coeffs[0]:+.4f}")
    print(f"  → 湿度每升 1%，A1 变化 {coeffs[1]:+.4f}")

    # 2. A1 vs 温度 的线性相关性
    corr_at = np.corrcoef(t_smooth, a1_smooth)[0, 1]
    print(f"\n  A1 vs 温度:  相关系数 r = {corr_at:.4f}")
    A2 = np.column_stack([np.ones_like(t_smooth), t_smooth])
    coeffs2, *_ = np.linalg.lstsq(A2, a1_smooth, rcond=None)
    print(f"  线性回归:    A1 = {coeffs2[1]:+.4f} × 温度 {coeffs2[0]:+.4f}")
    print(f"  → 温度每升 1°C，A1 变化 {coeffs2[1]:+.4f}")

    # 3. 温湿度交互: A1 ~ 湿度 + 温度
    A3 = np.column_stack([np.ones_like(h_smooth), h_smooth, t_smooth])
    coeffs3, *_ = np.linalg.lstsq(A3, a1_smooth, rcond=None)
    print(f"\n  双变量回归:  A1 = {coeffs3[1]:+.4f}×湿 {coeffs3[2]:+.4f}×温 {coeffs3[0]:+.4f}")

    # 4. A1 的变化范围
    print(f"\n  A1 范围:     {a1_smooth.min():.4f} ~ {a1_smooth.max():.4f} (Δ={a1_smooth.max()-a1_smooth.min():.4f})")
    print(f"  湿度范围:    {h_smooth.min():.1f}% ~ {h_smooth.max():.1f}%")
    print(f"  温度范围:    {t_smooth.min():.1f}°C ~ {t_smooth.max():.1f}°C")

# ── 跨电压对比 ──
print(f"\n{'='*60}")
print(f"  跨电压对比：A1漂移有多少是湿度解释的？")
print(f"{'='*60}")
for v in [-55, -40, 97]:
    r = results[v]
    # A1 总标准差 / 湿度回归残差标准差
    a1_std = r["a1"].std()
    pred_a1 = coeffs[1] * r["humid"] + coeffs[0]  # 用该电压的湿度回归
    resid_std = np.std(r["a1"] - pred_a1)
    print(f"\n  {v:>4}V: A1_std={a1_std:.4f}, 湿度解释后残差std={resid_std:.4f}, "
          f"解释率={(1 - resid_std/a1_std)*100:.1f}%")
