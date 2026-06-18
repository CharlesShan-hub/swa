"""
物理模型拟合：用傅里叶级数从 512 点波形中提取物理参数。

场磨传感器输出波形模型：
  y(t) = C + A₁·sin(ωt+φ₁) + A₃·sin(3ωt+φ₃) + A₅·sin(5ωt+φ₅)

其中:
  - ω = 转子旋转角频率（由 RPM 或 FFT 峰值确定）
  - A₁ = 基波幅值（主响应）
  - A₃、A₅ = 3 次/5 次谐波幅值（畸变成分）
  - C = 直流偏置

然后用线性模型: |voltage| = k₀ + k₁·A₁ + k₃·A₃ + k₅·A₅
或者只:     |voltage| = k₀ + k₁·A₁（如果谐波不贡献额外信息）

对比方案:
  1. Vpp+Kurt+Skew (3 特征)
  2. 物理模型参数 A₁+A₃+A₅ (3 特征)
  3. Full 16-feat (16 特征)

用法:
    uv run python scripts/physical_model.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.linear_model import LinearRegression
from collections import defaultdict

from scripts.utils.loader import load_jsonl
from src.swa.estimation.feature_extractor import extract_from_record

FS = 15873  # 采样率 Hz
MAX_PER_VOLTAGE = 5000


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.array(y) - np.array(p))**2)))


def fit_waveform(wave, rpm=None):
    """
    用最小二乘拟合傅里叶级数: y = C + A₁sin(ωt+φ₁) + A₃sin(3ωt+φ₃) + A₅sin(5ωt+φ₅)
    
    Returns:
        dict: {"amp1", "amp3", "amp5", "phase1", "phase3", "phase5", "dc", "r_squared"}
    """
    n = len(wave)
    ac = wave - np.mean(wave)
    
    # 确定基频
    if rpm is not None and rpm > 0:
        f0 = rpm / 60.0
    else:
        # FFT 找峰值
        fft_mag = np.abs(np.fft.fft(ac))[:n // 2]
        peak_idx = 1 + np.argmax(fft_mag[1:])
        f0 = peak_idx * FS / n

    t = np.arange(n) / FS
    wt = 2 * np.pi * f0 * t
    
    # 设计矩阵: [1, cos(ωt), sin(ωt), cos(3ωt), sin(3ωt), cos(5ωt), sin(5ωt)]
    A = np.column_stack([
        np.ones(n),
        np.cos(wt), np.sin(wt),
        np.cos(3*wt), np.sin(3*wt),
        np.cos(5*wt), np.sin(5*wt),
    ])
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, wave, rcond=None)
    
    C = coeffs[0]
    amp1 = np.sqrt(coeffs[1]**2 + coeffs[2]**2)
    amp3 = np.sqrt(coeffs[3]**2 + coeffs[4]**2)
    amp5 = np.sqrt(coeffs[5]**2 + coeffs[6]**2)
    phase1 = np.arctan2(coeffs[2], coeffs[1])
    phase3 = np.arctan2(coeffs[4], coeffs[3])
    phase5 = np.arctan2(coeffs[6], coeffs[5])
    
    # R² 拟合优度
    ss_res = residuals[0] if len(residuals) > 0 else np.sum((wave - A @ coeffs)**2)
    ss_tot = np.sum((wave - np.mean(wave))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return {
        "amp1": amp1, "amp3": amp3, "amp5": amp5,
        "phase1": phase1, "phase3": phase3, "phase5": phase5,
        "dc": C, "r_squared": r2
    }


def main():
    records = load_jsonl("data/exported_data.jsonl", extract_features=True, max_per_voltage=MAX_PER_VOLTAGE)
    print(f"total: {len(records)}")

    # ── 提取特征 ──
    X_vpp = []     # [Vpp, Kurt, Skew]
    X_phys = []    # [amp1, amp3, amp5]
    X_full = []    # 16 维
    y = []
    T = []
    phys_stats = {"amp1": [], "amp3": [], "amp5": [], "r_squared": []}

    for rec in records:
        v = rec.get("ACTUAL_VOLTAGE")
        if v is None:
            continue
        try:
            x = extract_from_record(rec)
        except Exception:
            continue
        
        wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
        wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]
        rpm_val = rec.get("RTU_REGS_P00_ROTOR_RPM")
        rpm = float(rpm_val) if rpm_val is not None else 0
        
        # 物理模型拟合
        phys = fit_waveform(wave, rpm)
        for k in phys_stats:
            phys_stats[k].append(phys[k])
        
        X_vpp.append(x[13:16])         # Vpp, Kurt, Skew
        X_phys.append([phys["amp1"], phys["amp3"], phys["amp5"]])
        X_full.append(x)               # 16 维
        y.append(abs(float(v)))
        T.append(float(rec.get("RTU_REGS_P00_ENV_TEMP", 0)))

    X_vpp = np.array(X_vpp)
    X_phys = np.array(X_phys)
    X_full = np.array(X_full)
    y = np.array(y)
    T = np.array(T)
    print(f"used: {len(y)}")

    # ── 物理参数统计 ──
    print(f"\n{'='*60}")
    print(f"  物理模型拟合统计")
    print(f"{'='*60}")
    for k in ["amp1", "amp3", "amp5", "r_squared"]:
        vals = phys_stats[k]
        print(f"  {k:>8}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
              f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")
    # 谐波占比
    amp1_arr = np.array(phys_stats["amp1"])
    amp3_arr = np.array(phys_stats["amp3"])
    amp5_arr = np.array(phys_stats["amp5"])
    total = amp1_arr + amp3_arr + amp5_arr
    mask = total > 0
    print(f"  amp3/amp1: mean={np.mean(amp3_arr[mask]/amp1_arr[mask]):.4f}")
    print(f"  amp5/amp1: mean={np.mean(amp5_arr[mask]/amp1_arr[mask]):.4f}")

    # ── 同一切分对比 ──
    np.random.seed(42)
    perm = np.random.permutation(len(y))
    n_test = len(y) // 10
    train_idx = perm[n_test:]
    test_idx = perm[:n_test]

    FEATURE_SETS = {
        "Vpp+Kurt+Skew (3)":                X_vpp,
        "Phys: A1+A3+A5 (3)":               X_phys,
        "Phys: A1+RPM (2)":                 np.column_stack([X_phys[:, 0], X_full[:, 12]]),
        "Phys: A1+RPM+A3+A5 (4)":           np.hstack([X_phys, X_full[:, 12:13]]),
        "Phys + env (6)":                   np.hstack([X_phys, X_full[:, 10:13]]),
        "Full 16-feat (16)":                X_full,
    }

    results = {}
    for name, X_feat in FEATURE_SETS.items():
        X_tr, X_te = X_feat[train_idx], X_feat[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model = LinearRegression()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rmse = _rmse(y_te, pred)
        results[name] = {"rmse": rmse, "pred": pred, "n_params": model.coef_.shape[0] + 1}
        print(f"\n  {name}")
        print(f"  params: {model.coef_.shape[0] + 1}, Test RMSE: {rmse:.4f} V")
        # 打印系数
        for i, c in enumerate(model.coef_):
            print(f"    w{i} = {c:+.4f}")
        print(f"    bias = {model.intercept_:+.4f}")

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"  Summary (balanced: max {MAX_PER_VOLTAGE} per voltage)")
    print(f"{'='*60}")
    print(f"  {'Feature Set':^30} | {'Params':>7} | {'Test RMSE':>10}")
    print(f"  {'-'*30}-+-{'-'*7}-+-{'-'*10}")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["rmse"]):
        print(f"  {name:>30} | {r['n_params']:>7} | {r['rmse']:>10.4f}")

    # ── 用物理模型预测未知数据 ──
    print(f"\n{'='*60}")
    print(f"  预测未知数据（物理模型 vs Vpp-only）")
    print(f"{'='*60}")
    for fname, true_v in [("u1", 43), ("u2", 36), ("u3", 72)]:
        recs = load_jsonl(f"data/{fname}.jsonl", extract_features=True, max_per_voltage=0)
        print(f"\n  {fname}.jsonl (真值={true_v}V, {len(recs)} 条):")
        
        # 收集
        X_vpp_u, X_phys_u, X_full_u = [], [], []
        for rec in recs:
            try:
                x = extract_from_record(rec)
            except Exception:
                continue
            wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
            wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]
            rpm_val = rec.get("RTU_REGS_P00_ROTOR_RPM")
            rpm = float(rpm_val) if rpm_val is not None else 0
            phys = fit_waveform(wave, rpm)
            X_vpp_u.append(x[13:16])
            X_phys_u.append([phys["amp1"], phys["amp3"], phys["amp5"]])
            X_full_u.append(x)
        
        if not X_phys_u:
            continue
        
        X_vpp_u = np.array(X_vpp_u)
        X_phys_u = np.array(X_phys_u)
        X_full_u = np.array(X_full_u)
        
        for name, X_feat, X_te in [
            ("Vpp+Kurt+Skew", X_vpp, X_vpp_u),
            ("Phys A1+A3+A5", X_phys, X_phys_u),
            ("Phys A1+RPM", np.column_stack([X_phys[:, 0], X_full[:, 12]]),
                           np.column_stack([np.array(X_phys_u)[:, 0], np.array(X_full_u)[:, 12]])),
            ("Full 16-feat", X_full, X_full_u),
        ]:
            model = LinearRegression()
            model.fit(X_feat[train_idx], y[train_idx])
            pred = np.abs(model.predict(X_te))
            err = np.abs(pred - true_v)
            print(f"    {name:20s}: avg={np.mean(pred):.2f}V, avg_err={np.mean(err):.2f}V")


if __name__ == "__main__":
    main()
