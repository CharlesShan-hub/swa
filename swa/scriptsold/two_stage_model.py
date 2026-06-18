"""
两阶段分解模型：先学纯信号(固定温度)，再学温湿度漂移校正。

Stage 1: 在固定温湿度窗口(30°C±0.5°C, 40%±2%)内，训练纯信号线性模型
         features = [sine_amp, Kurt, Skew] → |voltage|  (sine_amp = 正弦拟合幅值替代 Vpp)
Stage 2: 对全部数据，用 Stage 1 预测 → 残差 = |真值| - |预测|
         拟合残差 ~ f(T, RH) 校正模型

最终: V_pred = pure_signal(sine_amp, Kurt, Skew) + correction(T, RH)

用法:
    uv run python scripts/two_stage_model.py
    uv run python scripts/two_stage_model.py --data data/exported_data.jsonl
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.linear_model import LinearRegression

from scripts.utils.loader import load_jsonl, split_jsonl
from src.swa.estimation.feature_extractor import extract_from_record


# ── Config ──
T_WINDOW = (29.5, 30.5)   # Stage 1 固定温度窗口
RH_WINDOW = (38.0, 42.0)  # Stage 1 固定湿度窗口
T_ORDER = 1                # 温度阶数（1=线性）
RH_ORDER = 1               # 湿度阶数（1=线性）
INTERACT = False           # 是否包含 T×RH 交互项
USE_SINE_FIT = True        # True: 用正弦拟合幅值替代 Vpp

FS = 15873                 # 采样率 Hz


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def _sine_amplitude(wave, rpm):
    """
    用最小二乘拟合正弦波: y = C + A*sin(2π*f0*t + φ)
    返回幅值 A（更抗噪，利用全部 512 点）
    """
    n = len(wave)
    ac = wave - np.mean(wave)
    if rpm is None or rpm <= 0:
        # 不知道转速，用 FFT 找基频
        fft_mag = np.abs(np.fft.fft(ac))[:n // 2]
        # 跳过直流，找峰值
        peak_idx = 1 + np.argmax(fft_mag[1:])
        f0 = peak_idx * FS / n
    else:
        f0 = rpm / 60.0  # RPM → Hz

    t = np.arange(n) / FS
    # 设计矩阵: [1, cos(2πf0t), sin(2πf0t)]
    cos_wave = np.cos(2 * np.pi * f0 * t)
    sin_wave = np.sin(2 * np.pi * f0 * t)
    A = np.column_stack([np.ones(n), cos_wave, sin_wave])
    coeffs, *_ = np.linalg.lstsq(A, wave, rcond=None)
    amp = np.sqrt(coeffs[1]**2 + coeffs[2]**2)
    return amp


def _pure_signal_features(X):
    """从 16 维特征中提取纯信号特征: Vpp, Kurt, Skew (Vpp 列会被覆盖为 sine amp)"""
    return X[:, 13:16]


def _make_env_features(T, RH):
    """构造环境特征: T, RH（一阶线性）"""
    feats = [T, RH]
    if T_ORDER >= 2:
        feats.append(T ** 2)
    if RH_ORDER >= 2:
        feats.append(RH ** 2)
    if INTERACT:
        feats.append(T * RH)
    return np.column_stack(feats)


def analyze_data(args):
    records = load_jsonl(args.data, extract_features=True)

    # ── 提取特征和标签 ──
    X_all = []
    y_all = []      # |voltage|
    T_all = []
    RH_all = []
    skipped = 0

    for rec in records:
        v = rec.get("ACTUAL_VOLTAGE")
        t = rec.get("RTU_REGS_P00_ENV_TEMP")
        h = rec.get("RTU_REGS_P00_ENV_HUMIDITY")
        if v is None or t is None or h is None:
            skipped += 1
            continue
        try:
            x = extract_from_record(rec)
        except Exception:
            skipped += 1
            continue

        # 用正弦拟合幅值替换 Vpp
        if USE_SINE_FIT:
            wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
            wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]
            rpm_val = rec.get("RTU_REGS_P00_ROTOR_RPM")
            try:
                rpm = float(rpm_val) if rpm_val is not None else 0
            except (ValueError, TypeError):
                rpm = 0
            amp = _sine_amplitude(wave, rpm)
            x[13] = amp  # 替换 Vpp 列

        X_all.append(x)
        y_all.append(abs(float(v)))
        T_all.append(float(t))
        RH_all.append(float(h))

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    T_all = np.array(T_all)
    RH_all = np.array(RH_all)

    print(f"total records: {len(records)}, skipped: {skipped}, used: {len(y_all)}")

    # ══════════════════════════════════════
    # Stage 1: 纯信号模型 (固定温度窗口)
    # ══════════════════════════════════════
    mask = (T_all >= T_WINDOW[0]) & (T_all <= T_WINDOW[1]) \
          & (RH_all >= RH_WINDOW[0]) & (RH_all <= RH_WINDOW[1])
    X_sig = _pure_signal_features(X_all[mask])
    y_sig = y_all[mask]
    T_win = T_all[mask]
    RH_win = RH_all[mask]

    print(f"\n{'='*60}")
    print(f"  Stage 1: Pure signal model at T ∈ [{T_WINDOW[0]}, {T_WINDOW[1]}]°C, RH ∈ [{RH_WINDOW[0]}, {RH_WINDOW[1]}]%")
    print(f"{'='*60}")
    print(f"  samples in window: {len(y_sig)}")
    # 统计窗口内的电压分布
    unique_v, counts_v = np.unique(y_sig, return_counts=True)
    print(f"  voltage range: {min(y_sig):.0f} ~ {max(y_sig):.0f} V")
    for vv, cc in zip(unique_v, counts_v):
        print(f"    {vv:5.0f}V: {cc} samples ({cc/len(y_sig)*100:.1f}%)")
    print(f"  temp range in window: {min(T_win):.1f} ~ {max(T_win):.1f} °C")
    print(f"  humid range in window: {min(RH_win):.1f} ~ {max(RH_win):.1f} %")

    model1 = LinearRegression()
    model1.fit(X_sig, y_sig)
    pred_sig_train = model1.predict(X_sig)
    rmse1_train = _rmse(y_sig, pred_sig_train)
    print(f"  feature: {'sine_amp' if USE_SINE_FIT else 'Vpp'} + Kurt + Skew")
    print(f"\n  Stage 1 training RMSE (within window): {rmse1_train:.4f} V")
    coef_names = ["sine_amp" if USE_SINE_FIT else "Vpp","Kurt","Skew"]
    print(f"  Coefficients:")
    for i, name in enumerate(coef_names):
        print(f"    {name:>6} = {model1.coef_[i]:>+10.4f}")
    print(f"    {'bias':>6} = {model1.intercept_:>+10.4f}")

    # ── Stage 1 应用到全部数据 ──
    X_sig_all = _pure_signal_features(X_all)
    pred1_all = model1.predict(X_sig_all)
    residual = y_all - pred1_all   # 残差 = 真值 - 纯信号预测

    mae1_all = _rmse(y_all, pred1_all)
    print(f"\n  Stage 1 applied to ALL data: RMSE = {mae1_all:.4f} V")

    # ══════════════════════════════════════
    # Stage 2: 温湿度漂移校正
    # ══════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Stage 2: T/RH drift correction")
    print(f"{'='*60}")

    X_env = _make_env_features(T_all, RH_all)
    model2 = LinearRegression()
    model2.fit(X_env, residual)

    # 校正后预测
    correction = model2.predict(X_env)
    pred2_all = pred1_all + correction
    mae2 = _rmse(y_all, pred2_all)
    improvement = mae1_all - mae2
    print(f"\n  RMSE before correction:  {mae1_all:.4f} V")
    print(f"  RMSE after correction:   {mae2:.4f} V")
    print(f"  Improvement:            {improvement:.4f} V ({improvement/mae1_all*100:.1f}%)")

    # 校正模型系数
    env_names = ["T", "RH"]
    if T_ORDER >= 2:
        env_names.append("T²")
    if RH_ORDER >= 2:
        env_names.append("RH²")
    if INTERACT:
        env_names.append("T×RH")
    print(f"\n  Correction coeffs:")
    for i, name in enumerate(env_names):
        print(f"    {name:>6} = {model2.coef_[i]:>+10.4f}")
    print(f"    {'bias':>6} = {model2.intercept_:>+10.4f}")

    # ── 按温度分桶看校正效果 ──
    print(f"\n{'='*80}")
    print(f"  Per-temperature RMSE comparison")
    print(f"{'='*80}")
    print(f" {'temp':>6} | {'count':>6} | {'before':>8} | {'after':>8} | {'improve':>8}")
    print(f" {'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    temp_buckets = sorted(set(round(t / 5) * 5 for t in T_all))
    for tb in temp_buckets:
        tm = (T_all >= tb - 2) & (T_all < tb + 3)
        n = tm.sum()
        if n == 0:
            continue
        rmse_b = _rmse(y_all[tm], pred1_all[tm])
        rmse_a = _rmse(y_all[tm], pred2_all[tm])
        imp = rmse_b - rmse_a
        print(f" {tb:>5}C | {n:>6} | {rmse_b:>7.4f} | {rmse_a:>7.4f} | {imp:>+7.4f}")

    # ── 按湿度分桶 ──
    print(f"\n{'='*80}")
    print(f"  Per-humidity RMSE comparison")
    print(f"{'='*80}")
    print(f" {'humid':>6} | {'count':>6} | {'before':>8} | {'after':>8} | {'improve':>8}")
    print(f" {'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    hum_buckets = sorted(set(round(h / 5) * 5 for h in RH_all))
    for hb in hum_buckets:
        hm = (RH_all >= hb - 2) & (RH_all < hb + 3)
        n = hm.sum()
        if n == 0:
            continue
        rmse_b = _rmse(y_all[hm], pred1_all[hm])
        rmse_a = _rmse(y_all[hm], pred2_all[hm])
        imp = rmse_b - rmse_a
        print(f" {hb:>4}% | {n:>6} | {rmse_b:>7.4f} | {rmse_a:>7.4f} | {imp:>+7.4f}")

    # ── 按电压分桶 ──
    print(f"\n{'='*80}")
    print(f"  Per-voltage RMSE comparison")
    print(f"{'='*80}")
    print(f" {'voltage':>7} | {'count':>6} | {'before':>8} | {'after':>8} | {'improve':>8}")
    print(f" {'-'*7}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    v_buckets = sorted(set(round(v / 10) * 10 for v in y_all))
    for vb in v_buckets:
        vm = (y_all >= vb - 5) & (y_all < vb + 5)
        n = vm.sum()
        if n == 0:
            continue
        rmse_b = _rmse(y_all[vm], pred1_all[vm])
        rmse_a = _rmse(y_all[vm], pred2_all[vm])
        imp = rmse_b - rmse_a
        print(f" {vb:>5}V | {n:>6} | {rmse_b:>7.4f} | {rmse_a:>7.4f} | {imp:>+7.4f}")

    # ── 保存模型参数 ──
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        model_data = {
            "stage1_coef": model1.coef_.tolist(),
            "stage1_intercept": model1.intercept_,
            "stage1_features": coef_names,
            "stage2_coef": model2.coef_.tolist(),
            "stage2_intercept": model2.intercept_,
            "stage2_features": env_names,
            "temp_window": list(T_WINDOW),
            "rmse_before": mae1_all,
            "rmse_after": mae2,
        }
        with open(args.save, "w") as f:
            json.dump(model_data, f, indent=2)
        print(f"\n  Model saved to: {args.save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-stage model: pure signal + T/RH correction"
    )
    parser.add_argument("--data", default="data/exported_data.jsonl",
                        help="data file path")
    parser.add_argument("--save", default="data/two_stage_model.json",
                        help="save model params to (default: data/two_stage_model.json)")
    args = parser.parse_args()
    analyze_data(args)
