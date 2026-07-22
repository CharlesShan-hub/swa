"""
A/B Test: 带 RPM vs 不带 RPM

对比 score + harm_a1 + temperature + humidity ± rpm
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from swa.detection import least_squares

PROJECT_DIR = r"d:\project\work\swa\swa\src\data\projects\new"
TRAIN_V = [70, 90, 110, 130, 150]
TEST_V = [80, 100, 120, 140]

base_kwargs = dict(
    project_dir=PROJECT_DIR,
    train_voltages=TRAIN_V,
    test_voltages=TEST_V,
    window_size=8,
    window_method="mean",
    device_mapping=True,
    ref_device_id=None,
    humidity_correction=False,
    max_noise_pct=1.0,
    noise_correction=False,
    max_samples_per_voltage=0,
    device_id=None,
)


def run_once(features: list[str], label: str):
    """修改 _FEATURE_NAMES 后执行一次 run。"""
    least_squares._FEATURE_NAMES = features
    result = least_squares.run(**base_kwargs)
    metrics = result["metrics"]
    test_metrics = metrics["test"]

    # 逐设备/逐电压计算
    test_results = result["test_results"]
    df = pd.DataFrame(test_results)
    df["voltage"] = df["actual"]

    per_device = {}
    if "device_id" in df.columns:
        for dev, grp in df.groupby("device_id"):
            per_device[dev[-4:] if dev else "?"] = {
                "mae": float(np.mean(np.abs(grp["actual"] - grp["pred"]))),
                "n": len(grp),
            }

    per_voltage = {}
    for v, grp in df.groupby("voltage"):
        per_voltage[f"{v:.0f}V"] = {
            "mae": float(np.mean(np.abs(grp["actual"] - grp["pred"]))),
            "n": len(grp),
        }

    return {
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_mape": test_metrics["mape"],
        "train_count": metrics["train_count"],
        "test_count": metrics["test_count"],
        "coefficients": result["coefficients"],
        "intercept": result["intercept"],
        "per_device": per_device,
        "per_voltage": per_voltage,
    }


# ── A: 有 RPM ──
print("A: 带 RPM  (score, harm_a1, temperature, humidity, rpm)")
print("-" * 50)
r_a = run_once(
    ["score", "harm_a1", "temperature", "humidity", "rpm"],
    "带 RPM",
)

# ── B: 无 RPM ──
print("\nB: 无 RPM  (score, harm_a1, temperature, humidity)")
print("-" * 50)
r_b = run_once(
    ["score", "harm_a1", "temperature", "humidity"],
    "无 RPM",
)

# ── 对比 ──
print("\n" + "=" * 65)
print("A/B 对比结果")
print("=" * 65)

rows = [
    ("MAE",       r_a["test_mae"], r_b["test_mae"], False),
    ("RMSE",      r_a["test_rmse"], r_b["test_rmse"], False),
    ("R²",        r_a["test_r2"], r_b["test_r2"], True),
    ("MAPE(%)",   r_a["test_mape"], r_b["test_mape"], False),
    ("训练样本",  r_a["train_count"], r_b["train_count"], None),
    ("测试样本",  r_a["test_count"], r_b["test_count"], None),
]

print(f"\n{'指标':>12s}  {'带 RPM':>12s}  {'无 RPM':>12s}  {'差值':>12s}  {'改善':>8s}")
print("-" * 62)
for name, va, vb, higher_better in rows:
    if isinstance(va, int):
        print(f"{name:>12s}  {va:>12d}  {vb:>12d}")
    else:
        diff = va - vb
        if higher_better:
            imp = "✅更好" if va > vb else "❌更差" if va < vb else "="
        else:
            imp = "✅更好" if va < vb else "❌更差" if va > vb else "="
        print(f"{name:>12s}  {va:>12.4f}  {vb:>12.4f}  {diff:>+12.4f}  {imp:>8s}")

# 回归系数对比
print(f"\n{'─'*62}")
print("回归系数对比:")
print(f"{'─'*62}")
print(f"{'特征':>20s}  {'带 RPM':>12s}  {'无 RPM':>12s}")
all_feats = sorted(set(r_a["coefficients"]) | set(r_b["coefficients"]))
for f in all_feats:
    ca = r_a["coefficients"].get(f, 0)
    cb = r_b["coefficients"].get(f, 0)
    s = f" (截距={r_a['intercept']:.2f})" if f == all_feats[0] else ""
    print(f"  {f:>18s}  {ca:>+10.4f}  {cb:>+10.4f}{s}")

# 每设备 MAE
print(f"\n{'─'*62}")
print("每设备 MAE:")
print(f"{'─'*62}")
print(f"{'设备':>10s}  {'带 RPM':>12s}  {'无 RPM':>12s}  {'差值':>12s}  {'n':>6s}")
all_devs = sorted(set(r_a["per_device"]) | set(r_b["per_device"]))
for d in all_devs:
    da = r_a["per_device"].get(d, {})
    db = r_b["per_device"].get(d, {})
    ma = da.get("mae", float("nan"))
    mb = db.get("mae", float("nan"))
    na = da.get("n", 0)
    diff = mb - ma
    imp = "✅" if diff > 0 else "❌" if diff < 0 else "="
    print(f"  {d:>8s}  {ma:>10.4f}  {mb:>10.4f}  {diff:>+10.4f}  {na:>5d}  {imp}")

# 每电压 MAE
print(f"\n{'─'*62}")
print("每电压 MAE:")
print(f"{'─'*62}")
print(f"{'电压':>8s}  {'带 RPM':>12s}  {'无 RPM':>12s}  {'差值':>12s}  {'n':>6s}")
all_v = sorted(set(r_a["per_voltage"]) | set(r_b["per_voltage"]))
for v in all_v:
    va_d = r_a["per_voltage"].get(v, {})
    vb_d = r_b["per_voltage"].get(v, {})
    ma = va_d.get("mae", float("nan"))
    mb = vb_d.get("mae", float("nan"))
    nv = va_d.get("n", 0)
    diff = mb - ma
    imp = "✅" if diff > 0 else "❌" if diff < 0 else "="
    print(f"  {v:>6s}  {ma:>10.4f}  {mb:>10.4f}  {diff:>+10.4f}  {nv:>5d}  {imp}")

print(f"\n{'='*62}")
print("结论:")
print(f"  RPM 对 MAE 的影响: {(r_b['test_mae'] - r_a['test_mae']):+.4f}")
print(f"  无 RPM 下 R² = {r_b['test_r2']:.4f}, 有 RPM 下 R² = {r_a['test_r2']:.4f}")
