"""实验：FFT 谐波数量对精度的影响"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import xgboost as xgb
from src.swa.signal_process.loader import load_jsonl
from src.swa.estimation.feature_extractor import extract_features

records = load_jsonl("data/exported_data.jsonl")
np.random.seed(42)
np.random.shuffle(records)

# 提取特征和标签
def extract_with_n_harmonics(records, n_harmonics):
    X, y = [], []
    for r in records:
        wave_str = r.get("RTU_REGS_P00_WAVE_DATA", "")
        wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]

        def _f(v, d=0.0):
            try: return float(v)
            except: return d
        temp = _f(r.get("RTU_REGS_P00_ENV_TEMP"))
        humid = _f(r.get("RTU_REGS_P00_ENV_HUMIDITY"))
        rpm = _f(r.get("RTU_REGS_P00_ROTOR_RPM"))

        feats = extract_features(wave, temp, humid, rpm)
        feats = np.concatenate([feats[:n_harmonics], feats[10:13]])  # A1~An + T, RH, RPM

        v_str = str(r.get("ACTUAL_VOLTAGE", "")).lower().replace("v", "").strip()
        try: voltage = float(v_str)
        except: continue
        X.append(feats)
        y.append(voltage)
    return np.array(X), np.array(y)

train_n = 34200
results = []

for n in range(1, 11):
    X, y = extract_with_n_harmonics(records, n)

    X_train, y_train = X[:train_n], y[:train_n]
    X_test, y_test = X[train_n:train_n + 3800], y[train_n:train_n + 3800]

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.1, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = float(np.mean(np.abs(y_pred - y_test)))
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    max_err = float(np.max(np.abs(y_pred - y_test)))
    results.append((n, mae, rmse, max_err))
    print(f"  A1~A{n:>2}:  MAE={mae:.4f}  RMSE={rmse:.4f}  MaxErr={max_err:.4f}")

print("\n=== 结果汇总 ===")
print(f"{'谐波数':>6}  {'MAE':>8}  {'RMSE':>8}  {'最大误差':>10}")
for n, mae, rmse, max_err in results:
    print(f"     A1~A{n:<2}:  {mae:>8.4f}  {rmse:>8.4f}  {max_err:>10.4f}")
