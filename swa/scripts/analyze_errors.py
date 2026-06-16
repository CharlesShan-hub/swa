"""分析预测误差最大的数据"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.swa.signal_process.loader import load_jsonl
from src.swa.estimation.feature_extractor import extract_from_record
from src.swa.estimation import xgboost_model

records = load_jsonl("data/exported_data.jsonl")
X_list, y_list = [], []
for rec in records:
    v = rec.get("ACTUAL_VOLTAGE")
    if v is None: continue
    s = str(v).strip().lower().replace("v", "")
    try:
        y_list.append(float(s))
        X_list.append(extract_from_record(rec))
    except: continue

X = np.array(X_list)
y = np.array(y_list)
model = xgboost_model.train(X, y)
y_pred = xgboost_model.predict(model, X)

errs = np.abs(y_pred - y)
idx = np.argsort(-errs)[:30]
print("误差最大的30条:")
print(f"{'真实电压':>8} {'预测电压':>8} {'误差':>8}")
for i in idx:
    print(f"{y[i]:>8.1f} {y_pred[i]:>8.1f} {errs[i]:>8.1f}")

print(f"\n误差分布:")
for p in [50, 80, 90, 95, 99]:
    print(f"  {p}% 数据误差 < {np.percentile(errs, p):.2f} V")
