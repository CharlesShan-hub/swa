"""Analyze prediction errors: find worst predictions."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from src.swa.signal_process.loader import load_jsonl
from src.swa.estimation.feature_extractor import extract_from_record
from src.swa.estimation import xgboost_model

records = load_jsonl("data/exported_data.jsonl")
print(f"Loaded {len(records)} records")

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
print(f"Extracted features for {len(X)} records")

model = xgboost_model.train(X, y)
y_pred = xgboost_model.predict(model, X)

errs = np.abs(y_pred - y)
idx = np.argsort(-errs)[:30]
print("\nTop 30 worst predictions:")
print(f"{'True':>8} {'Pred':>8} {'Error':>8}")
for i in idx:
    print(f"{y[i]:>8.1f} {y_pred[i]:>8.1f} {errs[i]:>8.1f}")

print(f"\nError distribution:")
for p in [50, 80, 90, 95, 99]:
    print(f"  {p}% of errors < {np.percentile(errs, p):.2f} V")
