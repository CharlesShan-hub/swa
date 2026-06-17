"""Analyze prediction errors: find worst predictions."""
import sys, os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import importlib
from scripts.utils.loader import load_jsonl, split_jsonl
from src.swa.estimation.feature_extractor import extract_from_record

parser = argparse.ArgumentParser(description="分析最差预测结果")
parser.add_argument("-d", "--data", default="data/exported_data.jsonl",
                    help="数据文件路径 (默认: data/exported_data.jsonl)")
parser.add_argument("--algo", default="xgboost_model",
                    help="算法名称 (默认: xgboost_model)")
args = parser.parse_args()

records = load_jsonl(args.data)
_, _, test_records = split_jsonl(records, full_dataset=True, limit=0,
                                 train_ratio=0.9, val_ratio=0.0, test_ratio=0.1, seed=42)
print(f"Loaded {len(records)} records, test set: {len(test_records)}")

X_list, y_list = [], []
for rec in test_records:
    v = rec.get("ACTUAL_VOLTAGE")
    if v is None: continue
    try:
        y_list.append(float(v))
        X_list.append(extract_from_record(rec))
    except: continue

X = np.array(X_list)
y = np.array(y_list)
print(f"Extracted features for {len(X)} records")

module = importlib.import_module(f"scripts.traditional.{args.algo}")
model = module.train(X, y)
y_pred = module.predict(model, X)

errs = np.abs(y_pred - y)
idx = np.argsort(-errs)[:30]
print(f"\nTop 30 worst predictions ({args.algo}):")
print(f"{'True':>8} {'Pred':>8} {'Error':>8}")
for i in idx:
    print(f"{y[i]:>8.1f} {y_pred[i]:>8.1f} {errs[i]:>8.1f}")

print(f"\nError distribution:")
for p in [50, 80, 90, 95, 99]:
    print(f"  {p}% of errors < {np.percentile(errs, p):.2f} V")
