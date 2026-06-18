"""
确认 extract_features=False 对预测结果的影响
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import importlib
from scripts.utils.loader import load_jsonl, split_jsonl, get_dataset_model_path
from src.swa.estimation.feature_extractor import extract_from_record

# 用 extract_features=True 和 False 各加载一条数据对比
recs_true = load_jsonl("data/exported_data.jsonl", extract_features=True)
recs_false = load_jsonl("data/exported_data.jsonl", extract_features=False)

# 取第一条
r_true = recs_true[0]
r_false = recs_false[0]

print("--- 特征对比 ---")
f_true = extract_from_record(r_true)
f_false = extract_from_record(r_false)
names = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","T","RH","RPM","Vpp","Kurt","Skew"]
for i in range(16):
    diff = f_true[i] - f_false[i]
    mark = " ← 不同!" if abs(diff) > 0.001 else ""
    print(f"  {names[i]:>5}: True={f_true[i]:>10.4f}  False={f_false[i]:>10.4f}  diff={diff:>10.4f}{mark}")

# 预测对比
model_base = get_dataset_model_path("data/exported_data.jsonl", "data/model_params")
module = importlib.import_module("scripts.traditional.lightgbm_model")
import lightgbm as lgb
model = {"model": lgb.Booster(model_file=f"{model_base}_lightgbm_model.txt")}

pred_true = float(module.predict(model, f_true.reshape(1, -1))[0])
pred_false = float(module.predict(model, f_false.reshape(1, -1))[0])

print(f"\n--- LightGBM 预测对比 ---")
print(f"  extract_features=True  : {pred_true:.4f} V")
print(f"  extract_features=False : {pred_false:.4f} V")
print(f"  差值: {pred_false - pred_true:.4f} V")
