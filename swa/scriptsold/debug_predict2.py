"""
精确对比：用 evaluate_by_voltage 相同的逻辑预测第一条数据
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
import numpy as np
from scripts.utils.loader import load_jsonl, get_dataset_model_path
from src.swa.estimation.feature_extractor import extract_from_record

# 1. 加载模型（完全复用 evaluate_by_voltage 的 load_model）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate_by_voltage import load_model, predict

model_path = "data/exported_data/model_params_random_forest_model.joblib"
model_info = load_model(model_path)
print(f"模型类型: {model_info['type']}, 算法: {model_info.get('algorithm')}")

# 2. 加载数据
records = load_jsonl("data/exported_data.jsonl", extract_features=True)

# 3. 预测前10条
print(f"\n{'#':>3} | {'真值':>6} | {'预测':>8} | {'误差':>8} | {'用例'}")
print("-" * 55)
for i in range(10):
    rec = records[i]
    truth = rec['ACTUAL_VOLTAGE']
    pred = predict(model_info, rec)
    err = pred - truth
    print(f"{i+1:>3} | {truth:>6.0f} | {pred:>8.2f} | {err:>+8.2f} | {rec.get('TEST_CASE_CODE', '')}")
