"""
验证预测结果是否正确：对比 predict_all 和 evaluate_by_voltage
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
import numpy as np
import importlib
from scripts.utils.loader import get_dataset_model_path
from src.swa.estimation.feature_extractor import extract_from_record

# 取第一行数据
with open("data/exported_data.jsonl") as f:
    line = f.readline()
rec = json.loads(line)

# 打印原始数据
print("原始数据:")
print(f"  TEST_CASE_CODE: {rec['TEST_CASE_CODE']}")
print(f"  ACTUAL_VOLTAGE: {rec['ACTUAL_VOLTAGE']!r}")
print(f"  WAVE points: {len(rec['RTU_REGS_P00_WAVE_DATA'].split(','))}")

# 提取特征
features = extract_from_record(rec)
X = features.reshape(1, -1)
print(f"\n特征向量 (16维):")
names = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","T","RH","RPM","Vpp","Kurt","Skew"]
for i in range(16):
    print(f"  {names[i]:>5} = {features[i]:.6f}")

# 用 LightGBM 预测
print("\n各模型预测结果:")
model_base = get_dataset_model_path("data/exported_data.jsonl", "data/model_params")

algorithms = ["linear_model", "catboost_model", "lightgbm_model", "random_forest_model"]
for algo in algorithms:
    module = importlib.import_module(f"scripts.traditional.{algo}")

    ext_map = {
        "linear_model": ".json", "catboost_model": ".cbm",
        "lightgbm_model": ".txt", "random_forest_model": ".joblib"
    }

    path = f"{model_base}_{algo}{ext_map[algo]}"
    print(f"\n  {algo}:")
    print(f"    模型文件: {path}")
    print(f"    文件存在: {os.path.exists(path)}")

    if algo == "linear_model":
        with open(path) as f:
            meta = json.load(f)
        model = np.array(meta["params"])
    elif algo == "catboost_model":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(verbose=0)
        model.load_model(path)
        model = {"model": model}
    elif algo == "lightgbm_model":
        import lightgbm as lgb
        model = {"model": lgb.Booster(model_file=path)}
    else:
        import joblib
        model = joblib.load(path)

    pred = float(module.predict(model, X)[0])
    print(f"    预测: {pred:.4f} V")

print(f"\n真值: {rec['ACTUAL_VOLTAGE']} V")
