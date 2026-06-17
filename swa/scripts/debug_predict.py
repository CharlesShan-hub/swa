"""
直接对比 predict_all 和 evaluate_by_voltage 两种预测方式
取 exported_data.jsonl 第一条（真值 110V）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
import numpy as np
import importlib
from scripts.utils.loader import load_jsonl, get_dataset_model_path
from src.swa.estimation.feature_extractor import extract_from_record

# 取第一条数据
records = load_jsonl("data/exported_data.jsonl", extract_features=True)
rec = records[0]
X = extract_from_record(rec).reshape(1, -1)

print(f"真值: {rec['ACTUAL_VOLTAGE']} V")
print(f"特征: Vpp={rec.get('vpp')}, Kurt={rec.get('kurtosis')}, Skew={rec.get('skewness')}\n")

model_base = get_dataset_model_path("data/exported_data.jsonl", "data/model_params")

for algo in ["random_forest_model", "catboost_model", "lightgbm_model", "linear_model"]:
    module = importlib.import_module(f"scripts.traditional.{algo}")

    if algo == "linear_model":
        model_path = f"{model_base}_{algo}.json"
        with open(model_path) as f:
            meta = json.load(f)
        model = np.array(meta["params"])
    elif algo == "catboost_model":
        model_path = f"{model_base}_{algo}.cbm"
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(verbose=0)
        m.load_model(model_path)
        model = {"model": m}
    elif algo == "lightgbm_model":
        model_path = f"{model_base}_{algo}.txt"
        import lightgbm as lgb
        model = {"model": lgb.Booster(model_file=model_path)}
    else:
        model_path = f"{model_base}_{algo}.joblib"
        import joblib
        loaded = joblib.load(model_path)
        print(f"  {algo}: joblib.load 返回类型 = {type(loaded).__name__}")
        print(f"            keys = {list(loaded.keys()) if isinstance(loaded, dict) else 'N/A'}")
        model = loaded

    pred = float(module.predict(model, X)[0])
    print(f"  {algo}: 预测={pred:.4f} V")
