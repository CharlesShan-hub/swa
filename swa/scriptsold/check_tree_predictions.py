"""
检查树模型预测值是否集中在训练集见过的电压上。
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.utils.loader import load_jsonl
from src.swa.estimation.feature_extractor import extract_from_record

# 训练集电压分布（均衡后）
records = load_jsonl("data/exported_data.jsonl", extract_features=False, max_per_voltage=5000)
vs = [abs(float(r["ACTUAL_VOLTAGE"])) for r in records if r.get("ACTUAL_VOLTAGE") is not None]
print("=== 均衡后训练集电压分布 ===")
for v in sorted(set(round(v / 10) * 10 for v in vs)):
    n = sum(1 for x in vs if abs(x - v) < 5)
    print(f"  {v:5.0f}V: {n} 条")

# 预测 u1~u3 每条的预测值
ALGOS = ["random_forest_model", "catboost_model", "xgboost_model", "lightgbm_model"]
MODEL_DIR = "data/exported_data"

for fname, true_v in [("u1", 43), ("u2", 36), ("u3", 72)]:
    recs = load_jsonl(f"data/{fname}.jsonl", extract_features=True, max_per_voltage=0)
    print(f"\n=== {fname}.jsonl (真值={true_v}V) 各模型预测值分布 ===")
    for algo in ALGOS:
        if algo == "xgboost_model":
            import xgboost as xgb
            bst = xgb.Booster()
            bst.load_model(f"{MODEL_DIR}/model_params_{algo}.ubj")
            model = {"model": bst}
        elif algo == "lightgbm_model":
            import lightgbm as lgb
            bst = lgb.Booster(model_file=f"{MODEL_DIR}/model_params_{algo}.txt")
            model = {"model": bst}
        elif algo == "catboost_model":
            from catboost import CatBoostRegressor
            m = CatBoostRegressor(verbose=0)
            m.load_model(f"{MODEL_DIR}/model_params_{algo}.cbm")
            model = {"model": m}
        else:
            import joblib
            model = joblib.load(f"{MODEL_DIR}/model_params_{algo}.joblib")

        preds = []
        for rec in recs:
            X = extract_from_record(rec).reshape(1, -1)
            X = X[:, 10:16]  # har=0: [T,RH,RPM,Vpp,Kurt,Skew]
            if algo == "xgboost_model":
                p = float(model["model"].predict(xgb.DMatrix(X))[0])
            elif algo == "lightgbm_model":
                p = float(model["model"].predict(X)[0])
            else:
                p = float(model["model"].predict(X)[0])
            preds.append(abs(p))

        uniq, counts = np.unique(np.round(preds, 1), return_counts=True)
        top5 = sorted(zip(counts, uniq), reverse=True)[:5]
        print(f"  {algo:22s} avg={np.mean(preds):.1f}V  min={np.min(preds):.1f}  max={np.max(preds):.1f}")
        print(f"    top5: {', '.join(f'{v:.1f}V(x{c})' for c, v in top5)}")
        # 看唯一值数量 — 越少说明越"背"
        n_unique = len(uniq)
        n_total = len(preds)
        print(f"    唯一值: {n_unique}/{n_total} ({n_unique/n_total*100:.0f}% 不同的值)")
