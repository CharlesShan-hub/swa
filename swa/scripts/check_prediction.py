"""看看 -40V 的预测效果"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.utils.loader import load_jsonl
from src.swa.estimation.predictor import predict_from_record

records = load_jsonl("data/exported_data.jsonl")

# 日期范围
dates = sorted(set(str(r["SYSTEM_TIME"])[:10] for r in records if r.get("SYSTEM_TIME")))
print(f"日期范围: {dates[0]} ~ {dates[-1]}\n")

# 取前 5 条 -40V
count = 0
for r in records:
    v = str(r.get("ACTUAL_VOLTAGE", ""))
    if "40" in v:
        pred = predict_from_record(r)
        status = "投" if abs(pred) > 50 else "退"
        print(f"  真实={v:>6}  预测电压={pred:>+7.2f}V  {status}")
        count += 1
        if count >= 5:
            break
