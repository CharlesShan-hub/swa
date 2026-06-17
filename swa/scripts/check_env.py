"""看看数据里温湿度的实际变化范围"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scripts.utils.loader import load_jsonl

records = load_jsonl("data/exported_data.jsonl", extract_features=False)

temps = []
hums = []
for r in records:
    t = r.get("RTU_REGS_P00_ENV_TEMP")
    h = r.get("RTU_REGS_P00_ENV_HUMIDITY")
    if t: temps.append(float(t))
    if h: hums.append(float(h))

print(f"温度范围: {min(temps):.1f} ~ {max(temps):.1f} °C  (均值 {np.mean(temps):.1f}, 标准差 {np.std(temps):.2f})")
print(f"湿度范围: {min(hums):.1f} ~ {max(hums):.1f} %  (均值 {np.mean(hums):.1f}, 标准差 {np.std(hums):.2f})")

# 线性模型的系数看看T和RH的权重
import joblib
model = joblib.load("data/exported_data/model_params_linear_model.json", mmap_mode=None)
import json
with open("data/exported_data/model_params_linear_model.json") as f:
    meta = json.load(f)

coef = meta["params"]
print(f"\n线性模型系数:")
names = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","T","RH","RPM","Vpp","Kurt","Skew","bias"]
for i in range(min(len(coef), 17)):
    print(f"  {names[i] if i < 17 else f'x{i}':>6} = {coef[i]:>+10.4f}")
