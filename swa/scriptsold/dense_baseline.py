"""
第一步：只选温湿度最密集的数据，训练纯信号模型。
然后输出：该模型在其他温湿度下的误差分布（为第二步做准备）。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from collections import defaultdict
from scripts.utils.loader import load_jsonl
from src.swa.estimation.feature_extractor import extract_from_record

# 加载数据
records = load_jsonl("data/exported_data.jsonl", extract_features=True)

# 统计温湿度联合分布
env_buckets = defaultdict(list)
for r in records:
    t = round(float(r.get("RTU_REGS_P00_ENV_TEMP", 0)))
    h = round(float(r.get("RTU_REGS_P00_ENV_HUMIDITY", 0)))
    env_buckets[(t, h)].append(r)

# 找出样本最多的温湿度组合
sorted_env = sorted(env_buckets.items(), key=lambda x: -len(x[1]))
print("=== 温湿度分布 Top 10 ===")
for (t, h), recs in sorted_env[:10]:
    print(f"  {t}C / {h}%: {len(recs):>6}条")

# 温湿度最密集的区域：32~34C, 40~42%
dense_records = [r for r in records 
    if 32 <= float(r.get("RTU_REGS_P00_ENV_TEMP", 0)) <= 34
    and 40 <= float(r.get("RTU_REGS_P00_ENV_HUMIDITY", 0)) <= 42]
print(f"\n密集区 (32~34C, 40~42%) 共 {len(dense_records)} 条")

# dense 数据的电压分布
volt_dist = defaultdict(int)
for r in dense_records:
    v = round(r["ACTUAL_VOLTAGE"] / 10) * 10
    volt_dist[v] += 1
print("\n密集区电压分布:")
for v in sorted(volt_dist):
    print(f"  {v:>+5}V: {volt_dist[v]:>6}条 ({volt_dist[v]/len(dense_records)*100:.1f}%)")

# 训练纯信号模型（只用 A1~A10 + Vpp + Kurt + Skew）
def select_features(X):
    return np.hstack([X[:, 0:10], X[:, 13:16]])

X_list = []
y_list = []
for r in dense_records:
    feat = extract_from_record(r)
    X_list.append(select_features(feat.reshape(1, -1))[0])
    y_list.append(abs(r["ACTUAL_VOLTAGE"]))

X_train = np.array(X_list)
y_train = np.array(y_list)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print(f"\n密集区模型训练完成")

# 对所有数据预测，看误差 vs 温度 / 湿度
X_all = select_features(np.array([extract_from_record(r) for r in records]))
y_all = np.abs(np.array([r["ACTUAL_VOLTAGE"] for r in records]))
y_pred = model.predict(X_all)
errors = y_pred - y_all  # 正 = 高估, 负 = 低估

# 按温度统计误差
print(f"\n=== 误差 vs 温度 ===")
temp_err = defaultdict(list)
for r, e in zip(records, errors):
    t = round(float(r.get("RTU_REGS_P00_ENV_TEMP", 0)))
    temp_err[t].append(e)
print(f"{'温度':>6} | {'样本数':>6} | {'平均误差':>10} | {'误差标准差':>10}")
for t in sorted(temp_err):
    errs = temp_err[t]
    print(f"{t:>4}C | {len(errs):>6} | {np.mean(errs):>+9.4f}V | {np.std(errs):>9.4f}")

# 按湿度统计误差
print(f"\n=== 误差 vs 湿度 ===")
hum_err = defaultdict(list)
for r, e in zip(records, errors):
    h = round(float(r.get("RTU_REGS_P00_ENV_HUMIDITY", 0)))
    hum_err[h].append(e)
print(f"{'湿度':>6} | {'样本数':>6} | {'平均误差':>10} | {'误差标准差':>10}")
for h in sorted(hum_err):
    errs = hum_err[h]
    print(f"{h:>4}% | {len(errs):>6} | {np.mean(errs):>+9.4f}V | {np.std(errs):>9.4f}")

# 未知数据预测
print(f"\n=== 未知数据预测 ===")
for fname in ["unknow.jsonl", "u1.jsonl", "u2.jsonl"]:
    path = f"data/{fname}"
    if not os.path.exists(path): continue
    import json
    with open(path) as f:
        urecs = [json.loads(l) for l in f if l.strip()]
    
    upreds = []
    for ur in urecs:
        feat = extract_from_record(ur)
        X = select_features(feat.reshape(1, -1))
        p = model.predict(X)[0]
        upreds.append(p)
    
    print(f"  {fname}: 平均预测 {np.mean(upreds):.2f}V (真值 {urecs[0].get('ACTUAL_VOLTAGE', '?')})")
