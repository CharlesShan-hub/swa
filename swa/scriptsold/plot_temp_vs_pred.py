"""看看 -40V 下，温度对预测值的影响"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import json
from collections import defaultdict

from scripts.utils.loader import load_jsonl
from src.swa.estimation.feature_extractor import extract_from_record
from scripts.evaluate_by_voltage import load_model, predict

# 加载线性模型
model_info = load_model("data/exported_data/model_params_linear_model.json")
print(f"模型类型: {model_info['type']}")

# 筛选 -40V 的数据
records = load_jsonl("data/exported_data.jsonl", extract_features=True)
neg40 = [r for r in records if r["ACTUAL_VOLTAGE"] == -40.0]
print(f"-40V 数据共 {len(neg40)} 条")

# 按温度分组统计预测值
temp_groups = defaultdict(list)
for rec in neg40:
    t = round(float(rec.get("RTU_REGS_P00_ENV_TEMP", 0)) * 2) / 2  # 精确到 0.5°C
    pred = predict(model_info, rec)
    temp_groups[t].append(pred)

# 按温度排序
sorted_temps = sorted(temp_groups.keys())
means = [np.mean(temp_groups[t]) for t in sorted_temps]
stds = [np.std(temp_groups[t]) for t in sorted_temps]
counts = [len(temp_groups[t]) for t in sorted_temps]

print(f"\n{'温度':>6} | {'样本数':>6} | {'平均预测':>10} | {'标准差':>8}")
print("-" * 40)
for t, m, s, c in zip(sorted_temps, means, stds, counts):
    print(f"{t:>5}°C | {c:>6} | {m:>8.2f}V | {s:>7.3f}")

# 画图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# 上：预测值随温度变化
ax1.errorbar(sorted_temps, means, yerr=stds, fmt='o-', capsize=4, 
             color='#2196F3', ecolor='#BBDEFB', markersize=6)
ax1.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='Ground Truth 40V')
ax1.set_ylabel('Predicted Voltage (V)')
ax1.set_title('-40V: Predicted Value vs Temperature (Linear Model)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 下：样本分布
ax2.bar(sorted_temps, counts, color='#4CAF50', alpha=0.7, width=0.4)
ax2.set_xlabel('Temperature (C)')
ax2.set_ylabel('Sample Count')
ax2.set_title('Sample Distribution by Temperature')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/temp_vs_pred_40V.png", dpi=150)
print(f"Figure saved: data/temp_vs_pred_40V.png")
plt.show()
