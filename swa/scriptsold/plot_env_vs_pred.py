"""查看 -40V 下，温度 / 湿度对预测值的影响"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from collections import defaultdict

from scripts.utils.loader import load_jsonl
from scripts.evaluate_by_voltage import load_model, predict

model_info = load_model("data/exported_data/model_params_linear_model.json")
records = load_jsonl("data/exported_data.jsonl", extract_features=True)
neg40 = [r for r in records if r["ACTUAL_VOLTAGE"] == -40.0]
print(f"-40V 数据共 {len(neg40)} 条\n")

# === 温度 ===
temp_groups = defaultdict(list)
for rec in neg40:
    t = round(float(rec.get("RTU_REGS_P00_ENV_TEMP", 0)) * 2) / 2
    pred = predict(model_info, rec)
    temp_groups[t].append(pred)

sorted_temps = sorted(temp_groups.keys())
t_means = [np.mean(temp_groups[t]) for t in sorted_temps]
t_stds = [np.std(temp_groups[t]) for t in sorted_temps]
t_counts = [len(temp_groups[t]) for t in sorted_temps]

print("=== 温度 ===")
print(f"{'温度':>6} | {'样本数':>6} | {'平均预测':>10} | {'标准差':>8}")
print("-" * 40)
for t, m, s, c in zip(sorted_temps, t_means, t_stds, t_counts):
    print(f"{t:>5}C | {c:>6} | {m:>8.2f}V | {s:>7.3f}")

# === 湿度 ===
hum_groups = defaultdict(list)
for rec in neg40:
    h = round(float(rec.get("RTU_REGS_P00_ENV_HUMIDITY", 0)))
    pred = predict(model_info, rec)
    hum_groups[h].append(pred)

sorted_hums = sorted(hum_groups.keys())
h_means = [np.mean(hum_groups[h]) for h in sorted_hums]
h_stds = [np.std(hum_groups[h]) for h in sorted_hums]
h_counts = [len(hum_groups[h]) for h in sorted_hums]

print("\n=== 湿度 ===")
print(f"{'湿度':>6} | {'样本数':>6} | {'平均预测':>10} | {'标准差':>8}")
print("-" * 40)
for h, m, s, c in zip(sorted_hums, h_means, h_stds, h_counts):
    print(f"{h:>4}% | {c:>6} | {m:>8.2f}V | {s:>7.3f}")

# === 画图 ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 温度预测
ax = axes[0, 0]
ax.errorbar(sorted_temps, t_means, yerr=t_stds, fmt='o-', capsize=4,
             color='#2196F3', ecolor='#BBDEFB', markersize=5)
ax.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='GT 40V')
ax.set_ylabel('Predicted Voltage (V)')
ax.set_title('Temp vs Predicted')
ax.legend()
ax.grid(True, alpha=0.3)

# 温度分布
ax = axes[1, 0]
ax.bar(sorted_temps, t_counts, color='#4CAF50', alpha=0.7, width=0.4)
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Count')
ax.set_title('Sample Dist by Temp')
ax.grid(True, alpha=0.3)

# 湿度预测
ax = axes[0, 1]
ax.errorbar(sorted_hums, h_means, yerr=h_stds, fmt='s-', capsize=4,
             color='#FF9800', ecolor='#FFE0B2', markersize=5)
ax.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='GT 40V')
ax.set_ylabel('Predicted Voltage (V)')
ax.set_title('Humidity vs Predicted')
ax.legend()
ax.grid(True, alpha=0.3)

# 湿度分布
ax = axes[1, 1]
ax.bar(sorted_hums, h_counts, color='#FF9800', alpha=0.7, width=2)
ax.set_xlabel('Humidity (%)')
ax.set_ylabel('Count')
ax.set_title('Sample Dist by Humidity')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/env_vs_pred_40V.png", dpi=150)
print(f"\nFigure saved: data/env_vs_pred_40V.png")
plt.show()
