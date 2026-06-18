"""
画温湿度漂移曲线图 — 密集区模型在各温湿度下的误差
"""
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
from src.swa.estimation.feature_extractor import extract_from_record
from sklearn.linear_model import LinearRegression

# 加载数据
records = load_jsonl("data/exported_data.jsonl", extract_features=True)

# 只选密集区 (32~34C, 40~42%) 训练纯信号模型
def select_features(X):
    return np.hstack([X[:, 0:10], X[:, 13:16]])

dense = [r for r in records 
    if 32 <= float(r.get("RTU_REGS_P00_ENV_TEMP", 0)) <= 34
    and 40 <= float(r.get("RTU_REGS_P00_ENV_HUMIDITY", 0)) <= 42]

X_dense = select_features(np.array([extract_from_record(r) for r in dense]))
y_dense = np.abs(np.array([r["ACTUAL_VOLTAGE"] for r in dense]))

model = LinearRegression()
model.fit(X_dense, y_dense)

# 对所有数据预测，算误差
X_all = select_features(np.array([extract_from_record(r) for r in records]))
y_true = np.abs(np.array([r["ACTUAL_VOLTAGE"] for r in records]))
y_pred = model.predict(X_all)
errors = y_pred - y_true

# 按温度统计
temp_err = defaultdict(list)
for r, e in zip(records, errors):
    t = float(r.get("RTU_REGS_P00_ENV_TEMP", 0))
    temp_err[round(t)].append(e)

# 按湿度统计
hum_err = defaultdict(list)
for r, e in zip(records, errors):
    h = float(r.get("RTU_REGS_P00_ENV_HUMIDITY", 0))
    hum_err[round(h)].append(e)

# 画图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 温度 - 散点 + 均值曲线
ax = axes[0, 0]
temps_sorted = sorted(temp_err.keys())
t_means = [np.mean(temp_err[t]) for t in temps_sorted]
t_stds = [np.std(temp_err[t]) for t in temps_sorted]
t_counts = [len(temp_err[t]) for t in temps_sorted]

ax.errorbar(temps_sorted, t_means, yerr=t_stds, fmt='o-', capsize=4,
            color='#2196F3', ecolor='#BBDEFB', markersize=6)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Prediction Error (V)')
ax.set_title('Error vs Temperature (positive = overestimate)')
ax.grid(True, alpha=0.3)

# 温度 - 样本分布
ax = axes[1, 0]
ax.bar(temps_sorted, t_counts, color='#4CAF50', alpha=0.7, width=0.6)
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Sample Count')
ax.set_title('Sample Distribution by Temperature')
ax.grid(True, alpha=0.3)
# 标注密集区
ax.axvspan(32, 34, color='red', alpha=0.08, label='dense region')
ax.legend()

# 湿度 - 散点 + 均值曲线
ax = axes[0, 1]
hums_sorted = sorted(hum_err.keys())
# 只画样本数>=10的，避免少量样本的噪音
hums_plot = [h for h in hums_sorted if len(hum_err[h]) >= 10]
h_means = [np.mean(hum_err[h]) for h in hums_plot]
h_stds = [np.std(hum_err[h]) for h in hums_plot]
h_counts = [len(hum_err[h]) for h in hums_plot]

ax.errorbar(hums_plot, h_means, yerr=h_stds, fmt='s-', capsize=4,
            color='#FF9800', ecolor='#FFE0B2', markersize=6)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Humidity (%)')
ax.set_ylabel('Prediction Error (V)')
ax.set_title('Error vs Humidity (positive = overestimate)')
ax.grid(True, alpha=0.3)

# 湿度 - 样本分布
ax = axes[1, 1]
ax.bar(hums_plot, h_counts, color='#FF9800', alpha=0.7, width=1)
ax.set_xlabel('Humidity (%)')
ax.set_ylabel('Sample Count')
ax.set_title('Sample Distribution by Humidity')
ax.grid(True, alpha=0.3)
ax.axvspan(40, 42, color='red', alpha=0.08, label='dense region')
ax.legend()

plt.suptitle('Drift Analysis: Pure Signal Model Error vs Environment', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("data/drift_curves.png", dpi=150)
print(f"Figure saved: data/drift_curves.png")
plt.show()
