"""画电压分布图"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from collections import Counter
from scripts.utils.loader import load_jsonl

records = load_jsonl("data/exported_data.jsonl")
cnt = Counter(r.get("ACTUAL_VOLTAGE", "") for r in records)

# 按电压数值排序
def sort_key(v):
    try:
        return float(str(v).replace("V", "").strip())
    except:
        return 0
sorted_items = sorted(cnt.items(), key=lambda x: sort_key(x[0]))
labels = [str(v) for v, _ in sorted_items]
values = [n for _, n in sorted_items]
total = sum(values)
pcts = [n/total*100 for n in values]

# 突出显示 30V
colors = ['#ff7f7f' if '30' in l else '#7fbf7f' for l in labels]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(labels, pcts, color=colors, edgecolor='white', linewidth=0.5)

# 在柱子上标条数
for bar, pct, val in zip(bars, pcts, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('电压 (V)', fontsize=11)
ax.set_ylabel('占比 (%)', fontsize=11)
ax.set_title('数据集电压分布（38000 条）', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(pcts) * 1.15)

# 图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#7fbf7f', label='其他电压'),
    Patch(facecolor='#ff7f7f', label='30V（2026/6/16 新增，7718条）'),
]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/voltage_distribution.png", dpi=150)
print("已保存: assets/voltage_distribution.png")
