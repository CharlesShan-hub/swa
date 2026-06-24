"""
对每段 A1 时间序列做曲线拟合，看看段内数据的变化规律。

用法:
    uv run python test/segment_fit.py                # 处理所有 103 段
    uv run python test/segment_fit.py --head 5        # 只测试前 5 段
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.rcParams["font.family"] = "Microsoft YaHei"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.signal import savgol_filter
from lib.data import load
from lib.metrics import extend

# ── 参数 ──
MAX_SEG = 500
MIN_SEG = 100

parser = argparse.ArgumentParser()
parser.add_argument("--head", type=int, default=0, help="只处理前 N 段（0=全部）")
args = parser.parse_args()

# ── 加载数据 ──
records = load("default")
records = extend(records, ["a1", "temp", "humid"])

all_data = []
for rec in records:
    v = rec["ACTUAL_VOLTAGE"]
    a1 = rec["_a1"]
    t = rec.get("RTU_REGS_P00_ENV_TEMP")
    h = rec.get("RTU_REGS_P00_ENV_HUMIDITY")
    when = rec.get("SYSTEM_TIME", "")
    if not isinstance(v, (int, float)) or a1 is None or t is None or h is None:
        continue
    try:
        dt = datetime.strptime(when[:19], "%Y-%m-%d %H:%M:%S")
        all_data.append((dt, round(v), a1, float(t), float(h)))
    except:
        continue

all_data.sort(key=lambda x: x[0])

# ── 分段 ──
segments = []
cur_v = None
current = []

for d in all_data:
    dt, v, a1, t, h = d
    if v != cur_v or len(current) >= MAX_SEG:
        if current and len(current) >= MIN_SEG:
            segments.append([cur_v, current])
        elif current and len(current) < MIN_SEG and segments:
            segments[-1][1].extend(current)
        current = []
        cur_v = v
    current.append((dt, a1, t, h))

if current:
    if len(current) >= MIN_SEG:
        segments.append([cur_v, current])
    elif segments:
        segments[-1][1].extend(current)

n_seg = len(segments) if args.head == 0 else min(args.head, len(segments))
segments = segments[:n_seg]
print(f"处理 {n_seg} 段")

# ── 对每段做拟合 ──
fig, axes = plt.subplots(n_seg, 3, figsize=(16, 1.8 * n_seg))
if n_seg == 1:
    axes = axes.reshape(1, 3)

for row, seg in enumerate(segments):
    v = seg[0]; data = seg[1]
    times = np.array([(d[0] - data[0][0]).total_seconds() / 60 for d in data])
    a1s = np.array([d[1] for d in data])
    temps = np.array([d[2] for d in data])
    hums = np.array([d[3] for d in data])
    dt_start = data[0][0].strftime("%m/%d %H:%M")
    dt_end = data[-1][0].strftime("%m/%d %H:%M")

    # SavGol 窗口
    window = min(51, len(a1s) if len(a1s) % 2 == 1 else len(a1s) - 1)
    if window >= 5 and window % 2 == 1:
        sg_a1 = savgol_filter(a1s, window, 3)
        sg_temp = savgol_filter(temps, window, 3)
        sg_hum = savgol_filter(hums, window, 3)
    else:
        sg_a1, sg_temp, sg_hum = a1s, temps, hums
        window = "N/A"

    # 左图：A1
    ax = axes[row, 0]
    ax.plot(times, a1s, "-", linewidth=0.4, color="#2196F3", alpha=0.3, label="原始")
    ax.plot(times, sg_a1, "-", linewidth=2, color="#D32F2F", label=f"SavGol (w={window})")
    ax.set_ylabel("A1", fontsize=8)
    ax.set_title(f"第{row+1}段: {v}V ({len(data)}点)", fontsize=9, pad=2)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

    # 中图：温度
    ax = axes[row, 1]
    ax.plot(times, temps, "-", linewidth=0.4, color="#FF5722", alpha=0.3, label="原始")
    ax.plot(times, sg_temp, "-", linewidth=2, color="#D32F2F", label=f"SavGol")
    ax.set_ylabel("温度 (°C)", fontsize=8)
    ax.set_title(f"第{row+1}段: 温度", fontsize=9, pad=2)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

    # 右图：湿度
    ax = axes[row, 2]
    ax.plot(times, hums, "-", linewidth=0.4, color="#4CAF50", alpha=0.3, label="原始")
    ax.plot(times, sg_hum, "-", linewidth=2, color="#D32F2F", label=f"SavGol")
    ax.set_ylabel("湿度 (%)", fontsize=8)
    ax.set_title(f"第{row+1}段: 湿度", fontsize=9, pad=2)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

    # 时间标签
    for col in range(3):
        ax = axes[row, col]
        if len(times) > 1:
            step = max(1, len(times) // 4)
            ax.set_xticks(times[::step])
            ax.set_xticklabels([f"{times[i]:.0f}m" for i in range(0, len(times), step)], fontsize=5)

    # 输出统计
    a1_mean = a1s.mean()
    a1_std = a1s.std()
    print(f"  第{row+1}段 {v}V: A1均值={a1_mean:.4f}, std={a1_std:.4f}, "
          f"noise_ratio={a1_std/a1_mean*100:.1f}%  "
          f"温度均值={temps.mean():.1f}°C 湿度均值={hums.mean():.1f}%")

fig.suptitle("各段 A1 / 温度 / 湿度 SavGol 平滑", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("test/segment_fit.png", dpi=150)
plt.close()
print(f"\n已保存: test/segment_fit.png（{n_seg} 段）")
