"""
模拟削波对 A1 的影响
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

n = 512
cycles = 7
t = np.arange(n)
true_amp = 1.0
pure = true_amp * np.cos(2 * np.pi * cycles * t / n)

clip_levels = np.arange(0.5, 1.01, 0.02)

true_a1 = np.abs(np.fft.rfft(pure))[cycles]
print(f"原始(无削波) A1 = {true_a1:.2f}")
print(f"\n削波程度 → A1 → 低估%")
print("-" * 40)

results = []
for cl in clip_levels:
    clipped = np.clip(pure, -cl, cl)
    mag = np.abs(np.fft.rfft(clipped))
    a1_m = mag[cycles]
    loss = (a1_m - true_a1) / true_a1 * 100
    results.append((cl, a1_m, loss))
    print(f"  峰值保留{cl*100:5.1f}%  A1={a1_m:8.2f}  低估{loss:+.2f}%")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
for cl in [0.6, 0.8, 0.95, 1.0]:
    clipped = np.clip(pure, -cl, cl)
    ax.plot(clipped, label=f"保留{cl*100:.0f}%", linewidth=1)
ax.set_title("不同削波程度的波形", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
cls = [r[0] for r in results]
a1s = [r[1] for r in results]
ax.plot(cls, a1s, 'b-o', markersize=4)
ax.axhline(true_a1, color='r', linestyle='--', label='真实A1')
ax.set_xlabel('峰值保留比例')
ax.set_ylabel('测量 A1')
ax.set_title('A1 随削波程度的变化', fontsize=11)
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[2]
losses = [r[2] for r in results]
ax.plot(cls, losses, 'r-o', markersize=4)
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel('峰值保留比例')
ax.set_ylabel('A1 低估 (%)')
ax.set_title('削波导致的 A1 低估', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = r'd:\project\work\swa\swa\scripts\clipping_simulation.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\n已保存: {out}")
plt.close(fig)
