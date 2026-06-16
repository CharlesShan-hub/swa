"""
可视化模块

绘制 2×2 分析图表（原始信号 + 局部放大 + FFT频谱 + 结果文本）。

来自同事的 plot_analysis_results()，适配 AnalysisResult dataclass。
"""

import matplotlib.pyplot as plt
import numpy as np

from ..signal_process.analyzer import AnalysisResult

# 中文字体支持
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_analysis(result: AnalysisResult, save_path: str = "analysis.png"):
    """
    绘制 2×2 分析图表。

    布局：
        ┌──────────────────────┬──────────────────────┐
        │  原始信号 + 拟合对比   │  局部放大（前100点）   │
        ├──────────────────────┼──────────────────────┤
        │  FFT频谱图            │  分析结果文本          │
        └──────────────────────┴──────────────────────┘

    Args:
        result: 分析结果
        save_path: 图片保存路径
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("正弦波信号分析结果", fontsize=16, fontweight="bold")

    t = result.time
    orig = result.raw_signal
    fitted = result.fitted_signal

    # ---- 1. 原始 + 拟合 ----
    ax1.plot(t, orig, "b-", lw=1.5, label="原始信号", alpha=0.8)
    ax1.plot(t, fitted, "r--", lw=2, label="正弦波拟合", alpha=0.9)
    ax1.set_title("原始信号与正弦波拟合对比", fontsize=12, fontweight="bold")
    ax1.set_xlabel("时间 (s)")
    ax1.set_ylabel("信号幅值")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- 2. 局部放大 ----
    n_zoom = min(100, len(t))
    ax2.plot(t[:n_zoom], orig[:n_zoom], "b-", lw=2, label="原始信号", alpha=0.8)
    ax2.plot(t[:n_zoom], fitted[:n_zoom], "r--", lw=2.5, label="正弦波拟合", alpha=0.9)
    ax2.set_title(f"信号局部放大（前{n_zoom}个采样点）", fontsize=12, fontweight="bold")
    ax2.set_xlabel("时间 (s)")
    ax2.set_ylabel("信号幅值")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ---- 3. FFT 频谱 ----
    dom_idx = np.argmin(np.abs(result.freq_axis - result.frequency))
    ax3.plot(result.freq_axis, result.fft_magnitude, "g-", lw=1.5, label="FFT幅度谱")
    ax3.scatter(
        result.frequency, result.fft_magnitude[dom_idx],
        color="red", s=80, zorder=5,
        label=f"主频率: {result.frequency:.2f} Hz",
    )
    ax3.set_title("信号FFT频谱分析", fontsize=12, fontweight="bold")
    ax3.set_xlabel("频率 (Hz)")
    ax3.set_ylabel("FFT幅度")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, result.freq_axis.max() / 5)

    # ---- 4. 结果文本 ----
    ax4.axis("off")
    text = (
        f"正弦波信号分析结果\n\n"
        f"1. 正弦波置信度: {result.confidence:.4f}\n"
        f"   ({result.confidence * 100:.1f}%，≥0.5判定为正弦波)\n\n"
        f"2. 信号频率:    {result.frequency:.4f} Hz\n\n"
        f"3. 有效值 (RMS): {result.rms_value:.6f}\n\n"
        f"4. 初始相位:    {result.phase_rad:.4f} rad\n"
        f"   ({result.phase_deg:.2f}°)\n\n"
        f"5. 振幅:        {result.amplitude:.6f}\n\n"
        f"6. 直流偏移:    {result.dc_offset:.6f}\n\n"
        f"7. 采样参数:\n"
        f"   - 采样点数: {len(result.raw_signal)}\n"
        f"   - 采样频率: {result.sample_rate:.1f} Hz\n"
        f"   - 总采样时间: {result.time[-1]:.3f} s"
    )
    ax4.text(
        0.1, 0.9, text, transform=ax4.transAxes,
        fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"分析图表已保存至: {save_path}")
