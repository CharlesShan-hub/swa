"""
信号分析流水线

整合加载 → 预处理 → FFT → 正弦波拟合 → 结果输出的完整流程。

来自同事的分析逻辑，封装为可复用的函数和数据结构。
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import curve_fit

from ..config.settings import config
from .amplitude import remove_dc, rms, sine_confidence
from .models import sine_wave


@dataclass
class AnalysisResult:
    """完整分析结果"""
    raw_signal: np.ndarray          # 原始信号 (512,)
    time: np.ndarray                # 时间轴 (s)
    sample_rate: float              # 采样率 (Hz)
    dc_offset: float                # 直流偏移
    amplitude: float                # 拟合振幅
    rms_value: float                # RMS有效值
    frequency: float                # 信号频率 (Hz)
    phase_rad: float                # 初始相位 (rad)
    phase_deg: float                # 初始相位 (°)
    confidence: float               # 正弦波置信度 [0, 1]
    fit_error: float                # 拟合平均绝对误差
    fitted_signal: np.ndarray       # 拟合信号
    freq_axis: np.ndarray           # FFT频率轴（正频率）
    fft_magnitude: np.ndarray       # FFT幅度谱（正频率）

    def __str__(self) -> str:
        return (
            f"AnalysisResult("
            f"f={self.frequency:.1f}Hz, "
            f"A={self.amplitude:.4f}, "
            f"RMS={self.rms_value:.4f}, "
            f"DC={self.dc_offset:.4f}, "
            f"conf={self.confidence:.1%})"
        )

    def short_summary(self) -> str:
        """简短文本总结"""
        lines = [
            "─" * 50,
            f"  频率:        {self.frequency:>8.2f} Hz",
            f"  振幅:        {self.amplitude:>8.4f}",
            f"  RMS有效值:   {self.rms_value:>8.4f}",
            f"  直流偏移:    {self.dc_offset:>8.4f}",
            f"  初始相位:    {self.phase_rad:>8.4f} rad ({self.phase_deg:.2f}°)",
            f"  置信度:      {self.confidence:>8.1%}",
            f"  信号判定:    {'正弦波' if self.confidence >= 0.5 else '非正弦波'}",
            "─" * 50,
        ]
        return "\n".join(lines)


def analyze_wave(
    signal_data: np.ndarray,
    sample_rate: Optional[float] = None,
    verbose: bool = False,
) -> AnalysisResult:
    """
    对 512 点波形数据进行完整分析。

    流程：
        1. 构建时间轴
        2. RMS 计算
        3. FFT 频谱分析 + 找主频
        4. 置信度计算
        5. 正弦波曲线拟合（含回退策略）
        6. 组装 AnalysisResult

    Args:
        signal_data: 采集到的 512 点波形
        sample_rate: 采样率，默认使用 config 中的 15873 Hz
        verbose: 是否打印拟合过程中的调试信息

    Returns:
        AnalysisResult 包含所有分析结果
    """
    if sample_rate is None:
        sample_rate = config.signal.sample_rate

    n = len(signal_data)
    dt = 1.0 / sample_rate
    time = np.arange(n) / sample_rate

    # ---- RMS ----
    rms_value = rms(signal_data)

    # ---- FFT 频谱分析 ----
    ac = remove_dc(signal_data)
    fft_result = fft(ac)
    freq_all = np.fft.fftfreq(n, d=dt)
    fft_mag = np.abs(fft_result)

    mask = freq_all >= 0
    freq_pos = freq_all[mask]
    fft_pos_mag = fft_mag[mask]

    # 找主频（跳过直流）
    dom_idx = np.argmax(fft_pos_mag[1:]) + 1
    dom_freq = float(freq_pos[dom_idx])

    # ---- 置信度 ----
    confidence = sine_confidence(fft_mag, dom_idx, n)

    # ---- 正弦波拟合 ----
    dc_guess = float(np.mean(signal_data))
    amp_guess = float((np.max(signal_data) - np.min(signal_data)) / 2.0)

    if verbose:
        print(f"  拟合初值: A={amp_guess:.4f}, f={dom_freq:.2f}, DC={dc_guess:.4f}")

    try:
        popt, _ = curve_fit(
            sine_wave, time, signal_data,
            p0=[amp_guess, dom_freq, 0.0, dc_guess],
            maxfev=10000,
            bounds=(
                [0, 0, -np.pi, -np.inf],
                [np.inf, sample_rate / 2, np.pi, np.inf],
            ),
        )
        fit_amp, fit_freq, fit_phase, fit_dc = popt
        fit_phase = fit_phase % (2 * np.pi)
    except Exception as e:
        if verbose:
            print(f"  拟合失败 ({e})，回退到 FFT 估计")
        fit_freq = dom_freq
        fit_amp = fft_pos_mag[dom_idx] / (n / 2)
        fit_phase = float(np.angle(fft_result[dom_idx]) % (2 * np.pi))
        fit_dc = dc_guess

    fitted = sine_wave(time, fit_amp, fit_freq, fit_phase, fit_dc)
    fit_err = float(np.mean(np.abs(signal_data - fitted)))

    return AnalysisResult(
        raw_signal=signal_data,
        time=time,
        sample_rate=sample_rate,
        dc_offset=fit_dc,
        amplitude=fit_amp,
        rms_value=rms_value,
        frequency=fit_freq,
        phase_rad=fit_phase,
        phase_deg=np.degrees(fit_phase),
        confidence=confidence,
        fit_error=fit_err,
        fitted_signal=fitted,
        freq_axis=freq_pos,
        fft_magnitude=fft_pos_mag,
    )
