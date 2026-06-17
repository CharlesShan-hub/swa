"""
特征提取器

从 512 点波形 + 环境参数中提取特征向量（16 维）：
    [A1, A2, ..., A10, T, RH, RPM, Vpp, Kurtosis, Skewness]

其中 A1~A10 是 FFT 前 10 次谐波的幅值。
"""

import numpy as np
from scipy.fftpack import fft


def extract_features(
    wave: np.ndarray,
    temperature: float = 0.0,
    humidity: float = 0.0,
    rpm: float = 0.0,
    vpp: float = 0.0,
    kurtosis: float = 0.0,
    skewness: float = 0.0,
    sample_rate: int = 15873,
) -> np.ndarray:
    """从 512 点波形 + 环境参数 + 时域统计特征提取特征向量（16 维）。"""
    ac = wave - np.mean(wave)
    n = len(ac)
    fft_result = fft(ac)
    fft_mag = np.abs(fft_result)[: n // 2]

    harmonics = np.zeros(10)
    for i in range(10):
        idx = i + 1
        if idx < len(fft_mag):
            harmonics[i] = 2.0 * fft_mag[idx] / n

    features = np.concatenate([
        harmonics,
        [temperature, humidity, rpm, vpp, kurtosis, skewness],
    ])
    return features


def extract_from_record(record: dict) -> np.ndarray:
    """从一条记录提取特征"""
    wave_str = record.get("RTU_REGS_P00_WAVE_DATA", "")
    wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)[:512]

    def _to_float(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    temp = _to_float(record.get("RTU_REGS_P00_ENV_TEMP", 0))
    humid = _to_float(record.get("RTU_REGS_P00_ENV_HUMIDITY", 0))
    rpm = _to_float(record.get("RTU_REGS_P00_ROTOR_RPM", 0))
    vpp = _to_float(record.get("vpp", 0))
    kurt = _to_float(record.get("kurtosis", 0))
    skew_val = _to_float(record.get("skewness", 0))

    return extract_features(wave, temp, humid, rpm, vpp, kurt, skew_val)
