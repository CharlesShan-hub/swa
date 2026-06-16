"""
标定转换模块

ADC原始值 ↔ 电压值 的线性转换。
标定系数需通过实验确定。
"""

from dataclasses import dataclass


@dataclass
class Calibration:
    """标定参数"""
    # y = k * x + b, 其中 x 为 ADC 原始值，y 为电压 (V)
    k: float = 0.0   # 斜率（待标定）
    b: float = 0.0   # 截距（待标定）


_cal = Calibration()


def adc_to_voltage(adc_value: float) -> float:
    """ADC原始值 → 电压值"""
    return _cal.k * adc_value + _cal.b


def voltage_to_adc(voltage: float) -> float:
    """电压值 → ADC原始值"""
    return (voltage - _cal.b) / _cal.k if _cal.k != 0 else 0.0


def set_calibration(k: float, b: float):
    """设置标定系数"""
    _cal.k = k
    _cal.b = b
