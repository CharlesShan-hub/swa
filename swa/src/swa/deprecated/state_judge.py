"""
状态判别模块

根据幅值特征量判断压板投/退状态，带迟滞防抖。
"""

from enum import Enum
from dataclasses import dataclass

from ..config.settings import config


class PlatenState(Enum):
    """压板状态"""
    UNKNOWN = 0
    PUT_IN = 1       # 投（接通）→ 有电场
    PULL_OUT = 2     # 退（断开）→ 无电场


@dataclass
class JudgeResult:
    """单次判别结果"""
    state: PlatenState
    amplitude: float       # 幅值特征量
    threshold: float       # 当前使用的阈值
    confidence: float      # 置信度 0~1


def judge_from_amplitude(
    amplitude: float,
    threshold: float = None,
    hysteresis: float = None,
) -> PlatenState:
    """
    根据幅值判断压板投/退状态（带迟滞）。

    迟滞逻辑：
        - 当前为"退"，幅值 > (threshold + hysteresis/2) → 投
        - 当前为"投"，幅值 < (threshold - hysteresis/2) → 退
    """
    if threshold is None:
        threshold = config.detection.threshold_adc
    if hysteresis is None:
        hysteresis = config.detection.hysteresis

    # 暂用简单阈值（不带迟滞，后续完善）
    if amplitude > threshold:
        return PlatenState.PUT_IN
    else:
        return PlatenState.PULL_OUT


def batch_judge(
    amplitudes: list[float],
    threshold: float = None,
) -> list[JudgeResult]:
    """批量判别"""
    threshold = threshold or config.detection.threshold_adc
    results = []
    for amp in amplitudes:
        state = judge_from_amplitude(amp, threshold)
        results.append(JudgeResult(
            state=state,
            amplitude=amp,
            threshold=threshold,
            confidence=1.0,  # TODO: 实现置信度计算
        ))
    return results
