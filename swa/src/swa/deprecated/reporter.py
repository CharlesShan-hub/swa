"""
结果输出模块

将判别结果以多种格式输出。
"""

from dataclasses import dataclass, asdict
from typing import Optional
import json

from ..detection.state_judge import JudgeResult, PlatenState


@dataclass
class ChannelResult:
    """单个通道的完整判别结果"""
    channel_id: str               # 如 "P1_0"
    slave_id: int                 # 从机地址
    system_time: str              # 设备系统时间
    amplitude: float              # 幅值特征量
    state: str                    # "投" / "退" / "未知"
    confidence: float             # 置信度
    temperature: Optional[float] = None
    humidity: Optional[float] = None


def to_dict(result: ChannelResult) -> dict:
    """转为字典"""
    return asdict(result)


def to_json(result: ChannelResult, indent: int = 2) -> str:
    """转为 JSON 字符串"""
    return json.dumps(to_dict(result), ensure_ascii=False, indent=indent)


def print_result(result: ChannelResult):
    """打印到控制台"""
    print(f"[{result.channel_id}] "
          f"状态: {result.state} | "
          f"幅值: {result.amplitude:.2f} | "
          f"置信度: {result.confidence:.1%}")
