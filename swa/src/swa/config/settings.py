"""
配置管理模块

集中管理所有可配置参数：
- 采样率、滤波参数、阈值
- 数据库连接参数（从 local_config.py 读取）

配置方式（优先级从高到低）：
    1. src/swa/config/local_config.py — 本地配置，已 gitignore
    2. 环境变量 DM8_HOST / DM8_PORT / DM8_USER / DM8_PASSWORD
"""

import os
from dataclasses import dataclass, field


# ============================================================
# 本地配置加载
# ============================================================

def _load_db_config() -> dict:
    """尝试从 local_config.py 读取数据库配置，没有则用环境变量"""
    try:
        from . import local_config
        return {
            "host": getattr(local_config, "DM8_HOST", None) or os.environ.get("DM8_HOST", "localhost"),
            "port": int(getattr(local_config, "DM8_PORT", None) or os.environ.get("DM8_PORT", "5236")),
            "user": getattr(local_config, "DM8_USER", None) or os.environ.get("DM8_USER", ""),
            "password": getattr(local_config, "DM8_PASSWORD", None) or os.environ.get("DM8_PASSWORD", ""),
        }
    except (ImportError, ModuleNotFoundError):
        return {
            "host": os.environ.get("DM8_HOST", "localhost"),
            "port": int(os.environ.get("DM8_PORT", "5236")),
            "user": os.environ.get("DM8_USER", ""),
            "password": os.environ.get("DM8_PASSWORD", ""),
        }


_db_cfg = _load_db_config()


# ============================================================
# 全局常量
# ============================================================

SAMPLE_RATE: int = 15873         # 采样率 (Hz)，来自场磨传感器硬件
WAVE_POINTS: int = 512            # 每帧波形采样点数
ROTOR_FREQ_DEFAULT: float = 213.8  # 默认转子频率 (Hz)


# ============================================================
# 数据库配置
# ============================================================

@dataclass
class DatabaseConfig:
    """达梦DM8数据库连接配置"""
    host: str = _db_cfg["host"]
    port: int = _db_cfg["port"]
    user: str = _db_cfg["user"]
    password: str = _db_cfg["password"]


# ============================================================
# 数据源配置
# ============================================================

@dataclass
class DataSourceConfig:
    """数据源配置

    mode 可选值：
        "db"    — 从达梦数据库读取（需配置 local_config.py）
        "local" — 从本地 JSONL 文件读取（回家离线用）

    local_path：mode="local" 时，JSONL 文件的路径
    """
    mode: str = "local"                     # "db" 或 "local"
    local_path: str = "data/exported_data.jsonl"


# ============================================================
# 信号处理配置
# ============================================================

@dataclass
class SignalProcessConfig:
    """信号处理参数"""
    sample_rate: int = SAMPLE_RATE
    # 带通滤波
    bandpass_enabled: bool = True
    bandpass_center: float = ROTOR_FREQ_DEFAULT
    bandpass_bw: float = 50.0
    # 幅值提取方法: rms | fft | peak_to_peak
    amplitude_method: str = "rms"


# ============================================================
# 状态判别配置
# ============================================================

@dataclass
class DetectionConfig:
    """状态判别与异常检测参数"""
    threshold_adc: float = 50.0   # ADC原始值阈值（用于 报文.json）
    threshold_volt: float = 0.8   # 电压阈值（用于 raw.csv）
    hysteresis: float = 0.1       # 迟滞带（防抖动）
    confirm_count: int = 3         # 确认次数
    false_input_enabled: bool = True
    false_input_confirm: int = 5


# ============================================================
# 电压估计算法配置
# ============================================================

@dataclass
class EstimationConfig:
    """电压估算模型配置"""
    # 算法名称，取消注释即启用
    # algorithm: str = "linear_basic"
    # algorithm: str = "linear_with_env"
    # algorithm: str = "linear_full"
    # algorithm: str = "quadratic_model"
    algorithm: str = "lightgbm_model"
    # algorithm: str = "xgboost_model"
    # algorithm: str = "lenet"
    # algorithm: str = "lenet_hybrid"

    # 训练/测试划分（三者之和应 ≤ 总数据量）
    train_size: int = 30400   # 用于训练的条数（8份）
    val_size: int = 3800      # 用于验证的条数（1份）
    test_size: int = 3800     # 用于测试的条数（1份）

    # 神经网络训练参数
    max_epochs: int = -1      # 最大训练轮数（-1 表示无限，靠早停停止）
    lrn: int = 3               # 学习率衰减次数
    lr_decay: float = 0.5      # 每次衰减的倍数（衰减到 0.5 倍）
    early_stop_patience: int = 29  # 早停耐心值（配合 warmup=30，触发点在整十轮）
    loss_fn: str = "mse"       # 损失函数: "mse" / "huber" / "l1"
    huber_delta: float = 10.0  # Huber Loss 的 delta 参数

    # 投/退 判断阈值（预测电压的绝对值超过此值判定为"投"）
    threshold_abs: float = 30.0

    # 模型文件路径
    model_path: str = "data/model_params.json"


# ============================================================
# 统一配置入口
# ============================================================

@dataclass
class Config:
    """全局配置"""
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    signal: SignalProcessConfig = field(default_factory=SignalProcessConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)


# 全局单例
config = Config()
