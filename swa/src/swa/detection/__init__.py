"""
swa.detection — 检测方法模块

每个检测方法一个独立文件，导出统一的接口:

    name: str          # 方法显示名称
    run(project_dir, train_voltages, test_voltages) -> dict
        # 返回 {"metrics": {...}, "train_pred": [...], "test_pred": [...], ...}
"""

from . import least_squares

METHODS = {
    "least_squares": least_squares,
}

__all__ = ["METHODS"]
