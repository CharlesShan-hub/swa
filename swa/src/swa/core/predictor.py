"""
电压预测器 — 校准、训练、预测、模型持久化
"""

import json
from typing import Optional
import numpy as np
from sklearn.linear_model import LinearRegression

from swa.core.scoring import compute_score, compute_alpha7, s20_smooth
from swa.core.calibration import CalibrationParams


class VoltagePredictor:
    """
    电压预测器。

    支持两种模式:
      - "score": compute_score 加权评分
      - "alpha7": alpha_7 余弦分量（抗干扰）

    Args:
        window_size: S20 滑动窗口大小（默认 20，1 为不滑动）
        f1: 第一个周期数（默认 7.0）
        f2: 第二个周期数（默认 8.1）
        w: beta 权重（默认 0.25）
        mode: "score" 或 "alpha7"
    """

    def __init__(
        self,
        window_size: int = 20,
        f1: float = 7.0,
        f2: float = 8.1,
        w: float = 0.25,
        mode: str = "score",
    ):
        self.window_size = window_size
        self.f1 = f1
        self.f2 = f2
        self.w = w
        self.mode = mode
        self.calib = CalibrationParams()
        self.score_buffer: list[float] = []

    # ── 特征计算 ────────────────────────────────────────────────

    def compute_feature(self, wave: np.ndarray) -> Optional[float]:
        """根据 mode 计算单个波形的特征值。"""
        if self.mode == "alpha7":
            return compute_alpha7(wave)
        return compute_score(wave, f1=self.f1, f2=self.f2, w=self.w)

    # ── 校准 ────────────────────────────────────────────────────

    def fit_calibration(
        self,
        score_list: list[float],
        voltage_list: list[float],
    ) -> CalibrationParams:
        """
        线性校准 V = a * S + b。
        """
        X = np.array(score_list).reshape(-1, 1)
        y = np.array(voltage_list)
        reg = LinearRegression().fit(X, y)
        self.calib = CalibrationParams(
            a=float(reg.coef_[0]),
            b=float(reg.intercept_),
            is_fitted=True,
        )
        return self.calib

    def fit_calibration_with_humidity(
        self,
        score_list: list[float],
        voltage_list: list[float],
        humidity_list: list[float],
        h0: float = 50.0,
    ) -> CalibrationParams:
        """
        湿度补偿校准 V = a * S + c * (H - h0) + b。
        """
        X = np.column_stack([
            np.array(score_list),
            np.array(humidity_list) - h0,
        ])
        y = np.array(voltage_list)
        reg = LinearRegression().fit(X, y)
        self.calib = CalibrationParams(
            a=float(reg.coef_[0]),
            b=float(reg.intercept_),
            c=float(reg.coef_[1]),
            h0=h0,
            use_humidity=True,
            is_fitted=True,
        )
        return self.calib

    # ── 预测 ────────────────────────────────────────────────────

    def predict_single(
        self,
        wave: np.ndarray,
        humidity: Optional[float] = None,
    ) -> Optional[float]:
        """
        预测单条波形的电压。
        """
        feat = self.compute_feature(wave)
        if feat is None or not self.calib.is_fitted:
            return None

        self.score_buffer.append(feat)
        if len(self.score_buffer) > self.window_size * 2:
            self.score_buffer = self.score_buffer[-self.window_size * 2 :]

        h = humidity if humidity is not None else 50.0
        return self.calib.predict(feat, h)

    def predict_batch(
        self,
        waves: list[np.ndarray],
        humidities: Optional[list[float]] = None,
    ) -> list[Optional[float]]:
        """
        批量预测。
        """
        if humidities is None:
            humidities = [50.0] * len(waves)
        return [
            self.predict_single(w, h)
            for w, h in zip(waves, humidities)
        ]

    def predict_smoothed(
        self,
        waves: list[np.ndarray],
        humidities: Optional[list[float]] = None,
    ) -> list[Optional[float]]:
        """
        预测 + S20 滑动平均后返回。
        """
        raw = self.predict_batch(waves, humidities)
        valid = [v for v in raw if v is not None]
        if len(valid) < self.window_size:
            return [None] * len(raw)

        smoothed = s20_smooth(valid, self.window_size)
        # 对齐到原始长度，前面补 None
        none_count = len(raw) - len(valid)
        return [None] * none_count + smoothed

    # ── 模型持久化 ──────────────────────────────────────────────

    def save(self, path: str):
        data = {
            "mode": self.mode,
            "window_size": self.window_size,
            "f1": self.f1,
            "f2": self.f2,
            "w": self.w,
            "calib": self.calib.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.mode = data.get("mode", "score")
        self.window_size = data.get("window_size", 20)
        self.f1 = data.get("f1", 7.0)
        self.f2 = data.get("f2", 8.1)
        self.w = data.get("w", 0.25)
        self.calib = CalibrationParams.from_dict(data.get("calib", {}))
        self.score_buffer = []
