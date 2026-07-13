"""
校准参数与湿度补偿模型
"""

from dataclasses import dataclass


@dataclass
class CalibrationParams:
    """线性校准参数 V = a * S + b （或带湿度补偿 V = a * S + c * H + d）"""

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0       # 湿度系数
    h0: float = 50.0     # 参考湿度（%）
    use_humidity: bool = False
    is_fitted: bool = False

    def predict(self, score: float, humidity: float = 50.0) -> float:
        if self.use_humidity:
            return self.a * score + self.c * (humidity - self.h0) + self.b
        return self.a * score + self.b

    def to_dict(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "h0": self.h0,
            "use_humidity": self.use_humidity,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationParams":
        return cls(
            a=d.get("a", 1.0),
            b=d.get("b", 0.0),
            c=d.get("c", 0.0),
            h0=d.get("h0", 50.0),
            use_humidity=d.get("use_humidity", False),
            is_fitted=d.get("is_fitted", False),
        )
