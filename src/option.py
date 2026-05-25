"""期权对象定义。

把期权的基本参数封装成一个不可变数据类，避免在各定价函数间到处传一堆散参数。
所有定价器统一接受 Option 对象作为输入，便于扩展和测试。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OptionType(str, Enum):
    """期权类型：看涨（Call）或看跌（Put）。"""
    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, Enum):
    """行权方式：欧式（到期日行权）或美式（任意时刻可行权）。"""
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(frozen=True)
class Option:
    """欧式或美式香草期权。

    参数
    ----
    S : float
        标的资产现价（spot price）
    K : float
        行权价（strike price）
    T : float
        距到期时间（年化）
    r : float
        无风险利率（连续复利，年化）
    sigma : float
        标的资产波动率（年化）
    q : float
        股息率（连续复利，年化），默认 0
    option_type : OptionType
        看涨 / 看跌
    style : ExerciseStyle
        欧式 / 美式
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0
    option_type: OptionType = OptionType.CALL
    style: ExerciseStyle = ExerciseStyle.EUROPEAN

    def __post_init__(self) -> None:
        # 在系统边界做参数校验，内部模块就可以放心使用
        if self.S <= 0:
            raise ValueError(f"标的价格 S 必须为正，得到 {self.S}")
        if self.K <= 0:
            raise ValueError(f"行权价 K 必须为正，得到 {self.K}")
        if self.T < 0:
            raise ValueError(f"到期时间 T 不能为负，得到 {self.T}")
        if self.sigma < 0:
            raise ValueError(f"波动率 sigma 不能为负，得到 {self.sigma}")

    def payoff(self, S_T: float) -> float:
        """给定到期标的价格 S_T，返回期权到期收益。"""
        if self.option_type == OptionType.CALL:
            return max(S_T - self.K, 0.0)
        return max(self.K - S_T, 0.0)

    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL

    def is_american(self) -> bool:
        return self.style == ExerciseStyle.AMERICAN
