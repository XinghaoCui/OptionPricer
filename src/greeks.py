"""Greeks 计算（解析法 + 有限差分法）。

Greeks 是期权价格对各参数的偏导数，衡量风险敏感度：
    Delta = ∂V/∂S       —— 对标的价格的敏感度（对冲首选）
    Gamma = ∂²V/∂S²     —— Delta 的变化率（曲率风险）
    Vega  = ∂V/∂σ       —— 对波动率的敏感度
    Theta = ∂V/∂T       —— 对时间衰减的敏感度
    Rho   = ∂V/∂r       —— 对利率的敏感度

本模块同时提供两种实现：
1. **解析法**：直接调用 black_scholes.bs_greeks（精确但仅限欧式）
2. **有限差分法（finite difference）**：通用方法，适用于任何定价器（含美式期权），
   通过对参数小幅扰动后重新定价、做差分得到近似偏导数。

差分方案使用"中心差分"（central difference），收敛阶 O(h²)，精度比单侧差分高。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict

from .option import Option, ExerciseStyle
from .black_scholes import bs_greeks


# 默认定价器：欧式用 BS，美式回退到二叉树
def _default_pricer(option: Option) -> float:
    if option.style == ExerciseStyle.EUROPEAN:
        from .black_scholes import bs_price
        return bs_price(option)
    from .binomial_tree import binomial_price
    return binomial_price(option, steps=200)


def greeks_analytical(option: Option) -> Dict[str, float]:
    """解析 Greeks，仅支持欧式期权。"""
    return bs_greeks(option)


def greeks_finite_diff(
    option: Option,
    pricer: Callable[[Option], float] | None = None,
    h_S: float | None = None,
    h_sigma: float = 1e-4,
    h_T: float = 1.0 / 365,
    h_r: float = 1e-4,
) -> Dict[str, float]:
    """有限差分 Greeks，可用于任意定价器（包括美式期权树定价器）。

    参数
    ----
    option : Option
    pricer : 可调用对象 (Option) -> float
        定价函数，默认欧式用 BS，美式用 200 步二叉树
    h_S, h_sigma, h_T, h_r : float
        各参数的差分步长。h_S 默认 = 0.01·S（相对扰动），其他默认绝对扰动。
    """
    pricer = pricer or _default_pricer
    if h_S is None:
        h_S = 0.01 * option.S

    # 用 dataclasses.replace 创建扰动后的不可变 Option，
    # 这样不会污染原对象，符合"数据流而非状态"的函数式设计风格
    def perturb(**kwargs) -> Option:
        return replace(option, **kwargs)

    V0 = pricer(option)

    # Delta = [V(S+h) - V(S-h)] / (2h)
    V_S_up = pricer(perturb(S=option.S + h_S))
    V_S_down = pricer(perturb(S=option.S - h_S))
    delta = (V_S_up - V_S_down) / (2.0 * h_S)

    # Gamma = [V(S+h) - 2V(S) + V(S-h)] / h²
    gamma = (V_S_up - 2.0 * V0 + V_S_down) / (h_S * h_S)

    # Vega = [V(σ+h) - V(σ-h)] / (2h)
    V_sig_up = pricer(perturb(sigma=option.sigma + h_sigma))
    V_sig_down = pricer(perturb(sigma=max(option.sigma - h_sigma, 1e-8)))
    vega = (V_sig_up - V_sig_down) / (2.0 * h_sigma)

    # Theta = -[V(T-h) - V(T)] / h  （时间是反向流逝的，所以加负号约定）
    V_T_down = pricer(perturb(T=max(option.T - h_T, 1e-8)))
    theta = (V_T_down - V0) / h_T  # 注意符号：T 减小相当于时间前进一步

    # Rho = [V(r+h) - V(r-h)] / (2h)
    V_r_up = pricer(perturb(r=option.r + h_r))
    V_r_down = pricer(perturb(r=option.r - h_r))
    rho = (V_r_up - V_r_down) / (2.0 * h_r)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
