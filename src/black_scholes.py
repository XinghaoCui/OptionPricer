"""Black-Scholes 解析解。

作为所有数值方法（二叉树、三叉树、蒙特卡洛）的基准，用于：
1. 验证数值方法的收敛性与正确性；
2. 控制变量法（control variate）中的解析对照；
3. 作为隐含波动率求解的目标函数。

仅适用于欧式期权；美式期权无解析解。
"""

from __future__ import annotations

import math
from typing import Dict

from .option import Option, OptionType, ExerciseStyle


# ---------- 标准正态分布 CDF / PDF（不依赖 scipy，自己实现一份） ----------

def _norm_cdf(x: float) -> float:
    """标准正态分布累积分布函数 N(x)。

    使用 math.erf 实现，避免对 scipy.stats 的依赖，方便部署。
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """标准正态分布概率密度函数 φ(x)。"""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---------- 主接口：定价 + Greeks ----------

def bs_price(option: Option) -> float:
    """Black-Scholes 欧式期权定价公式。

    Call : C = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    Put  : P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
    其中  d1 = [ln(S/K) + (r - q + σ²/2) T] / (σ√T),  d2 = d1 - σ√T
    """
    if option.style != ExerciseStyle.EUROPEAN:
        raise ValueError("Black-Scholes 公式只适用于欧式期权")

    S, K, T, r, sigma, q = option.S, option.K, option.T, option.r, option.sigma, option.q

    # 处理到期时刻或零波动率的边界情形，直接返回内在价值
    if T == 0.0 or sigma == 0.0:
        return option.payoff(S)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option.option_type == OptionType.CALL:
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def bs_greeks(option: Option) -> Dict[str, float]:
    """Black-Scholes 解析 Greeks。

    返回字典：{"delta", "gamma", "vega", "theta", "rho"}
    其中 vega、rho 已按"每 1 个单位"返回（即不除以 100，调用方需要时自己换算）。
    """
    if option.style != ExerciseStyle.EUROPEAN:
        raise ValueError("Black-Scholes Greeks 解析式只适用于欧式期权")

    S, K, T, r, sigma, q = option.S, option.K, option.T, option.r, option.sigma, option.q

    if T == 0.0:
        # 到期日 Greeks 退化，返回零或不定形式的极限处理
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    pdf_d1 = _norm_pdf(d1)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    # Gamma、Vega 与 Call/Put 无关
    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_T)
    vega = S * disc_q * pdf_d1 * sqrt_T

    if option.option_type == OptionType.CALL:
        delta = disc_q * _norm_cdf(d1)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2.0 * sqrt_T)
            - r * K * disc_r * _norm_cdf(d2)
            + q * S * disc_q * _norm_cdf(d1)
        )
        rho = K * T * disc_r * _norm_cdf(d2)
    else:
        delta = -disc_q * _norm_cdf(-d1)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2.0 * sqrt_T)
            + r * K * disc_r * _norm_cdf(-d2)
            - q * S * disc_q * _norm_cdf(-d1)
        )
        rho = -K * T * disc_r * _norm_cdf(-d2)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
