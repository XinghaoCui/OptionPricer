"""蒙特卡洛期权定价（含方差缩减技术）。

蒙特卡洛方法的优点是对维度不敏感（适合高维问题、路径依赖期权），缺点是收敛速度
慢——标准误差按 O(1/√n) 衰减，要把误差减半需要 4 倍样本。为此发展了多种"方差缩减
技术"（Variance Reduction Techniques），本模块实现两种最经典的：

1. **对偶变量法（Antithetic Variates）**
   每次抽样 Z 后同时使用 -Z，构造一对负相关样本。当 payoff 关于 Z 单调时，
   Cov(f(Z), f(-Z)) < 0，方差严格减小，且仅需一半随机数生成调用。

2. **控制变量法（Control Variates）**
   引入一个已知期望的相关变量 Y，构造修正估计 X' = X - β(Y - E[Y])。
   当 Y 与 X 高度相关时方差大幅缩减。本实现用"标的资产终值 S_T"作为控制变量，
   其折现期望为 S·e^(-qT)，是欧式期权的天然控制变量。

返回值统一为 (价格, 标准误差) 元组，便于做收敛性分析与置信区间。
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .option import Option, ExerciseStyle


# ---------- 内部工具 ----------

def _terminal_prices(option: Option, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """在风险中性测度下模拟标的资产到期价格 S_T（几何布朗运动）。

    S_T = S · exp[(r - q - σ²/2)·T + σ·√T · Z],   Z ~ N(0,1)
    """
    S, T, r, sigma, q = option.S, option.T, option.r, option.sigma, option.q
    Z = rng.standard_normal(n_paths)
    drift = (r - q - 0.5 * sigma * sigma) * T
    diffusion = sigma * math.sqrt(T) * Z
    return S * np.exp(drift + diffusion)


def _payoffs_to_estimate(payoffs: np.ndarray, discount: float) -> Tuple[float, float]:
    """对 payoff 数组贴现求均值，并返回标准误差。"""
    discounted = discount * payoffs
    price = float(np.mean(discounted))
    # 标准误差 = 样本标准差 / √n（ddof=1 用无偏估计）
    stderr = float(np.std(discounted, ddof=1) / math.sqrt(len(discounted)))
    return price, stderr


# ---------- 三种 MC 估计 ----------

def mc_price(option: Option, n_paths: int = 100_000, seed: int | None = None) -> Tuple[float, float]:
    """普通蒙特卡洛欧式期权定价。

    仅适用于欧式期权（美式 MC 需 Longstaff-Schwartz 等回归方法，暂未实现）。
    """
    if option.style != ExerciseStyle.EUROPEAN:
        raise ValueError("本 MC 实现仅支持欧式期权")

    rng = np.random.default_rng(seed)
    S_T = _terminal_prices(option, n_paths, rng)

    if option.is_call():
        payoffs = np.maximum(S_T - option.K, 0.0)
    else:
        payoffs = np.maximum(option.K - S_T, 0.0)

    discount = math.exp(-option.r * option.T)
    return _payoffs_to_estimate(payoffs, discount)


def mc_price_antithetic(option: Option, n_paths: int = 100_000, seed: int | None = None) -> Tuple[float, float]:
    """对偶变量蒙特卡洛。

    将样本量 n_paths 一半用于 +Z、一半用于 -Z，得到一对负相关 payoff，
    其平均的方差通常远小于独立抽样的方差。
    """
    if option.style != ExerciseStyle.EUROPEAN:
        raise ValueError("本 MC 实现仅支持欧式期权")
    if n_paths % 2 != 0:
        n_paths += 1  # 凑成偶数，方便配对

    rng = np.random.default_rng(seed)
    S, T, r, sigma, q, K = option.S, option.T, option.r, option.sigma, option.q, option.K

    half = n_paths // 2
    Z = rng.standard_normal(half)
    drift = (r - q - 0.5 * sigma * sigma) * T
    sqrtT = math.sqrt(T) * sigma

    S_T_plus = S * np.exp(drift + sqrtT * Z)
    S_T_minus = S * np.exp(drift - sqrtT * Z)

    if option.is_call():
        pay_plus = np.maximum(S_T_plus - K, 0.0)
        pay_minus = np.maximum(S_T_minus - K, 0.0)
    else:
        pay_plus = np.maximum(K - S_T_plus, 0.0)
        pay_minus = np.maximum(K - S_T_minus, 0.0)

    # 配对平均后再做样本统计——这一步是 antithetic 方差缩减生效的关键
    paired_payoffs = 0.5 * (pay_plus + pay_minus)
    discount = math.exp(-r * T)
    return _payoffs_to_estimate(paired_payoffs, discount)


def mc_price_control_variate(option: Option, n_paths: int = 100_000, seed: int | None = None) -> Tuple[float, float]:
    """控制变量蒙特卡洛，使用 S_T 作为控制变量。

    设 X = e^(-rT) · payoff(S_T) 为待估值；
    取 Y = e^(-rT) · S_T，其真实期望为 E[Y] = S · e^(-qT)；
    构造修正估计 X' = X - β·(Y - E[Y])，β = Cov(X,Y)/Var(Y) 由样本估计。
    """
    if option.style != ExerciseStyle.EUROPEAN:
        raise ValueError("本 MC 实现仅支持欧式期权")

    rng = np.random.default_rng(seed)
    S, T, r, q, K = option.S, option.T, option.r, option.q, option.K
    discount = math.exp(-r * T)

    S_T = _terminal_prices(option, n_paths, rng)
    if option.is_call():
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    X = discount * payoffs
    Y = discount * S_T
    EY = S * math.exp(-q * T)  # Y 的解析期望

    # β = Cov(X,Y)/Var(Y) 的样本估计
    cov_matrix = np.cov(X, Y, ddof=1)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]

    X_cv = X - beta * (Y - EY)
    price = float(np.mean(X_cv))
    stderr = float(np.std(X_cv, ddof=1) / math.sqrt(n_paths))
    return price, stderr
