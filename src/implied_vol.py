"""隐含波动率（Implied Volatility, IV）求解。

给定市场观察到的期权价格 V_market，反解满足 BS(σ) = V_market 的 σ。
这是一个一维方程求根问题，但 BS 价格关于 σ 不是初等可逆函数，
必须用数值方法。本模块实现两种经典算法：

1. **牛顿法（Newton-Raphson）**
   利用 Vega = ∂V/∂σ 作为导数，迭代式：σ_{n+1} = σ_n - (V(σ_n) - V*) / Vega(σ_n)
   收敛阶 ≈ 2（平方收敛），通常 5-10 步即可达到 1e-8 精度。

2. **二分法（Bisection）**
   作为兜底：当牛顿法迭代失败（如 Vega 接近 0、初值太差导致越界等），自动切换到
   二分法保证收敛。收敛阶 = 1（线性收敛）但绝对稳定。

这是"数值算法"与"鲁棒性设计"结合的典型模式：先用高阶方法快速逼近，
失败后回退到鲁棒方法兜底，是工程实践中的常用模式。
"""

from __future__ import annotations

from dataclasses import replace

from .option import Option
from .black_scholes import bs_price, bs_greeks


def implied_volatility(
    option: Option,
    market_price: float,
    initial_guess: float = 0.2,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """求解隐含波动率。

    先尝试牛顿法，若不收敛则回退到二分法。

    参数
    ----
    option : Option
        市场期权对象。其 sigma 字段会被忽略（待求量）。
    market_price : float
        市场观察到的期权价格
    initial_guess : float
        牛顿法的初值
    tol : float
        收敛容差（价格误差绝对值）
    max_iter : int
        最大迭代次数
    """
    # 套利边界检查：欧式期权价格必须在内在价值与最大可能价值之间
    intrinsic = option.payoff(option.S)  # 立即行权价值（粗略下界）
    if market_price < intrinsic - tol:
        raise ValueError(
            f"市场价格 {market_price} 低于内在价值 {intrinsic}，存在套利或输入错误"
        )

    # ---------- 步骤 1：牛顿法 ----------
    sigma = max(initial_guess, 1e-4)
    for _ in range(max_iter):
        trial = replace(option, sigma=sigma)
        price = bs_price(trial)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        vega = bs_greeks(trial)["vega"]
        if vega < 1e-10:
            # Vega 太小，牛顿法可能发散，切换到二分法
            break

        sigma_new = sigma - diff / vega

        # 若迭代值越界（σ < 0 或过大），也切换到二分法
        if sigma_new <= 1e-8 or sigma_new > 10.0:
            break

        sigma = sigma_new
    else:
        # 用尽迭代次数仍未收敛，转二分法
        pass

    # ---------- 步骤 2：二分法兜底 ----------
    return _bisection_iv(option, market_price, tol=tol, max_iter=200)


def _bisection_iv(
    option: Option,
    market_price: float,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """二分法求解隐含波动率，绝对稳定但收敛较慢。"""
    f_lo = bs_price(replace(option, sigma=lo)) - market_price
    f_hi = bs_price(replace(option, sigma=hi)) - market_price

    if f_lo * f_hi > 0:
        # 区间内无根，可能 market_price 超出可达范围
        raise ValueError(
            f"二分区间 [{lo}, {hi}] 内无解：f(lo)={f_lo:.4e}, f(hi)={f_hi:.4e}"
        )

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = bs_price(replace(option, sigma=mid)) - market_price

        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid

        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return 0.5 * (lo + hi)
