"""OptionPricer —— 期权定价工具包

模块组成：
    option         : 期权对象定义
    black_scholes  : Black-Scholes 解析解
    binomial_tree  : CRR 二叉树定价（欧式 + 美式）
    trinomial_tree : Boyle 三叉树定价
    monte_carlo    : 蒙特卡洛模拟（含方差缩减）
    greeks         : 希腊字母（Greeks）计算
    implied_vol    : 隐含波动率求解
    visualizer     : 可视化模块
"""

from .option import Option, OptionType, ExerciseStyle
from .black_scholes import bs_price, bs_greeks
from .binomial_tree import binomial_price
from .trinomial_tree import trinomial_price
from .monte_carlo import mc_price, mc_price_antithetic, mc_price_control_variate
from .greeks import greeks_analytical, greeks_finite_diff
from .implied_vol import implied_volatility

__all__ = [
    "Option", "OptionType", "ExerciseStyle",
    "bs_price", "bs_greeks",
    "binomial_price",
    "trinomial_price",
    "mc_price", "mc_price_antithetic", "mc_price_control_variate",
    "greeks_analytical", "greeks_finite_diff",
    "implied_volatility",
]
