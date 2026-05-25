"""Black-Scholes 公式的单元测试。

测试要点：
1. 经典教科书参考值复现（Hull 教科书 Example 13.6）
2. Put-Call Parity 恒等式验证：C - P = S·e^(-qT) - K·e^(-rT)
3. 零波动率退化为内在价值
4. 到期日退化为内在价值
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.option import Option, OptionType, ExerciseStyle
from src.black_scholes import bs_price


class TestBlackScholes(unittest.TestCase):

    def test_hull_textbook_example(self) -> None:
        """复现 Hull《期权、期货及其他衍生品》第 13.6 节例题：
        S=42, K=40, r=0.10, σ=0.20, T=0.5 → Call ≈ 4.7594, Put ≈ 0.8086
        """
        opt_call = Option(S=42, K=40, T=0.5, r=0.10, sigma=0.20,
                          option_type=OptionType.CALL, style=ExerciseStyle.EUROPEAN)
        opt_put = Option(S=42, K=40, T=0.5, r=0.10, sigma=0.20,
                         option_type=OptionType.PUT, style=ExerciseStyle.EUROPEAN)
        self.assertAlmostEqual(bs_price(opt_call), 4.7594, places=3)
        self.assertAlmostEqual(bs_price(opt_put), 0.8086, places=3)

    def test_put_call_parity(self) -> None:
        """Put-Call Parity: C - P = S·e^(-qT) - K·e^(-rT)
        这是无套利定价的基石，任何欧式期权定价器都必须满足。
        """
        S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.02
        call = Option(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=OptionType.CALL)
        put = Option(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=OptionType.PUT)

        lhs = bs_price(call) - bs_price(put)
        rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
        self.assertAlmostEqual(lhs, rhs, places=8)

    def test_zero_volatility_returns_intrinsic(self) -> None:
        """σ=0 时退化为远期合约的折现内在价值。"""
        opt = Option(S=110, K=100, T=1.0, r=0.0, sigma=0.0, option_type=OptionType.CALL)
        self.assertAlmostEqual(bs_price(opt), 10.0, places=8)

    def test_at_expiry_returns_payoff(self) -> None:
        """T=0 时直接返回 payoff。"""
        opt = Option(S=105, K=100, T=0.0, r=0.05, sigma=0.20, option_type=OptionType.CALL)
        self.assertAlmostEqual(bs_price(opt), 5.0, places=8)


if __name__ == "__main__":
    unittest.main()
