"""隐含波动率求解的单元测试。

测试要点：往返一致性——先用已知 σ 通过 BS 算出价格，再用 IV 求解器反解，
看能否恢复原 σ（数值方法的"自洽性测试"）。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.option import Option, OptionType
from src.black_scholes import bs_price
from src.implied_vol import implied_volatility


class TestImpliedVolatility(unittest.TestCase):

    def _roundtrip(self, S: float, K: float, T: float, r: float, sigma_true: float,
                   option_type: OptionType = OptionType.CALL) -> None:
        opt = Option(S=S, K=K, T=T, r=r, sigma=sigma_true, option_type=option_type)
        market = bs_price(opt)
        sigma_back = implied_volatility(opt, market_price=market)
        self.assertAlmostEqual(sigma_back, sigma_true, places=6,
                               msg=f"IV roundtrip failed: σ_true={sigma_true}, σ_back={sigma_back}")

    def test_at_the_money_call(self) -> None:
        self._roundtrip(S=100, K=100, T=1.0, r=0.05, sigma_true=0.20)

    def test_deep_in_the_money_call(self) -> None:
        self._roundtrip(S=150, K=100, T=1.0, r=0.05, sigma_true=0.30)

    def test_out_of_the_money_call(self) -> None:
        self._roundtrip(S=80, K=100, T=1.0, r=0.05, sigma_true=0.40)

    def test_put_option(self) -> None:
        self._roundtrip(S=100, K=110, T=0.5, r=0.03, sigma_true=0.25,
                        option_type=OptionType.PUT)

    def test_high_volatility(self) -> None:
        """高波动率（80%）下的稳健性测试。"""
        self._roundtrip(S=100, K=100, T=1.0, r=0.05, sigma_true=0.80)

    def test_short_maturity(self) -> None:
        """短期期权（1 周）下的稳健性测试。"""
        self._roundtrip(S=100, K=100, T=7 / 365, r=0.05, sigma_true=0.25)


if __name__ == "__main__":
    unittest.main()
