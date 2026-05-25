"""蒙特卡洛定价的单元测试。

测试要点：
1. MC 估计值落在 BS 解析价的 95% 置信区间内（统计正确性）
2. 对偶变量法标准误差 < 普通 MC（方差缩减生效）
3. 控制变量法标准误差远小于普通 MC（对欧式期权效果尤其显著）
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.option import Option, OptionType, ExerciseStyle
from src.black_scholes import bs_price
from src.monte_carlo import mc_price, mc_price_antithetic, mc_price_control_variate


SEED = 20260525  # 固定种子保证测试可重现


class TestMonteCarlo(unittest.TestCase):

    def setUp(self) -> None:
        self.option = Option(
            S=100, K=100, T=1.0, r=0.05, sigma=0.20,
            option_type=OptionType.CALL, style=ExerciseStyle.EUROPEAN,
        )
        self.bs_val = bs_price(self.option)

    def test_mc_in_confidence_interval(self) -> None:
        """普通 MC 估计应落在 BS 真值的 ±3 个标准误内（>99.7% 置信度）。"""
        price, stderr = mc_price(self.option, n_paths=100_000, seed=SEED)
        self.assertLess(abs(price - self.bs_val), 3 * stderr)

    def test_antithetic_reduces_variance(self) -> None:
        """对偶变量法的标准误差应小于普通 MC（相同样本数下）。"""
        _, se_plain = mc_price(self.option, n_paths=100_000, seed=SEED)
        _, se_anti = mc_price_antithetic(self.option, n_paths=100_000, seed=SEED)
        self.assertLess(se_anti, se_plain)

    def test_control_variate_reduces_variance(self) -> None:
        """控制变量法的标准误差应远小于普通 MC（典型缩减 5-10 倍）。"""
        _, se_plain = mc_price(self.option, n_paths=100_000, seed=SEED)
        _, se_cv = mc_price_control_variate(self.option, n_paths=100_000, seed=SEED)
        self.assertLess(se_cv, se_plain * 0.5)  # 至少缩减一半

    def test_put_call_parity_via_mc(self) -> None:
        """通过 MC 估计 Call 和 Put，验证 put-call parity（统计意义上）。"""
        import math
        call = Option(S=100, K=100, T=1.0, r=0.05, sigma=0.20,
                      option_type=OptionType.CALL, style=ExerciseStyle.EUROPEAN)
        put = Option(S=100, K=100, T=1.0, r=0.05, sigma=0.20,
                     option_type=OptionType.PUT, style=ExerciseStyle.EUROPEAN)
        c, se_c = mc_price(call, n_paths=200_000, seed=SEED)
        p, se_p = mc_price(put, n_paths=200_000, seed=SEED + 1)
        lhs = c - p
        rhs = 100 - 100 * math.exp(-0.05 * 1.0)
        # 误差容忍设为 3 倍合并标准误
        tol = 3 * (se_c + se_p)
        self.assertLess(abs(lhs - rhs), tol)


if __name__ == "__main__":
    unittest.main()
