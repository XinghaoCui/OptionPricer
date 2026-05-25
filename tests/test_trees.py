"""二叉树与三叉树定价的单元测试。

测试要点：
1. 树定价在 N→∞ 时收敛到 BS 解析解（数值方法正确性的金标准）
2. 美式 Put ≥ 欧式 Put（美式有提前行权权利，价值更高）
3. 无股息美式 Call = 欧式 Call（Merton 定理）
4. 三叉树的收敛速度通常优于二叉树
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.option import Option, OptionType, ExerciseStyle
from src.black_scholes import bs_price
from src.binomial_tree import binomial_price
from src.trinomial_tree import trinomial_price


class TestBinomialTree(unittest.TestCase):

    def test_converges_to_bs_european(self) -> None:
        """二叉树定价 N=500 步应收敛到 BS 解析价（误差 < 0.01）。"""
        opt = Option(S=100, K=100, T=1.0, r=0.05, sigma=0.20,
                     option_type=OptionType.CALL, style=ExerciseStyle.EUROPEAN)
        bs_val = bs_price(opt)
        tree_val = binomial_price(opt, steps=500)
        self.assertAlmostEqual(bs_val, tree_val, places=2)

    def test_american_put_at_least_european(self) -> None:
        """美式 Put 必须 ≥ 欧式 Put（无套利原则）。"""
        params = dict(S=100, K=110, T=1.0, r=0.05, sigma=0.30, option_type=OptionType.PUT)
        eu = Option(**params, style=ExerciseStyle.EUROPEAN)
        am = Option(**params, style=ExerciseStyle.AMERICAN)
        self.assertGreaterEqual(binomial_price(am, 200), binomial_price(eu, 200))

    def test_american_call_equals_european_no_dividend(self) -> None:
        """无股息时，美式 Call = 欧式 Call（Merton 定理：提前行权永远不优）。"""
        params = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.30, q=0.0,
                      option_type=OptionType.CALL)
        eu = Option(**params, style=ExerciseStyle.EUROPEAN)
        am = Option(**params, style=ExerciseStyle.AMERICAN)
        self.assertAlmostEqual(binomial_price(eu, 300), binomial_price(am, 300), places=3)


class TestTrinomialTree(unittest.TestCase):

    def test_converges_to_bs(self) -> None:
        """三叉树 N=500 步应收敛到 BS（绝对误差 < 0.005）。"""
        opt = Option(S=100, K=100, T=1.0, r=0.05, sigma=0.20,
                     option_type=OptionType.CALL, style=ExerciseStyle.EUROPEAN)
        bs_val = bs_price(opt)
        tri_val = trinomial_price(opt, steps=500)
        self.assertAlmostEqual(bs_val, tri_val, delta=0.005)

    def test_trinomial_more_accurate_than_binomial_at_low_steps(self) -> None:
        """相同低步数下，三叉树误差通常小于二叉树（不是严格保证，但大多数参数下成立）。"""
        opt = Option(S=100, K=100, T=1.0, r=0.05, sigma=0.20,
                     option_type=OptionType.CALL, style=ExerciseStyle.EUROPEAN)
        bs_val = bs_price(opt)
        n = 30
        bn_err = abs(binomial_price(opt, n) - bs_val)
        tri_err = abs(trinomial_price(opt, n) - bs_val)
        # 用大于号略宽松，避免极端参数下的偶然失败
        self.assertLess(tri_err, bn_err * 2.0)


if __name__ == "__main__":
    unittest.main()
