"""演示一：四种定价方法的对照

对同一只欧式 Call 期权（S=100, K=100, T=1, r=5%, σ=20%），
分别用 Black-Scholes、CRR 二叉树、Boyle 三叉树、蒙特卡洛（三种变体）定价，
打印对比结果。

也展示了美式 Put 的提前行权溢价（Early Exercise Premium）。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.option import Option, OptionType, ExerciseStyle
from src.black_scholes import bs_price
from src.binomial_tree import binomial_price
from src.trinomial_tree import trinomial_price
from src.monte_carlo import mc_price, mc_price_antithetic, mc_price_control_variate


def time_it(func, *args, **kwargs):
    """返回 (结果, 耗时秒数)。"""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - t0


def main() -> None:
    print("=" * 70)
    print("演示一：欧式 Call 期权多方法定价对照")
    print("=" * 70)

    euro_call = Option(
        S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0,
        option_type=OptionType.CALL, style=ExerciseStyle.EUROPEAN,
    )
    print(f"\n参数：S={euro_call.S}, K={euro_call.K}, T={euro_call.T}, "
          f"r={euro_call.r}, σ={euro_call.sigma}\n")

    # Black-Scholes 基准
    bs_val, bs_t = time_it(bs_price, euro_call)
    print(f"{'Black-Scholes 解析解':<24} = {bs_val:>10.6f}   耗时 {bs_t*1000:.3f} ms  ← 基准")

    # 树方法
    for n in [50, 200, 500]:
        bn_val, bn_t = time_it(binomial_price, euro_call, n)
        tri_val, tri_t = time_it(trinomial_price, euro_call, n)
        print(f"  CRR 二叉树 (N={n:>3})         = {bn_val:>10.6f}   耗时 {bn_t*1000:>6.2f} ms"
              f"   误差 = {abs(bn_val-bs_val):.2e}")
        print(f"  Boyle 三叉树 (N={n:>3})       = {tri_val:>10.6f}   耗时 {tri_t*1000:>6.2f} ms"
              f"   误差 = {abs(tri_val-bs_val):.2e}")

    # 蒙特卡洛
    print()
    for func, name in [(mc_price, "普通 MC"),
                       (mc_price_antithetic, "对偶变量 MC"),
                       (mc_price_control_variate, "控制变量 MC")]:
        (price, stderr), t = time_it(func, euro_call, 100_000, 42)
        print(f"  {name:<22} = {price:>10.6f}   标准误 = {stderr:.4f}   耗时 {t*1000:>6.2f} ms")

    # ---------- 美式 vs 欧式 Put ----------
    print()
    print("=" * 70)
    print("演示二：美式 Put 的提前行权溢价")
    print("=" * 70)

    euro_put = Option(
        S=100, K=110, T=1.0, r=0.05, sigma=0.30, q=0.0,
        option_type=OptionType.PUT, style=ExerciseStyle.EUROPEAN,
    )
    amer_put = Option(
        S=100, K=110, T=1.0, r=0.05, sigma=0.30, q=0.0,
        option_type=OptionType.PUT, style=ExerciseStyle.AMERICAN,
    )

    print(f"\n参数：S=100, K=110（in-the-money put）, T=1, r=5%, σ=30%\n")
    eu_val = binomial_price(euro_put, steps=500)
    am_val = binomial_price(amer_put, steps=500)
    print(f"  欧式 Put          = {eu_val:.6f}")
    print(f"  美式 Put          = {am_val:.6f}")
    print(f"  提前行权溢价      = {am_val - eu_val:.6f}（{(am_val-eu_val)/eu_val*100:.2f}%）")

    # ---------- Merton 定理验证 ----------
    print()
    print("=" * 70)
    print("演示三：Merton 定理——无股息时美式 Call = 欧式 Call")
    print("=" * 70)

    euro_call_nodiv = Option(S=100, K=100, T=1.0, r=0.05, sigma=0.30, q=0.0,
                             option_type=OptionType.CALL, style=ExerciseStyle.EUROPEAN)
    amer_call_nodiv = Option(S=100, K=100, T=1.0, r=0.05, sigma=0.30, q=0.0,
                             option_type=OptionType.CALL, style=ExerciseStyle.AMERICAN)
    eu = binomial_price(euro_call_nodiv, 500)
    am = binomial_price(amer_call_nodiv, 500)
    print(f"\n  欧式 Call         = {eu:.6f}")
    print(f"  美式 Call         = {am:.6f}")
    print(f"  差额              = {abs(am - eu):.2e}   （Merton：理论上为 0）")


if __name__ == "__main__":
    main()
