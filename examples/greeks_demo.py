"""演示：Greeks 计算与可视化。

运行后会在终端打印解析 Greeks 与有限差分 Greeks 的对照（验证差分实现正确性），
并在 docs/screenshots/ 下保存 Greeks 关于标的价的曲线图。
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.option import Option, OptionType, ExerciseStyle
from src.greeks import greeks_analytical, greeks_finite_diff
from src.implied_vol import implied_volatility
from src.black_scholes import bs_price
from src.visualizer import plot_greeks_vs_spot


def main() -> None:
    SHOTS = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
    SHOTS.mkdir(parents=True, exist_ok=True)

    opt = Option(
        S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.02,
        option_type=OptionType.CALL, style=ExerciseStyle.EUROPEAN,
    )

    # ---------- Greeks 对照 ----------
    print("=" * 70)
    print("Greeks 计算：解析法 vs 有限差分法（中心差分）")
    print("=" * 70)
    print(f"\n参数：S={opt.S}, K={opt.K}, T={opt.T}, r={opt.r}, σ={opt.sigma}, q={opt.q}\n")

    g_ana = greeks_analytical(opt)
    g_fd = greeks_finite_diff(opt)

    print(f"{'Greek':<8}{'解析法':>14}{'有限差分':>14}{'相对误差':>14}")
    for key in ["delta", "gamma", "vega", "theta", "rho"]:
        rel_err = abs(g_ana[key] - g_fd[key]) / (abs(g_ana[key]) + 1e-12)
        print(f"{key:<8}{g_ana[key]:>14.6f}{g_fd[key]:>14.6f}{rel_err:>14.2e}")

    # ---------- 隐含波动率求解演示 ----------
    print()
    print("=" * 70)
    print("隐含波动率求解：给定市场价反解 σ")
    print("=" * 70)
    print("\n用真值 σ=0.20 算出 BS 价，再用 IV 求解器反解，看能否恢复：\n")
    market = bs_price(opt)
    iv = implied_volatility(opt, market_price=market, initial_guess=0.5)
    print(f"  市场价格 = {market:.6f}")
    print(f"  IV 反解  = {iv:.8f}   （真值 0.20000000）")
    print(f"  误差     = {abs(iv - 0.20):.2e}")

    # ---------- Greeks 曲线 ----------
    print()
    print("绘制 Greeks 关于标的价的曲线 ...")
    plot_greeks_vs_spot(opt, savepath=str(SHOTS / "greeks_vs_spot.png"))
    print(f"已保存到 {SHOTS / 'greeks_vs_spot.png'}")


if __name__ == "__main__":
    main()
