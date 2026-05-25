"""可视化模块。

提供四类图：
    1. plot_binomial_tree       —— 二叉树定价结构（小步数下的展开图）
    2. plot_convergence         —— 树步数 vs BS 解析价的收敛曲线
    3. plot_mc_convergence      —— MC 样本数 vs 标准误的收敛曲线（含方差缩减对比）
    4. plot_greeks_vs_spot      —— Greeks 关于标的价的曲线（风险敏感度可视化）

中文字体：Windows 下使用 SimHei（黑体），macOS 用 PingFang SC，Linux 用 Noto Sans CJK。
"""

from __future__ import annotations

from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .option import Option, ExerciseStyle
from .black_scholes import bs_price
from .binomial_tree import binomial_price, binomial_tree_full
from .monte_carlo import mc_price, mc_price_antithetic, mc_price_control_variate
from .greeks import greeks_analytical


# ---------- 中文字体配置 ----------
# 优先用 SimHei（Windows）、PingFang SC（mac）、Noto Sans CJK（Linux）
_CJK_FONTS = ["SimHei", "Microsoft YaHei", "PingFang SC", "Heiti SC",
              "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Arial Unicode MS"]
matplotlib.rcParams["font.sans-serif"] = _CJK_FONTS + matplotlib.rcParams["font.sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题


# ---------- 1. 二叉树展开图 ----------

def plot_binomial_tree(option: Option, steps: int = 5, savepath: str | None = None) -> None:
    """绘制二叉树的价格树与价值树。仅适用于小步数（≤ 8）下做演示。"""
    if steps > 8:
        raise ValueError("二叉树展开图仅支持 steps<=8，否则节点重叠不可读")

    price_tree, value_tree = binomial_tree_full(option, steps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, tree, title in [
        (axes[0], price_tree, f"标的价格树（CRR, N={steps}）"),
        (axes[1], value_tree, f"期权价值树（{option.style.value}, {option.option_type.value}）"),
    ]:
        for n in range(steps + 1):
            for i in range(n + 1):
                x = n
                y = 2 * i - n  # 纵坐标用偏移量，让上涨在上、下跌在下
                ax.scatter(x, y, s=400, c="lightblue", edgecolors="navy", zorder=3)
                ax.text(x, y, f"{tree[n][i]:.2f}", ha="center", va="center", fontsize=8, zorder=4)
                # 画两条出边（除最后一层外）
                if n < steps:
                    ax.plot([x, x + 1], [y, y + 1], "k-", alpha=0.3, zorder=1)
                    ax.plot([x, x + 1], [y, y - 1], "k-", alpha=0.3, zorder=1)
        ax.set_xlabel("时间步")
        ax.set_ylabel("节点偏移（↑涨 / ↓跌）")
        ax.set_title(title)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=120, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------- 2. 树定价收敛曲线 ----------

def plot_convergence(option: Option, max_steps: int = 200, savepath: str | None = None) -> None:
    """绘制二叉树定价价格随步数变化、收敛到 BS 解析价的曲线。

    展示"奇偶振荡"现象——CRR 二叉树价格关于 N 是上下振荡收敛的，
    这是树类算法的经典特征，体现了对算法收敛性的理解。
    """
    if option.style != ExerciseStyle.EUROPEAN:
        raise ValueError("收敛曲线对比图仅对欧式期权有效（美式无解析解）")

    steps_range = list(range(5, max_steps + 1, 2))  # 间隔 2 展示振荡
    bs_val = bs_price(option)
    bn_vals = [binomial_price(option, n) for n in steps_range]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps_range, bn_vals, "o-", markersize=3, label="CRR 二叉树", color="steelblue")
    ax.axhline(bs_val, color="red", linestyle="--", label=f"Black-Scholes 解析解 = {bs_val:.4f}")
    ax.set_xlabel("时间步数 N")
    ax.set_ylabel("期权价格")
    ax.set_title(f"二叉树定价收敛性（S={option.S}, K={option.K}, T={option.T}, σ={option.sigma}）")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=120, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------- 3. MC 方差缩减对比 ----------

def plot_mc_convergence(option: Option, savepath: str | None = None, seed: int = 42) -> None:
    """对比普通 MC、对偶变量 MC、控制变量 MC 的收敛速度（标准误 vs 样本数）。"""
    if option.style != ExerciseStyle.EUROPEAN:
        raise ValueError("MC 收敛对比仅对欧式期权有效")

    bs_val = bs_price(option)
    sample_sizes = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]

    methods: List[tuple] = [
        ("普通 MC", mc_price, "C0"),
        ("对偶变量 MC", mc_price_antithetic, "C1"),
        ("控制变量 MC", mc_price_control_variate, "C2"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, func, color in methods:
        prices, stderrs = [], []
        for n in sample_sizes:
            p, se = func(option, n_paths=n, seed=seed)
            prices.append(p)
            stderrs.append(se)

        ax1.plot(sample_sizes, prices, "o-", label=name, color=color, markersize=4)
        ax2.plot(sample_sizes, stderrs, "o-", label=name, color=color, markersize=4)

    ax1.axhline(bs_val, color="red", linestyle="--", alpha=0.6, label=f"BS = {bs_val:.4f}")
    ax1.set_xscale("log")
    ax1.set_xlabel("样本数 n")
    ax1.set_ylabel("估计价格")
    ax1.set_title("MC 估计值收敛")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("样本数 n")
    ax2.set_ylabel("标准误差")
    ax2.set_title("MC 标准误差对比（log-log）")
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=120, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------- 4. Greeks 曲线 ----------

def plot_greeks_vs_spot(option: Option, savepath: str | None = None) -> None:
    """绘制 Delta、Gamma、Vega、Theta 关于标的价格 S 的曲线（欧式期权）。"""
    if option.style != ExerciseStyle.EUROPEAN:
        raise ValueError("Greeks 曲线仅对欧式期权有效")

    from dataclasses import replace

    S_range = np.linspace(0.5 * option.K, 1.5 * option.K, 100)
    greeks_data = {k: [] for k in ["delta", "gamma", "vega", "theta"]}

    for S in S_range:
        g = greeks_analytical(replace(option, S=float(S)))
        for k in greeks_data:
            greeks_data[k].append(g[k])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # 偏微分符号 ∂ 用 matplotlib mathtext 渲染，避免中文字体缺失 glyph
    titles = {
        "delta": r"Delta ($\partial V / \partial S$)",
        "gamma": r"Gamma ($\partial^2 V / \partial S^2$)",
        "vega":  r"Vega ($\partial V / \partial \sigma$)",
        "theta": r"Theta ($\partial V / \partial T$)",
    }
    colors = {"delta": "C0", "gamma": "C1", "vega": "C2", "theta": "C3"}

    for ax, key in zip(axes.flat, ["delta", "gamma", "vega", "theta"]):
        ax.plot(S_range, greeks_data[key], color=colors[key], linewidth=2)
        ax.axvline(option.K, color="gray", linestyle=":", alpha=0.6, label=f"K={option.K}")
        ax.set_xlabel("标的价格 S")
        ax.set_ylabel(titles[key])
        ax.set_title(titles[key])
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"{option.option_type.value.upper()} 期权 Greeks 曲线"
        f"（K={option.K}, T={option.T}, σ={option.sigma}, r={option.r}）",
        fontsize=13,
    )
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=120, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
