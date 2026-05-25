"""Boyle 三叉树定价。

三叉树是二叉树的推广——每一步价格有三种可能走向：上涨、不变、下跌。
相比 CRR 二叉树，三叉树有两大优势：
1. **收敛更快**：相同步数下精度更高（因为每步信息量更大）；
2. **数值更稳定**：可自由调节 λ 控制 u 与 d 的关系，避免 p 越界。

Boyle (1986) 经典参数化：
    u = exp(λσ√Δt),   d = 1/u,   m = 1
    p_u = 1/(2λ²) + (r - q - σ²/2)√Δt / (2λσ)
    p_d = 1/(2λ²) - (r - q - σ²/2)√Δt / (2λσ)
    p_m = 1 - p_u - p_d
    其中 λ = √3 时收敛性最好（Kamrad-Ritchken 推荐）。

数据结构层面，三叉树是"重组树"（u·d=1, m=1），第 n 步只有 2n+1 个不同节点，
仍可用一维数组高效表示，空间 O(N)。
"""

from __future__ import annotations

import math

from .option import Option, ExerciseStyle


def trinomial_price(option: Option, steps: int = 200, lam: float = math.sqrt(3.0)) -> float:
    """Boyle 三叉树定价。

    参数
    ----
    option : Option
        待定价的期权
    steps : int
        时间步数 N
    lam : float
        参数 λ，默认 √3（Kamrad-Ritchken 最优收敛参数）
    """
    if steps < 1:
        raise ValueError(f"步数必须 >= 1，得到 {steps}")
    if lam <= 0:
        raise ValueError(f"参数 lambda 必须为正，得到 {lam}")

    S, K, T, r, sigma, q = option.S, option.K, option.T, option.r, option.sigma, option.q

    # ---------- 1. 构造三叉树参数 ----------
    dt = T / steps
    sqrt_dt = math.sqrt(dt)
    u = math.exp(lam * sigma * sqrt_dt)
    d = 1.0 / u

    drift = r - q - 0.5 * sigma * sigma
    p_u = 1.0 / (2.0 * lam * lam) + drift * sqrt_dt / (2.0 * lam * sigma)
    p_d = 1.0 / (2.0 * lam * lam) - drift * sqrt_dt / (2.0 * lam * sigma)
    p_m = 1.0 - p_u - p_d

    if not (0.0 <= p_u <= 1.0 and 0.0 <= p_m <= 1.0 and 0.0 <= p_d <= 1.0):
        raise ValueError(
            f"三叉树概率越界：p_u={p_u:.4f}, p_m={p_m:.4f}, p_d={p_d:.4f}；"
            f"请增大 steps 或调整 lambda"
        )

    disc = math.exp(-r * dt)

    # ---------- 2. 叶子层 payoff ----------
    # 第 N 步共 2N+1 个节点；用偏移量 j ∈ [-N, N] 表示净涨跌次数
    # 节点 j 的价格 = S · u^j（j>0 上涨 j 次，j<0 下跌 |j| 次）
    n_nodes = 2 * steps + 1
    values = [option.payoff(S * (u ** (j - steps))) for j in range(n_nodes)]

    # ---------- 3. 自底向上回溯 ----------
    for n in range(steps - 1, -1, -1):
        n_nodes_n = 2 * n + 1
        new_values = [0.0] * n_nodes_n
        for j in range(n_nodes_n):
            # 节点 j（位于本步）的三个子节点分别是下一步的 j, j+1, j+2
            # （下、不变、上对应数组索引偏移 0、1、2）
            continuation = disc * (
                p_d * values[j] + p_m * values[j + 1] + p_u * values[j + 2]
            )
            if option.style == ExerciseStyle.AMERICAN:
                S_nj = S * (u ** (j - n))
                intrinsic = option.payoff(S_nj)
                new_values[j] = max(intrinsic, continuation)
            else:
                new_values[j] = continuation
        values = new_values

    return values[0]
