"""CRR（Cox-Ross-Rubinstein）二叉树定价。

二叉树定价是"树"结构与"动态规划"的经典结合：
1. **树结构**：从根节点 S 出发，每步价格上涨为 S·u 或下跌为 S·d，
   N 步后形成一棵高度为 N 的二叉树，共 (N+1)(N+2)/2 个节点。
2. **空间优化**：由于树是"重组的"（recombining tree，即 ud = du），
   第 n 步只有 n+1 个不同节点，整棵树用一维数组就能表示，空间从
   O(N²) 压缩到 O(N)，这是"用数组实现完全二叉树"技巧的应用。
3. **动态规划**：从叶子节点的 payoff 出发，按 V = e^(-rΔt)·[p·V_up + (1-p)·V_down]
   倒推到根节点，是典型的"自底向上"DP。
4. **美式期权 → 最优停时问题**：每个内部节点取 max(立即行权价值, 持有价值)，
   是 Bellman 方程的离散化，体现 DP 的"最优子结构"。

参数选择（CRR 经典参数化）：
    u = exp(σ√Δt),   d = 1/u
    p = (exp((r-q)Δt) - d) / (u - d)   ——风险中性概率
"""

from __future__ import annotations

import math
from typing import Tuple

from .option import Option, OptionType, ExerciseStyle


def binomial_price(option: Option, steps: int = 200) -> float:
    """CRR 二叉树定价，欧式和美式统一接口。

    时间复杂度 O(N²)（N 步共需回溯 N(N+1)/2 个节点），
    空间复杂度 O(N)（仅维护一层节点的价值数组）。

    参数
    ----
    option : Option
        待定价的期权
    steps : int
        时间步数 N，越大越精确但耗时增加。N=200 一般可保证 4 位小数精度。
    """
    if steps < 1:
        raise ValueError(f"步数必须 >= 1，得到 {steps}")

    S, K, T, r, sigma, q = option.S, option.K, option.T, option.r, option.sigma, option.q

    # ---------- 1. 构造 CRR 参数 ----------
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)                       # 单步贴现因子
    p = (math.exp((r - q) * dt) - d) / (u - d)     # 风险中性概率

    # 数值稳定性检查（如果 σ 过小或 r 过大可能出现非概率）
    if not (0.0 < p < 1.0):
        raise ValueError(
            f"风险中性概率 p={p:.4f} 不在 (0,1) 内，"
            f"请增大 steps 或检查参数（dt={dt}, σ√dt={sigma*math.sqrt(dt)}）"
        )

    # ---------- 2. 叶子层 payoff ----------
    # 第 N 步共 N+1 个节点，从全跌（i=0）到全涨（i=N）
    # 节点 i 的标的价格 = S · u^i · d^(N-i)
    values = [option.payoff(S * (u ** i) * (d ** (steps - i))) for i in range(steps + 1)]

    # ---------- 3. 自底向上 DP 回溯 ----------
    for n in range(steps - 1, -1, -1):
        # 第 n 步有 n+1 个节点；原地更新 values[0..n]
        for i in range(n + 1):
            continuation = disc * (p * values[i + 1] + (1.0 - p) * values[i])

            if option.style == ExerciseStyle.AMERICAN:
                # 美式期权：每个节点比较"立即行权"与"继续持有"
                # —— 这就是最优停时问题的 Bellman 方程
                S_ni = S * (u ** i) * (d ** (n - i))
                intrinsic = option.payoff(S_ni)
                values[i] = max(intrinsic, continuation)
            else:
                values[i] = continuation

    return values[0]


def binomial_tree_full(option: Option, steps: int) -> Tuple[list, list]:
    """返回完整的二叉树（价格树与价值树），仅供可视化使用。

    注意这会消耗 O(N²) 空间，仅适合 steps 较小（≤ 50）的演示场景。

    返回
    ----
    price_tree : List[List[float]]
        price_tree[n][i] 表示第 n 步、上涨 i 次时的标的价格
    value_tree : List[List[float]]
        value_tree[n][i] 表示第 n 步、上涨 i 次时的期权价值
    """
    if steps > 100:
        raise ValueError(f"全树展开仅支持 steps<=100，得到 {steps}（请使用 binomial_price）")

    S, T, r, sigma, q = option.S, option.T, option.r, option.sigma, option.q
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)

    # 价格树：自顶向下构造
    price_tree = [[S * (u ** i) * (d ** (n - i)) for i in range(n + 1)] for n in range(steps + 1)]

    # 价值树：自底向上回溯
    value_tree: list = [[0.0] * (n + 1) for n in range(steps + 1)]
    value_tree[steps] = [option.payoff(price_tree[steps][i]) for i in range(steps + 1)]

    for n in range(steps - 1, -1, -1):
        for i in range(n + 1):
            continuation = disc * (p * value_tree[n + 1][i + 1] + (1.0 - p) * value_tree[n + 1][i])
            if option.style == ExerciseStyle.AMERICAN:
                intrinsic = option.payoff(price_tree[n][i])
                value_tree[n][i] = max(intrinsic, continuation)
            else:
                value_tree[n][i] = continuation

    return price_tree, value_tree
