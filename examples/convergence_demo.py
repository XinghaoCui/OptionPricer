"""演示：树定价收敛性 + 蒙特卡洛方差缩减效果可视化。

运行后会在 docs/screenshots/ 下保存：
    - convergence_binomial.png    （二叉树定价随步数振荡收敛到 BS 解析价）
    - convergence_mc.png          （MC 三种方法的标准误对比，log-log 图）
    - binomial_tree_small.png     （N=5 的小二叉树展开示意图）
"""

import sys
from pathlib import Path

# 用非交互后端，便于在脚本/CI 中直接保存图片而不弹窗
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.option import Option, OptionType, ExerciseStyle
from src.visualizer import (
    plot_binomial_tree,
    plot_convergence,
    plot_mc_convergence,
)


def main() -> None:
    SHOTS = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
    SHOTS.mkdir(parents=True, exist_ok=True)

    opt = Option(
        S=100, K=100, T=1.0, r=0.05, sigma=0.20,
        option_type=OptionType.CALL, style=ExerciseStyle.EUROPEAN,
    )

    print("[1/3] 绘制 N=5 二叉树展开示意图 ...")
    plot_binomial_tree(opt, steps=5, savepath=str(SHOTS / "binomial_tree_small.png"))

    print("[2/3] 绘制二叉树定价收敛曲线 ...")
    plot_convergence(opt, max_steps=200, savepath=str(SHOTS / "convergence_binomial.png"))

    print("[3/3] 绘制 MC 方差缩减对比曲线 ...")
    plot_mc_convergence(opt, savepath=str(SHOTS / "convergence_mc.png"))

    print(f"\n已保存到 {SHOTS}")


if __name__ == "__main__":
    main()
