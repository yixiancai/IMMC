# -*- coding: utf-8 -*-
"""
线性优化（线性规划 LP）数学建模完整示例
=========================================
经典生产计划问题：

    某工厂生产甲、乙两种产品，x1、x2 分别为计划产量（单位：件）。
    每件甲利润 2 万元，每件乙利润 3 万元。三种资源（原料 A/B/C）限量如下：

        max  z = 2*x1 + 3*x2                 （总利润最大化）
        s.t. x1 + 2*x2 <= 8                 （原料 A，kg）
             4*x1       <= 16               （原料 B，kg）
             4*x2       <= 12               （原料 C，kg）
             x1, x2     >= 0                （非负约束）

    理论最优解：x1=4, x2=2, z*=14（可用图解法/顶点法验证）

本脚本演示 4 种实现方式：
  1. SciPy     —— linprog（矩阵形式，科研通用）
  2. PuLP      —— 面向对象建模（语法贴近数学表达式，推荐建模比赛使用）
  3. 顶点枚举  —— 手写图解法原理（理解 LP 几何意义）
  4. 灵敏度分析 —— 影子价格与对偶价值（运筹学进阶）
最后绘制可行域与等值线图。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无界面环境渲染图片
import matplotlib.pyplot as plt

# ---------- 全局绘图设置（中文字体） ----------
from matplotlib import font_manager as _fm
for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    _fm.fontManager.addfont(_f)          # 手动注册系统 CJK 字体
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# =====================================================================
# 方法一：SciPy linprog —— 矩阵式建模
# 注意：scipy 只接受 minimization，因此 max z = c@x 需转化为 min -c@x
# 约束统一写成 A_ub @ x <= b_ub
# =====================================================================
def solve_with_scipy():
    print("=" * 60)
    print("方法一：SciPy linprog（HiGHS 求解器）")
    print("=" * 60)

    # 目标函数系数：max 2x1+3x2  ->  min -2x1-3x2
    c = np.array([-2, -3])

    # 不等式约束 A_ub @ x <= b_ub
    A_ub = np.array([
        [1, 2],    # x1 + 2*x2 <= 8
        [4, 0],    # 4*x1       <= 16
        [0, 4],    #       4*x2 <= 12
    ])
    b_ub = np.array([8, 16, 12])

    # 变量边界 x1, x2 >= 0（默认即非负，此处显式写出）
    bounds = [(0, None), (0, None)]

    res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if res.status != 0:
        print("求解失败：", res.message)
        return None

    x = res.x
    z = -res.fun  # 还原最大化目标值
    print(f"最优解    x* = ({x[0]:.4f}, {x[1]:.4f})")
    print(f"最优目标  z* = {z:.4f} 万元")
    print(f"求解状态    : {res.message}")

    # 影子价格（对偶变量）：资源约束右端项每增加 1 单位，目标值改变量
    marginals = res.ineqlin.marginals  # <= 0，取负后即影子价格
    shadow = -np.array(marginals)
    names = ["原料A", "原料B", "原料C"]
    print("\n--- 灵敏度分析：影子价格（对偶价值）---")
    for n, s in zip(names, shadow):
        print(f"  {n} 影子价格 = {s:+.4f} 万元/kg"
              + ("  <- 紧约束，值得增购资源" if s > 1e-9 else "  <- 松弛资源，增购无收益"))
    return x


# =====================================================================
# 方法二：PuLP —— 数学语言式建模（最直观，推荐）
# =====================================================================
def solve_with_pulp():
    import pulp
    print("\n" + "=" * 60)
    print("方法二：PuLP 面向对象建模")
    print("=" * 60)

    # 1. 创建问题（默认最小化，这里声明最大化）
    prob = pulp.LpProblem("生产计划", pulp.LpMaximize)

    # 2. 定义决策变量（lowBound=0 即非负约束）
    x1 = pulp.LpVariable("x1", lowBound=0, cat="Continuous")
    x2 = pulp.LpVariable("x2", lowBound=0, cat="Continuous")

    # 3. 目标函数：max 2x1 + 3x2
    prob += 2 * x1 + 3 * x2, "总利润"

    # 4. 约束条件（+= 即添加约束，写法与数学式一致）
    prob += x1 + 2 * x2 <= 8, "原料A限制"
    prob += 4 * x1 <= 16,     "原料B限制"
    prob += 4 * x2 <= 12,     "原料C限制"

    # 5. 求解（默认 CBC 求解器，PuLP 自带）
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    print(f"求解状态    : {pulp.LpStatus[status]}")
    print(f"最优解    x* = ({pulp.value(x1):.4f}, {pulp.value(x2):.4f})")
    print(f"最优目标  z* = {pulp.value(prob.objective):.4f} 万元")

    # 6. 对偶信息 + 松弛量（用到的资源 vs 闲置的资源）
    print("\n--- 灵敏度分析：影子价格 与 资源余量 ---")
    for name, con in prob.constraints.items():
        print(f"  {name:<8} 使用量={-con.value() if con.value() is not None else 0:>7.2f}"
              f"  影子价格={con.pi if con.pi is not None else 0:>+.4f}")
    return np.array([pulp.value(x1), pulp.value(x2)])


# =====================================================================
# 方法三：顶点枚举法 —— 图解法的程序化实现
# 原理：LP 的最优解必在可行域顶点处取得（线性规划基本定理）
# =====================================================================
def solve_by_vertex_enumeration():
    print("\n" + "=" * 60)
    print("方法三：顶点枚举法（基本定理验证）")
    print("=" * 60)

    c = np.array([2, 3])  # max c@x
    A_ub = np.array([[1, 2], [4, 0], [0, 4]])
    b_ub = np.array([8, 16, 12])
    nonneg = np.eye(2)    # x >= 0 也当作约束参与求交点

    A_all = np.vstack([A_ub, -nonneg])          # -x <= 0
    b_all = np.concatenate([b_ub, [0, 0]])
    m = len(b_all)

    vertices = []
    for i, j in itertools.combinations(range(m), 2):     # 两两约束求交点
        A_pair = np.array([A_all[i], A_all[j]])
        if abs(np.linalg.det(A_pair)) < 1e-12:           # 平行线无交点
            continue
        p = np.linalg.solve(A_pair, np.array([b_all[i], b_all[j]]))
        if np.all(A_all @ p <= b_all + 1e-9):            # 判断是否可行顶点
            vertices.append(p)

    best_v, best_z = None, -np.inf
    print("可行域顶点及对应目标值：")
    for p in sorted({(round(p[0], 6), round(p[1], 6)) for p in vertices}):
        z = c @ np.array(p)
        flag = ""
        if z > best_z:
            best_z, best_v, flag = z, np.array(p), "  <-- 最优"
        print(f"  顶点 ({p[0]:.1f}, {p[1]:.1f})  z = {z:>6.2f}{flag}")

    print(f"\n枚举结论：x* = ({best_v[0]:.1f}, {best_v[1]:.1f}), z* = {best_z:.1f}")
    return best_v


# =====================================================================
# 可视化：可行域 + 目标函数等值线 + 最优解
# =====================================================================
def visualize(x_star):
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=120)

    xx, yy = np.meshgrid(np.linspace(0, 6, 400), np.linspace(0, 5, 400))

    # 可行域：所有约束同时满足的区域
    feasible = (xx + 2 * yy <= 8) & (4 * xx <= 16) & (4 * yy <= 12) & (xx >= 0) & (yy >= 0)
    ax.imshow(feasible.astype(int), extent=(0, 6, 0, 5), origin="lower",
              cmap="Greens", alpha=0.35)

    # 三条资源约束边界线
    y1 = (8 - xx) / 2                     # x1 + 2x2 = 8
    y2 = np.full_like(xx, 4.0)            # 4x1 = 16 -> x1 = 4（竖线）
    y3 = np.full_like(xx, 3.0)            # 4x2 = 12 -> x2 = 3（横线）
    ax.plot(xx[0], y1[0], color="C0", lw=2, label=r"$x_1+2x_2 \leq 8$（原料A）")
    ax.axvline(4, color="C1", lw=2, label=r"$4x_1 \leq 16$（原料B）")
    ax.axhline(3, color="C2", lw=2, label=r"$4x_2 \leq 12$（原料C）")

    # 目标函数等值线 2x1+3x2 = z
    cs = ax.contour(xx, yy, 2 * xx + 3 * yy, levels=8, colors="gray",
                    linewidths=0.8, linestyles="dashed")
    ax.clabel(cs, fmt="z=%.0f", fontsize=8)

    # 最优点：等值线沿梯度方向平移，离开可行域前的最后触点
    ax.plot(*x_star, "r*", markersize=18, zorder=5,
            label=f"最优点 ({x_star[0]:.0f}, {x_star[1]:.0f}), z*={2*x_star[0]+3*x_star[1]:.0f}")
    ax.annotate("等值线上移至此触到可行域边界，\n即取得最大值（顶点处）",
                xy=x_star, xytext=(4.15, 1.0), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="red"))

    ax.set_xlim(0, 6); ax.set_ylim(0, 5)
    ax.set_xlabel(r"产品甲产量 $x_1$（件）")
    ax.set_ylabel(r"产品乙产量 $x_2$（件）")
    ax.set_title("线性规划图解法：可行域（绿色）与等值线", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = "./feasible_region.png"
    fig.savefig(out)
    print(f"\n[图已保存] {out}")


if __name__ == "__main__":
    import itertools
    import scipy.optimize

    x_sci = solve_with_scipy()        # 1. SciPy
    x_pulp = solve_with_pulp()        # 2. PuLP
    x_vtx = solve_by_vertex_enumeration()  # 3. 顶点枚举

    assert np.allclose(x_sci, x_pulp) and np.allclose(x_sci, x_vtx), "三种方法结果不一致！"
    print("\n[校验] 三种方法结果一致 ✓")

    visualize(x_sci)                  # 4. 可视化
