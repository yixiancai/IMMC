# -*- coding: utf-8 -*-
"""
Etosha 保护资源分配 —— 完整建模与求解套件
==========================================
本文件是论文数值结果的总入口，输出全部图表与关键指标。

模块结构:
  §1 空间栅格化 & 保护优先级权重 w_i
  §2 单期资源分配 MIP（战略部署）
  §3 多期动态模型（护林员轮换 / 巡逻排班 / 时间上的保护水平）
  §4 鲁棒优化（威胁强度区间不确定集）
  §5 敏感性 & 场景分析（人力、装备、预算、排班）
  §6 跨保护区适配（2 个其他大陆的保护区的参数迁移）
  §7 可视化输出
"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/workspace/im2c_2026/outputs"
os.makedirs(OUT, exist_ok=True)

RNG = np.random.default_rng(2026)


# =====================================================================
# §1  空间栅格化
# =====================================================================
GRID_N = 12
PARK_AREA = 22935.0
CELL_AREA = PARK_AREA / (GRID_N ** 2)


def build_cells(n=GRID_N):
    rng = np.random.default_rng(2026)
    cells = []
    half = n / 2.0
    for r in range(n):
        for c in range(n):
            x = (c + 0.5 - half) / half
            y = (r + 0.5 - half) / half
            d = math.hypot(x, y)
            is_pan = d < 0.45                      # Etosha Pan ≈ 21% 面积
            access = min(1.0, 0.30 + 0.70 * d)     # 外围道路密度高
            eco = 0.05 if is_pan else 0.55 + 0.45 * min(1.0, d)
            poach = float(np.clip((0.28 + 0.58 * access) * (1 - 0.88 * is_pan)
                                  * rng.uniform(0.62, 1.38), 0, 1))
            fire = 0.0 if is_pan else float(np.clip((0.18 + 0.72 * eco)
                                                    * rng.uniform(0.5, 1.3), 0, 1))
            cells.append(dict(id=len(cells), row=r, col=c, is_pan=is_pan,
                              eco=round(eco, 3), access=round(access, 3),
                              poach=round(poach, 3), fire=round(fire, 3)))
    return cells


def add_priority(cells):
    for c in cells:
        if c["is_pan"]:
            c["w"] = 0.02
        else:
            threat = 0.65 * c["poach"] + 0.35 * c["fire"]
            c["w"] = round(float(np.clip(c["eco"] * (0.25 + 0.75 * threat), 0, 1)), 4)
    return cells


# =====================================================================
# §2  资源定义
# =====================================================================
# 每条资源单元的年度成本($k)、人力需求、采购上限、
# span = 单个资源单元一年内可有效覆盖的网格数（受续航/维护/轮换限制）
RES = {
    #                    成本$k  人力    上限  覆盖跨度  覆盖强度  可移动性
    "foot_patrol":    dict(cost=3.0,  staff=1.0, cap=60, span=4,  kappa=0.34, mobility=1.0, label="Foot patrol"),
    "vehicle_patrol": dict(cost=9.0,  staff=2.0, cap=30, span=8,  kappa=0.30, mobility=3.0, label="Vehicle patrol"),
    "fixed_sensor":   dict(cost=2.0,  staff=0.1, cap=50, span=2,  kappa=0.26, mobility=0.0, label="Fixed sensor"),
    "drone_unit":     dict(cost=6.0,  staff=1.5, cap=20, span=10, kappa=0.20, mobility=5.0, label="Drone unit"),
    "satellite_feed": dict(cost=1.0,  staff=0.0, cap=1,  span=30, kappa=0.10, mobility=0.0, label="Satellite feed"),
    "ai_analytics":   dict(cost=4.0,  staff=0.5, cap=1,  span=50, kappa=0.14, mobility=0.0, label="AI analytics"),
}


def terrain_factor(rtype, cell):
    """地形/可达性适配：盐沼对地面力量屏蔽；车辆依赖道路密度。"""
    if cell["is_pan"]:
        return 0.0 if rtype in ("foot_patrol", "vehicle_patrol") else 0.45
    if rtype == "vehicle_patrol":
        return 1.0 if cell["access"] > 0.45 else 0.2
    return 1.0


def efficacy(u):
    """保护效能 E(u) = 1 - exp(-u)；凹函数 -> 边际递减 -> 战略部署。"""
    return 1.0 - np.exp(-np.asarray(u, dtype=float))


# 单网格覆盖强度上限：任何资源组合在 159 km² 的格子里，
# 一年内能建立的有效监视强度有一个物理上限 u_max（续航、人员轮换、地形遮蔽）。
# 这个饱和上限 + 效能函数的凹性，共同保证了"无法靠堆资源全覆盖"，
# 从而让战略取舍（优先保护哪些格子）成为真正的决策问题。
U_MAX = 1.6

# 人力排班可用率：295 人不可能全年无休，需轮班、休假、后勤。
# η < 1 使名义人数 ≠ 有效人力，是"人力需求"分析的关键。
STAFF_AVAILABILITY = 0.72


def weighted_P(u_vec, w_vec):
    """全域加权保护指数 P ∈ [0,1]。"""
    return float(np.sum(w_vec * efficacy(u_vec)) / np.sum(w_vec))


if __name__ == "__main__":
    cells = add_priority(build_cells())
    w = np.array([c["w"] for c in cells])
    print(f"§1 栅格: {len(cells)} 格 x {CELL_AREA:.0f} km²/格")
    print(f"   权重 w: mean={w.mean():.3f}, max={w.max():.3f}, pan格数={sum(c['is_pan'] for c in cells)}")
    print(f"§2 资源类型: {len(RES)} 类；效能函数示例 E(0.5)={efficacy(0.5):.3f}, E(1)={efficacy(1):.3f}, E(2)={efficacy(2):.3f}")
