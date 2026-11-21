# -*- coding: utf-8 -*-
"""
任务二：弹性分布式算力网络完整优化模型
=========================================
输入：../clean/ 下的清洗数据（源自 Epoch AI，用户下载的数据集）
  - data_centers_clean.csv : 节点容量/成本/算力密度/国家
  - chip_quantities_clean.csv : 芯片异构性
模型：见 对话记录_IMMC2027任务一.md 第16.5节
  min Z = 启用成本 + λ1·网络代价 + λ2·环境足迹 + λ3·公平性(Gini近似)
  s.t. 需求守恒 / 节点容量 / 水承载上限 / SLA时延 / 预算

输出：../outputs/task2_full_results.json

运行：python3 07_task2_full_model.py
"""
import os
import json
import numpy as np
import pandas as pd
import pulp

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(BASE, ".."))
CLEAN = os.path.join(ROOT, "clean")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

# =====================================================================
# 1. 读取下载并清洗的数据集
# =====================================================================
D = pd.read_csv(os.path.join(CLEAN, "data_centers_clean.csv"))
CHIP = pd.read_csv(os.path.join(CLEAN, "chip_quantities_clean.csv"))

# 取算力最大的 N 个站点作为候选节点
N_NODES = 15
nodes = D[D["h100e"] > 0].head(N_NODES).reset_index(drop=True)

# =====================================================================
# 2. 参数（数据集字段 + 必要假设，均标注）
# =====================================================================
H_EFF = 7884.0                    # 等效满负荷小时
W_PER_IT = 9.84                   # 耗水强度 m3/MWh（UNU 反推，任务一）
Q_PERMIT = 3e6                    # 批水量基准 m3/年（假设）

# 节点参数（来自数据集）
c = nodes["h100e"].values                       # 算力容量 H100e [数据]
b = nodes["cost_B"].fillna(nodes["cost_B"].median()).values  # 启用成本 $B [数据]
rho = nodes["h100e_per_MW"].fillna(nodes["h100e_per_MW"].median()).values  # 算力密度 [数据]
country = nodes["Country"].values               # 国家 [数据]
name = nodes["Name"].values                     # 名称 [数据]

# 时延代理（数据集无时延字段 → 按国家假设）
def latency(c1, c2):
    if c1 == c2:
        return 5.0        # 同国 5ms
    return 15.0           # 跨国 15ms
LATENCY = np.array([[latency(country[i], country[j]) for j in range(N_NODES)]
                    for i in range(N_NODES)])

# 网络能耗代价系数（假设：单位算力迁移的能耗代理，随距离/时延增大）
NETWORK_UNIT = np.array([[0.3 if country[i] == country[j] else 1.0
                          for j in range(N_NODES)] for i in range(N_NODES)])

# 环境足迹 g_j：用水强度（统一用 W_PER_IT，可替换为节点实测）
g = np.full(N_NODES, W_PER_IT)

# 水承载上限折算的算力上限 [数据+任务一模型]
water_cap_MW = Q_PERMIT / (H_EFF * W_PER_IT)
water_cap_H100e = water_cap_MW * rho

# 权重、预算、SLA、需求
LAM1, LAM2, LAM3 = 0.5, 1.0, 2.0   # 网络/环境/公平权重
BUDGET = 400.0                     # 预算 $B
TAU_MAX = 150.0                    # SLA 时延上限 ms
TASK_DEMAND = 300e3                # 总需求 H100e

# =====================================================================
# 3. 构建并求解 MIP
# 说明：Gini 含绝对值与比值，精确线性化复杂；这里用"节点负载率偏差之和"
#       作为公平性代理（等价思想：负载越均衡惩罚越小），保持 MIP 可解。
# =====================================================================
def solve(objective="min_cost", fairness=True, sla=True):
    prob = pulp.LpProblem("DistributedComputeNetwork", pulp.LpMinimize)

    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(N_NODES)]
    y = [[pulp.LpVariable(f"y_{j}", lowBound=0) for j in range(N_NODES)]]  # 单任务i=0
    yv = y[0]

    # 目标：启用成本 + λ1网络 + λ2环境 + λ3公平代理
    obj = pulp.lpSum(b[j] * x[j] for j in range(N_NODES))
    obj += LAM1 * pulp.lpSum(NETWORK_UNIT[0][j] * yv[j] / 1e3 for j in range(N_NODES))
    obj += LAM2 * pulp.lpSum(g[j] * yv[j] / 1e6 for j in range(N_NODES))

    # 公平代理：用 max-min 偏差（各节点负载率落在一带内，带宽越窄越公平）
    if fairness:
        zmax = pulp.LpVariable("zmax", lowBound=0)
        zmin = pulp.LpVariable("zmin", lowBound=0)
        M = 1.0  # 大M：利用率上界约1，取M=1
        for j in range(N_NODES):
            util = yv[j] * (1.0 / max(c[j], 1))   # c_j 为常数，乘倒数保持线性
            prob += util <= zmax
            # 大M法线性化：x_j=1 时 util>=zmin；x_j=0 时放松
            prob += util >= zmin - M * (1 - x[j])
        obj += LAM3 * (zmax - zmin) * 1e3

    if objective == "max_coverage":
        obj = -pulp.lpSum(yv) + 0.01 * obj
    prob += obj, "TotalCost"

    # 约束
    prob += pulp.lpSum(yv) >= TASK_DEMAND, "Demand"
    for j in range(N_NODES):
        prob += yv[j] <= c[j] * x[j], f"Cap_{j}"
        prob += yv[j] <= water_cap_H100e[j] * x[j], f"Water_{j}"
        if sla and LATENCY[0][j] > 5.0:
            # 时延硬约束：跨国(15ms)节点承接量受限，同国(5ms)不限制
            prob += yv[j] <= 0.3 * TASK_DEMAND * x[j], f"SLA_{j}"
    prob += pulp.lpSum(b[j] * x[j] for j in range(N_NODES)) <= BUDGET, "Budget"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status_str = pulp.LpStatus[status]
    feasible = status_str == "Optimal"

    if feasible:
        sel = [j for j in range(N_NODES) if x[j].varValue > 0.5]
        alloc = {name[j]: round(yv[j].varValue or 0, 1) for j in sel}
        total = sum(yv[j].varValue or 0 for j in range(N_NODES))
        # 计算公平性 Gini（后验，真实指标）
        utils = np.array([(yv[j].varValue or 0) / c[j] for j in sel])
        if len(utils) > 1 and utils.mean() > 0:
            gini = np.abs(utils[:, None] - utils[None, :]).sum() / (2 * len(utils)**2 * utils.mean())
        else:
            gini = 0.0
        return dict(feasible=True, status=status_str, n_nodes=len(sel),
                    selected=[name[j] for j in sel], allocation=alloc,
                    total_H100e=total, gini=round(float(gini), 4),
                    obj_value=pulp.value(prob.objective))
    else:
        return dict(feasible=False, status=status_str)


def robustness():
    """鲁棒性：需求+20% / 预算-30%"""
    global TASK_DEMAND, BUDGET
    orig = (TASK_DEMAND, BUDGET)
    TASK_DEMAND, BUDGET = orig[0] * 1.2, orig[1]
    r1 = solve()["feasible"]
    TASK_DEMAND, BUDGET = orig[0], orig[1] * 0.7
    r2 = solve()["feasible"]
    TASK_DEMAND, BUDGET = orig
    return dict(demand_plus20=r1, budget_minus30=r2)


def main():
    base = solve()
    no_fair = solve(fairness=False)
    no_sla = solve(sla=False)
    rob = robustness()

    results = dict(
        data_source="Epoch AI (用户下载数据集) → clean/data_centers_clean.csv",
        parameters=dict(N_NODES=N_NODES, BUDGET=BUDGET, TASK_DEMAND=TASK_DEMAND,
                        LAM1=LAM1, LAM2=LAM2, LAM3=LAM3, TAU_MAX=TAU_MAX,
                        W_PER_IT=W_PER_IT, H_EFF=H_EFF,
                        note="时延/网络能耗为假设(数据集无此字段);其余来自数据"),
        base_model=base,
        without_fairness=no_fair,
        without_sla=no_sla,
        robustness=rob,
        data_limitations="数据集无带宽/网速/时延/碳强度字段，网络维度按国家地理代理",
    )
    with open(os.path.join(OUT, "task2_full_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print("=" * 66)
    print("任务二：弹性分布式算力网络完整模型结果")
    print("=" * 66)
    print(f"求解状态: {base['status']}  启用节点: {base.get('n_nodes',0)}/{N_NODES}")
    print(f"满足需求: {base.get('total_H100e',0)/1e3:.0f}K / {TASK_DEMAND/1e3:.0f}K H100e")
    print(f"公平性 Gini: {base.get('gini','N/A')}  (越小越公平)")
    print("\n启用节点及分配:")
    for nm, amt in base.get("allocation", {}).items():
        print(f"  {nm[:38]:40} {amt/1e3:7.1f}K H100e")
    print("\n对比:")
    print(f"  无公平约束: Gini={no_fair.get('gini','N/A')}, 节点={no_fair.get('n_nodes','N/A')}")
    print(f"  无SLA约束 : 节点={no_sla.get('n_nodes','N/A')}")
    print(f"  鲁棒性: 需求+20%={'可行' if rob['demand_plus20'] else '不可行'}, "
          f"预算-30%={'可行' if rob['budget_minus30'] else '不可行'}")
    print(f"\n结果: {os.path.join(OUT, 'task2_full_results.json')}")


if __name__ == "__main__":
    main()
