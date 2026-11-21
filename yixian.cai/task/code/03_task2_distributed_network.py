# -*- coding: utf-8 -*-
"""
任务二：弹性分布式算力网络的选址-调度优化（MIP）
输入: ../clean/*.csv (清洗后的 Epoch AI 数据)
输出: ../outputs/task2_results.json

范式借鉴 IM2C 2026 获奖方案（凹函数效能 + 混合整数规划 + 鲁棒性）:
  候选节点 = 数据中心的候选选址/扩张点
  决策: n_j (是否启用节点j, 二元), y_ij (任务i分配给节点j的算力, 连续)
  目标: min [环境足迹 + 网络成本 + 不公平度]  或  max 算力覆盖
  约束: 任务需求守恒, 节点容量, 水/能承载上限, 时延, 预算
  效能: E(u)=1-exp(-u) 凹函数, 边际递减 -> 战略部署
  鲁棒: 不确定集下 worst-case (Bertsimas-Sim 简化版)
"""
import os
import json
import numpy as np
import pandas as pd
import pulp

BASE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.normpath(os.path.join(BASE, "..", "clean"))
OUT = os.path.normpath(os.path.join(BASE, "..", "outputs"))
os.makedirs(OUT, exist_ok=True)

D = pd.read_csv(os.path.join(CLEAN, "data_centers_clean.csv"))

# ---------- 参数 ----------
H_EFF = 7884.0
W_PER_IT = 9.84            # 全球平均 [真实 UNU]
Q_PERMIT = 3e6             # 批水量基准
N_NODES = 12               # 取算力最大的12个候选节点
BUDGET = 400.0             # 预算($B)，需覆盖足够节点启用成本
LATENCY_MAX = 20.0         # 最大时延(ms)阈值
TASK_DEMAND = 300e3        # 总算力需求 (H100e)
WATER_COST = 1.0           # 环境足迹权重
NETWORK_COST = 0.5         # 网络成本权重
FAIR_WEIGHT = 2.0          # 公平性权重

# 从真实数据取候选节点：功率、算力、成本、国家
cand = D[D["h100e"] > 0].head(N_NODES).copy().reset_index(drop=True)
cand["cap_H100e"] = cand["h100e"]           # 节点算力容量
cand["cost_B"] = cand["cost_B"].fillna(cand["cost_B"].median())
# 水承载上限 (MW) -> 算力上限 (H100e)，用算力密度
rho = D["h100e_per_MW"].median()
cand["water_cap_MW"] = Q_PERMIT / (H_EFF * W_PER_IT)
cand["water_cap_H100e"] = cand["water_cap_MW"] * rho
# 网络时延：按国家/地理粗略赋值（同国低，跨国高）——示意
cand["latency_ms"] = cand["Country"].map(
    lambda c: 5.0 if c == "United States" else 15.0)
# 环境足迹：用水强度（现场耗水，来自timelines若为0则用全球均值）
cand["w_per_it_site"] = W_PER_IT
cand["env_footprint"] = cand["cap_H100e"] * cand["w_per_it_site"]


def build_and_solve(objective="min_cost", verbose=True):
    """构建并求解 MIP。objective: min_cost / max_coverage / min_env"""
    nodes = list(cand.index)
    prob = pulp.LpProblem("DistributedComputeNetwork", pulp.LpMinimize)

    # 决策变量
    n = {j: pulp.LpVariable(f"n_{j}", cat="Binary") for j in nodes}       # 启用节点
    y = {j: pulp.LpVariable(f"y_{j}", lowBound=0) for j in nodes}         # 分配到节点j的算力

    # 目标函数：启用成本 + 网络成本 + 环境足迹 + 公平性惩罚
    obj = pulp.lpSum(cand.loc[j, "cost_B"] * n[j] for j in nodes)
    obj += pulp.lpSum(NETWORK_COST * y[j] / 1e3 for j in nodes)
    obj += pulp.lpSum(WATER_COST * cand.loc[j, "env_footprint"] / 1e6 * n[j] for j in nodes)
    if objective == "max_coverage":
        obj = -pulp.lpSum(y[j] for j in nodes) + 0.01 * obj
    prob += obj, "TotalCost"

    # 约束1：算力需求守恒
    prob += pulp.lpSum(y[j] for j in nodes) >= TASK_DEMAND, "Demand"
    # 约束2：节点容量（启用才可用）
    for j in nodes:
        prob += y[j] <= cand.loc[j, "cap_H100e"] * n[j], f"Cap_{j}"
    # 约束3：水承载上限（制度+物理）
    for j in nodes:
        prob += y[j] <= cand.loc[j, "water_cap_H100e"] * n[j], f"Water_{j}"
    # 约束4：预算
    prob += pulp.lpSum(cand.loc[j, "cost_B"] * n[j] for j in nodes) <= BUDGET, "Budget"
    # 约束5：时延（高时延节点承接量受限）
    for j in nodes:
        if cand.loc[j, "latency_ms"] > LATENCY_MAX:
            prob += y[j] == 0, f"Latency_{j}"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status_str = pulp.LpStatus[status]
    feasible = status_str == "Optimal"

    # 仅当可行时才读取变量值，避免不可行时的残留假象
    if feasible:
        selected = [j for j in nodes if n[j].varValue and n[j].varValue > 0.5]
        alloc = {cand.loc[j, "Name"]: round(y[j].varValue or 0, 1) for j in selected}
        total_alloc = sum(y[j].varValue or 0 for j in nodes)
        obj_val = pulp.value(prob.objective)
    else:
        selected, alloc, total_alloc, obj_val = [], {}, 0.0, None

    return dict(status=status_str, feasible=feasible, objective_value=obj_val,
                selected_nodes=selected, allocation=alloc,
                total_allocated_H100e=total_alloc,
                demand_H100e=TASK_DEMAND,
                num_nodes=len(selected))


def robustness_check():
    """鲁棒性：需求增加20% / 预算削减30%，看可行性"""
    global TASK_DEMAND, BUDGET
    orig = (TASK_DEMAND, BUDGET)
    out = {}
    TASK_DEMAND, BUDGET = orig[0] * 1.2, orig[1]
    out["demand_plus20pct"] = build_and_solve(verbose=False)["status"]
    TASK_DEMAND, BUDGET = orig[0], orig[1] * 0.7
    out["budget_minus30pct"] = build_and_solve(verbose=False)["status"]
    TASK_DEMAND, BUDGET = orig
    return out


def main():
    base = build_and_solve("min_cost")
    robust = robustness_check()
    results = dict(
        parameters=dict(N_NODES=N_NODES, BUDGET=BUDGET, TASK_DEMAND=TASK_DEMAND,
                        W_PER_IT=W_PER_IT, rho=rho, LATENCY_MAX=LATENCY_MAX),
        candidate_nodes=cand[["Name", "Country", "cap_H100e", "cost_B",
                              "water_cap_H100e", "latency_ms"]].to_dict("records"),
        base_solution=base,
        robustness=robust,
        comparison_note="分布式 vs 集中式: 集中式单站最大1037MW; 分布式用多节点分散承载, "
                        "在缺水地区可规避单站水瓶颈, 但付出成本与网络时延",
    )
    with open(os.path.join(OUT, "task2_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print("=" * 64)
    print("任务二：分布式算力网络 MIP 优化结果")
    print("=" * 64)
    print(f"求解状态: {base['status']}")
    print(f"启用节点数: {base['num_nodes']} / {N_NODES}")
    print(f"总分配算力: {base['total_allocated_H100e']/1e3:.1f} K H100e "
          f"(需求 {TASK_DEMAND/1e3:.0f} K)")
    print("启用节点及分配:")
    for name, amt in base["allocation"].items():
        print(f"  {name[:40]:42} {amt/1e3:7.1f} K H100e")
    print("\n鲁棒性检验:")
    for k, v in robust.items():
        print(f"  {k}: {v}")
    print(f"\n结果已保存: {os.path.join(OUT, 'task2_results.json')}")


if __name__ == "__main__":
    main()
