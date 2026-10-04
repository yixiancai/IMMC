# -*- coding: utf-8 -*-
"""
§3 单期资源分配 MIP（紧凑建模，可秒级求解）
==========================================
关键规模优化：不再把每个资源单元当独立二元变量，而是
  n_k  = 类型 k 的采购数量（整数）
  y_ki = 类型 k 部署到网格 i 的"投入量"（连续 [0, span_k * n_k]）
这样变量数从 O(units x cells) 降到 O(types x cells) = 6 x 144 = 864，
且能正确刻画"同类资源多单元可分散部署"。

目标（战略部署）：
  max  Σ_i w_i * (1 - exp(-u_i))
  u_i = Σ_k y_ki * κ_k * terrain_ki     有效覆盖强度
约束：
  Σ_k y_ki <= Σ_k span_k * n_k?  否 —— 用分配守恒：
  Σ_i y_ki <= span_k * n_k            （部署总量不超过采购能力）
  Σ_k n_k * staff_k <= H              人力
  Σ_k n_k * cost_k  <= B              预算
  n_k ∈ Z+
"""
import numpy as np
import pulp

from model_core import (GRID_N, RES, U_MAX, STAFF_AVAILABILITY, add_priority,
                        build_cells, efficacy, terrain_factor, weighted_P)


def build_and_solve(personnel=295, budget=900.0, grid_n=GRID_N,
                    kappa_scale=None, staff_avail=STAFF_AVAILABILITY,
                    seed=2026, time_limit=90, verbose=True):
    cells = add_priority(build_cells(grid_n))
    cells_by_id = {c["id"]: c for c in cells}
    ids = [c["id"] for c in cells]
    w = {c["id"]: c["w"] for c in cells}
    types = list(RES.keys())

    # 威胁-资源匹配乘子：地面力量更擅长反盗猎(隐蔽/快速反应)，
    # 空基/传感更擅长火情与广域监视
    if kappa_scale is None:
        kappa_scale = {t: 1.0 for t in types}

    prob = pulp.LpProblem("Etosha_Protection_SP", pulp.LpMaximize)

    # 采购数量
    n = {k: pulp.LpVariable(f"n_{k}", lowBound=0, upBound=RES[k]["cap"], cat="Integer")
         for k in types}
    # 部署投入量 y_ki（连续，允许同类资源分散到多格）
    y = {}
    for k in types:
        for c in cells:
            tf = terrain_factor(k, cells_by_id[c["id"]])
            if tf > 0:
                y[(k, c["id"])] = pulp.LpVariable(f"y_{k}_{c['id']}", lowBound=0)

    # 有效覆盖强度 u_i（受物理饱和上限 U_MAX 约束）
    u = {c["id"]: pulp.LpVariable(f"u_{c['id']}", lowBound=0, upBound=U_MAX) for c in cells}

    # ---- 分段线性近似 E(u)（凹函数上包络）----
    S = 12
    xs = np.linspace(0, U_MAX, S + 1)
    es = efficacy(xs)
    sl = np.diff(es) / np.diff(xs)
    z = {c["id"]: pulp.LpVariable(f"z_{c['id']}", lowBound=0, upBound=1.0) for c in cells}
    for c in cells:
        i = c["id"]
        for s in range(S):
            prob += z[i] <= es[s] + sl[s] * (u[i] - xs[s])

    # ---- 目标 ----
    prob += pulp.lpSum(w[i] * z[i] for i in ids)

    # ---- 覆盖合成 u_i = Σ_k y_ki * κ_k * terrain ----
    for c in cells:
        i = c["id"]
        terms = []
        for k in types:
            if (k, i) in y:
                kk = RES[k]["kappa"] * kappa_scale[k] * terrain_factor(k, c)
                terms.append(y[(k, i)] * kk)
        if terms:
            prob += u[i] <= pulp.lpSum(terms)
        else:
            prob += u[i] == 0

    # ---- 部署能力上限 ----
    for k in types:
        dep = [y[(k, i)] for i in ids if (k, i) in y]
        if dep:
            prob += pulp.lpSum(dep) <= RES[k]["span"] * n[k]

    # ---- 人力（关键：用排班可用率把名义人数折算为有效人力）----
    # 有效可用人时 <= personnel * staff_avail
    prob += pulp.lpSum(RES[k]["staff"] * n[k] for k in types) <= personnel * staff_avail
    # ---- 预算 ----
    prob += pulp.lpSum(RES[k]["cost"] * n[k] for k in types) <= budget

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    n_val = {k: int(round(pulp.value(n[k]) or 0)) for k in types}
    u_val = np.array([pulp.value(u[i]) or 0.0 for i in ids])
    P = weighted_P(u_val, np.array([w[i] for i in ids]))
    y_val = {(k, i): (pulp.value(y[(k, i)]) or 0.0) for (k, i) in y}
    cost = sum(RES[k]["cost"] * n_val[k] for k in types)
    staff = sum(RES[k]["staff"] * n_val[k] for k in types)

    out = dict(status=status, P=P, n=n_val, u=u_val, y=y_val, ids=ids,
               cells=cells, cost=cost, staff=staff,
               coverage=(u_val > 0.35).mean())
    if verbose:
        print(f"[{status}] P = {P:.4f} | 成本 {cost:.1f}万$ | 人力 {staff:.1f}人 "
              f"| 有效覆盖格数 {(u_val>0.35).sum()}/{len(ids)}")
        print("   采购:", {k: v for k, v in n_val.items() if v > 0})
    return out


if __name__ == "__main__":
    r = build_and_solve()
