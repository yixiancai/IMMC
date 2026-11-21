# -*- coding: utf-8 -*-
"""
任务二：接入真实外部数据（碳强度 + 地理坐标）后的分布式网络优化
=====================================================================
数据源（用户下载，位于 ../datasets_task2/）：
  - owid_carbon_intensity_full.csv : 各国电网碳强度 gCO2/kWh (Ember/OWID, CC-BY)
  - country_centroids.csv          : 各国质心经纬度 (360-info/geoboundaries)
  - ../clean/data_centers_clean.csv: 节点算力/成本/密度/国家 (Epoch AI, CC-BY)

相对旧模型(07)的改进：
  1) 环境足迹 g_j 用真实电网碳强度（不再固定9.84）
  2) 时延用真实大圆距离推算（不再同国5/跨国15的粗代理）
  3) 新增"碳感知调度"：优先把负载放到低碳国家

输出：../outputs/task2_realdata_results.json

运行：python3 08_task2_with_real_data.py
"""
import os, json, math
import numpy as np
import pandas as pd
import pulp

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(BASE, ".."))
T2 = os.path.join(ROOT, "datasets_task2")
CLEAN = os.path.join(ROOT, "clean")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

# =====================================================================
# 1. 读取三份数据
# =====================================================================
D = pd.read_csv(os.path.join(CLEAN, "data_centers_clean.csv"))
CARB = pd.read_csv(os.path.join(T2, "owid_carbon_intensity_full.csv"))
GEO = pd.read_csv(os.path.join(T2, "country_centroids.csv"))

# 按国家选代表节点（每国取算力最大站点），保证地理/碳强度多样性
Dv = D[D["h100e"] > 0].copy()
Dv = Dv.sort_values("h100e", ascending=False)
nodes = Dv.groupby("Country", as_index=False).head(1).reset_index(drop=True)
N_NODES = len(nodes)

# 碳强度：取2024年最新
carb_2024 = CARB[CARB["year"] == 2024].set_index("entity")["co2_intensity__gco2_kwh"].to_dict()
# 国家名标准化（数据集国家名 → 碳数据entity名，多数一致）
name_map = {"United States": "United States", "China": "China", "Malaysia": "Malaysia",
            "Norway": "Norway", "United Kingdom": "United Kingdom", "Finland": "Finland",
            "Indonesia": "Indonesia", "Australia": "Australia", "Portugal": "Portugal",
            "Iceland": "Iceland", "United Arab Emirates": "United Arab Emirates", "Sweden": "Sweden"}
# 质心坐标按 name 索引
geo_by_name = GEO.set_index("name")[["lat", "lon"]].to_dict("index")

# =====================================================================
# 2. 构造节点参数（真实数据驱动）
# =====================================================================
c = nodes["h100e"].values
b = nodes["cost_B"].fillna(nodes["cost_B"].median()).values
rho = nodes["h100e_per_MW"].fillna(nodes["h100e_per_MW"].median()).values
countries = nodes["Country"].values
names = nodes["Name"].values

# 2.1 碳强度 g_j (gCO2/kWh) —— 真实数据
g_carbon = np.array([carb_2024.get(name_map.get(cn, cn), 400.0) for cn in countries])

# 2.2 地理坐标 → 大圆距离 → 时延
def get_latlon(cn):
    key = name_map.get(cn, cn)
    if key in geo_by_name:
        return geo_by_name[key]["lat"], geo_by_name[key]["lon"]
    return None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

coords = [get_latlon(cn) for cn in countries]
# 时延代理：光纤中光速约 2e5 km/s，往返+路由开销系数 → 约 0.01 ms/km + 固定开销
LATENCY = np.zeros((N_NODES, N_NODES))
DIST = np.zeros((N_NODES, N_NODES))
for i in range(N_NODES):
    for j in range(N_NODES):
        if coords[i][0] is None or coords[j][0] is None:
            DIST[i,j] = 0.0; LATENCY[i,j] = 0.0
            continue
        d = haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
        DIST[i,j] = d
        LATENCY[i,j] = 2*d/200.0 + 5.0   # ms，含路由开销

# 网络能耗代价：随距离增大（归一化）
NETWORK_UNIT = DIST / (DIST.max() + 1)

# 水承载上限（沿用任务一）
H_EFF = 7884.0; W_PER_IT = 9.84; Q_PERMIT = 3e6
water_cap_H100e = (Q_PERMIT/(H_EFF*W_PER_IT)) * rho

# =====================================================================
# 3. MIP 模型（碳感知 + 时延硬约束 + 公平性）
# =====================================================================
LAM_NET, LAM_CARBON, LAM_FAIR = 0.5, 10.0, 2.0
BUDGET = 400.0
TAU_MAX = 150.0
TASK_DEMAND = 300e3

def solve(fairness=True, carbon_aware=True, sla=True):
    prob = pulp.LpProblem("Task2_RealData", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(N_NODES)]
    y = [pulp.LpVariable(f"y_{j}", lowBound=0) for j in range(N_NODES)]

    obj = pulp.lpSum(b[j]*x[j] for j in range(N_NODES))            # 启用成本
    obj += LAM_NET*pulp.lpSum(NETWORK_UNIT[0][j]*y[j]/1e3 for j in range(N_NODES))
    if carbon_aware:
        obj += LAM_CARBON*pulp.lpSum(g_carbon[j]*y[j]/1e5 for j in range(N_NODES))
    # 碳盲模型：目标无碳项，只看成本+网络+公平

    if fairness:
        zmax = pulp.LpVariable("zmax", lowBound=0)
        zmin = pulp.LpVariable("zmin", lowBound=0)
        for j in range(N_NODES):
            util = y[j]*(1.0/max(c[j],1))
            prob += util <= zmax
            prob += util >= zmin - 1.0*(1-x[j])   # 大M线性化
        obj += LAM_FAIR*(zmax-zmin)*1e3
    prob += obj, "Objective"

    prob += pulp.lpSum(y) >= TASK_DEMAND, "Demand"
    for j in range(N_NODES):
        prob += y[j] <= c[j]*x[j], f"Cap_{j}"
        prob += y[j] <= water_cap_H100e[j]*x[j], f"Water_{j}"
        if sla and LATENCY[0][j] > TAU_MAX:
            prob += y[j] <= 0.2*TASK_DEMAND*x[j], f"SLA_{j}"
    prob += pulp.lpSum(b[j]*x[j] for j in range(N_NODES)) <= BUDGET, "Budget"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    ok = pulp.LpStatus[status] == "Optimal"
    if not ok:
        return dict(feasible=False, status=pulp.LpStatus[status])
    sel = [j for j in range(N_NODES) if x[j].varValue > 0.5]
    alloc = {names[j]: round(y[j].varValue or 0, 1) for j in sel}
    total = sum(y[j].varValue or 0 for j in range(N_NODES))
    utils = np.array([(y[j].varValue or 0)/c[j] for j in sel])
    gini = (np.abs(utils[:,None]-utils[None,:]).sum()/(2*len(utils)**2*utils.mean())
            if len(utils)>1 and utils.mean()>0 else 0.0)
    # 加权平均碳强度（按分配算力）
    alloc_carbon = sum((y[j].varValue or 0)*g_carbon[j] for j in sel)/max(total,1)
    return dict(feasible=True, n_nodes=len(sel), selected=[names[j] for j in sel],
                allocation=alloc, total_H100e=total, gini=round(float(gini),4),
                avg_carbon_g_per_kwh=round(float(alloc_carbon),1))

def main():
    base = solve(carbon_aware=True)
    blind = solve(carbon_aware=False)
    # 单集中式基准：最大站点
    single_carbon = g_carbon[np.argmax(c)]
    results = dict(
        data_sources={
            "carbon":"Ember via Our World in Data (owid_carbon_intensity_full.csv, CC-BY)",
            "geo":"360-info country-centroids (country_centroids.csv)",
            "nodes":"Epoch AI (data_centers_clean.csv, CC-BY)"},
        parameters=dict(N_NODES=N_NODES, BUDGET=BUDGET, TASK_DEMAND=TASK_DEMAND,
                        TAU_MAX=TAU_MAX, carbon_data_year=2024),
        node_carbon={names[i]: round(float(g_carbon[i]),1) for i in range(N_NODES)},
        node_coordinates={names[i]: [round(coords[i][0],2) if coords[i][0] else None,
                                      round(coords[i][1],2) if coords[i][1] else None]
                          for i in range(N_NODES)},
        carbon_aware_model=base,
        carbon_blind_model=blind,
        single_site_benchmark=dict(site=names[int(np.argmax(c))],
                                   carbon_g_per_kwh=round(float(single_carbon),1)),
        note="时延由大圆距离推算(2d/200+5ms)；碳强度为国家级2024年数据，非站点实测",
    )
    with open(os.path.join(OUT,"task2_realdata_results.json"),"w",encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print("="*66); print("任务二：接入真实碳强度+坐标数据后的结果"); print("="*66)
    print(f"碳感知模型: {base['n_nodes']}节点, Gini={base['gini']}, "
          f"平均碳强度={base['avg_carbon_g_per_kwh']} gCO2/kWh")
    print(f"碳盲模型  : {blind['n_nodes']}节点, Gini={blind['gini']}, "
          f"平均碳强度={blind['avg_carbon_g_per_kwh']} gCO2/kWh")
    print(f"单站点基准: {results['single_site_benchmark']['site']}, "
          f"{results['single_site_benchmark']['carbon_g_per_kwh']} gCO2/kWh")
    print("\n碳感知启用节点:")
    for nm,amt in base["allocation"].items():
        j=list(names).index(nm)
        print(f"  {nm[:36]:38} {amt/1e3:6.1f}K  (碳 {g_carbon[j]:.0f} g/kWh)")
    reduction = (results['single_site_benchmark']['carbon_g_per_kwh']-base['avg_carbon_g_per_kwh'])/results['single_site_benchmark']['carbon_g_per_kwh']*100
    print(f"\n相对单站点碳强度下降: {reduction:.1f}%")
    print(f"结果: {os.path.join(OUT,'task2_realdata_results.json')}")

if __name__ == "__main__":
    main()
