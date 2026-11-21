# -*- coding: utf-8 -*-
"""
任务一：集中式 AI 数据中心最大可持续规模（承载上限）模型
输入: ../clean/*.csv (清洗后的 Epoch AI 数据)
输出: ../outputs/task1_results.json + 控制台报告

模型（v4 真实数据版）:
  P_max(t) = min[P_E,max(t), P_W,max(t), P_L,max(t)]
  P_W,max(t) = min[W_avail(t), Q_permit(t)] / (H_eff * w_per_IT)
  P_E,max = C_grid * eta_firm / PUE + P_self(t)   (含7年燃机交付延迟)
  P_L,max = A_avail*(1-alpha_eco) / (a_DC + a_power)
逐期取最小值点 P_max* = min_t P_max(t)
"""
import os
import json
import math
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.normpath(os.path.join(BASE, "..", "clean"))
OUT = os.path.normpath(os.path.join(BASE, "..", "outputs"))
os.makedirs(OUT, exist_ok=True)

D = pd.read_csv(os.path.join(CLEAN, "data_centers_clean.csv"))
T = pd.read_csv(os.path.join(CLEAN, "timelines_clean.csv"))

# ---------- 参数（来自 v4 论文 / 数据字典） ----------
H_EFF = 7884.0          # 等效满负荷小时
PUE = 1.2               # [假设]
ETA_FIRM = 0.3          # 风光可靠容量系数 [假设]
GRID_CAPACITY = 2500.0  # 电网可分配容量 MW（西北情景）[假设]
W_PER_IT = 9.84         # 全球平均强度 UNU 反推 [真实]
W_PER_IT_SERIES = {"traditional": 9.84, "liquid": 6.3, "liquid_wind_solar": 3.5}
SELF_BUILD_MW = 500.0   # 自建燃机 MW
SELF_BUILD_YEAR = 3     # 7年交付的简化：第3年投运
Q_PERMIT_0 = 3e6        # 初始批水量 m3/年（西北情景）
GAMMA = 0.02            # 批水量年收紧率
CHI = 0.5               # 水压力折减系数
WSI = 0.6               # 西北水压力指数
A_AVAIL = 5e6           # 可用土地 m2 (500公顷)
ALPHA_ECO = 0.1         # 生态敏感占比
A_DC = 300.0            # 本体占地 m2/MW
A_POWER = 500.0         # 配套电力占地 m2/MW（电网供电情景）


def water_upper(W_avail, w_per_it, q_permit_0, year=0):
    """水资源上限 P_W,max (MW)：物理水与批水量取小（制度闸门）"""
    Q_permit = q_permit_0 * (1 - GAMMA) ** year * (1 - CHI * WSI)
    W_supply = min(W_avail, Q_permit)
    return W_supply / (H_EFF * w_per_it), W_supply


def energy_upper(year, grid_remain=GRID_CAPACITY * ETA_FIRM / PUE, allow_self=True):
    """能源上限 P_E,max (MW)，含自建燃机交付延迟"""
    p_self = SELF_BUILD_MW if (allow_self and year >= SELF_BUILD_YEAR) else 0.0
    return grid_remain + p_self


def land_upper():
    """土地上限 P_L,max (MW)"""
    return A_AVAIL * (1 - ALPHA_ECO) / (A_DC + A_POWER)


def scenario(name, W_avail, q_permit_0, grid_remain, years=(0, 3, 5, 8, 10)):
    """单场景逐年计算三类上限与综合上限"""
    rows = []
    for k in years:
        p_w, w_supply = water_upper(W_avail, W_PER_IT, q_permit_0, k)
        p_e = energy_upper(k, grid_remain)
        p_l = land_upper()
        p_max = min(p_e, p_w, p_l)
        bottleneck = {p_e: "energy", p_w: "water", p_l: "land"}[p_max]
        rows.append(dict(year=k, W_avail=W_avail, W_supply=w_supply,
                         P_E=p_e, P_W=p_w, P_L=p_l, P_max=p_max,
                         bottleneck=bottleneck))
    return rows


def main():
    # 场景 A：南京丰水（W_avail=2.75亿 m3，批水量充足，能源瓶颈来自电网剩余304MW）
    nj = scenario("Nanjing", 2.75e8, q_permit_0=4e8, grid_remain=304.0)
    # 场景 B：西北缺水（W_avail=300万 m3，批水量300万，能源625MW）
    xb = scenario("Northwest", 3e6, q_permit_0=3e6, grid_remain=625.0)

    # 技术敏感性（西北，不同 w_per_it，批水量300万）
    tech = {}
    for tname, w in W_PER_IT_SERIES.items():
        p, ws = water_upper(3e6, w, q_permit_0=3e6, year=0)
        tech[tname] = dict(w_per_IT=w, W_supply=ws, P_W=p)

    # 临界批水量（使 P_W = 能源上限 304MW）
    p_e_base = 304.0
    q_star = p_e_base * H_EFF * W_PER_IT / (1 - CHI * WSI)
    q_star_liquid = p_e_base * H_EFF * W_PER_IT_SERIES["liquid"] / (1 - CHI * WSI)

    # 算力换算
    rho = D["h100e_per_MW"].median()
    max_compute_nj = max(r["P_max"] for r in nj) * rho
    max_compute_xb = max(r["P_max"] for r in xb) * rho

    results = dict(
        parameters=dict(H_eff=H_EFF, PUE=PUE, eta_firm=ETA_FIRM,
                        grid_capacity_MW=GRID_CAPACITY, w_per_IT=W_PER_IT,
                        Q_permit_0=Q_PERMIT_0, chi=CHI, WSI=WSI),
        scenario_Nanjing=nj,
        scenario_Northwest=xb,
        technology_sensitivity=tech,
        critical_permit=dict(global_w=f"{q_star/1e4:.0f}万m3/年",
                             liquid=f"{q_star_liquid/1e4:.0f}万m3/年"),
        compute_density=rho,
        max_compute=dict(Nanjing_H100e=max_compute_nj,
                         Northwest_H100e=max_compute_xb),
        global_compute_from_data=float(D["h100e"].sum()),
    )
    with open(os.path.join(OUT, "task1_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 报告
    print("=" * 64)
    print("任务一：集中式承载上限模型结果")
    print("=" * 64)
    print(f"{'场景':<10}{'年':>4}{'P_E':>9}{'P_W':>9}{'P_L':>9}{'P_max':>9}  瓶颈")
    for name, rows in [("南京", nj), ("西北", xb)]:
        for r in rows:
            print(f"{name:<10}{r['year']:>4}{r['P_E']:>9.0f}{r['P_W']:>9.0f}"
                  f"{r['P_L']:>9.0f}{r['P_max']:>9.0f}  {r['bottleneck']}")
    print("\n技术敏感性(西北):")
    for k, v in tech.items():
        print(f"  {k:20} w={v['w_per_IT']:.2f}  P_W={v['P_W']:.0f} MW")
    print(f"\n临界批水量: 全球口径 {q_star/1e4:.0f}万m3/年, 液冷 {q_star_liquid/1e4:.0f}万m3/年")
    print(f"算力密度: {rho:.0f} H100e/MW")
    print(f"最大可支撑算力: 南京 {max_compute_nj/1e3:.0f}K H100e, 西北 {max_compute_xb/1e3:.0f}K H100e")
    print(f"数据集中全球算力总和: {D['h100e'].sum()/1e6:.2f}M H100e")
    print(f"\n结果已保存: {os.path.join(OUT, 'task1_results.json')}")


if __name__ == "__main__":
    main()
