# -*- coding: utf-8 -*-
"""
完整分析套件 v2（修复版）
==========================
修复要点：
  1. 收紧资源 span + 物理饱和上限 U_MAX -> 边际递减真实存在
  2. 人力排班可用率 η -> "人力需求"分析有意义
  3. 新增边际效益分析 -> 回答"需要多少护林员"这一核心问题
"""
import json
import math
import os

import numpy as np


class _NpEncoder(json.JSONEncoder):
    """让 numpy 标量/数组可序列化。"""
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _sanitize(obj):
    """递归把 tuple 键转为字符串（JSON 要求键为 str/int/float/bool/None）。"""
    if isinstance(obj, dict):
        return {("__".join(map(str, k)) if isinstance(k, tuple) else k): _sanitize(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

from model_core import (GRID_N, RES, OUT, STAFF_AVAILABILITY, U_MAX,
                        add_priority, build_cells, efficacy,
                        terrain_factor, weighted_P)
from solve_single import build_and_solve


# =====================================================================
# §3 多期动态
# =====================================================================
def seasonal_threat_profile(periods=12):
    """威胁时间调制 σ_t：旱季(6-10月)盗猎高发；旅游旺季冲突上升；月相影响。"""
    t = np.arange(periods)
    dry = 0.5 + 0.5 * np.cos(2 * np.pi * (t - 6) / periods)
    tourism = 0.4 + 0.6 * np.exp(-((t - 7) ** 2) / 8)
    moon = 0.55 + 0.45 * np.cos(2 * np.pi * t / 2.5)
    sigma = 0.35 + 0.65 * (0.55 * dry + 0.25 * tourism + 0.20 * moon)
    return np.clip(sigma, 0.25, 1.25)


def temporal_protection(deploy, periods=12):
    """给定部署，计算各期保护指数 P_t（含风险暴露时间维度）。"""
    ids = deploy["ids"]
    cells = {c["id"]: c for c in deploy["cells"]}
    w = np.array([cells[i]["w"] for i in ids])
    n = deploy["n"]
    sigma = seasonal_threat_profile(periods)

    Ps = []
    for t in range(periods):
        eta = 0.70 + 0.25 * np.cos(2 * np.pi * (t - 3) / periods)     # 排班效率
        weather = 0.78 + 0.22 * np.cos(2 * np.pi * (t - 6) / periods)  # 旱季沙尘限无人机
        u = np.zeros(len(ids))
        for idx, i in enumerate(ids):
            tot = 0.0
            for k, info in RES.items():
                if n[k] <= 0:
                    continue
                yki = deploy["y"].get((k, i), 0.0)
                if yki <= 0:
                    continue
                tf = terrain_factor(k, cells[i])
                if k in ("foot_patrol", "vehicle_patrol"):
                    mod = eta
                elif k == "drone_unit":
                    mod = weather
                else:
                    mod = 1.0
                tot += yki * info["kappa"] * tf * mod
            u[idx] = tot / sigma[t]           # 威胁升高稀释相对保护
        Ps.append(weighted_P(u, w))
    return np.array(Ps), sigma


# =====================================================================
# §4 鲁棒优化
# =====================================================================
def robust_analysis(deploy, gamma_levels=(0, 0.25, 0.5, 0.75, 1.0)):
    """
    威胁不确定集：名义威胁向上浮动 Δ 时保护指数的下界。
    借鉴 Bertsimas-Sim 预算不确定集：Γ 比例的高风险格子同时偏离。
    """
    ids = deploy["ids"]
    cells = {c["id"]: c for c in deploy["cells"]}
    w = np.array([cells[i]["w"] for i in ids])
    u0 = deploy["u"]
    delta = 0.60            # 威胁浮动 ±60%（盗猎情报误差的现实量级）

    out = []
    for gam in gamma_levels:
        k_full = int(np.floor(gam * len(ids)))
        frac = gam * len(ids) - k_full
        order = np.argsort(-(w * u0))        # 影响最大的格子优先偏离
        u_w = u0.copy()
        for rank, idx in enumerate(order):
            if rank < k_full:
                u_w[idx] = u0[idx] / (1 + delta)
            elif rank == k_full:
                u_w[idx] = u0[idx] / (1 + delta * frac)
        Pn = weighted_P(u0, w); Pw = weighted_P(u_w, w)
        out.append(dict(gamma=gam, P_nom=Pn, P_worst=Pw, ratio=Pw / Pn))
    return out


# =====================================================================
# §5 敏感性 & 边际效益
# =====================================================================
def marginal_value_analysis():
    """核心决策问题：每增加一名护林员，保护指数提升多少？何时饱和？"""
    print("\n" + "=" * 62)
    print("§5a 边际效益：人力投入 vs 保护水平（回答'需要多少护林员'）")
    print("=" * 62)
    Hs = [60, 80, 100, 120, 150, 180, 210, 250, 295, 350, 420]
    rows = []
    prev_P = None
    for H in Hs:
        r = build_and_solve(personnel=H, verbose=False)
        dP = None if prev_P is None else r["P"] - prev_P
        rows.append(dict(H=H, P=r["P"], staff=r["staff"], cost=r["cost"],
                         dP_per_10=None if prev_P is None else round(dP * 10 / 20, 5)))
        s = f"  H={H:>4}人  P={r['P']:.4f}  实配{r['staff']:.0f}人  成本{r['cost']:.0f}万"
        if dP is not None:
            s += f"   ΔP={dP:+.4f}"
        print(s)
        prev_P = r["P"]
    return rows


def sensitivity_analysis():
    print("\n" + "=" * 62)
    print("§5b 敏感性：单一资源移除")
    print("=" * 62)
    base = build_and_solve(verbose=False)
    rows = []
    for k in RES:
        r = build_and_solve(kappa_scale={t: (0.0 if t == k else 1.0) for t in RES},
                            verbose=False)
        rows.append(dict(resource=k, P=r["P"], dP=r["P"] - base["P"],
                         price=RES[k]["cost"]))
        print(f"  移除 {k:<16} P={r['P']:.4f}  ΔP={r['P']-base['P']:+.4f}  (单价${RES[k]['cost']}k)")

    print("\n  --- 预算敏感性 ---")
    budget_rows = []
    for B in (100, 150, 200, 250, 300, 400, 600, 900):
        r = build_and_solve(budget=B, verbose=False)
        budget_rows.append(dict(B=B, P=r["P"], cost=r["cost"]))
        print(f"  预算 ${B:>4}k  P={r['P']:.4f}  实耗 ${r['cost']:.0f}k")

    print("\n  --- 排班可用率敏感性（后勤效率；在人力紧张场景 H=150 下评估）---")
    avail_rows = []
    for av in (0.50, 0.60, 0.72, 0.80, 0.90, 1.00):
        r = build_and_solve(personnel=150, staff_avail=av, verbose=False)
        avail_rows.append(dict(avail=av, P=r["P"], staff=r["staff"], H=150))
        print(f"  η={av:.2f} (H=150)  P={r['P']:.4f}  实配{r['staff']:.0f}人")
    return dict(base=base, remove=rows, budget=budget_rows, avail=avail_rows)


def scenario_analysis():
    print("\n" + "=" * 62)
    print("§5c 场景分析：多因素叠加压力测试")
    print("=" * 62)
    scenarios = {
        "Base (295 staff, $900k)": {},
        "Budget -50% ($450k)": dict(budget=450),
        "Budget -70% ($270k)": dict(budget=270),
        "Drones grounded": dict(kappa_scale={t: (0.0 if t == "drone_unit" else 1.0) for t in RES}),
        "Satellite outage": dict(kappa_scale={t: (0.0 if t == "satellite_feed" else 1.0) for t in RES}),
        "Ground patrol lost": dict(
            kappa_scale={t: (0.0 if t in ("foot_patrol", "vehicle_patrol") else 1.0) for t in RES}),
        "Tight staff + no drones": dict(personnel=150,
                                        kappa_scale={t: (0.0 if t == "drone_unit" else 1.0) for t in RES}),
    }
    out = {}
    for name, kw in scenarios.items():
        r = build_and_solve(verbose=False, **kw)
        out[name] = dict(P=r["P"], cost=r["cost"], staff=r["staff"], n=r["n"])
        print(f"  {name:<30} P={r['P']:.4f}  成本${r['cost']:.0f}k  人力{r['staff']:.0f}")
    return out


# =====================================================================
# §6 跨保护区适配
# =====================================================================
def adaptation_analysis():
    print("\n" + "=" * 62)
    print("§6 跨保护区适配（结构不变，仅重校准输入）")
    print("=" * 62)
    parks = [
        dict(name="Etosha National Park", continent="Africa", area=22935,
             terrain=0.10, road_density=0.155, visibility=0.05,
             threat="Rhino/elephant poaching, wildfire",
             base_P=0.615),
        dict(name="Kaziranga National Park", continent="Asia", area=1090,
             terrain=0.45, road_density=0.30, visibility=0.55,
             threat="Rhino/tiger poaching, monsoon flooding",
             base_P=None),
        dict(name="Yellowstone N.P.", continent="N. America", area=8983,
             terrain=0.60, road_density=0.12, visibility=0.50,
             threat="Illegal collection, wildfire, visitor pressure",
             base_P=None),
    ]
    out = []
    P_BASE = 0.797   # Etosha 基准保护指数（由 §3 基准解得到）
    for p in parks:
        road_deficit = 1.0 - min(1.0, p["road_density"] / 0.35)
        D = 0.5 * p["terrain"] + 0.3 * road_deficit + 0.2 * p["visibility"]
        P_pred = P_BASE * (1 - 0.30 * D)
        staff_f = 1.0 + 0.9 * D
        budget_f = 1.0 + 0.7 * D
        est_staff = 295 * staff_f
        rec = dict(name=p["name"], continent=p["continent"], area=p["area"],
                   threat=p["threat"], difficulty=round(D, 3),
                   P_pred=round(P_pred, 3), staff_factor=round(staff_f, 2),
                   budget_factor=round(budget_f, 2),
                   est_staff=round(est_staff, 0))
        out.append(rec)
        print(f"  {p['name']:<26}({p['continent']:<11}) D={D:.2f}  "
              f"P≈{P_pred:.3f}  人力需求x{staff_f:.2f}(≈{est_staff:.0f}人)  预算x{budget_f:.2f}")
    return out


# =====================================================================
# 主入口
# =====================================================================
if __name__ == "__main__":
    report = {}

    print("=" * 62)
    print("§3 基准解：单期战略部署（Etosha）")
    print("=" * 62)
    base = build_and_solve()
    report["base"] = dict(P=base["P"], cost=base["cost"], staff=base["staff"], n=base["n"])

    print("\n" + "=" * 62)
    print("§4 多期动态：季节性保护水平")
    print("=" * 62)
    Ps, sigma = temporal_protection(base)
    print(f"  月度 P_t: mean={Ps.mean():.4f}  min={Ps.min():.4f}(期{Ps.argmin()+1}) "
          f" max={Ps.max():.4f}(期{Ps.argmax()+1})")
    print(f"  威胁 σ_t: mean={sigma.mean():.3f}  peak={sigma.max():.3f}(期{sigma.argmax()+1})")
    print(f"  最脆弱月份风险暴露 = {1-Ps.min():.3f}")
    report["temporal"] = dict(Ps=Ps.tolist(), sigma=sigma.tolist(),
                              P_mean=float(Ps.mean()), P_min=float(Ps.min()),
                              P_max=float(Ps.max()))

    print("\n" + "=" * 62)
    print("§4b 鲁棒优化：威胁不确定集")
    print("=" * 62)
    rob = robust_analysis(base)
    for r in rob:
        print(f"  Γ={r['gamma']:.2f}  P_nom={r['P_nom']:.4f}  "
              f"P_worst={r['P_worst']:.4f}  保持率={r['ratio']*100:.1f}%")
    report["robust"] = rob

    marg = marginal_value_analysis()
    report["marginal"] = marg

    sens = sensitivity_analysis()
    report["sensitivity"] = sens

    scen = scenario_analysis()
    report["scenarios"] = scen

    adapt = adaptation_analysis()
    report["adaptation"] = adapt

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(_sanitize(report), f, indent=2, cls=_NpEncoder)
    print(f"\n[结果已保存] {OUT}/results.json")
