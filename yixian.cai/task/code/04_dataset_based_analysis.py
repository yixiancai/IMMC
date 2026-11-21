# -*- coding: utf-8 -*-
"""
基于数据集真实站点的分析（不使用假设的南京/西北情景）
输入: ../clean/*.csv (Epoch AI 真实数据)
输出: ../outputs/dataset_based_results.json

回答：以数据集真实数据为准，能得到什么结论？
"""
import os
import json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.normpath(os.path.join(BASE, "..", "clean"))
OUT = os.path.normpath(os.path.join(BASE, "..", "outputs"))
os.makedirs(OUT, exist_ok=True)

D = pd.read_csv(os.path.join(CLEAN, "data_centers_clean.csv"))
T = pd.read_csv(os.path.join(CLEAN, "timelines_clean.csv"))

results = {}

# ---- 结论1：真实规模分布 ----
results["scale_distribution"] = {
    "n_sites": int(len(D)),
    "total_power_GW": round(float(D["power_MW"].sum()) / 1000, 2),
    "total_compute_H100e": float(D["h100e"].sum()),
    "median_power_MW": round(float(D["power_MW"].median()), 1),
    "max_power_MW": float(D["power_MW"].max()),
    "max_power_site": str(D.loc[D["power_MW"].idxmax(), "Name"]),
    "median_compute_density": round(float(D["h100e_per_MW"].median()), 0),
}

# ---- 结论2：真实耗水强度（仅现场冷却口径）----
w = T[T["w_per_IT"].notna() & (T["w_per_IT"] > 0)]["w_per_IT"]
results["water_intensity_onsite"] = {
    "n_records": int(len(w)),
    "median_m3_per_MWh": round(float(w.median()), 3),
    "min": round(float(w.min()), 4),
    "max": round(float(w.max()), 3),
    "note": "现场冷却耗水反推，不含发电耗水；与UNU含发电口径9.84不同",
}
# 有耗水记录的站点的真实水-功率配对
pair = T[T["w_per_IT"].notna() & (T["w_per_IT"] > 0)][
    ["data_center", "it_power_MW", "water_MGD", "w_per_IT"]].copy()
pair["w_per_IT"] = pair["w_per_IT"].round(3)
results["water_power_pairs"] = pair.to_dict("records")

# ---- 结论3：真实中国站点（数据集里真实存在）----
cn = D[D["Country"] == "China"][["Name", "power_MW", "h100e", "cost_B", "h100e_per_MW"]].copy()
cn.columns = ["site", "power_MW", "h100e", "cost_B", "compute_density"]
results["china_sites"] = cn.round(2).to_dict("records")
results["china_summary"] = {
    "n": int(len(cn)),
    "total_power_MW": round(float(cn["power_MW"].sum()), 1),
    "median_power_MW": round(float(cn["power_MW"].median()), 1),
}

# ---- 结论4：真实站点能支撑的最大算力（基于水承载上限）----
# 用真实站点的现场耗水记录反推：给定_site的实测耗水，算力上限 = 水/单算力耗水
rho = float(D["h100e_per_MW"].median())
for site in ["Anthropic-Amazon New Carlisle", "Google Pryor (North)"]:
    rec = T[(T["data_center"] == site) & (T["w_per_IT"].notna())]
    if len(rec):
        r = rec.iloc[0]
        results.setdefault("site_water_examples", {})[site] = {
            "IT_power_MW": float(r["it_power_MW"]),
            "water_MGD": float(r["water_MGD"]),
            "water_m3_per_year": round(float(r["water_m3_per_year"]), 0),
            "w_per_IT": round(float(r["w_per_IT"]), 3),
            "compute_H100e": round(float(D[D["Name"] == site]["h100e"].iloc[0]), 0),
        }

# ---- 结论5：用真实耗水中位数外推的承载上限 ----
# 若一个站点现场耗水强度取真实中位数0.20，给定水量能撑多大？
w_med = float(w.median())
H_EFF = 7884.0
for water_m3 in [3e6, 3e7, 1e8]:
    p_upper = water_m3 / (H_EFF * w_med)
    results.setdefault("water_based_upper", {})[f"{water_m3/1e4:.0f}万m3"] = {
        "P_MW": round(p_upper, 0),
        "compute_H100e": round(p_upper * rho, 0),
    }

with open(os.path.join(OUT, "dataset_based_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)

# ---- 打印结论 ----
print("=" * 68)
print("以数据集真实数据为准的结论（非假设情景）")
print("=" * 68)
s = results["scale_distribution"]
print(f"\n[规模] {s['n_sites']}个真实站点, 总功率{s['total_power_GW']}GW, "
      f"总算力{s['total_compute_H100e']/1e6:.2f}M H100e")
print(f"       功率中位数{s['median_power_MW']}MW, 最大{s['max_power_MW']}MW ({s['max_power_site']})")
print(f"       算力密度中位数{s['median_compute_density']:.0f} H100e/MW")
wi = results["water_intensity_onsite"]
print(f"\n[耗水] 真实现场耗水记录{wi['n_records']}条, 强度中位数{wi['median_m3_per_MWh']}m3/MWh")
print(f"       范围{wi['min']}~{wi['max']} (差约{wi['max']/max(wi['min'],0.01):.0f}倍) -> 冷却技术决定耗水")
print(f"\n[中国站点] {results['china_summary']['n']}个真实站点, 合计{results['china_summary']['total_power_MW']}MW")
for r in results["china_sites"]:
    print(f"       {r['site']:26} {r['power_MW']:6.1f}MW  {r['h100e']/1e3:6.0f}K H100e")
print("\n[水→算力上限] 用真实现场耗水中位数0.20反推:")
for k, v in results["water_based_upper"].items():
    print(f"       {k:12}: {v['P_MW']:7.0f}MW -> {v['compute_H100e']/1e3:7.0f}K H100e")
print(f"\n结果已保存: {os.path.join(OUT, 'dataset_based_results.json')}")
