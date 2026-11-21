# -*- coding: utf-8 -*-
"""
IMMC 2027 A 题任务一/二 —— 数据清洗脚本
输入: ../datasets/ 下的 5 个原始 CSV (Epoch AI)
输出: ../clean/ 下的标准化 CSV，供建模代码直接读取

清洗内容:
  1. 主表: 数值列转 float，缺失文本填 'Unknown'，计算算力密度
  2. 时间线: 日期转标准格式，耗水 MGD 转 m3/年，反推单位耗水强度
  3. 芯片表: 数量列转 float，去重
  4. 冷水机/冷却塔: 数值列转 float，面积列补全，剔除全空行
  5. 生成 data_dictionary.json 数据字典
"""
import os
import json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.normpath(os.path.join(BASE, "..", "datasets"))
CLEAN = os.path.normpath(os.path.join(BASE, "..", "clean"))
os.makedirs(CLEAN, exist_ok=True)

# 单位换算常数
MGD_TO_M3Y = 3.78541e6 * 365 / 1000   # 1 MGD -> m^3/year
LF = 0.9                                # 负载率
HOURS = 8760                            # 年运行小时
HE = LF * HOURS                         # 等效满负荷小时 = 7884


def num(s):
    """安全转数值：无法解析的置 NaN。"""
    return pd.to_numeric(s, errors="coerce")


def clean_main():
    df = pd.read_csv(os.path.join(RAW, "data_centers.csv"))
    rename = {
        "Current H100 equivalents": "h100e",
        "Current power (MW)": "power_MW",
        "Current total capital cost (2025 USD billions)": "cost_B",
        "Current chip types": "chip_types",
        "All chip types": "all_chip_types",
    }
    df = df.rename(columns=rename)
    for c in ["h100e", "power_MW", "cost_B"]:
        df[c] = num(df[c])
    # 算力密度 H100e / MW
    df["h100e_per_MW"] = np.where(df["power_MW"] > 0,
                                  df["h100e"] / df["power_MW"], np.nan)
    # 文本列缺失填充
    for c in ["Owner", "Users", "Project", "chip_types", "all_chip_types",
              "Investors", "Construction companies", "Energy companies", "Address"]:
        df[c] = df[c].fillna("Unknown")
    df = df.sort_values("power_MW", ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(CLEAN, "data_centers_clean.csv"), index=False)
    return df


def clean_timelines():
    df = pd.read_csv(os.path.join(RAW, "data_center_timelines.csv"))
    rename = {
        "Data center": "data_center",
        "Date": "date",
        "Construction status": "status",
        "Buildings operational": "buildings_op",
        "IT power (MW)": "it_power_MW",
        "Power (MW)": "power_MW",
        "H100 equivalents": "h100e",
        "Performance (8-bit OP/s)": "perf_8bit",
        "Total capital cost (2025 USD billions)": "cost_B",
        "Compute cost (2025 USD billions)": "compute_cost_B",
        "Construction cost (2025 USD billions)": "construction_cost_B",
        "Annual operating cost (2025 USD billions)": "op_cost_B",
        "Water use (MGD)": "water_MGD",
        "Current H100 equivalents (from Data center)": "h100e_current",
    }
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    for c in ["buildings_op", "it_power_MW", "power_MW", "h100e", "perf_8bit",
              "cost_B", "compute_cost_B", "construction_cost_B", "op_cost_B",
              "water_MGD", "h100e_current"]:
        df[c] = num(df[c])
    # 耗水 MGD -> m^3/year（现场冷却耗水，非发电耗水）
    df["water_m3_per_year"] = df["water_MGD"] * MGD_TO_M3Y
    # 反推单位 IT 耗水强度 (m3/MWh)，仅当耗水与功率均>0
    df["w_per_IT"] = np.where(
        (df["water_m3_per_year"] > 0) & (df["it_power_MW"] > 0),
        df["water_m3_per_year"] / (df["it_power_MW"] * HE), np.nan)
    # 状态长文本截断为关键词
    def short_status(s):
        if not isinstance(s, str):
            return "unknown"
        s = s.lower()
        if "operational" in s or "live" in s:
            return "operational"
        if "land clearing" in s or "site preparation" in s or "ground broken" in s:
            return "land_clearing"
        if "roof" in s or "construction" in s or "under construction" in s:
            return "under_construction"
        return "other"
    df["status_simple"] = df["status"].apply(short_status)
    df = df.sort_values(["data_center", "date"]).reset_index(drop=True)
    df.to_csv(os.path.join(CLEAN, "timelines_clean.csv"), index=False)
    return df


def clean_chips():
    df = pd.read_csv(os.path.join(RAW, "data_centers_chip_quantities.csv"))
    rename = {
        "Handle": "handle", "Date": "date", "Chip type": "chip_type",
        "Number of Units": "units", "Chip type source": "chip_type_src",
        "Number of Units source": "units_src", "Owner": "owner", "User": "user",
        "Notes": "notes", "Data center": "data_center", "Timeline": "timeline",
    }
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["units"] = num(df["units"])
    df = df.drop_duplicates(subset=["data_center", "date", "chip_type", "units"])
    for c in ["owner", "user", "notes", "chip_type_src", "units_src"]:
        df[c] = df[c].fillna("Unknown")
    df.to_csv(os.path.join(CLEAN, "chip_quantities_clean.csv"), index=False)
    return df


def clean_cooling(fn, out):
    df = pd.read_csv(os.path.join(RAW, fn))
    df = df.dropna(how="all")
    # 把所有数值型列转 float
    for c in df.columns:
        if any(k in c for k in ["kW", "(m)", "(m^2)", "Diameter", "Fans", "Horsepower", "fans"]):
            df[c] = num(df[c])
    # 面积列若缺失但有长宽则补算
    if {"Length (m)", "Width (m)", "Area (m^2)"}.issubset(df.columns):
        missing = df["Area (m^2)"].isna() & df["Length (m)"].notna() & df["Width (m)"].notna()
        df.loc[missing, "Area (m^2)"] = df.loc[missing, "Length (m)"] * df.loc[missing, "Width (m)"]
    for c in ["Source", "Link", "Spot check", "Notes", "Type", "Manufacturer", "Model"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")
    df.to_csv(os.path.join(CLEAN, out), index=False)
    return df


def main():
    d = clean_main()
    t = clean_timelines()
    c = clean_chips()
    ch = clean_cooling("data_center_chillers.csv", "chillers_clean.csv")
    ct = clean_cooling("data_center_cooling_towers.csv", "cooling_towers_clean.csv")

    dictionary = {
        "generation": "2026-11-21",
        "source": "Epoch AI AI data centers database (CC-BY 4.0), https://epoch.ai/data/ai-data-centers",
        "constants": {"LF": LF, "HOURS": HOURS, "H_eff": HE,
                      "MGD_to_m3_per_year": MGD_TO_M3Y},
        "files": {
            "data_centers_clean.csv": {
                "rows": len(d), "desc": "96个AI数据中心主表",
                "key_cols": ["Name", "power_MW", "h100e", "cost_B", "h100e_per_MW", "Country"],
                "notes": "h100e_per_MW=算力密度(中位数约1137); power_MW=IT功率"},
            "timelines_clean.csv": {
                "rows": len(t), "desc": "建设时间线(逐站点逐事件)",
                "key_cols": ["data_center", "date", "year", "it_power_MW", "h100e", "water_MGD",
                             "water_m3_per_year", "w_per_IT", "status_simple"],
                "notes": "water_MGD是现场冷却耗水; w_per_IT由此反推(仅现场,不含发电耗水); "
                         "w_per_IT中位数约0.20,与UNU含发电的全口径9.84不同"},
            "chip_quantities_clean.csv": {
                "rows": len(c), "desc": "芯片数量明细", "key_cols": ["data_center", "date", "chip_type", "units"]},
            "chillers_clean.csv": {
                "rows": len(ch), "desc": "冷水机规格(冷却能力kW/占地m2)",
                "key_cols": ["Model", "Manufacturer", "Cooling capacity (kW)", "Area (m^2)"]},
            "cooling_towers_clean.csv": {
                "rows": len(ct), "desc": "冷却塔规格(冷却能力kW/占地m2)",
                "key_cols": ["Model", "Cooling capacity (central estimate, kW)", "Area (m^2)"]},
        },
    }
    with open(os.path.join(CLEAN, "data_dictionary.json"), "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)

    print("清洗完成，输出目录:", CLEAN)
    for k, v in dictionary["files"].items():
        print(f"  {k}: {v['rows']} 行")
    print("\n关键校验:")
    print(f"  主表算力密度中位数: {d['h100e_per_MW'].median():.0f} H100e/MW")
    wt = t["w_per_IT"].dropna()
    print(f"  现场耗水强度中位数: {wt.median():.3f} m3/MWh (有效{len(wt)}条)")
    print(f"  New Carlisle 现场耗水: {t[(t.data_center=='Anthropic-Amazon New Carlisle')]['water_MGD'].max():.1f} MGD")


if __name__ == "__main__":
    main()
