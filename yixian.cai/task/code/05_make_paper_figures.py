# -*- coding: utf-8 -*-
"""
把 dataset_based_results.json 做成论文可用的学术图表
输出: ../figures/ 下 4 张图，每张 PNG(300dpi) + PDF(矢量)

图1 规模分布直方图（96站点功率）
图2 耗水强度差异（核心发现：755倍）——对数轴柱状图
图3 中国站点 vs 美国超大站点规模对比
图4 水资源→算力上限关系

风格：学术、克制配色、中文标签、300dpi
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(BASE, ".."))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# ---------- 中文字体 ----------
FONT_PATH = "/System/Library/Fonts/PingFang.ttc"
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
else:
    plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11

# 配色（学术克制）
C_BLUE = "#2E5E8C"
C_BLUE_L = "#8FB3D9"
C_CORAL = "#C05A3A"
C_GRAY = "#6B7280"
C_TEAL = "#1D8A72"

with open(os.path.join(ROOT, "outputs", "dataset_based_results.json"), encoding="utf-8") as f:
    R = json.load(f)
D = pd.read_csv(os.path.join(ROOT, "clean", "data_centers_clean.csv"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  生成 {name}.png / {name}.pdf")


# ============ 图1：规模分布 ============
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    p = D["power_MW"].dropna()
    p = p[p > 0]
    n, bins, patches = ax.hist(p, bins=28, color=C_BLUE_L, edgecolor=C_BLUE, linewidth=0.6, alpha=0.9)
    # 渐变：高功率段加深
    for i, (patch, left) in enumerate(zip(patches, bins[:-1])):
        if left >= 500:
            patch.set_facecolor(C_BLUE)
    med = R["scale_distribution"]["median_power_MW"]
    mx = R["scale_distribution"]["max_power_MW"]
    ax.axvline(med, color=C_CORAL, linestyle="--", linewidth=1.4, label=f"中位数 {med:.0f} MW")
    ax.annotate(f"最大 {mx:.0f} MW\n(New Carlisle)", xy=(mx, 0), xytext=(mx*0.55, n.max()*0.55),
                fontsize=9, color=C_CORAL,
                arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=1.1))
    ax.set_xlabel("单站 IT 功率 (MW)")
    ax.set_ylabel("站点数量")
    ax.set_title("图1  全球最大 AI 数据中心规模分布（Epoch AI，96 站点）")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "图1_规模分布")


# ============ 图2：耗水强度差异（核心） ============
def fig2():
    pairs = R["water_power_pairs"]
    names = [x["data_center"].replace("Anthropic-Amazon ", "").replace("Microsoft ", "")
             for x in pairs]
    vals = [max(x["w_per_IT"], 1e-4) for x in pairs]  # 对数轴保护
    colors = [C_CORAL if v >= 1 else (C_TEAL if v < 0.05 else C_BLUE) for v in vals]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, alpha=0.9, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("单位 IT 耗水强度 w_per-IT (m³/MWh，对数轴)")
    med = R["water_intensity_onsite"]["median_m3_per_MWh"]
    ax.axvline(med, color=C_GRAY, linestyle="--", linewidth=1.2,
               label=f"中位数 {med:.3f}")
    ax.set_title("图2  真实现场耗水强度差异（15 条记录，极差约 755 倍）")
    ax.legend(frameon=False, loc="lower right")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.52, 0.04, "红 ≥1（高耗水）　蓝 0.05–1　绿 <0.05（近无水冷却）",
            transform=ax.transAxes, fontsize=8.5, va="bottom", ha="center", color=C_GRAY)
    save(fig, "图2_耗水强度差异")


# ============ 图3：中国站点 vs 美国超大 ============
def fig3():
    cn = R["china_sites"]
    names = [x["site"] for x in cn]
    pw = [x["power_MW"] for x in cn]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    y = np.arange(len(names))
    ax.barh(y, pw, color=C_TEAL, alpha=0.85, height=0.62, label="中国站点（华北/内蒙古）")
    # 美国最大站点参考线
    usmax = R["scale_distribution"]["max_power_MW"]
    ax.axvline(usmax, color=C_CORAL, linestyle="--", linewidth=1.5,
               label=f"美国最大站点 {usmax:.0f} MW")
    for i, v in enumerate(pw):
        ax.text(v + 5, i, f"{v:.0f}", va="center", fontsize=9, color=C_TEAL)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("IT 功率 (MW)")
    ax.set_title("图3  数据集里中国 5 站点规模 vs 全球最大站点")
    ax.legend(frameon=False, loc="lower right")
    ax.invert_yaxis()
    ax.set_xlim(0, usmax * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "图3_中国站点规模")


# ============ 图4：水资源→算力上限 ============
def fig4():
    up = R["water_based_upper"]
    labels = list(up.keys())
    pmw = [up[k]["P_MW"] for k in labels]
    comp = [up[k]["compute_H100e"] / 1e3 for k in labels]  # K H100e
    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(labels))
    b = ax1.bar(x, pmw, width=0.5, color=C_BLUE, alpha=0.85, label="可支撑功率 P (MW)")
    ax1.set_ylabel("可支撑功率 (MW)", color=C_BLUE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("年可用水量")
    for i, v in enumerate(pmw):
        # 功率标签放柱内顶部（白色字），避免与折线点重叠
        ax1.text(i, v*0.5, f"{v:,.0f}", ha="center", va="center", fontsize=9,
                 color="white", fontweight="bold")
    ax2 = ax1.twinx()
    ax2.plot(x, comp, "o-", color=C_CORAL, linewidth=2, markersize=7,
             label="可支撑算力 (K H100e)")
    ax2.set_ylabel("可支撑算力 (K H100e)", color=C_CORAL)
    ax2.set_ylim(0, max(comp)*1.18)
    for i, v in enumerate(comp):
        ax2.annotate(f"{v:,.0f}", xy=(i, v), xytext=(0, 10),
                     textcoords="offset points", ha="center", fontsize=9, color=C_CORAL)
    ax1.set_title("图4  水资源可支撑的最大算力\n（现场耗水中位数 0.195 m³/MWh，算力密度 1137 H100e/MW）")
    # 合并图例
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, frameon=False, loc="upper left")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    save(fig, "图4_水到算力上限")


def main():
    print("生成论文图表，输出目录:", FIG)
    fig1()
    fig2()
    fig3()
    fig4()
    print("\n完成。每张图均有 300dpi PNG 与矢量 PDF 两种格式。")


if __name__ == "__main__":
    main()
