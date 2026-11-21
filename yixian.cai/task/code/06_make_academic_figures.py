# -*- coding: utf-8 -*-
"""
=====================================================================
学术论文规范图表生成代码
IMMC 2027 A题 · 任务一/二 · 基于 Epoch AI 真实数据
=====================================================================
规范：
  1) 语言：Python (matplotlib + seaborn)，完整可运行，带注释；
  2) 字体：中文宋体(Songti SC)，英文/数字 Times New Roman；
  3) 字号：图标题12，坐标轴标签11，刻度/图例10；
  4) 配色：学术低饱和，黑白印刷可区分（辅以灰阶/线型）；
  5) 边框：保留上、右坐标轴边框；
  6) 坐标轴：带清晰标签与单位，图例不遮挡数据；
  7) 导出：.eps / .pdf / .svg 矢量图，dpi>=300；
  8) 图注：见文件末尾 FIGURE_CAPTIONS，可直接粘贴到论文。

运行：python3 06_make_academic_figures.py
输出：../figures_academic/ 下 4 张图 × 3 种矢量格式
=====================================================================
"""
import os
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 无界面后端，服务器/终端均可运行
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns

# ---------------------------------------------------------------------
# 0. 全局样式设置（严格按学术规范）
# ---------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(BASE, ".."))
OUTDIR = os.path.join(ROOT, "figures_academic")
os.makedirs(OUTDIR, exist_ok=True)

# 0.1 注册并配置字体：中文宋体，西文 Times New Roman
# matplotlib 按列表顺序查找字形：西文字符优先用 Times New Roman，
# 中文字符 Times New Roman 无字形时自动回退到宋体。
SONG_PATH = "/System/Library/Fonts/Supplemental/Songti.ttc"
TNR_PATH = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
for p in (SONG_PATH, TNR_PATH):
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams["font.family"] = ["Times New Roman", "Songti SC", "serif"]
plt.rcParams["axes.unicode_minus"] = False   # 负号正常显示

# 0.2 字号规范
plt.rcParams["font.size"] = 10                # 默认/刻度/图例 10
plt.rcParams["axes.titlesize"] = 12           # 图标题 12
plt.rcParams["axes.labelsize"] = 11           # 坐标轴标签 11
plt.rcParams["legend.fontsize"] = 10          # 图例 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

# 0.3 学术低饱和配色（色盲友好 + 灰阶可区分）
# 用深色/浅灰/线型组合，保证黑白打印仍能区分
C_DARK = "#37474F"     # 深石板灰（主数据）
C_MID = "#78909C"      # 中灰
C_LIGHT = "#B0BEC5"    # 浅灰
C_ACCENT = "#1A5F7A"   # 低饱和蓝（强调）
C_WARM = "#A64B3A"     # 低饱和砖红（强调2）

# 0.4 seaborn 样式：白底、保留全边框、灰色网格
sns.set_theme(style="ticks", rc={
    "font.family": ["Times New Roman", "Songti SC", "serif"],
    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
})
sns.set_context("paper", font_scale=1.0)

# ---------------------------------------------------------------------
# 1. 读取数据
# ---------------------------------------------------------------------
with open(os.path.join(ROOT, "outputs", "dataset_based_results.json"), encoding="utf-8") as f:
    R = json.load(f)
D = pd.read_csv(os.path.join(ROOT, "clean", "data_centers_clean.csv"))

SAVE_FORMATS = ["eps", "pdf", "svg"]


def save_figure(fig, name):
    """按规范导出：矢量图三种格式，dpi>=300"""
    fig.tight_layout()
    for fmt in SAVE_FORMATS:
        fig.savefig(
            os.path.join(OUTDIR, f"{name}.{fmt}"),
            format=fmt, dpi=300, bbox_inches="tight", facecolor="white"
        )
    plt.close(fig)
    print(f"  已导出 {name} (.eps/.pdf/.svg)")


def style_axes(ax):
    """统一坐标轴样式：保留上右边框，刻度朝内"""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#000000")
    ax.tick_params(direction="in", length=4, width=0.8)


# ---------------------------------------------------------------------
# 图1：AI 数据中心规模分布（直方图 + 累计分布）
# ---------------------------------------------------------------------
def figure_1():
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    p = D["power_MW"].dropna()
    p = p[p > 0]

    # 直方图：低饱和灰，密度归一化
    sns.histplot(p, bins=26, color=C_LIGHT, edgecolor=C_DARK,
                 linewidth=0.7, stat="density", ax=ax, label="频数密度")

    # 累计分布（次坐标轴，线型区分）
    ax2 = ax.twinx()
    xs = np.sort(p)
    cdf = np.arange(1, len(xs) + 1) / len(xs)
    ax2.plot(xs, cdf, color=C_ACCENT, linewidth=1.6, linestyle="-.",
             label="累计分布")

    # 中位数参考线
    med = R["scale_distribution"]["median_power_MW"]
    ax.axvline(med, color=C_WARM, linestyle="--", linewidth=1.3,
               label=f"中位数 = {med:.0f} MW")

    ax.set_xlabel("单站 IT 功率 P_IT (MW)")
    ax.set_ylabel("频数密度")
    ax2.set_ylabel("累计比例")
    ax.set_title("图1  全球最大 AI 数据中心 IT 功率分布")
    ax.set_xlim(0, 1100)

    # 合并图例，置于右上不遮挡主峰
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True,
              edgecolor="#999999", fancybox=False)

    style_axes(ax)
    style_axes(ax2)
    save_figure(fig, "图1_规模分布_学术版")


# ---------------------------------------------------------------------
# 图2：真实现场耗水强度差异（核心图，对数轴条形图）
# ---------------------------------------------------------------------
def figure_2():
    pairs = R["water_power_pairs"]
    # 简化站点名以便阅读
    names = [x["data_center"].replace("Anthropic-Amazon ", "")
             .replace("Microsoft ", "").replace("Google ", "G-")
             .replace("AWS ", "").replace("OpenAI ", "O-") for x in pairs]
    vals = np.array([max(x["w_per_IT"], 1e-4) for x in pairs])

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    y = np.arange(len(names))

    # 用灰度 + 线型区分三档耗水强度（黑白打印可辨）
    colors = []
    hatches = []
    for v in vals:
        if v >= 1:            # 高耗水：深灰 + 斜线
            colors.append(C_DARK); hatches.append("//")
        elif v >= 0.05:       # 中耗水：中灰 + 竖线
            colors.append(C_MID); hatches.append("|")
        else:                 # 近无水：浅灰 + 横线
            colors.append(C_LIGHT); hatches.append("--")
    bars = ax.barh(y, vals, color=colors, edgecolor="#000000",
                   linewidth=0.6, height=0.68)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("单位 IT 耗水强度 w (m³/MWh，对数轴)")
    ax.set_ylabel("数据中心站点")
    ax.set_title("图2  真实现场冷却耗水强度的分布与差异")

    # 中位数线
    med = R["water_intensity_onsite"]["median_m3_per_MWh"]
    ax.axvline(med, color=C_WARM, linestyle=":", linewidth=1.4,
               label=f"中位数 {med:.3f} m$^3$/MWh")

    # 自定义图例（用 Patch）
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=C_DARK, edgecolor="#000000", hatch="//", label="w ≥ 1  高耗水"),
        Patch(facecolor=C_MID, edgecolor="#000000", hatch="|", label="0.05 ≤ w < 1  中耗水"),
        Patch(facecolor=C_LIGHT, edgecolor="#000000", hatch="--", label="w < 0.05  近无水冷却"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True,
              edgecolor="#999999", fancybox=False, fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    style_axes(ax)
    save_figure(fig, "图2_耗水强度差异_学术版")


# ---------------------------------------------------------------------
# 图3：中国站点规模 vs 全球最大站点（水平条形图）
# ---------------------------------------------------------------------
def figure_3():
    cn = R["china_sites"]
    names = [x["site"] for x in cn]
    pw = np.array([x["power_MW"] for x in cn])

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(names))
    ax.barh(y, pw, color=C_MID, edgecolor=C_DARK, linewidth=0.7,
            height=0.6, label="中国站点（华北/内蒙古）")

    # 数值标注
    for i, v in enumerate(pw):
        ax.text(v + 4, i, f"{v:.0f}", va="center", fontsize=9)

    # 全球最大站点参考线
    usmax = R["scale_distribution"]["max_power_MW"]
    ax.axvline(usmax, color=C_WARM, linestyle="--", linewidth=1.4,
               label=f"全球最大站点 {usmax:.0f} MW")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("IT 功率 P_IT (MW)")
    ax.set_ylabel("数据中心站点")
    ax.set_title("图3  数据集内中国站点规模与全球最大站点对比")
    ax.set_xlim(0, usmax * 1.18)
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=True, edgecolor="#999999", fancybox=False)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    style_axes(ax)
    save_figure(fig, "图3_中国站点规模_学术版")


# ---------------------------------------------------------------------
# 图4：水资源可支撑的最大算力（双轴图）
# ---------------------------------------------------------------------
def figure_4():
    up = R["water_based_upper"]
    labels = list(up.keys())
    pmw = np.array([up[k]["P_MW"] for k in labels])
    comp = np.array([up[k]["compute_H100e"] / 1e3 for k in labels])  # K H100e

    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(labels))

    # 柱状：可支撑功率（浅灰 + 深色边）
    bars = ax1.bar(x, pmw, width=0.5, color=C_LIGHT, edgecolor=C_DARK,
                   linewidth=0.8, label="可支撑功率 (MW)")
    ymax = max(pmw)
    for b, v in zip(bars, pmw):
        # 矮柱标签放柱顶上方，高柱放柱内中部，避免与折线点重叠
        if v < ymax * 0.1:
            ax1.text(b.get_x() + b.get_width()/2, v + ymax*0.02, f"{v:,.0f}",
                     ha="center", va="bottom", fontsize=9, color="#000000")
        else:
            ax1.text(b.get_x() + b.get_width()/2, v*0.5, f"{v:,.0f}",
                     ha="center", va="center", fontsize=9, color="#000000")

    ax1.set_ylabel("可支撑 IT 功率 P_W,max (MW)")
    ax1.set_xlabel("年可用水资源量 W_supply")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    # 折线：可支撑算力（深色 + 实线 + 圆点）
    ax2 = ax1.twinx()
    ax2.plot(x, comp, marker="o", color=C_ACCENT, linewidth=1.8,
             markersize=6, label="可支撑算力 (K H100e)")
    ax2.set_ylabel("可支撑算力 (K H100e)")
    ax2.set_ylim(0, max(comp) * 1.18)
    for i, v in enumerate(comp):
        ax2.annotate(f"{v:,.0f}", xy=(i, v), xytext=(0, 9),
                     textcoords="offset points", ha="center", fontsize=9,
                     color=C_ACCENT)

    ax1.set_title("图4  水资源约束下的最大算力上限\n（现场耗水中位数法，算力密度 1137 H100e/MW）")

    # 合并图例置于左上
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True,
               edgecolor="#999999", fancybox=False)

    style_axes(ax1)
    style_axes(ax2)
    save_figure(fig, "图4_水到算力上限_学术版")


# ---------------------------------------------------------------------
# 图注（Figure captions）—— 可直接粘贴到论文
# ---------------------------------------------------------------------
FIGURE_CAPTIONS = """
图1  全球最大 AI 数据中心 IT 功率分布。
样本为 Epoch AI 开放数据库收录的 96 个最大 AI 数据中心（截至 2026 年 11 月）。
横轴为单站 IT 功率（MW），主纵轴为频数密度（直方图，26 组），次纵轴为累计比例。
虚线为样本中位数 118 MW，最大值为 1037 MW（Anthropic-Amazon New Carlisle）。
采用核密度以外的频率直方图与经验累计分布进行描述性统计。

图2  真实现场冷却耗水强度的分布与差异。
样本为 Epoch AI 时间线数据中报告现场耗水量（MGD）的 15 条有效记录。
耗水强度 w = 年耗水量 / (IT 功率 × 年等效满负荷小时 7884)，单位 m³/MWh，横轴取对数。
按强度分三档：w≥1（斜线）、0.05≤w<1（竖线）、w<0.05（横线），以保证黑白印刷可区分。
点线为中位数 0.195 m³/MWh；样本极差约 755 倍（0.001–7.548），反映冷却技术差异。

图3  数据集内中国站点规模与全球最大站点对比。
样本为 Epoch AI 数据库中 5 个中国数据中心站点（均位于华北/内蒙古）。
横轴为 IT 功率（MW）；竖虚线为全球最大站点 1037 MW 参考值。
中国站点功率区间为 118–221 MW，显著低于全球最大单体，支持缺水地区中小规模布局的判断。

图4  水资源约束下的最大算力上限。
基于现场耗水强度中位数 0.195 m³/MWh 与算力密度中位数 1137 H100e/MW 外推。
柱形（左轴）为可支撑 IT 功率（MW），折线（右轴）为可支撑算力（K H100e）。
三个水资源情景为 300 万、3000 万、10000 万 m³/年。
统计方法为确定性外推：P_W,max = W_supply / (H_eff × w)，算力 = P × 密度；未计不确定性区间。
"""


def main():
    print("生成学术论文规范图表（宋体 + Times New Roman，矢量导出）")
    print("输出目录:", OUTDIR)
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    with open(os.path.join(OUTDIR, "图注_FigureCaptions.txt"), "w", encoding="utf-8") as f:
        f.write(FIGURE_CAPTIONS)
    print("\n图注已保存至 图注_FigureCaptions.txt")
    print("全部完成：4 张图 × 3 种矢量格式（.eps/.pdf/.svg），dpi>=300。")


if __name__ == "__main__":
    main()
