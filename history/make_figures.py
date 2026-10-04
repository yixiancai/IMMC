# -*- coding: utf-8 -*-
"""
§7 可视化输出（v2，适配修复后模型）
===================================
图 1-8，覆盖 IM2C 题目要求的"视觉/结构化元素"：
  1 空间优先级热力图（问题1：保护的定义与优先级）
  2 最优部署与战略集中（问题2：资源分配）
  3 保护水平随时间（问题3：时间维度）
  4 人力边际效益（问题3：人员需求 —— 核心决策曲线）
  5 敏感性分析三联图（问题4）
  6 鲁棒性（问题4：不确定性）
  7 场景压力测试（问题4）
  8 跨大陆适配（问题6）
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from model_core import GRID_N, RES, OUT, U_MAX, add_priority, build_cells

plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False,
                     "figure.dpi": 110, "savefig.dpi": 150})

CMAP_RISK = LinearSegmentedColormap.from_list(
    "risk", ["#e8f5e9", "#fff59d", "#ef9a9a", "#c62828"])
CMAP_COV = LinearSegmentedColormap.from_list(
    "cov", ["#ffffff", "#c8e6c9", "#66bb6a", "#1b5e20"])


def grid_matrix(cells, key):
    M = np.full((GRID_N, GRID_N), np.nan)
    for c in cells:
        M[c["row"], c["col"]] = c[key]
    return M


def fig1_priority(cells):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (k, title, cm) in zip(axes, [
            ("eco", "Ecological value $e_i$", "YlGn"),
            ("poach", "Poaching pressure $p_i$", "OrRd"),
            ("w", "Protection priority $w_i$", CMAP_RISK)]):
        M = grid_matrix(cells, k)
        im = ax.imshow(M, cmap=cm, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("grid column"); ax.set_ylabel("grid row")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Fig. 1  Spatial heterogeneity of Etosha "
                 "(12×12 grid, ~159 km$^2$/cell; centre = salt pan)", fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "fig1_priority_map.png"); fig.savefig(p); plt.close(fig)
    return p


def fig2_deployment(base, tight=None):
    """双场景对比：充足预算 vs 紧张预算，展示战略集中随资源稀缺性的变化。"""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

    def cov_map(ax, deploy, title):
        M = np.full((GRID_N, GRID_N), np.nan)
        for c, uv in zip(deploy["cells"], deploy["u"]):
            M[c["row"], c["col"]] = uv
        im = ax.imshow(M, cmap=CMAP_COV, vmin=0, vmax=U_MAX)
        ax.set_title(f"{title}\nP = {deploy['P']:.3f}, cost = ${deploy['cost']:.0f}k", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        return im

    cov_map(axes[0], base, "(a) Full budget deployment")
    if tight is not None:
        cov_map(axes[1], tight, "(b) Stressed budget (30%) deployment")

    # (c) 优先级-覆盖关系（两场景）
    w = np.array([c["w"] for c in base["cells"]])
    axes[2].scatter(w, base["u"], s=30, facecolor="none", edgecolor="#1b5e20",
                    linewidth=0.9, label=f"full budget (slope={np.polyfit(w, base['u'], 1)[0]:.2f})")
    if tight is not None:
        axes[2].scatter(w, tight["u"], s=30, c="#c62828", marker="^",
                        linewidth=0.4, alpha=0.75,
                        label=f"stressed budget (slope={np.polyfit(w, tight['u'], 1)[0]:.2f})")
    axes[2].set_xlabel("protection priority $w_i$")
    axes[2].set_ylabel("coverage intensity $u_i$")
    axes[2].set_title("(c) Coverage concentrates on priority cells")
    axes[2].legend(fontsize=8.5); axes[2].grid(alpha=0.3)

    fig.suptitle("Fig. 2  Optimal strategic deployment under two budget regimes", fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "fig2_deployment.png"); fig.savefig(p); plt.close(fig)
    return p


def fig3_temporal(res):
    Ps = np.array(res["temporal"]["Ps"]); sigma = np.array(res["temporal"]["sigma"])
    t = np.arange(1, len(Ps) + 1)
    months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    fig, ax1 = plt.subplots(figsize=(9, 4.6))
    ax1.plot(t, Ps, "o-", color="#1b5e20", lw=2, label="Protection index $P_t$")
    ax1.axhline(Ps.mean(), color="#1b5e20", ls=":", alpha=0.6, label=f"mean {Ps.mean():.3f}")
    ax1.fill_between(t, Ps, Ps.min(), color="#1b5e20", alpha=0.08)
    ax1.set_xlabel("Month"); ax1.set_ylabel("Protection index $P_t$", color="#1b5e20")
    ax1.set_ylim(0, 1); ax1.set_xticks(t); ax1.set_xticklabels(months, fontsize=8)
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(t, sigma, "s--", color="#c62828", alpha=0.8, label=r"Threat level $\sigma_t$")
    ax2.set_ylabel(r"Threat modulation $\sigma_t$", color="#c62828")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
    imin = int(Ps.argmin())
    ax1.annotate("dry-season trough", xy=(t[imin], Ps[imin]),
                 xytext=(t[imin] + 0.6, Ps[imin] - 0.16),
                 arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8, color="gray")
    ax1.set_title("Fig. 3  Protection over time vs. seasonal threat")
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_temporal.png"); fig.savefig(p); plt.close(fig)
    return p


def fig4_marginal(res):
    m = res["marginal"]
    H = [r["H"] for r in m]; P = [r["P"] for r in m]
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.plot(H, P, "o-", color="#1b5e20", lw=2.2, ms=6, label="Protection index $P$")
    ax1.set_xlabel("Ranger personnel available, $H$")
    ax1.set_ylabel("Protection index $P$", color="#1b5e20")
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    dP = np.gradient(P, H)
    ax2.plot(H, dP, "s--", color="#c62828", lw=1.8, alpha=0.85,
             label=r"marginal value $\partial P/\partial H$")
    ax2.set_ylabel(r"Marginal value $\partial P/\partial H$", color="#c62828")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9)
    ax1.set_title("Fig. 4  Marginal value of personnel: how many rangers are needed?")
    fig.tight_layout()
    p = os.path.join(OUT, "fig4_marginal_personnel.png"); fig.savefig(p); plt.close(fig)
    return p


def fig5_sensitivity(res):
    sens = res["sensitivity"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    rows = sens["remove"]
    names = [r["resource"] for r in rows]; dP = [r["dP"] for r in rows]
    order = np.argsort(dP)
    axes[0].barh([names[i] for i in order], [dP[i] for i in order], color="#c62828")
    axes[0].axvline(0, color="k", lw=0.8); axes[0].grid(alpha=0.3, axis="x")
    axes[0].set_xlabel(r"$\Delta P$ if resource removed ($\times10^{3}$)")
    axes[0].set_title("(a) Criticality of each resource")
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    br = sens["budget"]
    B = [r["B"] for r in br]; Pb = [r["P"] for r in br]
    axes[1].plot(B, Pb, "s-", color="#1565c0", lw=2)
    axes[1].set_xlabel("Annual budget (\\$k)"); axes[1].set_ylabel("Protection index $P$")
    axes[1].set_title("(b) Protection vs. budget"); axes[1].grid(alpha=0.3)

    av = sens["avail"]
    A = [r["avail"] for r in av]; Pa = [r["P"] for r in av]
    axes[2].plot(A, Pa, "^-", color="#6a1b9a", lw=2)
    axes[2].set_xlabel("Logistics availability $\\eta$")
    axes[2].set_ylabel("Protection index $P$")
    axes[2].set_title("(c) Protection vs. scheduling efficiency")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Fig. 5  Sensitivity analysis", fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "fig5_sensitivity.png"); fig.savefig(p); plt.close(fig)
    return p


def fig6_robust(res):
    rob = res["robust"]
    g = [r["gamma"] for r in rob]; pn = [r["P_nom"] for r in rob]
    pw = [r["P_worst"] for r in rob]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(g, pn, "o-", color="#1b5e20", lw=2, label="Nominal $P$")
    ax.plot(g, pw, "s-", color="#c62828", lw=2, label="Worst-case $P$")
    ax.fill_between(g, pw, pn, alpha=0.18, color="orange", label="Robustness gap")
    ax.set_xlabel(r"Uncertainty budget $\Gamma$ (fraction of cells deviating)")
    ax.set_ylabel("Protection index $P$")
    ax.set_title("Fig. 6  Robustness to threat uncertainty (budget uncertainty set)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1)
    fig.tight_layout()
    p = os.path.join(OUT, "fig6_robust.png"); fig.savefig(p); plt.close(fig)
    return p


def fig7_scenarios(res):
    sc = res["scenarios"]
    names = list(sc.keys()); P = [sc[k]["P"] for k in names]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    colors = ["#1b5e20" if i == 0 else "#c62828" for i in range(len(names))]
    bars = ax.bar(range(len(names)), P, color=colors, alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("Protection index $P$")
    ax.set_title("Fig. 7  Scenario analysis: multi-factor stress tests")
    for b, v in zip(bars, P):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=8.5)
    ax.axhline(P[0], color="gray", ls=":", alpha=0.7)
    ax.set_ylim(0, 1); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = os.path.join(OUT, "fig7_scenarios.png"); fig.savefig(p); plt.close(fig)
    return p


def fig8_adaptation(res):
    ad = res["adaptation"]
    names = [a["name"].replace(" National Park", "").replace(" N.P.", "") for a in ad]
    conts = [a["continent"] for a in ad]
    D = [a["difficulty"] for a in ad]; Pp = [a["P_pred"] for a in ad]
    sf = [a["staff_factor"] for a in ad]; bf = [a["budget_factor"] for a in ad]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(names))
    axes[0].bar(x - 0.2, Pp, 0.4, label="Predicted $P$", color="#1b5e20")
    axes[0].bar(x + 0.2, D, 0.4, label="Terrain difficulty $D$", color="#f9a825")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{n}\n({c})" for n, c in zip(names, conts)], fontsize=8.5)
    axes[0].set_title("(a) Predicted protection vs. difficulty")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x - 0.2, sf, 0.4, label="Staff multiplier", color="#1565c0")
    axes[1].bar(x + 0.2, bf, 0.4, label="Budget multiplier", color="#6a1b9a")
    axes[1].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{n}\n({c})" for n, c in zip(names, conts)], fontsize=8.5)
    axes[1].set_title("(b) Required resource multipliers")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("Fig. 8  Cross-continental adaptation (structure fixed, inputs re-calibrated)",
                 fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, "fig8_adaptation.png"); fig.savefig(p); plt.close(fig)
    return p


def make_all():
    with open(os.path.join(OUT, "results.json")) as f:
        res = json.load(f)
    cells = add_priority(build_cells())
    from solve_single import build_and_solve
    base = build_and_solve(verbose=False)
    tight = build_and_solve(budget=270, verbose=False)   # 30% 预算的紧张场景
    paths = [fig1_priority(cells), fig2_deployment(base, tight), fig3_temporal(res),
             fig4_marginal(res), fig5_sensitivity(res), fig6_robust(res),
             fig7_scenarios(res), fig8_adaptation(res)]
    print("生成图表:")
    for p in paths:
        print("  ", p)
    return paths


if __name__ == "__main__":
    make_all()
