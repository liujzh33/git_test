"""Part 2 figures: published leaderboards redrawn from the papers' tables."""
import numpy as np

from common import *  # noqa: F401,F403

# Sekuboyina et al., MedIA 2021, Table 4 — hidden test phase (id.rate %, Dice %).
VERSE19 = {
    "Payer C.": (94.25, 89.80), "Lessmann N.": (90.42, 85.76), "Chen M.": (86.73, 82.56),
    "Amiranashvili T.": (73.32, 68.96), "Dong Y.": (67.21, 67.51), "Angermann C.": (54.85, 46.40),
    "Sekuboyina A.（基线）": (87.66, 83.18),
}
VERSE20 = {
    "Chen D.": (96.58, 91.23), "Payer C.": (92.82, 89.71), "Zhang A.": (96.22, 89.36),
    "Yeah T.": (94.65, 87.91), "Xiangshang Z.": (82.08, 85.07), "Hou F.": (90.47, 84.92),
    "Netherton T.": (89.08, 78.26), "Jakubicek R.": (49.54, 52.97),
    "Sekuboyina A.（基线）": (86.06, 79.52),
}
WINNER = {"VerSe'19": "Payer C.", "VerSe'20": "Payer C."}

# Label placement (dx, dy in font sizes, ha); points in the top-right corner are crowded.
LEFT, RIGHT = (-1.0, 0, "right"), (1.0, 0, "left")
PLACE = {
    "VerSe'19": {"Payer C.": LEFT, "Chen M.": (-1.0, -0.3, "right"),
                 "Sekuboyina A.（基线）": (0, -1.3, "center")},
    "VerSe'20": {"Chen D.": (-1.0, 0.5, "right"), "Payer C.": (-1.0, -0.5, "right"),
                 "Zhang A.": RIGHT, "Yeah T.": (0, -1.3, "center"), "Hou F.": (0, -1.3, "center"),
                 "Xiangshang Z.": LEFT, "Sekuboyina A.（基线）": LEFT},
}


def fig_verse_leader():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.0), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, (name, d) in zip(axes, [("VerSe'19", VERSE19), ("VerSe'20", VERSE20)]):
        for team, (idr, dice) in d.items():
            base = "基线" in team
            win = team == WINNER[name]
            ax.scatter(idr, dice, s=46 if win else 30, zorder=3,
                       color=BONE if win else ("white" if base else FILM),
                       edgecolors=SLATE if base else "none", linewidths=1)
            dx, dy, ha = PLACE[name].get(team, RIGHT)
            ax.annotate(team + ("（冠军）" if win else ""), (idr, dice), xytext=(dx, dy),
                        textcoords="offset fontsize", fontsize=7.5, ha=ha, va="center",
                        color=BONE if win else (SLATE if base else INK),
                        fontweight="bold" if win else "normal")
        ax.set_title(f"{name} 隐藏测试集", loc="left")
        ax.set_xlabel("椎骨识别率 id.rate (%)")
        ax.set_xlim(45, 100)
        ax.grid(color=RULE, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Dice (%)")
    axes[0].set_ylim(40, 95)
    fig.savefig(ASSETS / "verse_leader.svg")
    plt.close(fig)


# Deng et al., CTSpine1K (MELBA 2025), Table 2 — DSC per vertebra.
CTS = """C1 .951 .957 .446
C2 .974 .975 .923
C3 .934 .945 .605
C4 .946 .608 .613
C5 .573 .726 .231
C6 .802 .824 .121
C7 .930 .899 .228
T1 .952 .927 .623
T2 .960 .932 .883
T3 .955 .798 .817
T4 .899 .831 .723
T5 .681 .830 .614
T6 .814 .762 .483
T7 .881 .764 .400
T8 .856 .772 .395
T9 .949 .925 .597
T10 .952 .951 .813
T11 .959 .961 .887
T12 .931 .965 .889
L1 .965 .883 .925
L2 .965 .913 .902
L3 .963 .952 .830
L4 .965 .945 .795
L5 .969 .943 .723
L6 0 0 0"""


def fig_ctspine_pervert():
    rows = [r.split() for r in CTS.splitlines()]
    names = [r[0] for r in rows]
    v = np.array([[float(x) for x in r[1:]] for r in rows])
    fig, ax = plt.subplots(figsize=(9.6, 2.4))
    x = np.arange(len(names))
    for k in range(len(names)):
        lbl = k + 1
        ax.axvspan(k - 0.5, k + 0.5, color=VERT_COLORS[lbl], alpha=0.10, lw=0)
    ax.vlines(x, v.min(1), v.max(1), color=RULE, lw=2, zorder=1)
    ax.scatter(x, v[:, 0], color=FILM, s=22, zorder=3, label="公开测试集 (n=197)")
    ax.scatter(x, v[:, 1], color=BONE, s=22, marker="D", zorder=3, label="私有测试集 (n=198)")
    ax.scatter(x, v[:, 2], facecolor="white", edgecolor=SLATE, s=22, marker="o", zorder=3,
               label="VerSe 公开病例 (n=41，跨域)")
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylim(-0.05, 1.02)
    ax.set_ylabel("Dice")
    ax.axhline(0.869, color=FILM, lw=0.8, ls="--")
    ax.text(len(names) - 0.6, 0.869, "均值 0.869", color=FILM, fontsize=7.5, ha="right", va="bottom")
    ax.axhline(0.619, color=SLATE, lw=0.8, ls=":")
    ax.text(len(names) - 0.6, 0.619, "跨域均值 0.619", color=SLATE, fontsize=7.5, ha="right", va="bottom")
    ax.annotate("L6 全部为 0：\n训练集里太少见", (24, 0.0), xytext=(21.2, 0.2), fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.7))
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_xlim(-0.5, len(names) - 0.5)
    fig.savefig(ASSETS / "ctspine_pervert.svg")
    plt.close(fig)


if __name__ == "__main__":
    fig_verse_leader()
    fig_ctspine_pervert()
    print("ok")
