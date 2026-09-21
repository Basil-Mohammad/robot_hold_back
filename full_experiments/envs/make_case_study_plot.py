import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qp_core import solve_qp, active_constraints, multiplier_interval, \
    deletion_ground_truth

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e4df", "grid.linewidth": 0.7,
})
BLUE = "#2a78d6"; ORANGE = "#eb6834"; YELLOW = "#eda100"; AQUA = "#1baf7a"; GRAY = "#9a9990"; RED = "#d64545"

with open("../results/rollout_qp_dataset.pkl", "rb") as f:
    records = pickle.load(f)
rec = records[41823]
H, A, b, u_pi, pos = rec["H"], rec["A"], rec["b"], rec["u_pi"], rec["pos"]
obstacles_eff = rec["obstacles_eff"]
res = solve_qp(H, A, b, u_pi)
u_star = res["u_star"]
s = res["s"]
I = active_constraints(A, b, u_star)
y_true = deletion_ground_truth(H, A, b, u_pi, u_star)

cat_color = {"cause": RED, "shadowed": YELLOW, "counter-acting": BLUE, "bystander": GRAY}
labels = {"cause": "cause ($v_i,y_i$)", "shadowed": "shadowed ($v_i,\\neg y_i$)",
          "counter-acting": "counter-acting ($\\neg v_i,y_i$)", "bystander": "bystander"}

fig, ax = plt.subplots(figsize=(6.6, 6.9))
seen = set()
for i, (ox, oy, r) in enumerate(obstacles_eff):
    v_i = bool(s[i] < -1e-9)
    y_i = y_true[i]
    cat = ("cause" if v_i and y_i else "shadowed" if v_i and not y_i else
           "counter-acting" if (not v_i) and y_i else "bystander")
    lbl = labels[cat] if cat not in seen else None
    seen.add(cat)
    alpha_fill = 0.85 if cat != "bystander" else 0.25
    ax.add_patch(plt.Circle((ox, oy), r, color=cat_color[cat], alpha=alpha_fill, label=lbl, zorder=3 if cat!="bystander" else 1))
    ax.annotate(f"{i}", (ox, oy), ha="center", va="center", fontsize=8,
                color="white" if cat != "bystander" else "#52514e", zorder=4)

ax.plot(*pos, "o", color="black", markersize=7, zorder=5)
ax.annotate("robot", pos, textcoords="offset points", xytext=(8, 8), fontsize=9)

scale = 1.0
ax.annotate("", xy=pos + u_pi * scale, xytext=pos,
            arrowprops=dict(arrowstyle="->", color="#52514e", lw=2, linestyle="--"), zorder=6)
ax.annotate("$u_\\pi$ (nominal)", pos + u_pi * scale, textcoords="offset points", xytext=(4, -12), fontsize=9, color="#52514e")

ax.annotate("", xy=pos + u_star * scale, xytext=pos,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=2.4), zorder=6)
ax.annotate("$u^\\star$ (filtered)", pos + u_star * scale, textcoords="offset points", xytext=(4, 8), fontsize=9, color=BLUE)

ax.set_xlim(pos[0] - 2.0, pos[0] + 2.0)
ax.set_ylim(pos[1] - 2.0, pos[1] + 2.0)
ax.set_aspect("equal")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Illustrative instance: seed 6, episode 39, $t{=}155$\n"
              "($K{=}10$ nearest obstacles shown;\nnumbers are constraint index $i$)",
              fontsize=10.5)
handles, labels_ = ax.get_legend_handles_labels()
ax.legend(handles, labels_, loc="upper left", fontsize=8, framealpha=0.95)
fig.tight_layout()
fig.savefig("../figures/fig_case_study.png", dpi=200)
print("wrote ../figures/fig_case_study.png")
print("active set I =", I, "u_pi=", u_pi, "u_star=", u_star)
