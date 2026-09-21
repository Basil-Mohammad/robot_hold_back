import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e4df", "grid.linewidth": 0.7,
})

BLUE = "#2a78d6"; ORANGE = "#eb6834"; YELLOW = "#eda100"; AQUA = "#1baf7a"; GRAY = "#9a9990"
METHOD_COLORS = {"pred_interval": BLUE, "pred_single": YELLOW, "pred_naive": ORANGE}
METHOD_LABELS = {"pred_interval": "Proposed\n(deletion-interval)",
                  "pred_single": "Single multiplier\n(off-the-shelf KKT)",
                  "pred_naive": "Naive\nviolated-set"}

# ---------------- Fig A: training curves across seeds ----------------
logs = sorted(glob.glob("../models/train_log_seed*.json"))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
all_success = []
for lp in logs:
    with open(lp) as f:
        d = json.load(f)
    ts = [h["timesteps"] for h in d["history"]]
    sr = [h["success_rate"] for h in d["history"]]
    ret = [h["mean_return"] for h in d["history"]]
    axes[0].plot(ts, sr, color=BLUE, alpha=0.35, linewidth=1.3)
    axes[1].plot(ts, ret, color=BLUE, alpha=0.35, linewidth=1.3)
    all_success.append((ts, sr))

axes[0].set_xlabel("Training timesteps"); axes[0].set_ylabel("Success rate (10 eval episodes)")
axes[0].set_title("Success rate, all %d seeds" % len(logs))
axes[0].set_ylim(-0.05, 1.05)
axes[1].set_xlabel("Training timesteps"); axes[1].set_ylabel("Mean episodic return")
axes[1].set_title("Return, all %d seeds" % len(logs))
fig.tight_layout()
fig.savefig("../figures/fig_training_curves.png", dpi=200)
plt.close(fig)

# ---------------- Fig B: maze layout + example trajectory ----------------
import maze_layout as ml
fig, ax = plt.subplots(figsize=(6.5, 7.2))
for (cx, cy, hx, hy) in ml.wall_boxes_merged():
    ax.add_patch(plt.Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, color="#52514e"))
for (cx, cy, r) in ml.PILLARS:
    ax.add_patch(plt.Circle((cx, cy), r, color=ORANGE, alpha=0.9))
ax.plot(*ml.START, "o", color=AQUA, markersize=10, label="start region", zorder=5)
ax.plot(*ml.GOAL, "*", color=YELLOW, markersize=18, label="goal region", zorder=5)
ax.plot(ml.REFERENCE_PATH[:, 0], ml.REFERENCE_PATH[:, 1], "--", color=BLUE, linewidth=1.3,
        alpha=0.8, label="reference route")
ax.set_xlim(ml.WORLD_XMIN, ml.WORLD_XMAX); ax.set_ylim(ml.WORLD_YMIN, ml.WORLD_YMAX)
ax.set_aspect("equal"); ax.legend(loc="lower right", fontsize=9)
ax.set_title("Evaluation maze (moderate complexity:\n1 dead end, 1 four-pillar room)")
fig.tight_layout()
fig.savefig("../figures/fig_maze_layout.png", dpi=200)
plt.close(fig)

# ---------------- Fig C: attribution F1 (real trajectories) ----------------
with open("../results/real_analysis_summary.json") as f:
    summ = json.load(f)

methods = ["pred_naive", "pred_single", "pred_interval"]
f1s = [summ["attribution"]["overall"][m]["f1"] for m in methods]
f1_std = [summ["attribution"]["overall"][m]["f1_std_across_seeds"] for m in methods]

fig, ax = plt.subplots(figsize=(6.2, 4.2))
xs = np.arange(len(methods))
ax.bar(xs, f1s, width=0.55, color=[METHOD_COLORS[m] for m in methods],
       yerr=f1_std, capsize=4, error_kw=dict(linewidth=1.2, ecolor="#3a3a38"))
for x, v in zip(xs, f1s):
    ax.text(x, v + 0.03, f"{v:.3f}", ha="center", fontsize=10)
ax.set_xticks(xs); ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=9)
ax.set_ylabel("F1 vs. deletion ground truth")
ax.set_ylim(0, 1.15)
ax.set_title("Attribution faithfulness on REAL policy trajectories\n"
             "(%d instances, %d seeds; error bars = std across seeds)" % (
                 sum(summ["attribution"]["category_counts"].values()), summ["attribution"]["n_seeds"]))
fig.tight_layout()
fig.savefig("../figures/fig_real_attribution_f1.png", dpi=200)
plt.close(fig)

# ---------------- Fig D: taxonomy error rates (real trajectories) ----------------
cats = ["cause", "shadowed", "counter-acting", "bystander"]
err_matrix = np.zeros((len(methods), len(cats)))
n_by_cat = []
for j, cat in enumerate(cats):
    d = summ["attribution"]["by_category"].get(cat, {})
    n_by_cat.append(d.get("n", 0))
    for i, m in enumerate(methods):
        err_matrix[i, j] = d.get(m, np.nan)

fig, ax = plt.subplots(figsize=(7.6, 4.6))
width = 0.24
xs = np.arange(len(cats))
for i, m in enumerate(methods):
    offset = (i - 1) * width
    ax.bar(xs + offset, err_matrix[i] * 100, width=width * 0.92,
           color=METHOD_COLORS[m], label=METHOD_LABELS[m].replace("\n", " "))
    for x, v in zip(xs, err_matrix[i] * 100):
        if not np.isnan(v):
            ax.text(x + offset, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cats, n_by_cat)], fontsize=9)
ax.set_ylabel("Misclassification rate (%)")
ax.set_ylim(0, 115)
ax.set_title("Where naive / single-multiplier attribution fail\non real policy trajectories")
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.2))
fig.tight_layout()
fig.savefig("../figures/fig_real_taxonomy_errors.png", dpi=200)
plt.close(fig)

# ---------------- Fig E: stability scatter (real trajectories) ----------------
df_stab = pd.read_csv("../results/real_stability.csv").dropna(subset=["mean_sens_ratio"])
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.scatter(df_stab["kappa_stab"], df_stab["mean_sens_ratio"], s=10, alpha=0.35, color=BLUE, edgecolors="none")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"Conditioning indicator $\kappa_{\mathrm{stab}}=\sigma_{\min}(M_I)$")
ax.set_ylabel(r"Empirical multiplier sensitivity $\|\Delta\lambda\|/\|\Delta b\|$")
rho = summ["stability"]["spearman_rho_kappa_vs_sensitivity"]
ax.set_title("Conditioning vs. instability, real trajectories\n"
              r"Spearman $\rho$=%.2f, n=%d" % (rho, len(df_stab)))
fig.tight_layout()
fig.savefig("../figures/fig_real_stability.png", dpi=200)
plt.close(fig)

# ---------------- Fig F: counterfactual gap distribution ----------------
df_cf = pd.read_csv("../results/real_counterfactual.csv")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.hist(df_cf["rel_gap"] * 100, bins=40, color=BLUE, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Relative certificate gap (%)")
ax.set_ylabel("Number of QP instances")
ax.set_title("Counterfactual certificate gap, real trajectories\n(n=%d; bound held: %s)" %
              (len(df_cf), summ["counterfactual"]["bound_always_holds"]))
fig.tight_layout()
fig.savefig("../figures/fig_real_counterfactual_gap.png", dpi=200)
plt.close(fig)

print("all full-scale plots written")
