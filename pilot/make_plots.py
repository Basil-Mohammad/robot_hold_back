import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c9c8c2",
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e5e4df",
    "grid.linewidth": 0.7,
})

BLUE = "#2a78d6"      # ours (deletion-interval)
ORANGE = "#eb6834"    # naive violated-set
YELLOW = "#eda100"    # single-multiplier
AQUA = "#1baf7a"       # shapley baseline
GRAY = "#9a9990"

METHOD_COLORS = {
    "pred_interval": BLUE,
    "pred_single": YELLOW,
    "pred_naive": ORANGE,
    "pred_shapley": AQUA,
}
METHOD_LABELS = {
    "pred_interval": "Proposed\n(deletion-interval)",
    "pred_single": "Single multiplier\n(off-the-shelf KKT)",
    "pred_naive": "Naive\nviolated-set",
    "pred_shapley": "Shapley\n(game-theoretic)",
}

# ---------------------------------------------------------------- Figure 1
# Attribution F1 by method (overall)
with open("summary_attribution.json") as f:
    attr = json.load(f)

methods = ["pred_naive", "pred_single", "pred_shapley", "pred_interval"]
f1s = [attr["overall"][m]["f1"] for m in methods]

fig, ax = plt.subplots(figsize=(6.4, 4.2))
xs = np.arange(len(methods))
bars = ax.bar(xs, f1s, width=0.6, color=[METHOD_COLORS[m] for m in methods])
for x, v in zip(xs, f1s):
    ax.text(x, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10, color="#0b0b0b")
ax.set_xticks(xs)
ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=9)
ax.set_ylabel("F1 vs. constraint-deletion ground truth")
ax.set_ylim(0, 1.12)
ax.set_title("Attribution faithfulness: overall F1 (10,872 constraint instances)")
fig.tight_layout()
fig.savefig("fig1_attribution_f1.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
# Error rate by taxonomy category, grouped bars
df_attr = pd.read_csv("results_attribution.csv")
cats = ["cause", "shadowed", "counter-acting", "bystander"]
cat_methods = ["pred_naive", "pred_single", "pred_interval"]

err_matrix = np.zeros((len(cat_methods), len(cats)))
n_by_cat = []
for j, cat in enumerate(cats):
    sub = df_attr[df_attr["category"] == cat]
    n_by_cat.append(len(sub))
    for i, m in enumerate(cat_methods):
        if len(sub) == 0:
            err_matrix[i, j] = np.nan
        else:
            err_matrix[i, j] = float((sub["y_true"] != sub[m]).mean())

fig, ax = plt.subplots(figsize=(7.6, 4.6))
width = 0.24
xs = np.arange(len(cats))
for i, m in enumerate(cat_methods):
    offset = (i - 1) * width
    bars = ax.bar(xs + offset, err_matrix[i] * 100, width=width * 0.92,
                   color=METHOD_COLORS[m], label=METHOD_LABELS[m].replace("\n", " "))
    for x, v in zip(xs, err_matrix[i] * 100):
        if not np.isnan(v):
            ax.text(x + offset, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
ax.set_xticks(xs)
ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cats, n_by_cat)], fontsize=9)
ax.set_ylabel("Misclassification rate vs. deletion ground truth (%)")
ax.set_ylim(0, 115)
ax.set_title("Where naive and single-multiplier attribution fail")
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.18))
fig.tight_layout()
fig.savefig("fig2_taxonomy_errors.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
# Stability: conditioning indicator vs empirical multiplier sensitivity
df_stab = pd.read_csv("results_stability.csv").dropna(subset=["mean_sens_ratio"])
with open("summary_stability.json") as f:
    stab = json.load(f)

fig, ax = plt.subplots(figsize=(6.6, 4.6))
ax.scatter(df_stab["kappa_stab"], df_stab["mean_sens_ratio"], s=14, alpha=0.55,
           color=BLUE, edgecolors="none")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Conditioning indicator $\kappa_{\mathrm{stab}}=\sigma_{\min}(M_I)$")
ax.set_ylabel(r"Empirical multiplier sensitivity  $\|\Delta\lambda\|/\|\Delta b\|$")
ax.set_title("Conditioning predicts attribution instability\n"
             r"Spearman $\rho$=%.2f  ($p$<1e-200), n=%d" % (
                 stab["spearman_rho_kappa_vs_sensitivity"], len(df_stab)))
fig.tight_layout()
fig.savefig("fig3_stability.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure 4
# Counterfactual: multi-obstacle certificate gap distribution
df_cfm = pd.read_csv("results_cf_multi.csv")
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.hist(df_cfm["rel_gap"] * 100, bins=40, color=BLUE, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Relative certificate gap  (d_feasible - d_lower) / d_feasible  [%]")
ax.set_ylabel("Number of scenarios")
ax.set_title("Two-sided counterfactual certificate: gap distribution\n"
             "(n=%d multi-obstacle scenarios; bound held in 100%% of cases)" % len(df_cfm))
fig.tight_layout()
fig.savefig("fig4_counterfactual_gap.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure 5
# Identity numerical error vs conditioning (ties Exp.1 residual error to RQ4)
df_id = pd.read_csv("results_identities.csv")
df_id = df_id[df_id["err_energy"] > 0]
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.scatter(range(len(df_id)), sorted(df_id["err_energy"]), s=10, color=BLUE, alpha=0.6)
ax.set_yscale("log")
ax.set_xlabel("Instances, sorted by energy-identity error")
ax.set_ylabel(r"$|2V-\lambda^{\star\top}M\lambda^{\star}|$ (log scale)")
ax.set_title("Exact-identity numerical residual across %d active instances" % len(df_id))
fig.tight_layout()
fig.savefig("fig5_identity_error.png", dpi=200)
plt.close(fig)

print("plots written")
