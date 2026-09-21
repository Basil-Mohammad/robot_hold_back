import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import quad_layout as ql

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e4df", "grid.linewidth": 0.7,
})
BLUE = "#2a78d6"; ORANGE = "#eb6834"; YELLOW = "#eda100"; AQUA = "#1baf7a"

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

# Layout panel
ax = axes[0]
for (cx, cy, r) in ql.ZONES:
    ax.add_patch(plt.Circle((cx, cy), r, color=ORANGE, alpha=0.85))
ax.plot(*ql.START, "o", color=AQUA, markersize=10, label="start region", zorder=5)
ax.plot(*ql.GOAL, "*", color=YELLOW, markersize=18, label="goal region", zorder=5)
ax.set_xlim(ql.WORLD_XMIN, ql.WORLD_XMAX); ax.set_ylim(ql.WORLD_YMIN, ql.WORLD_YMAX)
ax.set_aspect("equal"); ax.legend(loc="lower right", fontsize=9)
ax.set_title("Third environment: open-field planar\nquadrotor, %d no-fly zones" % len(ql.ZONES))
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

# Training curves panel
logs = sorted(glob.glob("../models_quad/train_log_quad_seed*.json"))
for lp in logs:
    with open(lp) as f:
        d = json.load(f)
    ts = [h["timesteps"] for h in d["history"]]
    sr = [h["success_rate"] for h in d["history"]]
    axes[1].plot(ts, sr, color=BLUE, alpha=0.5, linewidth=1.4)
axes[1].set_xlabel("Training timesteps"); axes[1].set_ylabel("Success rate (10 eval episodes)")
axes[1].set_title("PPO success rate, all %d quadrotor seeds" % len(logs))
axes[1].set_ylim(-0.05, 1.05)

fig.tight_layout()
fig.savefig("../figures/fig_quad_layout_training.png", dpi=200)
print("wrote ../figures/fig_quad_layout_training.png")
