import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import arm_layout as al

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e4df", "grid.linewidth": 0.7,
})
BLUE = "#2a78d6"; ORANGE = "#eb6834"; YELLOW = "#eda100"; AQUA = "#1baf7a"; GRAY = "#52514e"

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

# Layout panel: arm at its start pose, obstacles, workspace boundary
ax = axes[0]
positions, _ = al.forward_kinematics(al.Q_START)
ax.plot(positions[:, 0], positions[:, 1], "-o", color=GRAY, linewidth=3,
        markersize=6, zorder=5, label="6-DOF arm (start pose)")
ax.plot(*al.BASE, "s", color="black", markersize=10, zorder=6, label="base")
ax.plot(*positions[-1], "D", color=AQUA, markersize=9, zorder=6, label="end effector")
for (ox, oy, r) in al.OBSTACLES:
    ax.add_patch(plt.Circle((ox, oy), r, color=ORANGE, alpha=0.85))
workspace = plt.Circle(al.BASE, al.REACH, fill=False, linestyle="--", color=GRAY, alpha=0.5)
ax.add_patch(workspace)
ax.plot(*al.sample_goal(np.random.default_rng(3)), "*", color=YELLOW, markersize=16,
        label="example goal", zorder=6)
lim = al.REACH * 1.1
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=7.5)
ax.set_title("Third generalization environment:\n6-DOF planar manipulator, %d keep-out zones" % len(al.OBSTACLES))
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

# Training curves panel
logs = sorted(glob.glob("../models_arm/train_log_arm_seed*.json"))
for lp in logs:
    with open(lp) as f:
        d = json.load(f)
    ts = [h["timesteps"] for h in d["history"]]
    sr = [h["success_rate"] for h in d["history"]]
    axes[1].plot(ts, sr, color=BLUE, alpha=0.6, linewidth=1.4)
axes[1].set_xlabel("Training timesteps"); axes[1].set_ylabel("Success rate (10 eval episodes)")
axes[1].set_title("PPO success rate, all %d arm seeds (m=6)" % len(logs))
axes[1].set_ylim(-0.05, 1.05)

fig.tight_layout()
fig.savefig("../figures/fig_arm_layout_training.png", dpi=200)
print("wrote ../figures/fig_arm_layout_training.png")
