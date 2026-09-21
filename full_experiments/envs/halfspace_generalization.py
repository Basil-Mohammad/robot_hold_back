"""
halfspace_generalization.py
-----------------------------
Generalization check: does the framework's identities / attribution /
counterfactual / stability results hold when the safety constraints are
NOT circular?

The main experiment (run_full_analysis.py) uses only circular CBF
constraints (the maze's pillars, and a circular approximation of the
walls, "wall_circles()"). This script reconstructs an ALTERNATIVE,
affine (linear half-space) CBF formulation for the wall boundaries --
using the actual axis-aligned wall geometry (wall_boxes_merged()) instead
of its circular approximation -- while leaving the pillar constraints
circular, giving a genuinely MIXED constraint-type QP (linear walls +
quadratic-in-position pillars, both still affine IN THE ACTION u, which
is all the CBF-QP machinery requires).

Crucially this re-derives (A, b) at exactly the same (position, nominal
action) pairs visited by the already-trained, already-validated policies
in rollout_qp_dataset.pkl -- so this is a controlled constraint-type
ablation on the SAME real trajectories, not a new training run, and
carries no risk of re-introducing the environment bugs that were fixed
for the main experiment.

Linear wall half-space CBF: for a wall box (cx, cy, hx, hy) and a query
position p outside it, let ex = p.x-cx, ey = p.y-cy. The box's dominant
overflow axis is x if (|ex|-hx) >= (|ey|-hy), else y. The corresponding
outward face gives an EXACTLY affine barrier
    h(p) = sign(e_axis) * (p_axis - center_axis) - half_extent_axis
           - (robot_radius + margin)
whose gradient a = alpha * n is constant (a genuinely different
functional form from the circular h(p) = ||p-o||^2 - r^2, whose gradient
depends on p). Pillars keep their original circular form unchanged.
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, ".")
import maze_layout as ml
from qp_core import solve_qp, active_constraints, multiplier_interval, \
    deletion_ground_truth, gram_conditioning

ALPHA = 2.5
ROBOT_R = 0.15
SAFETY_MARGIN = 0.03
K_NEAREST = 10

WALL_BOXES = ml.wall_boxes_merged()  # (cx, cy, hx, hy)
PILLARS = ml.PILLARS                  # (ox, oy, r)


def wall_halfspace_row(p, box):
    cx, cy, hx, hy = box
    ex, ey = p[0] - cx, p[1] - cy
    ox = abs(ex) - hx
    oy = abs(ey) - hy
    if ox >= oy:
        n = np.array([1.0 if ex >= 0 else -1.0, 0.0])
        h = ox - (ROBOT_R + SAFETY_MARGIN)
    else:
        n = np.array([0.0, 1.0 if ey >= 0 else -1.0])
        h = oy - (ROBOT_R + SAFETY_MARGIN)
    a_row = ALPHA * n
    b_val = ALPHA * (-h) * 0 + (-ALPHA * h)  # b = -alpha*h, kept explicit
    return a_row, -ALPHA * h, h


def pillar_circle_row(p, pillar):
    ox, oy, r = pillar
    d = p - np.array([ox, oy])
    r_eff = r + ROBOT_R + SAFETY_MARGIN
    h = d @ d - r_eff ** 2
    a_row = 2 * d
    b_val = -ALPHA * h
    return a_row, b_val, h


def build_mixed_constraints(pos, k=K_NEAREST):
    cand = []
    for box in WALL_BOXES:
        a_row, b_val, h = wall_halfspace_row(pos, box)
        cand.append((h, "wall_linear", a_row, b_val))
    for pillar in PILLARS:
        a_row, b_val, h = pillar_circle_row(pos, pillar)
        cand.append((h, "pillar_circular", a_row, b_val))
    cand.sort(key=lambda t: t[0])
    top = cand[:k]
    A = np.array([t[2] for t in top])
    b = np.array([t[3] for t in top])
    kinds = [t[1] for t in top]
    return A, b, kinds


def category(v_i, y_i):
    if v_i and y_i:
        return "cause"
    if v_i and not y_i:
        return "shadowed"
    if not v_i and y_i:
        return "counter-acting"
    return "bystander"


def main():
    with open("../results/rollout_qp_dataset.pkl", "rb") as f:
        records = pickle.load(f)
    print(f"loaded {len(records)} real trajectory records")

    rng = np.random.default_rng(0)
    N_SAMPLE = 8000
    idx = rng.choice(len(records), size=min(N_SAMPLE, len(records)), replace=False)
    sample = [records[i] for i in idx]

    H = np.eye(2)
    id_stationarity, id_energy = [], []
    attr_rows = []
    kappa_rows = []
    n_linear_active = 0
    n_circular_active = 0
    n_mixed_active = 0

    for rec in sample:
        pos, u_pi = rec["pos"], rec["u_pi"]
        A, b, kinds = build_mixed_constraints(pos)
        res = solve_qp(H, A, b, u_pi)
        if not res["feasible"] or len(res["active_set"]) == 0:
            continue
        u_star = res["u_star"]
        I = active_constraints(A, b, u_star)
        kinds_active = set(kinds[i] for i in I)
        if kinds_active == {"wall_linear"}:
            n_linear_active += 1
        elif kinds_active == {"pillar_circular"}:
            n_circular_active += 1
        elif len(kinds_active) > 1:
            n_mixed_active += 1

        res_chk = solve_qp(H, A, b, u_pi)
        id_stationarity.append(np.linalg.norm(res_chk["u_star"] - u_star))
        interval = multiplier_interval(H, A, b, u_pi, u_star, I)
        y_true = deletion_ground_truth(H, A, b, u_pi, u_star)
        s = res["s"]
        lam_full = res["lam_full"]
        Hinv = np.linalg.inv(H)
        V = 0.5 * (u_star - u_pi) @ H @ (u_star - u_pi)
        id_energy.append(abs(2 * V - (-lam_full @ s)))

        kappa = gram_conditioning(H, A, I)
        if np.isfinite(kappa) and kappa > 0:
            p_dim = A.shape[0]
            ratios = []
            n_switch = 0
            for _ in range(6):
                db = rng.normal(size=p_dim)
                db = db / np.linalg.norm(db) * 1e-4
                res2 = solve_qp(H, A, b + db, u_pi)
                if not res2["feasible"]:
                    continue
                I2 = active_constraints(A, b + db, res2["u_star"])
                if set(I2) != set(I):
                    n_switch += 1
                    continue
                ratios.append(np.linalg.norm(res2["lam_full"] - lam_full) / 1e-4)
            kappa_rows.append(dict(kappa_stab=float(kappa),
                                    mean_sens_ratio=float(np.mean(ratios)) if ratios else np.nan,
                                    switch_rate=n_switch / 6))

        for i, y_i in y_true.items():
            v_i = bool(s[i] < -1e-9)
            lmin, lmax = interval[i]
            attr_rows.append(dict(
                v_i=v_i, y_true=y_i, category=category(v_i, y_i), kind=kinds[i],
                pred_naive=v_i, pred_single=bool(lam_full[i] > 1e-6),
                pred_interval=bool(lmin > 1e-6)))

    df_attr = pd.DataFrame(attr_rows)
    df_kappa = pd.DataFrame(kappa_rows).dropna(subset=["mean_sens_ratio"])

    def prf1(y_true, y_pred):
        y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
        tp = np.sum(y_true & y_pred); fp = np.sum(~y_true & y_pred); fn = np.sum(y_true & ~y_pred)
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec_ = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) > 0 else float("nan")
        return prec, rec_, f1

    overall = {}
    for m in ["pred_naive", "pred_single", "pred_interval"]:
        p_, r_, f1_ = prf1(df_attr["y_true"], df_attr[m])
        overall[m] = dict(precision=p_, recall=r_, f1=f1_)

    if len(df_kappa) > 2:
        rho, pval = spearmanr(np.log10(1.0 / df_kappa["kappa_stab"].clip(lower=1e-300)),
                               df_kappa["mean_sens_ratio"])
    else:
        rho, pval = float("nan"), float("nan")

    summary = dict(
        n_instances_analyzed=len(sample),
        n_active_instances=len(id_stationarity),
        constraint_mix=dict(
            wall_linear_only_active=n_linear_active,
            pillar_circular_only_active=n_circular_active,
            mixed_active=n_mixed_active),
        identities=dict(
            max_err_stationarity=float(np.max(id_stationarity)) if id_stationarity else None,
            max_err_energy=float(np.max(id_energy)) if id_energy else None,
        ),
        attribution=dict(overall=overall, n_constraint_tests=len(df_attr)),
        stability=dict(n=len(df_kappa), spearman_rho=float(rho), spearman_p=float(pval)),
    )
    with open("../results/halfspace_generalization_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
