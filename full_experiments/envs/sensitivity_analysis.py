"""
sensitivity_analysis.py
--------------------------
One-factor-at-a-time hyperparameter sensitivity sweep for the CBF-QP
safety filter's three design constants (ALPHA, K_NEAREST, SAFETY_MARGIN),
addressing the reviewer-facing question "are the paper's headline results
an artifact of one particular filter tuning?".

Re-derives (A, b) exactly as safe_maze_env.py's _cbf_qp() does -- nearest-
K circular obstacles (walls + pillars, ml.all_obstacles()), same
h(p)=||p-o||^2 - r_eff^2, a_row = 2*(p-o), b = -alpha*h -- but swept over
alpha/k/margin, evaluated at the SAME real (position, nominal action)
pairs visited by the already-trained, already-validated policies
(rollout_qp_dataset.pkl). No retraining; only the filter's own
hyperparameters change, post-hoc, exactly as in halfspace_generalization.py.

For each setting we report: fraction of instances where the filter is
active, the max numerical-identity error (stationarity + energy, which
must stay at solver precision regardless of tuning), and attribution F1
for naive / single-multiplier / proposed (multiplier-interval) against
the deletion ground truth -- the same three methods compared throughout
the paper.
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import maze_layout as ml
from qp_core import solve_qp, active_constraints, multiplier_interval, \
    deletion_ground_truth

ROBOT_R = ml.ROBOT_RADIUS
BASE_ALPHA = 2.5
BASE_K = 10
BASE_MARGIN = 0.03
N_SAMPLE = 2500
RNG_SEED = 0

OBSTACLES = np.array(ml.all_obstacles(), dtype=np.float64)  # (N,3): cx,cy,r


def build_constraints(pos, alpha, k, margin):
    d_all = np.hypot(OBSTACLES[:, 0] - pos[0], OBSTACLES[:, 1] - pos[1])
    h_all = d_all - (OBSTACLES[:, 2] + ROBOT_R + margin)
    idx = np.argsort(h_all)[:k]
    obs_k = OBSTACLES[idx]
    p = obs_k.shape[0]
    A = np.zeros((p, 2))
    b = np.zeros(p)
    for i in range(p):
        ox, oy, r = obs_k[i]
        dd = pos - np.array([ox, oy])
        r_eff = r + ROBOT_R + margin
        h = dd @ dd - r_eff ** 2
        A[i] = 2 * dd
        b[i] = -alpha * h
    return A, b


def prf1(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = np.sum(y_true & y_pred); fp = np.sum(~y_true & y_pred); fn = np.sum(y_true & ~y_pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    return prec, rec, f1


def evaluate_setting(sample, alpha, k, margin):
    H = np.eye(2)
    rows = []
    id_stationarity, id_energy = [], []
    n_active = 0
    for pos, u_pi in sample:
        A, b = build_constraints(pos, alpha, k, margin)
        res = solve_qp(H, A, b, u_pi)
        if not res["feasible"]:
            continue
        u_star = res["u_star"]
        if len(res["active_set"]) == 0:
            continue  # filter inactive at this instance under this setting
        n_active += 1
        I = active_constraints(A, b, u_star)
        s = res["s"]
        lam_full = res["lam_full"]
        interval = multiplier_interval(H, A, b, u_pi, u_star, I)
        y_true = deletion_ground_truth(H, A, b, u_pi, u_star)

        res_chk = solve_qp(H, A, b, u_pi)
        id_stationarity.append(float(np.linalg.norm(res_chk["u_star"] - u_star)))
        V = 0.5 * (u_star - u_pi) @ H @ (u_star - u_pi)
        id_energy.append(float(abs(2 * V - (-lam_full @ s))))

        for i, y_i in y_true.items():
            v_i = bool(s[i] < -1e-9)
            lmin, _ = interval[i]
            rows.append(dict(y_true=y_i, pred_naive=v_i,
                              pred_single=bool(lam_full[i] > 1e-6),
                              pred_interval=bool(lmin > 1e-6)))
    df = pd.DataFrame(rows)
    out = dict(alpha=alpha, k=k, margin=margin,
               n_instances=len(sample), n_active=n_active,
               frac_active=n_active / len(sample) if sample else float("nan"),
               max_err_stationarity=float(np.max(id_stationarity)) if id_stationarity else None,
               max_err_energy=float(np.max(id_energy)) if id_energy else None,
               n_constraint_tests=len(df))
    for col, name in [("pred_naive", "naive"), ("pred_single", "single"), ("pred_interval", "proposed")]:
        if len(df):
            p_, r_, f1_ = prf1(df["y_true"], df[col])
        else:
            p_, r_, f1_ = float("nan"), float("nan"), float("nan")
        out[f"f1_{name}"] = f1_
        out[f"precision_{name}"] = p_
        out[f"recall_{name}"] = r_
    return out


def main():
    with open("../results/rollout_qp_dataset.pkl", "rb") as f:
        records = pickle.load(f)
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(len(records), size=min(N_SAMPLE, len(records)), replace=False)
    sample = [(records[i]["pos"], records[i]["u_pi"]) for i in idx]
    print(f"sensitivity sweep on {len(sample)} real (position, action) pairs")

    alpha_grid = [1.0, 1.5, 2.0, BASE_ALPHA, 3.5, 5.0]
    k_grid = [3, 5, BASE_K, 15, 20]
    margin_grid = [0.0, 0.01, BASE_MARGIN, 0.06, 0.10]

    results = {"alpha_sweep": [], "k_sweep": [], "margin_sweep": []}

    for a in alpha_grid:
        r = evaluate_setting(sample, a, BASE_K, BASE_MARGIN)
        results["alpha_sweep"].append(r)
        print(f"  ALPHA={a:5.2f}  frac_active={r['frac_active']:.3f}  "
              f"F1 naive/single/proposed = {r['f1_naive']:.3f}/{r['f1_single']:.3f}/{r['f1_proposed']:.3f}  "
              f"max_err_energy={r['max_err_energy']:.2e}")

    for k in k_grid:
        r = evaluate_setting(sample, BASE_ALPHA, k, BASE_MARGIN)
        results["k_sweep"].append(r)
        print(f"  K_NEAREST={k:3d}  frac_active={r['frac_active']:.3f}  "
              f"F1 naive/single/proposed = {r['f1_naive']:.3f}/{r['f1_single']:.3f}/{r['f1_proposed']:.3f}  "
              f"max_err_energy={r['max_err_energy']:.2e}")

    for m in margin_grid:
        r = evaluate_setting(sample, BASE_ALPHA, BASE_K, m)
        results["margin_sweep"].append(r)
        print(f"  MARGIN={m:5.3f}  frac_active={r['frac_active']:.3f}  "
              f"F1 naive/single/proposed = {r['f1_naive']:.3f}/{r['f1_single']:.3f}/{r['f1_proposed']:.3f}  "
              f"max_err_energy={r['max_err_energy']:.2e}")

    with open("../results/sensitivity_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Wrote ../results/sensitivity_summary.json")


if __name__ == "__main__":
    main()
