"""
noise_latency_robustness.py
----------------------------
Robustness of the explanation framework to realistic sensor imperfection,
addressing the reviewer-facing question "does this survive real, noisy
sensors, or only an idealized simulator with perfect state?".

Two physically-motivated perturbations are applied post-hoc to the SAME
real policy-driven rollout dataset used throughout the paper
(rollout_qp_dataset.pkl), re-deriving (A, b) exactly as
safe_maze_env.py's _cbf_qp() / sensitivity_analysis.py's build_constraints()
do -- no retraining, no change to the nominal action u_pi (which the policy
already committed to):

  1. Sensor noise: i.i.d. Gaussian perturbation of each candidate
     obstacle's SENSED position (ox, oy), sigma in {0.01, 0.02, 0.05}
     length units (1/2/5 cm at this maze's scale, ROBOT_RADIUS=0.15),
     representative of LiDAR/depth-camera position error at close range.
  2. Latency: the safety filter constructs (A, b) from the robot's
     position estimate tau control steps in the past (tau in {1,2,3} at
     the paper's own 20 Hz control rate, i.e. 50/100/150 ms sense-to-act
     delay), while the already-decided nominal action u_pi is unchanged --
     a standard perception-latency model for a static-obstacle workspace.

For every instance and every setting we ask three questions, mirroring
the paper's existing RQ1/RQ2/RQ4 pipeline exactly (same solve_qp,
multiplier_interval, deletion_ground_truth, gram_conditioning primitives):

  (a) Exactness (RQ1 sanity check): does the proposed deletion-interval
      method remain exactly faithful (F1=1.000) to the DELETION GROUND
      TRUTH OF THE NOISY/DELAYED QP ITSELF? (Should hold trivially, since
      the theory never assumed noiseless input -- reported as an explicit
      reassurance, not a new claim.)
  (b) Taxonomy stability (the real question): does the cause/shadowed/
      counter-acting/bystander LABEL for each candidate constraint, when
      computed from the noisy/delayed measurement, still match the label
      computed from the CLEAN measurement of the same physical instant?
  (c) kappa_stab under realistic noise (RQ4 extension): does kappa_stab
      (computed on the CLEAN instance, exactly as in Table VIII) still
      predict how much the multiplier vector actually moves under this
      physically-motivated perturbation, replacing the original
      arbitrary-direction b-perturbation with real sensor noise?

Usage: python3 noise_latency_robustness.py
"""
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, ".")
import maze_layout as ml
from qp_core import solve_qp, active_constraints, multiplier_interval, \
    deletion_ground_truth, gram_conditioning

ROBOT_R = ml.ROBOT_RADIUS
ALPHA = 2.5
K_NEAREST = 10
SAFETY_MARGIN = 0.03
N_SAMPLE = 800
M_DRAWS = 5          # Monte Carlo noise realizations per instance per sigma
RNG_SEED = 7
SIGMAS = [0.01, 0.02, 0.05]   # meters-equivalent sensor position noise
TAUS = [1, 2, 3]              # control-step latency (50/100/150 ms at 20 Hz)

OBSTACLES = np.array(ml.all_obstacles(), dtype=np.float64)  # (N,3): cx,cy,r


def build_constraints_from_obs(pos, obs_k):
    """Same construction as safe_maze_env._cbf_qp / sensitivity_analysis
    .build_constraints, but takes an explicit (possibly perturbed) obstacle
    array obs_k (p,3) rather than re-selecting nearest-K from scratch, so
    noise never changes WHICH physical obstacles are candidates -- only
    their sensed positions."""
    p = obs_k.shape[0]
    A = np.zeros((p, 2))
    b = np.zeros(p)
    for i in range(p):
        ox, oy, r = obs_k[i]
        d = pos - np.array([ox, oy])
        r_eff = r + ROBOT_R + SAFETY_MARGIN
        h = d @ d - r_eff ** 2
        A[i] = 2 * d
        b[i] = -ALPHA * h
    return A, b


def category(v_i, y_i):
    if v_i and y_i:
        return "cause"
    if v_i and not y_i:
        return "shadowed"
    if not v_i and y_i:
        return "counter-acting"
    return "bystander"


def clean_reference(pos, u_pi):
    """Recompute the clean (noiseless) instance from scratch using the
    SAME nearest-K obstacle selection as the recorded rollout, so we have
    the exact obstacle array (ox,oy,r) to perturb and a clean baseline to
    compare against."""
    d_all = np.hypot(OBSTACLES[:, 0] - pos[0], OBSTACLES[:, 1] - pos[1])
    h_all = d_all - (OBSTACLES[:, 2] + ROBOT_R + SAFETY_MARGIN)
    idx = np.argsort(h_all)[:K_NEAREST]
    obs_k = OBSTACLES[idx].copy()
    H = np.eye(2)
    A, b = build_constraints_from_obs(pos, obs_k)
    res = solve_qp(H, A, b, u_pi)
    if not res["feasible"] or len(res["active_set"]) == 0:
        return None
    u_star = res["u_star"]
    I = active_constraints(A, b, u_star)
    y_true = deletion_ground_truth(H, A, b, u_pi, u_star)
    interval = multiplier_interval(H, A, b, u_pi, u_star, I)
    s = res["s"]
    lam_full = res["lam_full"]
    kappa = gram_conditioning(H, A, I)
    cats = {}
    for i, y_i in y_true.items():
        v_i = bool(s[i] < -1e-9)
        cats[i] = category(v_i, y_i)
    return dict(H=H, obs_k=obs_k, A=A, b=b, u_pi=u_pi, u_star=u_star, I=I,
                y_true=y_true, interval=interval, s=s, lam_full=lam_full,
                kappa=kappa, cats=cats)


def evaluate_perturbed(clean, pos_used, obs_used):
    """Re-solve the QP from a perturbed (pos, obstacle-array) pair, using
    the SAME H, u_pi as the clean instance, and report (a) exactness of
    the proposed method against THIS instance's own deletion ground
    truth, (b) per-constraint taxonomy agreement with the clean labels,
    (c) multiplier displacement for the kappa_stab correlation."""
    H, u_pi = clean["H"], clean["u_pi"]
    A, b = build_constraints_from_obs(pos_used, obs_used)
    res = solve_qp(H, A, b, u_pi)
    if not res["feasible"] or len(res["active_set"]) == 0:
        return None
    u_star = res["u_star"]
    I = active_constraints(A, b, u_star)
    y_true = deletion_ground_truth(H, A, b, u_pi, u_star)
    interval = multiplier_interval(H, A, b, u_pi, u_star, I)
    s = res["s"]
    lam_full = res["lam_full"]

    # (a) exactness vs THIS (noisy) instance's own ground truth
    exact_hits, exact_total = 0, 0
    for i, y_i in y_true.items():
        lmin, _ = interval[i]
        pred = bool(lmin > 1e-6)
        exact_total += 1
        exact_hits += int(pred == y_i)

    # (b) taxonomy agreement vs the CLEAN labels, same constraint indices
    agree, compared = 0, 0
    for i in range(A.shape[0]):
        if i not in clean["cats"]:
            continue
        v_i = bool(s[i] < -1e-9)
        y_i = y_true.get(i, False)
        cat_noisy = category(v_i, y_i)
        compared += 1
        agree += int(cat_noisy == clean["cats"][i])

    # (c) multiplier displacement (full p-vector, zero-padded off-active)
    lam_full_padded = lam_full
    lam_clean_padded = clean["lam_full"]
    displacement = float(np.linalg.norm(lam_full_padded - lam_clean_padded))
    active_switch = bool(set(I) != set(clean["I"]))

    return dict(exact_hits=exact_hits, exact_total=exact_total,
                agree=agree, compared=compared,
                displacement=displacement, active_switch=active_switch)


def main():
    t0 = time.time()
    with open("../results/rollout_qp_dataset.pkl", "rb") as f:
        records = pickle.load(f)
    rng = np.random.default_rng(RNG_SEED)

    # index by (seed, episode) -> {t: dataset index} for latency lookback
    by_ep = {}
    for idx, rec in enumerate(records):
        key = (rec["seed"], rec["episode"])
        by_ep.setdefault(key, {})[rec["t"]] = idx

    sample_idx = rng.choice(len(records), size=min(N_SAMPLE, len(records)), replace=False)

    results = {"noise": [], "latency": []}

    # ---------------- sensor-noise settings ----------------
    for sigma in SIGMAS:
        rows = []
        n_instances = 0
        for i in sample_idx:
            rec = records[i]
            pos, u_pi = rec["pos"], rec["u_pi"]
            clean = clean_reference(pos, u_pi)
            if clean is None:
                continue
            n_instances += 1
            disp_draws, switch_draws = [], []
            exact_h, exact_t, agr_h, agr_t = 0, 0, 0, 0
            for _ in range(M_DRAWS):
                noise = rng.normal(scale=sigma, size=(clean["obs_k"].shape[0], 2))
                obs_noisy = clean["obs_k"].copy()
                obs_noisy[:, 0:2] += noise
                out = evaluate_perturbed(clean, pos, obs_noisy)
                if out is None:
                    continue
                exact_h += out["exact_hits"]; exact_t += out["exact_total"]
                agr_h += out["agree"]; agr_t += out["compared"]
                disp_draws.append(out["displacement"])
                switch_draws.append(out["active_switch"])
            if exact_t == 0:
                continue
            rows.append(dict(
                kappa_stab=clean["kappa"],
                exact_f1=exact_h / exact_t,
                taxonomy_agreement=agr_h / agr_t if agr_t else np.nan,
                mean_displacement=float(np.mean(disp_draws)) if disp_draws else np.nan,
                any_switch=bool(np.any(switch_draws)) if switch_draws else False,
            ))
        df = pd.DataFrame(rows)
        sub = df.dropna(subset=["mean_displacement"])
        sub = sub[np.isfinite(sub["kappa_stab"]) & (sub["kappa_stab"] > 0)]
        if len(sub) > 2:
            rho, pval = spearmanr(np.log10(1.0 / sub["kappa_stab"].clip(lower=1e-300)),
                                   sub["mean_displacement"])
        else:
            rho, pval = float("nan"), float("nan")
        summary = dict(
            sigma=sigma, n_instances=n_instances, n_valid=len(df),
            mean_exact_f1=float(df["exact_f1"].mean()),
            mean_taxonomy_agreement=float(df["taxonomy_agreement"].mean()),
            frac_any_switch=float(df["any_switch"].mean()),
            spearman_rho_kappa_vs_displacement=float(rho),
            spearman_p=float(pval),
        )
        results["noise"].append(summary)
        print(f"[noise sigma={sigma:.2f}] n={summary['n_valid']}  "
              f"exact_F1={summary['mean_exact_f1']:.4f}  "
              f"taxonomy_agreement={summary['mean_taxonomy_agreement']:.4f}  "
              f"frac_any_switch={summary['frac_any_switch']:.4f}  "
              f"rho(kappa,disp)={summary['spearman_rho_kappa_vs_displacement']:.3f} "
              f"(p={summary['spearman_p']:.2e})")

    # ---------------- latency settings ----------------
    for tau in TAUS:
        rows = []
        n_instances, n_skipped_no_history = 0, 0
        for i in sample_idx:
            rec = records[i]
            pos, u_pi = rec["pos"], rec["u_pi"]
            clean = clean_reference(pos, u_pi)
            if clean is None:
                continue
            key = (rec["seed"], rec["episode"])
            t_target = rec["t"] - tau
            if t_target < 0 or t_target not in by_ep.get(key, {}):
                n_skipped_no_history += 1
                continue
            n_instances += 1
            pos_delayed = records[by_ep[key][t_target]]["pos"]
            out = evaluate_perturbed(clean, pos_delayed, clean["obs_k"])
            if out is None:
                continue
            rows.append(dict(
                kappa_stab=clean["kappa"],
                exact_f1=(out["exact_hits"] / out["exact_total"]) if out["exact_total"] else np.nan,
                taxonomy_agreement=(out["agree"] / out["compared"]) if out["compared"] else np.nan,
                displacement=out["displacement"],
                any_switch=out["active_switch"],
                pos_shift=float(np.linalg.norm(pos_delayed - pos)),
            ))
        df = pd.DataFrame(rows)
        sub = df.dropna(subset=["displacement"])
        sub = sub[np.isfinite(sub["kappa_stab"]) & (sub["kappa_stab"] > 0)]
        if len(sub) > 2:
            rho, pval = spearmanr(np.log10(1.0 / sub["kappa_stab"].clip(lower=1e-300)),
                                   sub["displacement"])
        else:
            rho, pval = float("nan"), float("nan")
        summary = dict(
            tau=tau, n_instances=n_instances, n_skipped_no_history=n_skipped_no_history,
            n_valid=len(df),
            mean_pos_shift=float(df["pos_shift"].mean()) if len(df) else float("nan"),
            mean_exact_f1=float(df["exact_f1"].mean()),
            mean_taxonomy_agreement=float(df["taxonomy_agreement"].mean()),
            frac_any_switch=float(df["any_switch"].mean()),
            spearman_rho_kappa_vs_displacement=float(rho),
            spearman_p=float(pval),
        )
        results["latency"].append(summary)
        print(f"[latency tau={tau} steps] n={summary['n_valid']}  "
              f"mean_pos_shift={summary['mean_pos_shift']:.4f}  "
              f"exact_F1={summary['mean_exact_f1']:.4f}  "
              f"taxonomy_agreement={summary['mean_taxonomy_agreement']:.4f}  "
              f"frac_any_switch={summary['frac_any_switch']:.4f}  "
              f"rho(kappa,disp)={summary['spearman_rho_kappa_vs_displacement']:.3f} "
              f"(p={summary['spearman_p']:.2e})")

    with open("../results/noise_latency_robustness.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote ../results/noise_latency_robustness.json in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
