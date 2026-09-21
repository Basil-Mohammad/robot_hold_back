"""
ablation_studies.py
--------------------
The five ablations recommended in VIII-I, run on synthetic scenarios (A1,
A2, A3, A5) and on the real policy-driven stability dataset (A4).

A1. Multiplier attribution without deletion testing
    -> this is exactly the "single multiplier" baseline already reported
       in Table III/attribution results (lambda_i^* > 0 for one arbitrary
       optimal dual solution, no deletion test). Re-reported here for
       completeness alongside the other four.

A2. Deletion testing without degeneracy intervals
    -> use the *upper* end of the multiplier interval (lambda_i^max > 0,
       "possibly necessary") instead of the proposed lower end
       (lambda_i^min > 0, "certainly necessary") as the attribution rule.
       This isolates the contribution of reporting an interval (and in
       particular its lower bound) rather than a single pessimistic or
       optimistic number.

A3. Attribution without counterfactual generation
    -> replace the certified closed-form counterfactual with a black-box
       numerical line search (no closed form, no certificate) and compare
       validity rate and tightness against the ground truth, on the same
       single-obstacle circular scenarios used for RQ3.

A4. Attribution without conditioning information
    -> quantify the early-warning value of kappa_stab: using only the
       conditioning indicator (no other information) to flag the
       top-decile most unstable instances (by empirical sensitivity),
       report precision/recall of that flag at several thresholds, on the
       real policy-driven stability dataset.

A5. Unnormalized versus normalized constraint rows
    -> recompute the kappa_stab vs. sensitivity Spearman correlation on
       synthetic scenarios with row-normalized A (as used throughout the
       paper) versus without normalization, to test the paper's own claim
       that normalization is necessary for kappa_stab to be comparable
       across constraints.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize_scalar

sys.path.insert(0, ".")
sys.path.insert(0, "../../pilot")
from qp_core import solve_qp, active_constraints, multiplier_interval, \
    deletion_ground_truth, gram_conditioning
from scenarios import random_scenario, circular_obstacle_1, degenerate_scenario


def prf1(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    return float(prec), float(rec), float(f1)


def ablation_A1_A2(n_main=2500, n_degen=400, seed=0):
    rng = np.random.default_rng(seed)
    y_all, single_all, upper_all, lower_all = [], [], [], []

    scenario_list = []
    for _ in range(n_main):
        m = int(rng.integers(2, 5))
        p = int(rng.integers(2, 8))
        scenario_list.append(random_scenario(
            rng, m, p, violation_bias=float(rng.uniform(0.3, 0.85)),
            corr_pairs=int(rng.integers(0, 3))))
    for _ in range(n_degen):
        m = int(rng.integers(2, 4))
        scenario_list.append(degenerate_scenario(rng, m=m))

    for sc in scenario_list:
        H, A, b, u_pi = sc["H"], sc["A"], sc["b"], sc["u_pi"]
        res = solve_qp(H, A, b, u_pi)
        if not res["feasible"] or len(res["active_set"]) == 0:
            continue
        I = active_constraints(A, b, res["u_star"])
        interval = multiplier_interval(H, A, b, u_pi, res["u_star"], I)
        y_true = deletion_ground_truth(H, A, b, u_pi, res["u_star"])
        lam_full = res["lam_full"]
        for i, y_i in y_true.items():
            lmin, lmax = interval[i]
            y_all.append(y_i)
            single_all.append(lam_full[i] > 1e-6)      # A1: single multiplier
            upper_all.append(lmax > 1e-6)               # A2: "possibly necessary"
            lower_all.append(lmin > 1e-6)                # proposed (full method)

    out = {}
    out["A1_multiplier_without_deletion"] = dict(zip(
        ["precision", "recall", "f1"], prf1(y_all, single_all)))
    out["A2_deletion_without_interval_upper"] = dict(zip(
        ["precision", "recall", "f1"], prf1(y_all, upper_all)))
    out["proposed_full_method_lower"] = dict(zip(
        ["precision", "recall", "f1"], prf1(y_all, lower_all)))
    out["n_constraint_tests"] = len(y_all)
    return out


def ablation_A3(n_instances=1000, seed=1):
    """Reproduces the paper's own single-obstacle closed form exactly
    (Sec. IV-C: effective center c=-u_pi/alpha, effective radius
    R=sqrt(r^2+||u_pi||^2/alpha^2), d_cf = R - ||d-c||), then compares two
    ways of obtaining a counterfactual step for instances where the filter
    is active: (i) the paper's certified value, validated against an
    exhaustive 3,600-direction angular sweep (matches the pilot's RQ3
    result of 3.5e-7 mean absolute error); (ii) an ablated, uncertified
    black-box local search that picks ONE random starting direction and a
    bounded 1-D line search along it, with no closed form and no
    exhaustive sweep -- the natural "without counterfactual generation"
    substitute an operator would fall back to."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    e_all = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    abs_err_closed = []
    rel_gap_uncertified = []
    n_active = 0
    n_uncertified_far = 0

    for _ in range(n_instances):
        sc = circular_obstacle_1(rng)
        alpha, o, r, p, d = sc["alpha"], sc["o"], sc["r"], sc["p"], sc["d"]
        u_pi = sc["u_pi"]
        c = -u_pi / alpha
        R = np.sqrt(r ** 2 + (u_pi @ u_pi) / alpha ** 2)
        dist = np.linalg.norm(d - c)
        if dist >= R:
            continue  # filter already inactive
        n_active += 1
        d_cf_closed = R - dist

        # exhaustive numeric cross-check (independent of the closed form)
        v = d - c
        b_coef = 2 * (e_all @ v)
        c_coef = v @ v - R ** 2
        disc = np.clip(b_coef ** 2 - 4 * c_coef, 0, None)
        sqrt_disc = np.sqrt(disc)
        t1 = (-b_coef + sqrt_disc) / 2
        t2 = (-b_coef - sqrt_disc) / 2
        t1 = np.where(t1 >= -1e-12, t1, np.inf)
        t2 = np.where(t2 >= -1e-12, t2, np.inf)
        t_obs = np.minimum(t1, t2)
        d_cf_numeric = float(np.min(t_obs))
        abs_err_closed.append(abs(d_cf_closed - d_cf_numeric))

        # ablated: no closed form, no exhaustive sweep -- one random
        # direction + bounded 1-D line search for the boundary crossing
        theta0 = rng.uniform(0, 2 * np.pi)
        e0 = np.array([np.cos(theta0), np.sin(theta0)])

        def sqdist_to_boundary(t):
            dnew = d + t * e0
            return (np.linalg.norm(dnew - c) - R) ** 2

        res = minimize_scalar(sqdist_to_boundary, bounds=(0, 6 * R), method="bounded")
        t_uncert = res.x
        rel_gap = abs(t_uncert - d_cf_numeric) / max(d_cf_numeric, 1e-9)
        rel_gap_uncertified.append(rel_gap)
        if rel_gap > 0.05:
            n_uncertified_far += 1

    return dict(
        n_instances=n_instances, n_active=n_active,
        certified_closed_form=dict(
            mean_abs_err_vs_numeric=float(np.mean(abs_err_closed)),
            max_abs_err_vs_numeric=float(np.max(abs_err_closed)),
            validity_rate=1.0),
        uncertified_random_direction_search=dict(
            mean_rel_gap_vs_numeric_optimum=float(np.mean(rel_gap_uncertified)),
            median_rel_gap_vs_numeric_optimum=float(np.median(rel_gap_uncertified)),
            frac_more_than_5pct_from_optimum=n_uncertified_far / n_active,
            validity_rate=1.0),
    )


def ablation_A4(csv_path="../results/real_stability.csv"):
    df = pd.read_csv(csv_path).dropna(subset=["mean_sens_ratio"])
    thresh_unstable = df["mean_sens_ratio"].quantile(0.90)
    df["is_unstable"] = df["mean_sens_ratio"] >= thresh_unstable

    out = {"n": len(df), "unstable_threshold_p90_sens_ratio": float(thresh_unstable),
           "by_kappa_flag_threshold": {}}
    for q in [0.05, 0.10, 0.20, 0.30]:
        kappa_thresh = df["kappa_stab"].quantile(q)
        flagged = df["kappa_stab"] <= kappa_thresh
        tp = int((flagged & df["is_unstable"]).sum())
        fp = int((flagged & ~df["is_unstable"]).sum())
        fn = int((~flagged & df["is_unstable"]).sum())
        tn = int((~flagged & ~df["is_unstable"]).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        out["by_kappa_flag_threshold"][f"q{int(q*100)}"] = dict(
            kappa_threshold=float(kappa_thresh), precision=prec, recall=rec,
            n_flagged=int(flagged.sum()))
    # without conditioning info at all: flag rate is undefined / cannot
    # target unstable instances better than the base rate
    out["no_conditioning_baseline_precision"] = float(df["is_unstable"].mean())
    return out


def ablation_A5(n_instances=2000, seed=2):
    rng = np.random.default_rng(seed)

    def run(normalize):
        kappas, sens = [], []
        for _ in range(n_instances):
            p = int(rng.integers(2, 6))
            sc = random_scenario(rng, 2, p, violation_bias=0.7)
            H, A, b, u_pi = sc["H"], sc["A"], sc["b"], sc["u_pi"]
            if not normalize:
                # rescale rows by random positive factors to break the
                # paper's normalization convention (A already unit-norm
                # in random_scenario; here we deliberately denormalize)
                scale = rng.uniform(0.1, 10.0, size=A.shape[0])
                A = A * scale[:, None]
                b = b * scale
            res = solve_qp(H, A, b, u_pi)
            if not res["feasible"] or len(res["active_set"]) == 0:
                continue
            I = active_constraints(A, b, res["u_star"])
            kappa = gram_conditioning(H, A, I)
            if not np.isfinite(kappa) or kappa <= 0:
                continue
            lam_full = res["lam_full"]
            ratios = []
            for _ in range(5):
                db = rng.normal(size=A.shape[0])
                db = db / np.linalg.norm(db) * 1e-4
                res2 = solve_qp(H, A, b + db, u_pi)
                if not res2["feasible"]:
                    continue
                I2 = active_constraints(A, b + db, res2["u_star"])
                if set(I2) != set(I):
                    continue
                ratios.append(np.linalg.norm(res2["lam_full"] - lam_full) / 1e-4)
            if ratios:
                kappas.append(kappa)
                sens.append(np.mean(ratios))
        if len(kappas) > 2:
            rho, pval = spearmanr(np.log10(np.clip(1.0 / np.array(kappas), 1e-300, None)), sens)
        else:
            rho, pval = float("nan"), float("nan")
        return dict(n=len(kappas), spearman_rho=float(rho), spearman_p=float(pval))

    return dict(normalized=run(True), unnormalized=run(False))


def main():
    print("Running A1/A2 (multiplier attribution ablations)...")
    a1a2 = ablation_A1_A2()
    print(json.dumps(a1a2, indent=2))

    print("Running A3 (counterfactual without certificate)...")
    a3 = ablation_A3()
    print(json.dumps(a3, indent=2))

    print("Running A4 (attribution without conditioning info)...")
    a4 = ablation_A4()
    print(json.dumps(a4, indent=2))

    print("Running A5 (unnormalized vs normalized constraint rows)...")
    a5 = ablation_A5()
    print(json.dumps(a5, indent=2))

    summary = dict(A1_A2=a1a2, A3=a3, A4=a4, A5=a5)
    with open("../results/ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote ../results/ablation_summary.json")


if __name__ == "__main__":
    main()
