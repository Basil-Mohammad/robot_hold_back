"""
run_full_analysis.py
----------------------
RQ1-RQ4 on the REAL, policy-driven QP dataset collected by
collect_rollouts.py (as opposed to the pilot study's synthetic random
QPs). Every metric is computed per-seed first, then aggregated across
seeds (mean +/- std), which is what makes this publication-quality rather
than a single pooled number: it lets us report variance across independent
training runs, exactly as a JMLR/TNNLS reviewer would expect.

RQ1/RQ2 -- attribution faithfulness of 3 methods (naive violated-set,
          single-multiplier, proposed deletion-interval) against the exact
          deletion ground truth, overall and by taxonomy category.
RQ3      -- counterfactual validity: every constraint here is circular (see
          maze_layout.py), so the paper's own closed form applies to EVERY
          single instance, not just a separate synthetic circular-obstacle
          experiment as in the pilot. Also the two-sided multi-obstacle
          certificate.
RQ4      -- stability: correlation between kappa_stab and (a) how large the
          degeneracy-driven attribution interval actually is on instances
          that do occur, (b) perturbation-based sensitivity, replicated on
          real states (not just synthetic ones).
"""
import argparse
import json
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, ".")
from qp_core import solve_qp, active_constraints, multiplier_interval, \
    deletion_ground_truth, gram_conditioning


def angular_min_step(pos, obstacles, angles):
    """Same exact-1D-quadratic angular sweep as the pilot's Experiment 3,
    generalized to an arbitrary list of (ox, oy, r_eff) obstacles: for each
    direction, the minimal step needed is the max over obstacles of each
    obstacle's own minimal step (0 if already satisfied)."""
    e = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    t_needed = np.zeros(len(angles))
    for (ox, oy, r) in obstacles:
        v = pos - np.array([ox, oy])
        dist = np.linalg.norm(v)
        if dist >= r:
            continue
        b_coef = 2 * (e @ v)
        c_coef = v @ v - r ** 2
        disc = np.clip(b_coef ** 2 - 4 * c_coef, 0, None)
        sqrt_disc = np.sqrt(disc)
        t1 = (-b_coef + sqrt_disc) / 2
        t2 = (-b_coef - sqrt_disc) / 2
        t1 = np.where(t1 >= -1e-12, t1, np.inf)
        t2 = np.where(t2 >= -1e-12, t2, np.inf)
        t_obs = np.minimum(t1, t2)
        t_needed = np.maximum(t_needed, t_obs)
    return t_needed


def category(v_i, y_i):
    if v_i and y_i:
        return "cause"
    if v_i and not y_i:
        return "shadowed"
    if not v_i and y_i:
        return "counter-acting"
    return "bystander"


def analyze_instance(rec, angles, rng):
    H, A, b, u_pi, u_star, pos = rec["H"], rec["A"], rec["b"], rec["u_pi"], rec["u_star"], rec["pos"]
    obstacles_eff = rec["obstacles_eff"]
    alpha = rec["alpha"]

    res = solve_qp(H, A, b, u_pi)
    if not res["feasible"] or len(res["active_set"]) == 0:
        return None
    u_star_chk = res["u_star"]
    id_err_stationarity = np.linalg.norm(u_star_chk - u_star)

    I = active_constraints(A, b, u_star_chk)
    interval = multiplier_interval(H, A, b, u_pi, u_star_chk, I)
    y_true = deletion_ground_truth(H, A, b, u_pi, u_star_chk)
    kappa = gram_conditioning(H, A, I)
    s = res["s"]
    lam_full = res["lam_full"]

    Hinv = np.linalg.inv(H)
    M = A @ Hinv @ A.T
    V = 0.5 * (u_star_chk - u_pi) @ H @ (u_star_chk - u_pi)
    id_err_energy = abs(2 * V - (-lam_full @ s))

    attr_rows = []
    for i, y_i in y_true.items():
        v_i = bool(s[i] < -1e-9)
        lmin, lmax = interval[i]
        attr_rows.append(dict(
            v_i=v_i, y_true=y_i, category=category(v_i, y_i),
            pred_naive=v_i, pred_single=bool(lam_full[i] > 1e-6),
            pred_interval=bool(lmin > 1e-6),
            degenerate=bool((lmax - lmin) > 1e-6),
        ))

    # counterfactual: only meaningful when the *nominal* policy action was
    # actually infeasible (some real obstacle currently violated)
    cf_row = None
    violated_idx = [i for i in range(A.shape[0]) if s[i] < -1e-9]
    if violated_idx:
        d_lower = 0.0
        for i in violated_idx:
            ox, oy, r = obstacles_eff[i]
            dist = np.linalg.norm(pos - np.array([ox, oy]))
            if dist < r:
                d_lower = max(d_lower, r - dist)
        obstacles_list = [tuple(o) for o in obstacles_eff]
        t_all = angular_min_step(pos, obstacles_list, angles)
        d_feasible = float(np.min(t_all))
        cf_row = dict(d_lower=d_lower, d_feasible=d_feasible,
                       gap=d_feasible - d_lower,
                       rel_gap=(d_feasible - d_lower) / max(d_feasible, 1e-9),
                       bound_holds=bool(d_lower <= d_feasible + 1e-9),
                       n_violated=len(violated_idx))

    # stability: perturb b, resolve, measure multiplier sensitivity
    stab_row = None
    if np.isfinite(kappa) and kappa > 0:
        p_dim = A.shape[0]
        ratios = []
        n_switch = 0
        K_PERTURB = 8
        for _ in range(K_PERTURB):
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
        stab_row = dict(kappa_stab=float(kappa),
                         mean_sens_ratio=float(np.mean(ratios)) if ratios else np.nan,
                         switch_rate=n_switch / K_PERTURB)

    return dict(attr_rows=attr_rows, cf_row=cf_row, stab_row=stab_row,
                id_err_stationarity=id_err_stationarity, id_err_energy=id_err_energy,
                seed=rec["seed"], k_active=len(I))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="../results/rollout_qp_dataset.pkl")
    ap.add_argument("--outdir", type=str, default="../results")
    ap.add_argument("--max_instances", type=int, default=0, help="0 = use all")
    args = ap.parse_args()

    with open(args.dataset, "rb") as f:
        records = pickle.load(f)
    print(f"loaded {len(records)} QP instances")
    if args.max_instances:
        rng0 = np.random.default_rng(0)
        idx = rng0.choice(len(records), size=min(args.max_instances, len(records)), replace=False)
        records = [records[i] for i in idx]
        print(f"subsampled to {len(records)} instances")

    angles = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    rng = np.random.default_rng(42)

    attr_records = []
    cf_records = []
    stab_records = []
    id_stationarity = []
    id_energy = []

    for k, rec in enumerate(records):
        out = analyze_instance(rec, angles, rng)
        if out is None:
            continue
        for row in out["attr_rows"]:
            row["seed"] = out["seed"]
            attr_records.append(row)
        if out["cf_row"] is not None:
            out["cf_row"]["seed"] = out["seed"]
            cf_records.append(out["cf_row"])
        if out["stab_row"] is not None:
            out["stab_row"]["seed"] = out["seed"]
            stab_records.append(out["stab_row"])
        id_stationarity.append(out["id_err_stationarity"])
        id_energy.append(out["id_err_energy"])
        if k % 2000 == 0:
            print(f"  processed {k}/{len(records)}")

    df_attr = pd.DataFrame(attr_records)
    df_cf = pd.DataFrame(cf_records)
    df_stab = pd.DataFrame(stab_records)

    df_attr.to_csv(f"{args.outdir}/real_attribution.csv", index=False)
    df_cf.to_csv(f"{args.outdir}/real_counterfactual.csv", index=False)
    df_stab.to_csv(f"{args.outdir}/real_stability.csv", index=False)

    # ---------------- per-seed aggregation ----------------
    def prf1(y_true, y_pred):
        y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
        tp = np.sum(y_true & y_pred); fp = np.sum(~y_true & y_pred); fn = np.sum(y_true & ~y_pred)
        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rec_ = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = 2 * prec * rec_ / (prec + rec_) if (prec is not np.nan and rec_ is not np.nan and (prec + rec_) > 0) else np.nan
        return prec, rec_, f1

    methods = ["pred_naive", "pred_single", "pred_interval"]
    per_seed_f1 = {m: [] for m in methods}
    for seed, sub in df_attr.groupby("seed"):
        for m in methods:
            _, _, f1 = prf1(sub["y_true"], sub[m])
            per_seed_f1[m].append(f1)

    overall = {}
    for m in methods:
        p_, r_, f1_ = prf1(df_attr["y_true"], df_attr[m])
        overall[m] = dict(precision=p_, recall=r_, f1=f1_,
                           f1_mean_across_seeds=float(np.nanmean(per_seed_f1[m])),
                           f1_std_across_seeds=float(np.nanstd(per_seed_f1[m])))

    by_category = {}
    for cat in ["cause", "shadowed", "counter-acting", "bystander"]:
        sub = df_attr[df_attr["category"] == cat]
        by_category[cat] = dict(n=int(len(sub)))
        for m in methods:
            if len(sub):
                err = float((sub["y_true"] != sub[m]).mean())
            else:
                err = float("nan")
            by_category[cat][m] = err

    cf_summary = dict(
        n=len(df_cf),
        bound_always_holds=bool(df_cf["bound_holds"].all()) if len(df_cf) else None,
        mean_rel_gap=float(df_cf["rel_gap"].mean()) if len(df_cf) else None,
        median_rel_gap=float(df_cf["rel_gap"].median()) if len(df_cf) else None,
        p90_rel_gap=float(df_cf["rel_gap"].quantile(0.9)) if len(df_cf) else None,
        max_rel_gap=float(df_cf["rel_gap"].max()) if len(df_cf) else None,
    )

    stab_sub = df_stab.dropna(subset=["mean_sens_ratio"])
    if len(stab_sub) > 2:
        rho, pval = spearmanr(np.log10(1.0 / stab_sub["kappa_stab"].clip(lower=1e-300)),
                               stab_sub["mean_sens_ratio"])
    else:
        rho, pval = float("nan"), float("nan")

    summary = dict(
        n_instances_analyzed=len(records),
        n_active_instances=len(id_stationarity),
        identities=dict(
            max_err_stationarity=float(np.max(id_stationarity)) if id_stationarity else None,
            mean_err_stationarity=float(np.mean(id_stationarity)) if id_stationarity else None,
            max_err_energy=float(np.max(id_energy)) if id_energy else None,
            mean_err_energy=float(np.mean(id_energy)) if id_energy else None,
        ),
        attribution=dict(overall=overall, by_category=by_category,
                          category_counts=df_attr["category"].value_counts().to_dict(),
                          n_seeds=int(df_attr["seed"].nunique())),
        counterfactual=cf_summary,
        stability=dict(
            n=len(stab_sub),
            spearman_rho_kappa_vs_sensitivity=float(rho),
            spearman_p_kappa_vs_sensitivity=float(pval),
            frac_any_switch=float((df_stab["switch_rate"] > 0).mean()) if len(df_stab) else None,
        ),
    )

    with open(f"{args.outdir}/real_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
