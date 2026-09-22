"""
moving_obstacle_analysis.py
------------------------------
RQ1-RQ4 on the moving-obstacle rollout dataset (collect_rollouts_moving.py
output), reusing the exact same analyze_instance() machinery as
run_full_analysis.py (same solver, same attribution/counterfactual/
stability code -- nothing moving-obstacle-specific in qp_core.py or
analyze_instance itself, by design: the human row is just one more affine
constraint row to that code). This script ADDS one thing on top: every
attribution row is tagged with whether it came from the moving human row
or from a static wall/pillar row (rec["is_moving"][i]), so we can report
attribution/taxonomy accuracy broken down by moving vs. static -- the
central empirical question for this generalization check (does the method
degrade on the one row whose (A,b) changes with time?).

Usage: python3 moving_obstacle_analysis.py
"""
import json
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from qp_core import solve_qp, active_constraints, multiplier_interval, deletion_ground_truth
from run_full_analysis import analyze_instance, category


def analyze_instance_tagged(rec, angles, rng):
    """Same as analyze_instance, but tags each attribution row with
    is_moving (bool) using rec['is_moving'], and separately reports the
    moving row's own kappa-stab / degeneracy behavior."""
    out = analyze_instance(rec, angles, rng)
    if out is None:
        return None
    is_moving = rec.get("is_moving")
    if is_moving is not None:
        for i, row in enumerate(out["attr_rows"]):
            row["is_moving"] = bool(is_moving[i])
    return out


def prf1(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = np.sum(y_true & y_pred); fp = np.sum(~y_true & y_pred); fn = np.sum(y_true & ~y_pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec_ = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * prec * rec_ / (prec + rec_) if (prec == prec and rec_ == rec_ and (prec + rec_) > 0) else np.nan
    return prec, rec_, f1


def main():
    with open("../results/rollout_qp_dataset_moving.pkl", "rb") as f:
        records = pickle.load(f)
    print(f"loaded {len(records)} QP instances (moving-obstacle env)")

    angles = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    rng = np.random.default_rng(42)

    attr_records = []
    cf_records = []
    stab_records = []
    id_stationarity, id_energy = [], []
    n_human_binding = 0

    for k, rec in enumerate(records):
        out = analyze_instance_tagged(rec, angles, rng)
        if out is None:
            continue
        for row in out["attr_rows"]:
            row["seed"] = out["seed"]
            attr_records.append(row)
            if row.get("is_moving"):
                n_human_binding += 1 if row["v_i"] else 0
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
    df_attr.to_csv("../results/moving_attribution.csv", index=False)
    df_cf.to_csv("../results/moving_counterfactual.csv", index=False)
    df_stab.to_csv("../results/moving_stability.csv", index=False)

    methods = ["pred_naive", "pred_single", "pred_interval"]

    def block_summary(sub):
        out = {}
        for m in methods:
            p_, r_, f1_ = prf1(sub["y_true"], sub[m])
            out[m] = dict(precision=p_, recall=r_, f1=f1_, n=int(len(sub)))
        return out

    overall = block_summary(df_attr)
    moving_only = block_summary(df_attr[df_attr["is_moving"] == True])
    static_only = block_summary(df_attr[df_attr["is_moving"] == False])

    by_category = {}
    for cat in ["cause", "shadowed", "counter-acting", "bystander"]:
        sub = df_attr[df_attr["category"] == cat]
        by_category[cat] = dict(n=int(len(sub)))
        for m in methods:
            by_category[cat][m] = float((sub["y_true"] != sub[m]).mean()) if len(sub) else float("nan")

    cf_summary = dict(
        n=len(df_cf),
        bound_always_holds=bool(df_cf["bound_holds"].all()) if len(df_cf) else None,
        mean_rel_gap=float(df_cf["rel_gap"].mean()) if len(df_cf) else None,
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
        n_human_row_binding=int(n_human_binding),
        identities=dict(
            max_err_stationarity=float(np.max(id_stationarity)) if id_stationarity else None,
            max_err_energy=float(np.max(id_energy)) if id_energy else None,
        ),
        attribution=dict(overall=overall, moving_row_only=moving_only, static_rows_only=static_only,
                          by_category=by_category,
                          category_counts=df_attr["category"].value_counts().to_dict()),
        counterfactual=cf_summary,
        stability=dict(
            n=len(stab_sub),
            spearman_rho_kappa_vs_sensitivity=float(rho),
            spearman_p_kappa_vs_sensitivity=float(pval),
        ),
    )

    with open("../results/moving_obstacle_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
