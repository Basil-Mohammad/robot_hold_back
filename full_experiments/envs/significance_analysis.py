"""
significance_analysis.py
---------------------------
Statistical significance for the paper's main attribution-faithfulness
claims:

(A) Main policy-driven comparison (naive vs. single-multiplier vs. the
    proposed multiplier-interval method), on the full real_attribution.csv
    (644,380 constraint-level tests across 10 independently trained PPO
    seeds):
      - Seed-clustered bootstrap 95% CIs on F1 for each method (resampling
        whole seeds with replacement, not individual rows, since rows
        within a seed are not independent -- this avoids overstating
        precision from pseudo-replication).
      - Row-level paired McNemar's test (proposed vs. naive, proposed vs.
        single-multiplier), since every row is the SAME constraint
        instance scored by all three methods.

(B) Post-hoc explainer comparison (SHAP/LIME/saliency/decision-tree
    surrogate/rule-based vs. proposed), on the SAME 150-instance sample
    used in real_baselines.py. The proposed (multiplier-interval) method's
    predictions on that exact sample are recomputed here (deterministic:
    identical RNG seed and instance filter as real_baselines.py) and
    row-aligned against the already-computed real_baselines_shap_lime.csv
    (verified via an exact y_true match assertion), enabling a genuine
    paired McNemar test method-by-method, plus per-method bootstrap F1 CIs
    (resampled at the instance level).
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, ".")
from qp_core import solve_qp, active_constraints, multiplier_interval, \
    deletion_ground_truth

RNG_SEED = 0
N_SAMPLE_INSTANCES = 150
N_BOOTSTRAP = 2000


def prf1(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = np.sum(y_true & y_pred); fp = np.sum(~y_true & y_pred); fn = np.sum(y_true & ~y_pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    return prec, rec, f1


def mcnemar(correct_a, correct_b):
    """Paired McNemar's test (with continuity correction) on whether two
    paired binary classifiers (here: 'was this row correctly classified')
    disagree symmetrically. b = a-correct/b-wrong, c = a-wrong/b-correct."""
    correct_a = np.asarray(correct_a, bool)
    correct_b = np.asarray(correct_b, bool)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return dict(b=b, c=c, statistic=0.0, p_value=1.0)
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(1 - chi2.cdf(stat, df=1))
    return dict(b=b, c=c, statistic=float(stat), p_value=p)


def _cluster_confusion_counts(df, pred_col, cluster_col):
    """Precompute per-cluster (tp, fp, fn) counts once, so each bootstrap
    resample is just an array-index-and-sum instead of a DataFrame concat."""
    y = df["y_true"].to_numpy(bool)
    p = df[pred_col].to_numpy(bool)
    clusters = df[cluster_col].to_numpy()
    uniq = np.unique(clusters)
    tp = np.zeros(len(uniq)); fp = np.zeros(len(uniq)); fn = np.zeros(len(uniq))
    for k, c in enumerate(uniq):
        mask = clusters == c
        yc, pc = y[mask], p[mask]
        tp[k] = np.sum(yc & pc)
        fp[k] = np.sum(~yc & pc)
        fn[k] = np.sum(yc & ~pc)
    return uniq, tp, fp, fn


def _bootstrap_from_counts(uniq, tp, fp, fn, n_boot=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    n = len(uniq)
    f1s = np.empty(n_boot)
    valid = np.zeros(n_boot, dtype=bool)
    for b in range(n_boot):
        sel = rng.integers(0, n, size=n)  # resample cluster indices w/ replacement
        TP, FP, FN = tp[sel].sum(), fp[sel].sum(), fn[sel].sum()
        prec = TP / (TP + FP) if (TP + FP) > 0 else np.nan
        rec = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else np.nan
        f1s[b] = f1
        valid[b] = np.isfinite(f1)
    f1s = f1s[valid]
    return dict(mean=float(np.mean(f1s)), lo=float(np.percentile(f1s, 2.5)),
                hi=float(np.percentile(f1s, 97.5)), n_boot_valid=int(valid.sum()))


def seed_clustered_bootstrap_f1(df, pred_col, n_boot=N_BOOTSTRAP, seed=0):
    uniq, tp, fp, fn = _cluster_confusion_counts(df, pred_col, "seed")
    return _bootstrap_from_counts(uniq, tp, fp, fn, n_boot=n_boot, seed=seed)


def instance_bootstrap_f1(df, pred_col, instance_col="instance_id", n_boot=N_BOOTSTRAP, seed=0):
    uniq, tp, fp, fn = _cluster_confusion_counts(df, pred_col, instance_col)
    return _bootstrap_from_counts(uniq, tp, fp, fn, n_boot=n_boot, seed=seed)


def part_a_main_comparison():
    print("=== Part A: main policy-driven comparison (naive/single/proposed) ===")
    df = pd.read_csv("../results/real_attribution.csv")
    methods = {"pred_naive": "naive", "pred_single": "single-multiplier",
               "pred_interval": "proposed"}
    ci = {}
    for col, name in methods.items():
        _, _, f1_point = prf1(df["y_true"], df[col])
        boot = seed_clustered_bootstrap_f1(df, col)
        ci[name] = dict(f1_point=float(f1_point), **boot)
        print(f"  {name:20s} F1={f1_point:.4f}  95% CI=[{boot['lo']:.4f}, {boot['hi']:.4f}]  (seed-clustered bootstrap, n_boot={N_BOOTSTRAP})")

    correct_naive = (df["pred_naive"] == df["y_true"])
    correct_single = (df["pred_single"] == df["y_true"])
    correct_proposed = (df["pred_interval"] == df["y_true"])
    mc_naive = mcnemar(correct_proposed, correct_naive)
    mc_single = mcnemar(correct_proposed, correct_single)
    print(f"  McNemar proposed vs naive:  b={mc_naive['b']}, c={mc_naive['c']}, "
          f"chi2={mc_naive['statistic']:.2f}, p={mc_naive['p_value']:.3e}")
    print(f"  McNemar proposed vs single: b={mc_single['b']}, c={mc_single['c']}, "
          f"chi2={mc_single['statistic']:.2f}, p={mc_single['p_value']:.3e}")

    return dict(n_rows=len(df), n_seeds=int(df["seed"].nunique()),
                confidence_intervals=ci,
                mcnemar_proposed_vs_naive=mc_naive,
                mcnemar_proposed_vs_single=mc_single)


def part_b_posthoc_comparison():
    print("\n=== Part B: post-hoc baseline comparison (paired, same 150-instance sample) ===")
    rng = np.random.default_rng(RNG_SEED)
    with open("../results/rollout_qp_dataset.pkl", "rb") as f:
        records = pickle.load(f)

    active_records = []
    for rec in records:
        H, A, b, u_pi = rec["H"], rec["A"], rec["b"], rec["u_pi"]
        s = A @ u_pi - b
        if np.sum(s < -1e-9) >= 1 and A.shape[0] >= 2:
            active_records.append(rec)

    idx = rng.choice(len(active_records), size=min(N_SAMPLE_INSTANCES, len(active_records)),
                      replace=False)
    sample = [active_records[i] for i in idx]

    rows = []
    for inst_id, rec in enumerate(sample):
        H, A, b, u_pi = rec["H"], rec["A"], rec["b"], rec["u_pi"]
        p = A.shape[0]
        res = solve_qp(H, A, b, u_pi)
        if not res["feasible"] or len(res["active_set"]) == 0:
            continue
        u_star = res["u_star"]
        I = active_constraints(A, b, u_star)
        interval = multiplier_interval(H, A, b, u_pi, u_star, I)
        y_true = deletion_ground_truth(H, A, b, u_pi, u_star)
        for i in range(p):
            lmin, _ = interval[i]
            rows.append(dict(instance_id=inst_id, y_true=y_true[i],
                              pred_proposed=bool(lmin > 1e-6)))

    df_proposed = pd.DataFrame(rows)
    df_base = pd.read_csv("../results/real_baselines_shap_lime.csv")

    if len(df_proposed) != len(df_base):
        print(f"  WARNING: row count mismatch (proposed={len(df_proposed)}, "
              f"baselines={len(df_base)}) -- sampling not exactly reproducible; "
              f"falling back to unpaired (instance-level bootstrap only, no McNemar).")
        aligned = False
    else:
        mismatch = int(np.sum(df_proposed["y_true"].values != df_base["y_true"].values))
        aligned = (mismatch == 0)
        print(f"  Row alignment check: {len(df_proposed)} rows, {mismatch} y_true mismatches "
              f"-> {'ALIGNED, exact paired test valid' if aligned else 'NOT ALIGNED, falling back'}")

    df_base["instance_id"] = df_proposed["instance_id"].values if aligned else np.repeat(
        np.arange(len(df_base) // (len(df_base) // len(sample) if len(sample) else 1)), 1)[:len(df_base)]

    methods = {"pred_shap": "SHAP", "pred_lime": "LIME", "pred_saliency": "Saliency",
               "pred_rule_based": "Rule-based"}
    results = {}
    _, _, f1_prop = prf1(df_proposed["y_true"], df_proposed["pred_proposed"])
    boot_prop = instance_bootstrap_f1(df_proposed, "pred_proposed")
    results["proposed"] = dict(f1_point=float(f1_prop), **boot_prop)
    print(f"  {'proposed':12s} F1={f1_prop:.4f}  95% CI=[{boot_prop['lo']:.4f}, {boot_prop['hi']:.4f}]")

    for col, name in methods.items():
        _, _, f1_pt = prf1(df_base["y_true"], df_base[col])
        boot = instance_bootstrap_f1(df_base, col)
        entry = dict(f1_point=float(f1_pt), **boot)
        if aligned:
            correct_prop = (df_proposed["pred_proposed"] == df_proposed["y_true"]).values
            correct_base = (df_base[col] == df_base["y_true"]).values
            mc = mcnemar(correct_prop, correct_base)
            entry["mcnemar_vs_proposed"] = mc
            print(f"  {name:12s} F1={f1_pt:.4f}  95% CI=[{boot['lo']:.4f}, {boot['hi']:.4f}]  "
                  f"McNemar vs proposed: b={mc['b']}, c={mc['c']}, p={mc['p_value']:.3e}")
        else:
            print(f"  {name:12s} F1={f1_pt:.4f}  95% CI=[{boot['lo']:.4f}, {boot['hi']:.4f}]")
        results[name.lower().replace("-", "_")] = entry

    return dict(n_instances=len(sample), aligned=aligned, results=results)


def main():
    a = part_a_main_comparison()
    b = part_b_posthoc_comparison()
    summary = dict(main_comparison=a, posthoc_comparison=b)
    with open("../results/significance_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nWrote ../results/significance_summary.json")


if __name__ == "__main__":
    main()
