"""
significance_analysis_generic.py
------------------------------------
Reuses significance_analysis.py's seed-clustered bootstrap + paired
McNemar machinery (Part A only -- the naive/single/proposed comparison)
on an arbitrary results directory's real_attribution.csv, so the same
statistical treatment applied to the main PPO/maze comparison
(Sec. IX-J) also covers the SAC replication (Sec. IX-N) and the m=6 arm
dataset (Sec. IX-O), instead of leaving those two as point estimates.

Usage: python3 significance_analysis_generic.py --dir ../results_sac --label sac
       python3 significance_analysis_generic.py --dir ../results_arm --label arm
"""
import argparse
import json

import pandas as pd

from significance_analysis import prf1, mcnemar, seed_clustered_bootstrap_f1, N_BOOTSTRAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, required=True)
    ap.add_argument("--label", type=str, required=True)
    args = ap.parse_args()

    df = pd.read_csv(f"{args.dir}/real_attribution.csv")
    methods = {"pred_naive": "naive", "pred_single": "single-multiplier",
               "pred_interval": "proposed"}
    ci = {}
    for col, name in methods.items():
        _, _, f1_point = prf1(df["y_true"], df[col])
        boot = seed_clustered_bootstrap_f1(df, col)
        ci[name] = dict(f1_point=float(f1_point), **boot)
        print(f"  {name:20s} F1={f1_point:.4f}  95% CI=[{boot['lo']:.4f}, {boot['hi']:.4f}]  "
              f"(seed-clustered bootstrap, n_boot={N_BOOTSTRAP}, n_seeds={df['seed'].nunique()})")

    correct_naive = (df["pred_naive"] == df["y_true"])
    correct_single = (df["pred_single"] == df["y_true"])
    correct_proposed = (df["pred_interval"] == df["y_true"])
    mc_naive = mcnemar(correct_proposed, correct_naive)
    mc_single = mcnemar(correct_proposed, correct_single)
    print(f"  McNemar proposed vs naive:  b={mc_naive['b']}, c={mc_naive['c']}, "
          f"chi2={mc_naive['statistic']:.2f}, p={mc_naive['p_value']:.3e}")
    print(f"  McNemar proposed vs single: b={mc_single['b']}, c={mc_single['c']}, "
          f"chi2={mc_single['statistic']:.2f}, p={mc_single['p_value']:.3e}")

    out = dict(label=args.label, n_rows=len(df), n_seeds=int(df["seed"].nunique()),
               confidence_intervals=ci,
               mcnemar_proposed_vs_naive=mc_naive,
               mcnemar_proposed_vs_single=mc_single)
    outpath = f"../results/significance_summary_{args.label}.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()
