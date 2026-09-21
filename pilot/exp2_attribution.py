"""
Experiment 2 (RQ1 / RQ2): attribution faithfulness.

For every intervening instance and every constraint i we know the ground
truth y_i (does removing i change u*, via exact re-solve). We compare four
attribution methods' binary "is i responsible" prediction against y_i:

  1. naive-violated     : predict necessary iff s_i < 0  (the common
                           "which constraints did the robot violate"
                           heuristic used in practice)
  2. single-multiplier   : predict necessary iff the ONE multiplier returned
                           by an ordinary KKT/QP solve is > 0 (this is what
                           you get from cvxpy/OSQP etc. without degeneracy
                           handling -- an arbitrary vertex of the optimal
                           dual set when it is not a singleton)
  3. shapley-constraint  : cooperative-game (SHAP-style) credit assignment
                           over the power set of constraints, using
                           intervention energy v(S) = 2*V(S) as the
                           characteristic function; predict necessary iff
                           phi_i > eps. Only computed for p <= 6 (64
                           sub-solves/instance) as a second, more expensive
                           baseline.
  4. deletion-interval   : the paper's proposed method, predict necessary
                           iff lambda_i^min > 0 (Theorem: filter-
                           interventional faithfulness).

Metrics: precision/recall/F1 overall and split by the taxonomy category
(cause / shadowed / counter-acting / bystander).
"""
import itertools
import json
import math
import numpy as np
import pandas as pd

from qp_core import solve_qp, active_constraints, multiplier_interval, deletion_ground_truth
from scenarios import random_scenario, degenerate_scenario

rng = np.random.default_rng(7)

records = []


def category(v_i, y_i):
    if v_i and y_i:
        return "cause"
    if v_i and not y_i:
        return "shadowed"
    if not v_i and y_i:
        return "counter-acting"
    return "bystander"


def shapley_constraint_attribution(H, A, b, u_pi):
    p = A.shape[0]
    idx = list(range(p))
    Hinv = np.linalg.inv(H)

    def V(S):
        if len(S) == 0:
            return 0.0
        A_S = A[list(S), :]
        b_S = b[list(S)]
        res = solve_qp(H, A_S, b_S, u_pi)
        u_S = res["u_star"]
        return 0.5 * (u_S - u_pi) @ H @ (u_S - u_pi)

    fact = math.factorial
    phi = np.zeros(p)
    others = idx
    for i in idx:
        rest = [j for j in idx if j != i]
        total = 0.0
        for k in range(len(rest) + 1):
            for S in itertools.combinations(rest, k):
                w = fact(k) * fact(p - k - 1) / fact(p)
                total += w * (V(tuple(sorted(S + (i,)))) - V(S))
        phi[i] = total
    return phi


def process_instance(sc, tag, do_shapley):
    H, A, b, u_pi = sc["H"], sc["A"], sc["b"], sc["u_pi"]
    res = solve_qp(H, A, b, u_pi)
    if not res["feasible"] or len(res["active_set"]) == 0:
        return
    u_star, s = res["u_star"], res["s"]
    I = active_constraints(A, b, u_star)
    interval = multiplier_interval(H, A, b, u_pi, u_star, I)
    y_true = deletion_ground_truth(H, A, b, u_pi, u_star)

    phi = None
    if do_shapley and A.shape[0] <= 6:
        phi = shapley_constraint_attribution(H, A, b, u_pi)

    for i, y_i in y_true.items():
        v_i = bool(s[i] < -1e-9)
        lmin, lmax = interval[i]
        lam_single = res["lam_full"][i]
        rec = dict(
            tag=tag, p=A.shape[0], m=A.shape[1],
            v_i=v_i, y_true=y_i,
            category=category(v_i, y_i),
            pred_naive=v_i,
            pred_single=bool(lam_single > 1e-6),
            pred_interval=bool(lmin > 1e-6),
        )
        if phi is not None:
            rec["pred_shapley"] = bool(phi[i] > 1e-6)
            rec["phi_i"] = float(phi[i])
        records.append(rec)


N_MAIN = 2500
N_DEGEN = 400
N_SHAPLEY = 500

for _ in range(N_MAIN):
    m = rng.integers(2, 5)
    p = rng.integers(2, 8)
    sc = random_scenario(rng, m, p, violation_bias=rng.uniform(0.3, 0.85),
                          corr_pairs=rng.integers(0, 3))
    process_instance(sc, "random", do_shapley=False)

for _ in range(N_DEGEN):
    m = rng.integers(2, 4)
    sc = degenerate_scenario(rng, m=m)
    process_instance(sc, "degenerate", do_shapley=False)

for _ in range(N_SHAPLEY):
    m = rng.integers(2, 4)
    p = rng.integers(2, 6)
    sc = random_scenario(rng, m, p, violation_bias=rng.uniform(0.3, 0.85),
                          corr_pairs=rng.integers(0, 3))
    process_instance(sc, "shapley_subset", do_shapley=True)

df = pd.DataFrame(records)
df.to_csv("results_attribution.csv", index=False)


def prf1(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else float("nan"))
    return dict(precision=precision, recall=recall, f1=f1, support=int(y_true.sum()), n=len(y_true))


methods = ["pred_naive", "pred_single", "pred_interval"]
summary = {"overall": {}, "by_category": {}}

for meth in methods:
    summary["overall"][meth] = prf1(df["y_true"], df[meth])

def error_rate(y_true, y_pred):
    # Within a taxonomy category y_true is constant by construction
    # (cause/counter-acting => 1, shadowed/bystander => 0), so
    # precision/recall degenerate there; the informative statistic is the
    # misclassification rate against that constant ground truth (i.e. the
    # false-positive rate for shadowed/bystander, the miss/false-negative
    # rate for cause/counter-acting).
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    n = len(y_true)
    if n == 0:
        return dict(error_rate=float("nan"), n=0)
    wrong = np.sum(y_true != y_pred)
    return dict(error_rate=float(wrong / n), n=int(n), n_wrong=int(wrong))


for cat in ["cause", "shadowed", "counter-acting", "bystander"]:
    sub = df[df["category"] == cat]
    summary["by_category"][cat] = {
        meth: error_rate(sub["y_true"], sub[meth]) for meth in methods
    }
    summary["by_category"][cat]["n"] = int(len(sub))

sh = df[df["pred_shapley"].notna()] if "pred_shapley" in df.columns else df.iloc[0:0]
if len(sh):
    summary["overall"]["pred_shapley"] = prf1(sh["y_true"], sh["pred_shapley"])
    summary["by_category_shapley"] = {}
    for cat in ["cause", "shadowed", "counter-acting", "bystander"]:
        subsh = sh[sh["category"] == cat]
        summary["by_category_shapley"][cat] = error_rate(subsh["y_true"], subsh["pred_shapley"]) if len(subsh) else None

summary["category_counts"] = df["category"].value_counts().to_dict()
summary["n_instances_total_rows"] = len(df)

with open("summary_attribution.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(json.dumps(summary, indent=2, default=str))
