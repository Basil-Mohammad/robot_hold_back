"""
Experiment 1: numerical verification of the paper's exact identities and of
the filter-interventional-faithfulness theorem (RQ1 groundwork).

For every active (i.e. intervening) instance we check:
  (a) stationarity decomposition:  u*-u_pi == H^{-1} A^T lambda*
  (b) energy identity:             2V == lambda*^T M lambda* == -lambda*^T s
  (c) faithfulness theorem:        (u*_{-i} != u*)  <=>  (lambda_i^min > 0)
      for EVERY constraint i, using the deletion ground truth.
"""
import json
import numpy as np
import pandas as pd

from qp_core import solve_qp, active_constraints, multiplier_interval, deletion_ground_truth
from scenarios import random_scenario, degenerate_scenario

rng = np.random.default_rng(42)

N_RANDOM = 1500
N_DEGENERATE = 300

rows = []
faithfulness_mismatches = []
n_active_instances = 0
n_deletion_tests = 0

def run_instance(sc, tag):
    global n_active_instances, n_deletion_tests
    H, A, b, u_pi = sc["H"], sc["A"], sc["b"], sc["u_pi"]
    res = solve_qp(H, A, b, u_pi)
    if not res["feasible"]:
        return
    u_star, lam_full, s = res["u_star"], res["lam_full"], res["s"]
    if len(res["active_set"]) == 0:
        return  # filter inactive: identities are trivial, skip
    n_active_instances += 1

    Hinv = np.linalg.inv(H)
    du_pred = Hinv @ A.T @ lam_full
    du_true = u_star - u_pi
    err_stationarity = np.linalg.norm(du_pred - du_true)

    V = 0.5 * (u_star - u_pi) @ H @ (u_star - u_pi)
    M = A @ Hinv @ A.T
    energy_via_M = lam_full @ M @ lam_full
    energy_via_s = -lam_full @ s
    err_energy = max(abs(2 * V - energy_via_M), abs(2 * V - energy_via_s))

    I = active_constraints(A, b, u_star)
    interval = multiplier_interval(H, A, b, u_pi, u_star, I)
    y_true = deletion_ground_truth(H, A, b, u_pi, u_star)
    n_deletion_tests_local = len(y_true)

    mismatch_count = 0
    for i, y_i in y_true.items():
        lmin, lmax = interval[i]
        predicted_necessary = lmin > 1e-6
        if predicted_necessary != y_i:
            mismatch_count += 1
            faithfulness_mismatches.append(dict(
                tag=tag, i=i, y_true=y_i, lmin=lmin, lmax=lmax))

    rows.append(dict(
        tag=tag, p=A.shape[0], m=A.shape[1], k_active=len(I),
        err_stationarity=err_stationarity, err_energy=err_energy,
        n_constraints_tested=n_active_instances,
        n_deletion_tests=n_deletion_tests_local,
        mismatch_count=mismatch_count,
        degenerate=int(any(abs(interval[i][1] - interval[i][0]) > 1e-6 for i in I)),
    ))
    return n_deletion_tests_local


total_deletion_tests = 0
for _ in range(N_RANDOM):
    m = rng.integers(2, 5)
    p = rng.integers(2, 7)
    sc = random_scenario(rng, m, p, violation_bias=rng.uniform(0.3, 0.8),
                          corr_pairs=rng.integers(0, 2))
    n = run_instance(sc, "random")
    if n:
        total_deletion_tests += n

for _ in range(N_DEGENERATE):
    m = rng.integers(2, 4)
    sc = degenerate_scenario(rng, m=m)
    n = run_instance(sc, "degenerate")
    if n:
        total_deletion_tests += n

df = pd.DataFrame(rows)
df.to_csv("results_identities.csv", index=False)

summary = dict(
    n_active_instances=int(n_active_instances),
    n_deletion_tests_total=int(total_deletion_tests),
    max_err_stationarity=float(df["err_stationarity"].max()),
    mean_err_stationarity=float(df["err_stationarity"].mean()),
    max_err_energy=float(df["err_energy"].max()),
    mean_err_energy=float(df["err_energy"].mean()),
    n_faithfulness_mismatches=int(len(faithfulness_mismatches)),
    n_degenerate_instances=int(df["degenerate"].sum()),
    frac_degenerate=float(df["degenerate"].mean()),
)
with open("summary_identities.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
if faithfulness_mismatches:
    print("MISMATCHES:", faithfulness_mismatches[:5])
