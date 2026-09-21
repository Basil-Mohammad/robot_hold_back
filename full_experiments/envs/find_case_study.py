import pickle
import sys

import numpy as np

sys.path.insert(0, ".")
from qp_core import solve_qp, active_constraints, multiplier_interval, \
    deletion_ground_truth, gram_conditioning

with open("../results/rollout_qp_dataset.pkl", "rb") as f:
    records = pickle.load(f)

rng = np.random.default_rng(7)
idx = rng.choice(len(records), size=6000, replace=False)

best = None
for i in idx:
    rec = records[i]
    H, A, b, u_pi = rec["H"], rec["A"], rec["b"], rec["u_pi"]
    p = A.shape[0]
    if p < 4:
        continue
    res = solve_qp(H, A, b, u_pi)
    if not res["feasible"] or len(res["active_set"]) == 0:
        continue
    u_star = res["u_star"]
    s = res["s"]
    lam_full = res["lam_full"]
    I = active_constraints(A, b, u_star)
    if len(I) < 1:
        continue
    y_true = deletion_ground_truth(H, A, b, u_pi, u_star)
    interval = multiplier_interval(H, A, b, u_pi, u_star, I)
    kappa = gram_conditioning(H, A, I)

    cats = []
    for j in range(p):
        v_j = bool(s[j] < -1e-9)
        y_j = y_true[j]
        if v_j and y_j:
            cats.append("cause")
        elif v_j and not y_j:
            cats.append("shadowed")
        elif not v_j and y_j:
            cats.append("counter-acting")
        else:
            cats.append("bystander")
    n_cause = cats.count("cause")
    n_shadow = cats.count("shadowed")
    n_counter = cats.count("counter-acting")
    # want: exactly one cause, at least one shadowed, ideally a counter-acting too,
    # moderate kappa (not degenerate, not trivially huge), small p for a readable table
    score = 0
    if n_cause == 1:
        score += 3
    if n_shadow >= 1:
        score += 3
    if n_counter >= 1:
        score += 5
    if 4 <= p <= 6:
        score += 2
    if np.isfinite(kappa) and 0.01 < kappa < 2.0:
        score += 1
    if best is None or score > best[0]:
        best = (score, i, dict(p=p, cats=cats, kappa=float(kappa) if np.isfinite(kappa) else None,
                                 n_cause=n_cause, n_shadow=n_shadow, n_counter=n_counter))

print(best)
i = best[1]
rec = records[i]
print("seed", rec["seed"], "episode", rec["episode"], "t", rec["t"])
print("pos", rec["pos"])
print("u_pi", rec["u_pi"])
print("u_star", rec["u_star"])
print("A\n", rec["A"])
print("b", rec["b"])
print("obstacles_eff", rec["obstacles_eff"])
print("alpha", rec["alpha"])
