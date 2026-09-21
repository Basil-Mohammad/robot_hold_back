"""
benchmark_overhead_highdim.py
--------------------------------
Extends benchmark_overhead.py's scaling analysis from the paper's own
m=2 (single-integrator planar robot) operating point to higher action
dimensions m in {2, 4, 6}, representative of a multi-jointed manipulator
or a full 6-DOF (position+attitude) aerial/ground platform. Times QP
solve and full attribution overhead (multiplier interval + deletion
ground truth + conditioning) as a function of both p and m; the
counterfactual step is intentionally excluded here since its closed form
in the paper is specific to 2D circular obstacles (Sec. IV-C) and does
not directly generalize to m>2 without a separate derivation -- this
benchmark isolates the solver/attribution machinery, which IS already
fully dimension-generic (qp_core.py makes no assumption on m).
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "../../pilot")
from qp_core import solve_qp, active_constraints, multiplier_interval, \
    deletion_ground_truth, gram_conditioning
from scenarios import random_scenario

DT_CONTROL = 0.05


def main():
    rng = np.random.default_rng(0)
    m_values = [2, 4, 6]
    p_values = [2, 4, 6, 8, 10, 12]
    n_repeats = 150

    results = {"n_repeats": n_repeats, "dt_control_s": DT_CONTROL, "by_m": {}}

    for m in m_values:
        by_p = {}
        for p in p_values:
            if p < m:
                continue  # need at least m constraints for a full-rank active set to exist
            scenarios = [random_scenario(rng, m, p, violation_bias=0.9)
                         for _ in range(n_repeats)]
            t_solve, t_attr = [], []
            for sc in scenarios:
                H, A, b, u_pi = sc["H"], sc["A"], sc["b"], sc["u_pi"]
                t0 = time.perf_counter()
                res = solve_qp(H, A, b, u_pi)
                t_solve.append((time.perf_counter() - t0) * 1e3)
                if not res["feasible"] or len(res["active_set"]) == 0:
                    continue
                I = active_constraints(A, b, res["u_star"])
                t1 = time.perf_counter()
                multiplier_interval(H, A, b, u_pi, res["u_star"], I)
                deletion_ground_truth(H, A, b, u_pi, res["u_star"])
                gram_conditioning(H, A, I)
                t_attr.append((time.perf_counter() - t1) * 1e3)
            t_solve = np.array(t_solve)
            t_attr = np.array(t_attr) if t_attr else np.array([np.nan])
            by_p[p] = dict(
                qp_solve_ms=float(t_solve.mean()),
                attribution_ms=float(t_attr.mean()),
                total_ms=float(t_solve.mean() + t_attr.mean()),
                headroom_pct=float((1 - (t_solve.mean() + t_attr.mean()) / 1000.0 / DT_CONTROL) * 100),
            )
            print(f"m={m}  p={p:3d}  QP solve={t_solve.mean():.3f}ms  "
                  f"attribution={t_attr.mean():.3f}ms  total={t_solve.mean()+t_attr.mean():.3f}ms")
        results["by_m"][m] = by_p

    with open("../results/overhead_highdim_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote ../results/overhead_highdim_benchmark.json")


if __name__ == "__main__":
    main()
