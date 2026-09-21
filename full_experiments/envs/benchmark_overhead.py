"""
benchmark_overhead.py
----------------------
Computational-overhead benchmark for VIII-H (Computational Metrics):
  - QP solve time (the CBF-QP filter itself)
  - explanation overhead (multiplier interval + deletion ground truth +
    conditioning indicator, i.e. everything the runtime explanation
    Algorithm 1 needs beyond the bare filter)
  - counterfactual computation time (3,600-direction angular sweep)
  - scaling with the number of constraints p (m=2, single-integrator
    action dimension, matching the maze robot)
  - total control-cycle latency = QP solve + full explanation, compared
    against the environment's control period DT=0.05s (20 Hz)

All numbers are wall-clock, single CPU core, no batching/vectorization
beyond what qp_core.py already does, i.e. a conservative (pessimistic)
estimate of deployability.
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

DT_CONTROL = 0.05  # seconds, matches SafeMazeEnv


def angular_min_step(pos, obstacles, angles):
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


def time_block(fn, n_repeats=200):
    ts = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return dict(mean_ms=float(ts.mean() * 1e3), std_ms=float(ts.std() * 1e3),
                p50_ms=float(np.percentile(ts, 50) * 1e3),
                p95_ms=float(np.percentile(ts, 95) * 1e3),
                max_ms=float(ts.max() * 1e3))


def main():
    rng = np.random.default_rng(0)
    m = 2
    angles = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    p_values = [2, 4, 6, 8, 10, 12]
    n_repeats = 200

    results = {"m": m, "n_repeats": n_repeats, "dt_control_s": DT_CONTROL,
               "by_p": {}}

    for p in p_values:
        scenarios = [random_scenario(rng, m, p, violation_bias=0.9)
                     for _ in range(n_repeats)]

        # --- QP solve time ---
        t_solve = []
        for sc in scenarios:
            H, A, b, u_pi = sc["H"], sc["A"], sc["b"], sc["u_pi"]
            t0 = time.perf_counter()
            res = solve_qp(H, A, b, u_pi)
            t_solve.append(time.perf_counter() - t0)
        t_solve = np.array(t_solve) * 1e3

        # --- full explanation overhead (attribution + conditioning) ---
        t_attr = []
        t_cf = []
        t_total = []
        obstacles = [(rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0.3, 0.8))
                     for _ in range(p)]
        pos = np.zeros(2)
        for sc in scenarios:
            H, A, b, u_pi = sc["H"], sc["A"], sc["b"], sc["u_pi"]
            t0 = time.perf_counter()
            res = solve_qp(H, A, b, u_pi)
            t_qp = time.perf_counter() - t0
            if not res["feasible"] or len(res["active_set"]) == 0:
                continue
            I = active_constraints(A, b, res["u_star"])
            t1 = time.perf_counter()
            interval = multiplier_interval(H, A, b, u_pi, res["u_star"], I)
            y_true = deletion_ground_truth(H, A, b, u_pi, res["u_star"])
            kappa = gram_conditioning(H, A, I)
            t_attr_i = time.perf_counter() - t1
            t2 = time.perf_counter()
            _ = angular_min_step(pos, obstacles, angles)
            t_cf_i = time.perf_counter() - t2
            t_attr.append(t_attr_i * 1e3)
            t_cf.append(t_cf_i * 1e3)
            t_total.append((t_qp + t_attr_i + t_cf_i) * 1e3)

        results["by_p"][p] = dict(
            qp_solve_ms=dict(mean=float(t_solve.mean()), p95=float(np.percentile(t_solve, 95))),
            attribution_overhead_ms=dict(mean=float(np.mean(t_attr)), p95=float(np.percentile(t_attr, 95))) if t_attr else None,
            counterfactual_ms=dict(mean=float(np.mean(t_cf)), p95=float(np.percentile(t_cf, 95))) if t_cf else None,
            total_explained_control_cycle_ms=dict(mean=float(np.mean(t_total)), p95=float(np.percentile(t_total, 95))) if t_total else None,
        )
        print(f"p={p:3d}  QP solve mean={t_solve.mean():.3f}ms  "
              f"attribution mean={np.mean(t_attr) if t_attr else float('nan'):.3f}ms  "
              f"counterfactual mean={np.mean(t_cf) if t_cf else float('nan'):.3f}ms")

    # headline numbers at the operating point used in the real experiment (K=10)
    p_op = 10
    op = results["by_p"][p_op]
    results["headline_at_K10"] = dict(
        qp_solve_ms=op["qp_solve_ms"]["mean"],
        explanation_overhead_ms=(op["attribution_overhead_ms"]["mean"] if op["attribution_overhead_ms"] else None),
        counterfactual_ms=(op["counterfactual_ms"]["mean"] if op["counterfactual_ms"] else None),
        total_control_cycle_ms=(op["total_explained_control_cycle_ms"]["mean"] if op["total_explained_control_cycle_ms"] else None),
        control_period_ms=DT_CONTROL * 1e3,
        headroom_pct=(1 - (op["total_explained_control_cycle_ms"]["mean"] / 1000.0) / DT_CONTROL) * 100
            if op["total_explained_control_cycle_ms"] else None,
    )

    with open("../results/overhead_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["headline_at_K10"], indent=2))


if __name__ == "__main__":
    main()
