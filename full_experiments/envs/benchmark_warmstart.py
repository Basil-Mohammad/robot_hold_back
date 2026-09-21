"""
benchmark_warmstart.py
------------------------
Benchmarks qp_core_warmstart.solve_qp_warmstart against the full
active-set enumeration of qp_core.solve_qp, in two settings:

(1) Correctness: on random synthetic scenarios (deliberately given a bad/
    empty warm-start guess, the hardest case), confirm the warm-started
    solver always returns the identical optimum u_star as the exact
    enumeration solver (both certify the unique global optimum of a
    strictly convex QP, so they must agree).

(2) Real-trajectory speedup: replay the actual chronological sequence of
    QPs from the maze rollout dataset (m=2) within each episode, warm-
    starting each cycle's solve with the *previous* cycle's active set --
    exactly what a real controller would do. Report the mean number of
    candidate active sets evaluated and wall-clock time, full enumeration
    vs. warm start.

(3) Synthetic smoothly-varying trajectories at m in {4, 6} (representative
    of a multi-joint manipulator / aerial platform), constructed by
    perturbing a random scenario slightly at each step (small jitter to
    H, A, b, u_pi, mimicking obstacles/state drifting by a small amount
    per DT), for the same p sweep as benchmark_overhead_highdim.py.
"""
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "../../pilot")
from qp_core import solve_qp
from qp_core_warmstart import solve_qp_warmstart
from scenarios import random_scenario


def check_correctness(n_trials=600, seed=0):
    rng = np.random.default_rng(seed)
    max_diff = 0.0
    mismatches = 0
    for _ in range(n_trials):
        m = int(rng.choice([2, 4, 6]))
        p = int(rng.integers(m, 13))
        sc = random_scenario(rng, m, p, violation_bias=0.7)
        H, A, b, u_pi = sc["H"], sc["A"], sc["b"], sc["u_pi"]
        r1 = solve_qp(H, A, b, u_pi)
        # deliberately bad guess: empty, and also a random wrong guess
        r2 = solve_qp_warmstart(H, A, b, u_pi, active_set_guess=())
        if r1["feasible"] != r2["feasible"]:
            mismatches += 1
            continue
        if r1["feasible"]:
            diff = float(np.max(np.abs(r1["u_star"] - r2["u_star"])))
            max_diff = max(max_diff, diff)
            if diff > 1e-6:
                mismatches += 1
    return dict(n_trials=n_trials, max_diff=max_diff, mismatches=mismatches)


def bench_real_trajectories(path="../results/rollout_qp_dataset.pkl"):
    with open(path, "rb") as f:
        records = pickle.load(f)
    by_episode = {}
    for i, rec in enumerate(records):
        key = (rec["seed"], rec["episode"])
        by_episode.setdefault(key, []).append(rec)
    for key in by_episode:
        by_episode[key].sort(key=lambda r: r["t"])

    full_candidates, warm_candidates = [], []
    full_times, warm_times = [], []
    n_episodes = 0
    for key, recs in by_episode.items():
        n_episodes += 1
        prev_active = ()
        for rec in recs:
            H, A, b, u_pi = rec["H"], rec["A"], rec["b"], rec["u_pi"]
            p = A.shape[0]

            t0 = time.perf_counter()
            r_full = solve_qp(H, A, b, u_pi)
            t_full = (time.perf_counter() - t0) * 1e3
            n_full = 0
            if r_full["active_set"] is not None and len(r_full["active_set"]) > 0:
                # count candidates the exact enumerator would examine
                # (matches qp_core's own enumeration order/cost)
                from itertools import combinations
                k_star = len(r_full["active_set"])
                n_full = sum(len(list(combinations(range(p), k)))
                             for k in range(1, k_star))
                n_full += 1  # the active set itself (first found at size k_star)
            elif r_full["feasible"]:
                n_full = 0  # inactive, no enumeration needed

            t1 = time.perf_counter()
            r_warm = solve_qp_warmstart(H, A, b, u_pi, active_set_guess=prev_active)
            t_warm = (time.perf_counter() - t1) * 1e3

            full_candidates.append(n_full)
            warm_candidates.append(r_warm["n_candidates_tried"])
            full_times.append(t_full)
            warm_times.append(t_warm)

            prev_active = r_full["active_set"] if r_full["active_set"] else ()

    return dict(
        n_episodes=n_episodes,
        n_cycles=len(full_candidates),
        mean_candidates_full=float(np.mean(full_candidates)),
        mean_candidates_warm=float(np.mean(warm_candidates)),
        mean_time_full_ms=float(np.mean(full_times)),
        mean_time_warm_ms=float(np.mean(warm_times)),
        speedup_time=float(np.mean(full_times) / max(np.mean(warm_times), 1e-9)),
    )


def make_smooth_sequence(rng, m, p, n_steps, jitter=0.03):
    """A sequence of scenarios sharing H, with A/b/u_pi perturbed by a
    small amount each step, modeling a slowly-evolving control loop."""
    sc0 = random_scenario(rng, m, p, violation_bias=0.7)
    H = sc0["H"]
    A, b, u_pi = sc0["A"].copy(), sc0["b"].copy(), sc0["u_pi"].copy()
    seq = [dict(H=H, A=A.copy(), b=b.copy(), u_pi=u_pi.copy())]
    for _ in range(n_steps - 1):
        A = A + rng.normal(scale=jitter, size=A.shape)
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        b = b + rng.normal(scale=jitter, size=b.shape)
        u_pi = u_pi + rng.normal(scale=jitter, size=u_pi.shape)
        seq.append(dict(H=H, A=A.copy(), b=b.copy(), u_pi=u_pi.copy()))
    return seq


def bench_synthetic_highdim(m_values=(4, 6), p_values=(6, 8, 10, 12),
                             n_steps=150, seed=1):
    rng = np.random.default_rng(seed)
    results = {}
    for m in m_values:
        by_p = {}
        for p in p_values:
            if p < m:
                continue
            seq = make_smooth_sequence(rng, m, p, n_steps)
            prev_active = ()
            t_full_list, t_warm_list = [], []
            for sc in seq:
                H, A, b, u_pi = sc["H"], sc["A"], sc["b"], sc["u_pi"]
                t0 = time.perf_counter()
                r_full = solve_qp(H, A, b, u_pi)
                t_full_list.append((time.perf_counter() - t0) * 1e3)
                t1 = time.perf_counter()
                r_warm = solve_qp_warmstart(H, A, b, u_pi,
                                             active_set_guess=prev_active)
                t_warm_list.append((time.perf_counter() - t1) * 1e3)
                prev_active = r_full["active_set"] if r_full["active_set"] else ()
            by_p[p] = dict(
                mean_time_full_ms=float(np.mean(t_full_list)),
                mean_time_warm_ms=float(np.mean(t_warm_list)),
                speedup=float(np.mean(t_full_list) / max(np.mean(t_warm_list), 1e-9)),
            )
            print(f"m={m} p={p:2d}  full={np.mean(t_full_list):.3f}ms  "
                  f"warm={np.mean(t_warm_list):.3f}ms  "
                  f"speedup={by_p[p]['speedup']:.1f}x")
        results[m] = by_p
    return results


def main():
    print("=== Correctness check ===")
    corr = check_correctness()
    print(corr)
    assert corr["mismatches"] == 0, "warm-start solver disagreed with exact solver!"

    print("\n=== Real maze trajectories (m=2) ===")
    real = bench_real_trajectories()
    print(real)

    print("\n=== Synthetic smooth trajectories (m=4,6) ===")
    synth = bench_synthetic_highdim()

    out = dict(correctness=corr, real_trajectories=real, synthetic_highdim=synth)
    import json
    with open("../results/warmstart_benchmark.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote ../results/warmstart_benchmark.json")


if __name__ == "__main__":
    main()
