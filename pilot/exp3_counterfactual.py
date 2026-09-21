"""
Experiment 3 (RQ3): certified counterfactual explanations.

Part A -- single circular obstacle: compare the paper's closed-form
displacement  d_cf = R - ||d-c||  against an INDEPENDENT exact computation
(a dense angular sweep that, for every direction e, solves the exact
1-D quadratic for the minimal positive step reaching the constraint
boundary, then takes the global minimum over directions). This does not
reuse any of the paper's algebra, so it is a genuine external check.

Part B -- multiple circular obstacles sharing one perturbation of the
robot position: build the two-sided certificate
    d_lower = max_i (single-obstacle bound for obstacle i)
    d_feasible = best point found by the same angular sweep, now requiring
                 simultaneous feasibility of all obstacles
and check d_lower <= d_feasible always holds, and how tight the gap is.
"""
import json
import numpy as np
import pandas as pd

from scenarios import circular_obstacle_1, circular_obstacle_multi

rng = np.random.default_rng(123)

N_SINGLE = 1000
N_MULTI = 500
N_ANGLES = 3600


def angular_min_step(d, c, R, angles):
    """For each unit direction e=[cos,sin], minimal t>0 s.t. ||d+t e-c||=R."""
    e = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (N,2)
    v = d - c
    b_coef = 2 * (e @ v)
    c_coef = v @ v - R ** 2
    disc = b_coef ** 2 - 4 * c_coef
    disc = np.clip(disc, 0, None)
    sqrt_disc = np.sqrt(disc)
    t1 = (-b_coef + sqrt_disc) / 2
    t2 = (-b_coef - sqrt_disc) / 2
    t = np.where(t1 >= -1e-12, t1, np.inf)
    t2v = np.where(t2 >= -1e-12, t2, np.inf)
    t = np.minimum(t, t2v)
    return t  # array of length N_ANGLES, minimal nonneg step per direction


rows_single = []
angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)

for _ in range(N_SINGLE):
    sc = circular_obstacle_1(rng)
    alpha, o, r, p, d = sc["alpha"], sc["o"], sc["r"], sc["p"], sc["d"]
    u_pi = sc["u_pi"]
    c = -u_pi / alpha
    R = np.sqrt(r ** 2 + (u_pi @ u_pi) / alpha ** 2)
    dist = np.linalg.norm(d - c)
    if dist >= R:
        continue  # filter already inactive, no counterfactual needed
    d_cf_closed = R - dist

    t_all = angular_min_step(d, c, R, angles)
    d_cf_numeric = float(np.min(t_all))

    rows_single.append(dict(
        d_cf_closed=d_cf_closed, d_cf_numeric=d_cf_numeric,
        abs_err=abs(d_cf_closed - d_cf_numeric),
        rel_err=abs(d_cf_closed - d_cf_numeric) / max(d_cf_closed, 1e-9),
        R=R, dist=dist,
    ))

df_single = pd.DataFrame(rows_single)
df_single.to_csv("results_cf_single.csv", index=False)

rows_multi = []
for _ in range(N_MULTI):
    sc = circular_obstacle_multi(rng, n_obs=rng.integers(2, 5))
    alpha, obstacles, p, ds = sc["alpha"], sc["obstacles"], sc["p"], sc["ds"]
    u_pi = sc["u_pi"]
    c = -u_pi / alpha
    Rs, dists, individual_cf = [], [], []
    any_violated = False
    for (o, r), d in zip(obstacles, ds):
        R = np.sqrt(r ** 2 + (u_pi @ u_pi) / alpha ** 2)
        dist = np.linalg.norm(d - c)
        Rs.append(R)
        dists.append(dist)
        if dist < R:
            any_violated = True
            individual_cf.append(R - dist)
        else:
            individual_cf.append(0.0)
    if not any_violated:
        continue
    d_lower = max(individual_cf)

    # simultaneous feasibility sweep: for each direction, the required
    # step is the max over obstacles of their individual required step in
    # that direction (0 if already feasible for that obstacle)
    t_needed = np.zeros(N_ANGLES)
    for (o, r), d, R in zip(obstacles, ds, Rs):
        dist = np.linalg.norm(d - c)
        if dist >= R:
            continue
        t_all = angular_min_step(d, c, R, angles)
        t_needed = np.maximum(t_needed, t_all)
    d_feasible = float(np.min(t_needed))

    rows_multi.append(dict(
        n_obs=len(obstacles), d_lower=d_lower, d_feasible=d_feasible,
        gap=d_feasible - d_lower,
        rel_gap=(d_feasible - d_lower) / max(d_feasible, 1e-9),
        bound_holds=bool(d_lower <= d_feasible + 1e-9),
    ))

df_multi = pd.DataFrame(rows_multi)
df_multi.to_csv("results_cf_multi.csv", index=False)

summary = dict(
    single=dict(
        n=len(df_single),
        max_abs_err=float(df_single["abs_err"].max()) if len(df_single) else None,
        mean_abs_err=float(df_single["abs_err"].mean()) if len(df_single) else None,
        max_rel_err=float(df_single["rel_err"].max()) if len(df_single) else None,
        angular_resolution_deg=360.0 / N_ANGLES,
    ),
    multi=dict(
        n=len(df_multi),
        bound_always_holds=bool(df_multi["bound_holds"].all()) if len(df_multi) else None,
        mean_rel_gap=float(df_multi["rel_gap"].mean()) if len(df_multi) else None,
        median_rel_gap=float(df_multi["rel_gap"].median()) if len(df_multi) else None,
        p90_rel_gap=float(df_multi["rel_gap"].quantile(0.9)) if len(df_multi) else None,
        max_rel_gap=float(df_multi["rel_gap"].max()) if len(df_multi) else None,
    ),
)
with open("summary_counterfactual.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
