"""
Experiment 4 (RQ4): does the conditioning indicator kappa_stab =
sigma_min(M_I) predict empirical explanation instability?

For each intervening instance with active set I (|I|>=1):
  - compute kappa_stab = sigma_min(M_I)
  - apply K small random perturbations to b (magnitude eps) and re-solve
    the FULL QP (not just the linearized system) each time
  - record: (i) whether the active set switches, (ii) for perturbations
    that keep the active set fixed, the empirical multiplier sensitivity
    ratio ||Delta lambda_I|| / ||Delta b|| , which the paper's linearized
    formula predicts should scale like ||M_I^{-1}||  ~ 1/kappa_stab.

We then report the correlation (Spearman, rank-based -- robust to the
extreme skew of 1/kappa_stab) between kappa_stab and (a) mean empirical
sensitivity ratio, (b) active-set switching frequency.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from qp_core import solve_qp, active_constraints, gram_conditioning
from scenarios import random_scenario, degenerate_scenario

rng = np.random.default_rng(2024)

N_SCEN = 800
K_PERTURB = 25
EPS = 1e-4

rows = []

for trial in range(N_SCEN):
    if trial % 4 == 0:
        m = rng.integers(2, 4)
        sc = degenerate_scenario(rng, m=m)  # forces near/at singular M_I sometimes
    else:
        m = rng.integers(2, 5)
        p = rng.integers(2, 7)
        sc = random_scenario(rng, m, p, violation_bias=rng.uniform(0.3, 0.85),
                              corr_pairs=rng.integers(0, 3))
    H, A, b, u_pi = sc["H"], sc["A"], sc["b"], sc["u_pi"]
    res = solve_qp(H, A, b, u_pi)
    if not res["feasible"]:
        continue
    u_star = res["u_star"]
    I = active_constraints(A, b, u_star)
    if len(I) == 0:
        continue
    kappa = gram_conditioning(H, A, I)
    if not np.isfinite(kappa):
        continue

    p_dim = A.shape[0]
    sens_ratios = []
    n_switch = 0
    for _ in range(K_PERTURB):
        db = rng.normal(size=p_dim)
        db = db / np.linalg.norm(db) * EPS
        b2 = b + db
        res2 = solve_qp(H, A, b2, u_pi)
        if not res2["feasible"]:
            continue
        I2 = active_constraints(A, b2, res2["u_star"])
        if set(I2) != set(I):
            n_switch += 1
            continue
        dlam = res2["lam_full"] - res["lam_full"]
        ratio = np.linalg.norm(dlam) / np.linalg.norm(db)
        sens_ratios.append(ratio)

    rows.append(dict(
        trial=trial, k_active=len(I), kappa_stab=kappa,
        log_inv_kappa=np.log10(1.0 / kappa),
        mean_sens_ratio=float(np.mean(sens_ratios)) if sens_ratios else np.nan,
        max_sens_ratio=float(np.max(sens_ratios)) if sens_ratios else np.nan,
        switch_rate=n_switch / K_PERTURB,
        n_valid_nonswitch=len(sens_ratios),
    ))

df = pd.DataFrame(rows)
df.to_csv("results_stability.csv", index=False)

sub = df.dropna(subset=["mean_sens_ratio"])
rho_sens, p_sens = spearmanr(sub["log_inv_kappa"], sub["mean_sens_ratio"])
rho_switch, p_switch = spearmanr(df["log_inv_kappa"], df["switch_rate"])

summary = dict(
    n_scenarios=len(df),
    n_scenarios_with_nonswitch_data=len(sub),
    spearman_rho_kappa_vs_sensitivity=float(rho_sens),
    spearman_p_kappa_vs_sensitivity=float(p_sens),
    spearman_rho_kappa_vs_switchrate=float(rho_switch),
    spearman_p_kappa_vs_switchrate=float(p_switch),
    frac_scenarios_any_switch=float((df["switch_rate"] > 0).mean()),
    kappa_stab_min=float(df["kappa_stab"].min()),
    kappa_stab_max=float(df["kappa_stab"].max()),
)
with open("summary_stability.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
