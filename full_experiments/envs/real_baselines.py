"""
real_baselines.py
-------------------
Applies the five baselines promised in VIII-D (Baselines) -- SHAP, LIME,
saliency-based attribution, a decision-tree surrogate, and a rule-based
active-constraint heuristic -- to the composed controller on REAL
policy-driven QP instances (sampled from rollout_qp_dataset.pkl), and
scores every method's binary "is constraint i responsible" prediction
against the same exact deletion ground truth y_i used throughout the
paper. Results sit alongside naive/single-multiplier/proposed (already
reported in Table III/Fig. 4) for a complete Table III.

Common characteristic function used by SHAP and LIME (both explain a
black-box set function, exactly as an operator would apply them to the
composed policy+filter controller in deployment):

    f(z) = 2*V(S_z) = -lambda(S_z)^T s(S_z)   for z in {0,1}^K,
           S_z = {i : z_i = 1}

i.e. the intervention energy of solving the CBF-QP with only the
constraints selected by z. f({all constraints}) - f({}) is the total
intervention energy actually paid; SHAP/LIME distribute credit for that
total across the K constraints. This is exactly how one would apply
either method to a black-box safety filter without access to its KKT
structure -- which is the entire point of comparing them against a
method (ours) that *does* use that structure.

Saliency-based attribution: finite-difference gradient of the same energy
functional with respect to each constraint's offset b_i (analogous to a
gradient-based saliency map over "how much would tightening/loosening
constraint i change the intervention"), thresholded by magnitude.

Decision-tree surrogate: a shallow (depth<=4) CART trained on
(slack, raw multiplier, row norm, kappa_stab) to predict the DELETION
label, trained on seeds 0-6 and evaluated on held-out seeds 7-9 (temporal/
seed-disjoint split, avoiding leakage across the same training run).

Rule-based active-constraint heuristic: "predict necessary iff constraint
is violated AND is the most-violated (smallest slack) among violated
constraints in that instance" -- a representative example of the
active-constraint heuristics common in applied safety-filter deployments
(cf. Sec. II-D "rule-based active-constraint explanations").
"""
import json
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from qp_core import solve_qp, active_constraints, multiplier_interval, \
    deletion_ground_truth, gram_conditioning

N_SAMPLE_INSTANCES = 150   # SHAP/LIME are re-solve-heavy; a representative sample
SHAP_NSAMPLES = 256
LIME_NSAMPLES = 256
RNG_SEED = 0


def energy_given_mask(H, A, b, u_pi, mask):
    """f(z): intervention energy of the QP restricted to constraints in mask."""
    idx = np.where(mask > 0.5)[0]
    if len(idx) == 0:
        return 0.0
    A_s, b_s = A[idx, :], b[idx]
    res = solve_qp(H, A_s, b_s, u_pi)
    u_s = res["u_star"]
    if u_s is None:
        return 0.0
    return float(0.5 * (u_s - u_pi) @ H @ (u_s - u_pi))


def run_shap(H, A, b, u_pi, p, rng):
    import shap

    def f(Z):
        return np.array([energy_given_mask(H, A, b, u_pi, z) for z in Z])

    background = np.zeros((1, p))
    explainer = shap.KernelExplainer(f, background, seed=int(rng.integers(0, 1_000_000)))
    full = np.ones((1, p))
    sv = explainer.shap_values(full, nsamples=SHAP_NSAMPLES, silent=True)
    sv = np.asarray(sv).reshape(-1)
    return sv


def run_lime(H, A, b, u_pi, p, rng):
    from lime.lime_tabular import LimeTabularExplainer

    def f(Z):
        return np.array([energy_given_mask(H, A, b, u_pi, z) for z in Z])

    # training background: random binary masks
    train = rng.integers(0, 2, size=(LIME_NSAMPLES, p)).astype(float)
    explainer = LimeTabularExplainer(
        train, mode="regression", discretize_continuous=False,
        feature_names=[f"c{i}" for i in range(p)],
        random_state=int(rng.integers(0, 1_000_000)))
    full = np.ones(p)
    exp = explainer.explain_instance(full, f, num_features=p, num_samples=LIME_NSAMPLES)
    coefs = dict(exp.as_list())
    out = np.zeros(p)
    name_to_idx = {f"c{i}": i for i in range(p)}
    for key, val in coefs.items():
        key = key.strip()
        if key in name_to_idx:
            out[name_to_idx[key]] = val
    return out


def run_saliency(H, A, b, u_pi, h=1e-4):
    p = A.shape[0]
    res0 = solve_qp(H, A, b, u_pi)
    grad = np.zeros(p)
    for i in range(p):
        b_plus = b.copy(); b_plus[i] += h
        b_minus = b.copy(); b_minus[i] -= h
        rp = solve_qp(H, A, b_plus, u_pi)
        rm = solve_qp(H, A, b_minus, u_pi)
        Vp = 0.5 * (rp["u_star"] - u_pi) @ H @ (rp["u_star"] - u_pi) if rp["u_star"] is not None else 0.0
        Vm = 0.5 * (rm["u_star"] - u_pi) @ H @ (rm["u_star"] - u_pi) if rm["u_star"] is not None else 0.0
        grad[i] = (Vp - Vm) / (2 * h)
    return grad


def rule_based(s):
    """predict necessary iff violated AND most-violated among violated."""
    p = len(s)
    violated = s < -1e-9
    pred = np.zeros(p, dtype=bool)
    if violated.any():
        i_star = int(np.argmin(s))  # most negative slack
        pred[i_star] = True
    return pred


def prf1(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = np.sum(y_true & y_pred); fp = np.sum(~y_true & y_pred); fn = np.sum(y_true & ~y_pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    return prec, rec, f1


def main():
    rng = np.random.default_rng(RNG_SEED)
    with open("../results/rollout_qp_dataset.pkl", "rb") as f:
        records = pickle.load(f)
    print(f"loaded {len(records)} records")

    # keep only instances where the filter is actually active with >=2
    # candidate constraints (otherwise SHAP/LIME/rule-based collapse
    # trivially to the single-constraint case)
    active_records = []
    for rec in records:
        H, A, b, u_pi = rec["H"], rec["A"], rec["b"], rec["u_pi"]
        s = A @ u_pi - b
        if np.sum(s < -1e-9) >= 1 and A.shape[0] >= 2:
            active_records.append(rec)
    print(f"{len(active_records)} filter-active candidate records")

    idx = rng.choice(len(active_records), size=min(N_SAMPLE_INSTANCES, len(active_records)),
                      replace=False)
    sample = [active_records[i] for i in idx]

    rows_full = []       # for shap/lime/saliency/rule-based (small sample)
    rows_surrogate = []  # for decision-tree surrogate (larger, feature-based)

    for k, rec in enumerate(sample):
        H, A, b, u_pi, seed = rec["H"], rec["A"], rec["b"], rec["u_pi"], rec["seed"]
        p = A.shape[0]
        res = solve_qp(H, A, b, u_pi)
        if not res["feasible"] or len(res["active_set"]) == 0:
            continue
        u_star = res["u_star"]
        s = res["s"]
        lam_full = res["lam_full"]
        y_true = deletion_ground_truth(H, A, b, u_pi, u_star)

        sv = run_shap(H, A, b, u_pi, p, rng)
        lv = run_lime(H, A, b, u_pi, p, rng)
        gv = run_saliency(H, A, b, u_pi)
        rb = rule_based(s)

        for i in range(p):
            rows_full.append(dict(
                y_true=y_true[i],
                pred_shap=bool(sv[i] > 1e-6),
                pred_lime=bool(lv[i] > 1e-6),
                pred_saliency=bool(gv[i] > 1e-6),
                pred_rule_based=bool(rb[i]),
                seed=seed,
            ))
        if (k + 1) % 25 == 0:
            print(f"  processed {k+1}/{len(sample)} instances")

    df = pd.DataFrame(rows_full)
    df.to_csv("../results/real_baselines_shap_lime.csv", index=False)

    methods = ["pred_shap", "pred_lime", "pred_saliency", "pred_rule_based"]
    overall = {}
    for m in methods:
        p_, r_, f1_ = prf1(df["y_true"], df[m])
        overall[m] = dict(precision=p_, recall=r_, f1=f1_)

    # -------- decision-tree surrogate on a larger feature-based sample --------
    idx2 = rng.choice(len(active_records), size=min(20000, len(active_records)), replace=False)
    feats, labels, seeds_ = [], [], []
    for i in idx2:
        rec = active_records[i]
        H, A, b, u_pi, seed = rec["H"], rec["A"], rec["b"], rec["u_pi"], rec["seed"]
        res = solve_qp(H, A, b, u_pi)
        if not res["feasible"] or len(res["active_set"]) == 0:
            continue
        s = res["s"]; lam_full = res["lam_full"]
        I = active_constraints(A, b, res["u_star"])
        kappa = gram_conditioning(H, A, I)
        y_true = deletion_ground_truth(H, A, b, u_pi, res["u_star"])
        for j in range(A.shape[0]):
            feats.append([s[j], lam_full[j], np.linalg.norm(A[j]),
                          kappa if np.isfinite(kappa) else -1.0])
            labels.append(y_true[j])
            seeds_.append(seed)
    feats = np.array(feats); labels = np.array(labels); seeds_ = np.array(seeds_)

    train_mask = seeds_ <= 6
    test_mask = seeds_ >= 7
    clf = DecisionTreeClassifier(max_depth=4, random_state=0)
    clf.fit(feats[train_mask], labels[train_mask])
    pred_test = clf.predict(feats[test_mask])
    p_dt, r_dt, f1_dt = prf1(labels[test_mask], pred_test)
    overall["pred_decision_tree_surrogate"] = dict(
        precision=p_dt, recall=r_dt, f1=f1_dt,
        n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
        tree_depth=int(clf.get_depth()), tree_feature_importances=dict(
            zip(["slack", "raw_multiplier", "row_norm", "kappa_stab"],
                clf.feature_importances_.tolist())))

    summary = dict(n_shap_lime_instances=len(sample),
                    n_constraint_tests_shap_lime=len(df),
                    overall=overall)
    with open("../results/real_baselines_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(overall, indent=2, default=str))


if __name__ == "__main__":
    main()
