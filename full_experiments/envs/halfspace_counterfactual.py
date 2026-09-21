"""
halfspace_counterfactual.py
------------------------------
Closed-form certified counterfactual for the LINEAR (half-space) wall
constraints of halfspace_generalization.py's mixed constraint-type
dataset (Sec. IX-K / Table X), closing a remaining scope gap: the
paper's closed form (Sec. V-C) was derived for CIRCULAR obstacles only;
this script derives and validates the (even simpler, exactly affine)
counterpart for the linear wall barrier used in the non-circular-
constraint generalization experiment.

Derivation: for a wall row with outward unit normal n and barrier
h(p) = |e_axis| - half_extent - margin (affine along n by construction),
the row's slack is s = alpha*(h(p) + n.u_pi). Displacing the robot
position by t*n gives h(p + t*n) = h(p) + t EXACTLY (not to first order)
as long as the same face stays dominant, which a purely-outward move
along the already-dominant axis always preserves (moving along n only
increases the dominant-axis overflow and leaves the other axis
untouched). Hence s(p + t*n) = s(p) + alpha*t, so the minimal outward
displacement that makes the row's slack exactly zero is
    d_cf = -s(p) / alpha,
an exact closed form, independently verified below by perturbing p by
d_cf and re-evaluating the row's own slack from scratch.
"""
import pickle
import sys

import numpy as np

sys.path.insert(0, ".")
from qp_core import solve_qp
from halfspace_generalization import ALPHA, K_NEAREST, ROBOT_R, SAFETY_MARGIN, \
    pillar_circle_row
import maze_layout as ml

WALL_BOXES = ml.wall_boxes_merged()
PILLARS = ml.PILLARS


def wall_row_with_box(p, box):
    cx, cy, hx, hy = box
    ex, ey = p[0] - cx, p[1] - cy
    ox, oy = abs(ex) - hx, abs(ey) - hy
    if ox >= oy:
        n = np.array([1.0 if ex >= 0 else -1.0, 0.0])
        h = ox - (ROBOT_R + SAFETY_MARGIN)
    else:
        n = np.array([0.0, 1.0 if ey >= 0 else -1.0])
        h = oy - (ROBOT_R + SAFETY_MARGIN)
    a_row = ALPHA * n
    b_val = -ALPHA * h
    return a_row, b_val, h, n


def build_mixed_constraints_tracked(pos, k=K_NEAREST):
    """Same candidate selection as halfspace_generalization.build_mixed_constraints,
    but keeps a reference to the source wall box (or None for pillars) for each
    kept row, so the counterfactual can be computed and re-verified without
    guessing which geometric object a given row came from."""
    cand = []
    for box in WALL_BOXES:
        a_row, b_val, h, n = wall_row_with_box(pos, box)
        cand.append((h, "wall_linear", a_row, b_val, box, n))
    for pillar in PILLARS:
        a_row, b_val, h = pillar_circle_row(pos, pillar)
        cand.append((h, "pillar_circular", a_row, b_val, None, None))
    cand.sort(key=lambda t: t[0])
    top = cand[:k]
    A = np.array([t[2] for t in top])
    b = np.array([t[3] for t in top])
    kinds = [t[1] for t in top]
    boxes = [t[4] for t in top]
    normals = [t[5] for t in top]
    return A, b, kinds, boxes, normals


def main():
    with open("../results/rollout_qp_dataset.pkl", "rb") as f:
        records = pickle.load(f)
    rng = np.random.default_rng(0)
    N_SAMPLE = 8000
    idx = rng.choice(len(records), size=min(N_SAMPLE, len(records)), replace=False)
    sample = [records[i] for i in idx]

    H = np.eye(2)
    n_wall_rows = 0
    n_violated = 0
    n_verified = 0
    max_cert_err = 0.0
    d_cf_values = []

    for rec in sample:
        pos, u_pi = rec["pos"], rec["u_pi"]
        A, b, kinds, boxes, normals = build_mixed_constraints_tracked(pos)
        res = solve_qp(H, A, b, u_pi)
        if not res["feasible"]:
            continue
        s = res["s"]
        for i, kind in enumerate(kinds):
            if kind != "wall_linear":
                continue
            n_wall_rows += 1
            if s[i] >= -1e-9:
                continue
            n_violated += 1
            d_cf = -s[i] / ALPHA
            d_cf_values.append(d_cf)
            # independent re-verification: perturb pos by d_cf along this
            # row's own outward normal, rebuild the SAME wall's row from
            # scratch at the perturbed position, and confirm its slack is
            # (numerically) zero
            box, n_hat = boxes[i], normals[i]
            pos_pert = pos + d_cf * n_hat
            _, _, h_pert, n_pert = wall_row_with_box(pos_pert, box)
            s_pert = ALPHA * (h_pert + n_pert @ u_pi)
            cert_err = abs(s_pert)
            max_cert_err = max(max_cert_err, cert_err)
            n_verified += 1

    summary = dict(
        n_samples=len(sample), n_wall_rows_checked=n_wall_rows,
        n_violated_wall_rows=n_violated, n_certificates_verified=n_verified,
        max_certificate_residual=max_cert_err,
        mean_d_cf=float(np.mean(d_cf_values)) if d_cf_values else None,
        max_d_cf=float(np.max(d_cf_values)) if d_cf_values else None,
    )
    print(summary)
    import json
    with open("../results/halfspace_counterfactual_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote ../results/halfspace_counterfactual_summary.json")


if __name__ == "__main__":
    main()
