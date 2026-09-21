"""
qp_core.py
----------
Exact primal-dual machinery for the CBF-QP safety filter studied in
"Why Did the Robot Hold Back?"

    u* = argmin_u  1/2 (u - u_pi)^T H (u - u_pi)   s.t.  A u >= b

Everything here is deliberately exact / brute-force rather than calling a
general-purpose QP library: because the action dimension m and the number
of constraints p are small (as they are for a realistic CBF-QP filter,
typically m in {2,...,6}, p in {1,...,8}), we can enumerate candidate
active sets and pick the one that satisfies the full KKT system. This
removes solver-specific numerical noise from the pilot and lets us test
the paper's identities against a ground truth we fully control.
"""
import itertools
import numpy as np

TOL = 1e-9


def solve_qp(H, A, b, u_pi, tol=TOL):
    """
    Exact solve of  min 1/2 (u-u_pi)^T H (u-u_pi)  s.t. A u >= b
    via active-set enumeration (H SPD assumed).

    Returns dict with:
        u_star, active_set (tuple of indices with A_i u* = b_i within tol),
        feasible (bool), lam_full (a single valid KKT multiplier, zero off
        the active set), s (slack at u_pi), status
    """
    p, m = A.shape
    Hinv = np.linalg.inv(H)

    # Unconstrained optimum is u_pi itself (since objective is minimized
    # there). Check feasibility first (cheapest path, matches Prop. in
    # the paper: filter inactive iff s(theta) >= 0).
    s = A @ u_pi - b
    if np.all(s >= -tol):
        return dict(u_star=u_pi.copy(), active_set=(), feasible=True,
                     lam_full=np.zeros(p), s=s, status="inactive")

    best = None
    # Enumerate active sets by size, smallest first (typical CBF-QP active
    # sets are small relative to p for these pilot scenarios).
    idx_all = list(range(p))
    for k in range(1, min(p, m) + 1):
        for I in itertools.combinations(idx_all, k):
            A_I = A[list(I), :]
            b_I = b[list(I)]
            # Equality-constrained QP restricted to A_I u = b_I:
            # u = u_pi + Hinv A_I^T mu  solving A_I Hinv A_I^T mu = b_I - A_I u_pi
            M_I = A_I @ Hinv @ A_I.T
            rhs = b_I - A_I @ u_pi
            if np.linalg.matrix_rank(M_I, tol=1e-10) < M_I.shape[0]:
                continue  # redundant/degenerate active set at this stage
            mu = np.linalg.solve(M_I, rhs)
            if np.any(mu < -1e-7):
                continue  # dual infeasible for this active set
            u_cand = u_pi + Hinv @ A_I.T @ mu
            s_cand = A @ u_cand - b
            if np.all(s_cand >= -1e-7):
                # Feasible + dual feasible for this active set -> optimal
                # by convexity. Build a full multiplier vector (0 off I).
                lam_full = np.zeros(p)
                lam_full[list(I)] = np.maximum(mu, 0.0)
                return dict(u_star=u_cand, active_set=I, feasible=True,
                             lam_full=lam_full, s=s, status="active_set_%d" % k)
    return dict(u_star=None, active_set=None, feasible=False, lam_full=None,
                s=s, status="infeasible")


def active_constraints(A, b, u_star, tol=1e-7):
    s_star = A @ u_star - b
    return tuple(int(i) for i in np.where(np.abs(s_star) <= tol)[0])


def multiplier_interval(H, A, b, u_pi, u_star, I, tol=1e-9):
    """
    For fixed primal optimum u_star and active set I, compute
    [lambda_i^min, lambda_i^max] for every i via LP over the affine
    dual-feasible set:
        A_I^T lambda_I = H (u_star - u_pi),   lambda_I >= 0
    (off-active multipliers are identically 0 by complementary slackness).

    Returns dict: i -> (lmin, lmax) for i in I, and {} entries of 0 for i
    not in I (unique, since only i in I can carry positive multiplier).
    """
    from scipy.optimize import linprog

    p = A.shape[0]
    interval = {i: (0.0, 0.0) for i in range(p)}
    if len(I) == 0:
        return interval

    A_I = A[list(I), :]
    c = H @ (u_star - u_pi)  # stationarity RHS
    k = len(I)

    for pos, i in enumerate(I):
        obj = np.zeros(k)
        obj[pos] = 1.0
        # minimize lambda_i
        res_min = linprog(obj, A_eq=A_I.T, b_eq=c, bounds=[(0, None)] * k,
                           method="highs")
        # maximize lambda_i  <=>  minimize -lambda_i
        res_max = linprog(-obj, A_eq=A_I.T, b_eq=c, bounds=[(0, None)] * k,
                           method="highs")
        lmin = res_min.x[pos] if res_min.success else np.nan
        lmax = -res_max.fun if res_max.success else np.nan
        interval[i] = (max(lmin, 0.0), max(lmax, 0.0))
    return interval


def gram_conditioning(H, A, I):
    if len(I) == 0:
        return np.inf
    Hinv = np.linalg.inv(H)
    A_I = A[list(I), :]
    M_I = A_I @ Hinv @ A_I.T
    sing = np.linalg.svd(M_I, compute_uv=False)
    return sing[-1]  # sigma_min


def deletion_ground_truth(H, A, b, u_pi, u_star, tol=1e-7):
    """
    For every constraint i, solve the QP with constraint i removed and
    check whether the optimizer changes. Returns dict i -> bool
    (True = intervention-necessary, i.e. removing i changes u*).
    """
    p = A.shape[0]
    y = {}
    for i in range(p):
        mask = [j for j in range(p) if j != i]
        A_minus = A[mask, :]
        b_minus = b[mask]
        res = solve_qp(H, A_minus, b_minus, u_pi)
        u_minus = res["u_star"]
        if u_minus is None:
            # reduced problem infeasible -- cannot be compared meaningfully;
            # treat conservatively as "changes" only if it truly differs
            # (shouldn't happen for these scenarios: removing a constraint
            # cannot make a feasible problem infeasible)
            y[i] = True
            continue
        y[i] = bool(np.linalg.norm(u_minus - u_star) > tol)
    return y
