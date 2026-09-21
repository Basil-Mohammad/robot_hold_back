"""
qp_core_warmstart.py
---------------------
A warm-started variant of qp_core.solve_qp's exact active-set enumeration.

Motivation (Sec. IX-G of the paper): solve_qp's worst-case cost is
O(C(p,m) m^3) because it enumerates every candidate active set from
scratch every control cycle. In a real deployment, consecutive control
cycles see smoothly-varying obstacles/state (the environment moves by at
most ||v||*DT between cycles), so the optimal active set is usually
identical or very close (Hamming distance 1-2) to the previous cycle's
active set. This module exploits that correlation: it tries the previous
cycle's active set first, then does a small local search (add/drop one
constraint at a time, breadth-first) before falling back to full
enumeration identical to solve_qp for the (rare) cases where the warm
start is far from the true optimum.

Because the underlying QP is a strictly convex quadratic program (H is
SPD) subject to linear inequality constraints, any active set that
produces a primal- and dual-feasible KKT point is *the* global optimum
(uniqueness by strict convexity). So solve_qp_warmstart always returns
exactly the same u_star as solve_qp -- the only difference is how many
candidate active sets are evaluated to find it. This lets us report a
genuine, certified speedup rather than an approximation.
"""
import itertools

import numpy as np

TOL = 1e-9


def _try_active_set(H, Hinv, A, b, u_pi, I, tol=1e-7):
    """Attempt to certify I as the optimal active set. Returns a result
    dict if I is primal- and dual-feasible, else None."""
    p, m = A.shape
    if len(I) == 0 or len(I) > m:
        return None
    A_I = A[list(I), :]
    b_I = b[list(I)]
    M_I = A_I @ Hinv @ A_I.T
    if np.linalg.matrix_rank(M_I, tol=1e-10) < M_I.shape[0]:
        return None
    rhs = b_I - A_I @ u_pi
    mu = np.linalg.solve(M_I, rhs)
    if np.any(mu < -1e-7):
        return None
    u_cand = u_pi + Hinv @ A_I.T @ mu
    s_cand = A @ u_cand - b
    if np.all(s_cand >= -tol):
        lam_full = np.zeros(p)
        lam_full[list(I)] = np.maximum(mu, 0.0)
        return dict(u_star=u_cand, active_set=I, feasible=True,
                    lam_full=lam_full)
    return None


def solve_qp_warmstart(H, A, b, u_pi, active_set_guess=(), tol=TOL):
    """
    Same optimum as qp_core.solve_qp(H, A, b, u_pi), found via a
    warm-started local search around active_set_guess before falling
    back to full active-set enumeration. Returns the same dict shape as
    solve_qp, plus 'n_candidates_tried' (the number of active sets whose
    KKT system was actually solved -- the quantity that drives wall-clock
    cost) and 'status' in {'inactive', 'warmstart', 'fallback_enumeration',
    'infeasible'}.
    """
    p, m = A.shape
    Hinv = np.linalg.inv(H)
    s = A @ u_pi - b
    if np.all(s >= -tol):
        return dict(u_star=u_pi.copy(), active_set=(), feasible=True,
                    lam_full=np.zeros(p), s=s, status="inactive",
                    n_candidates_tried=0)

    tried = set()

    def attempt(I):
        I = tuple(sorted(set(I)))
        if I in tried:
            return None
        tried.add(I)
        return _try_active_set(H, Hinv, A, b, u_pi, I)

    guess = tuple(sorted(i for i in active_set_guess if 0 <= i < p))
    res = attempt(guess)
    if res is not None:
        res.update(s=s, status="warmstart", n_candidates_tried=len(tried))
        return res

    # Breadth-first local search: neighbors within growing Hamming
    # distance of the guess (drop one / add one constraint at a time).
    frontier = {guess}
    max_radius = min(p, m)
    for _radius in range(max_radius):
        next_frontier = set()
        for I in frontier:
            for i in I:
                cand = tuple(sorted(set(I) - {i}))
                if cand not in tried:
                    r = attempt(cand)
                    if r is not None:
                        r.update(s=s, status="warmstart",
                                 n_candidates_tried=len(tried))
                        return r
                    next_frontier.add(cand)
            if len(I) < m:
                for i in range(p):
                    if i not in I:
                        cand = tuple(sorted(set(I) | {i}))
                        if cand not in tried:
                            r = attempt(cand)
                            if r is not None:
                                r.update(s=s, status="warmstart",
                                         n_candidates_tried=len(tried))
                                return r
                            next_frontier.add(cand)
        frontier = next_frontier
        if not frontier:
            break

    # Fallback: full enumeration identical to qp_core.solve_qp, skipping
    # active sets already ruled out during the local search.
    for k in range(1, min(p, m) + 1):
        for I in itertools.combinations(range(p), k):
            if I in tried:
                continue
            r = attempt(I)
            if r is not None:
                r.update(s=s, status="fallback_enumeration",
                         n_candidates_tried=len(tried))
                return r

    return dict(u_star=None, active_set=None, feasible=False, lam_full=None,
                s=s, status="infeasible", n_candidates_tried=len(tried))
