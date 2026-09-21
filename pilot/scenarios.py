"""
scenarios.py
------------
Synthetic scenario generators for the pilot study. Three families:

1. random_scenario       -- general CBF-QP-like instances (random H,A,b,u_pi),
                             used for identity verification, deletion ground
                             truth and attribution faithfulness (RQ1, RQ2).
2. circular_obstacle_1   -- single circular obstacle, single-integrator robot,
                             matches the paper's closed-form counterfactual
                             (RQ3).
3. circular_obstacle_multi -- multiple circular obstacles, for the two-sided
                             counterfactual certificate (RQ3).
4. degenerate_scenario   -- forces two (near-)parallel active constraints so
                             the dual optimum is non-unique, for the
                             degeneracy / stability tests (RQ4).
"""
import numpy as np


def spd_matrix(rng, m, cond_floor=1.0):
    Q = rng.normal(size=(m, m))
    H = Q @ Q.T + cond_floor * np.eye(m)
    return H


def random_scenario(rng, m, p, violation_bias=0.6, corr_pairs=0):
    """
    General random CBF-QP-style instance.

    violation_bias in [0,1]: probability that a given raw constraint is
    drawn so it is violated at u_pi (this is what makes the filter active
    and lets shadowed / counter-acting constraints emerge with p>=3).

    corr_pairs: number of constraint rows that are forced to be near-
    duplicates of an existing row (small random noise), which is the
    mechanism that produces "shadowed" constraints (violated at u_pi but
    redundant with another, so not intervention-necessary) and occasional
    dual degeneracy.
    """
    H = spd_matrix(rng, m)
    u_pi = rng.normal(size=m) * rng.uniform(0.5, 2.0)

    A = rng.normal(size=(p, m))
    A /= np.linalg.norm(A, axis=1, keepdims=True)

    for _ in range(min(corr_pairs, p - 1)):
        i, j = rng.choice(p, size=2, replace=False)
        A[j] = A[i] + rng.normal(scale=0.03, size=m)
        A[j] /= np.linalg.norm(A[j])

    b = np.empty(p)
    for i in range(p):
        if rng.uniform() < violation_bias:
            # constraint violated at u_pi by a random positive margin
            margin = rng.uniform(0.05, 1.5)
            b[i] = A[i] @ u_pi + margin
        else:
            margin = rng.uniform(0.05, 1.5)
            b[i] = A[i] @ u_pi - margin
    return dict(H=H, A=A, b=b, u_pi=u_pi)


def degenerate_scenario(rng, m=2):
    """
    Two constraints whose rows are exactly parallel and both active at the
    optimum (classic dual-degenerate CBF-QP corner case: e.g. two barrier
    functions that happen to describe the same tangent hyperplane at the
    optimum).
    """
    H = spd_matrix(rng, m)
    u_pi = rng.normal(size=m) * rng.uniform(0.5, 1.5)
    a = rng.normal(size=m)
    a /= np.linalg.norm(a)
    A = np.stack([a, a.copy()], axis=0)  # exactly parallel rows
    # add a couple of extra, generic, non-active constraints for realism
    p_extra = rng.integers(0, 3)
    if p_extra > 0:
        A_extra = rng.normal(size=(p_extra, m))
        A_extra /= np.linalg.norm(A_extra, axis=1, keepdims=True)
        A = np.vstack([A, A_extra])
    p = A.shape[0]
    b = np.empty(p)
    margin = rng.uniform(0.2, 1.0)
    b[0] = A[0] @ u_pi + margin
    b[1] = A[1] @ u_pi + margin  # same slack -> both exactly active together
    for i in range(2, p):
        b[i] = A[i] @ u_pi - rng.uniform(0.2, 1.0)  # inactive, not violated
    return dict(H=H, A=A, b=b, u_pi=u_pi)


def circular_obstacle_1(rng, alpha=None):
    """
    Single-integrator robot (dot p = u, m=2), single circular obstacle:
        h(p) = ||p-o||^2 - r^2 ,  a = 2(p-o)^T ,  b = -alpha*h(p)
    Matches Sec. IV-C of the paper exactly (H = I).
    """
    alpha = rng.uniform(0.5, 2.5) if alpha is None else alpha
    o = rng.normal(size=2) * 2.0
    r = rng.uniform(0.5, 1.5)
    # place the robot so the obstacle is nearby (sometimes forcing s<0)
    p = o + rng.normal(size=2) * r * rng.uniform(0.3, 1.2)
    u_pi = rng.normal(size=2) * rng.uniform(0.5, 3.0)

    d = p - o
    a = (2 * d).reshape(1, 2)
    h = d @ d - r ** 2
    b = np.array([-alpha * h])
    H = np.eye(2)
    return dict(H=H, A=a, b=b, u_pi=u_pi, alpha=alpha, o=o, r=r, p=p, d=d)


def circular_obstacle_multi(rng, n_obs=3, alpha=None):
    alpha = rng.uniform(0.5, 2.0) if alpha is None else alpha
    p = rng.normal(size=2) * 0.5
    obstacles = []
    for _ in range(n_obs):
        o = p + rng.normal(size=2) * rng.uniform(0.6, 1.8)
        r = rng.uniform(0.4, 1.0)
        obstacles.append((o, r))
    u_pi = rng.normal(size=2) * rng.uniform(0.5, 3.0)

    rows = []
    bs = []
    ds = []
    for o, r in obstacles:
        d = p - o
        rows.append(2 * d)
        h = d @ d - r ** 2
        bs.append(-alpha * h)
        ds.append(d)
    A = np.array(rows)
    b = np.array(bs)
    H = np.eye(2)
    return dict(H=H, A=A, b=b, u_pi=u_pi, alpha=alpha, obstacles=obstacles,
                p=p, ds=ds)
