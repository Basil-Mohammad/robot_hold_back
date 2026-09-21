"""
arm_layout.py
--------------
Layout for the third generalization environment: a 6-DOF planar
manipulator (six revolute joints in series) reaching for a task-space
target while its end effector avoids circular keep-out regions. This is
the paper's genuine higher-dimensional (m=6) action-space platform,
trained with real PPO policies (Sec. IX-H), as opposed to the purely
synthetic timing benchmark of Sec. IX-G.
"""
import numpy as np

N_JOINTS = 6
LINK_LENGTHS = np.array([0.55, 0.55, 0.5, 0.5, 0.45, 0.45])
REACH = float(LINK_LENGTHS.sum())  # 3.0

Q_MIN = -np.pi * 0.85 * np.ones(N_JOINTS)
Q_MAX = np.pi * 0.85 * np.ones(N_JOINTS)

BASE = np.array([0.0, 0.0])
Q_START = np.array([0.35, -0.5, 0.4, -0.35, 0.3, -0.2])

# Circular keep-out obstacles the end effector must avoid, placed within
# the arm's reachable workspace (annulus roughly 1.0-2.7 from the base).
_obs_polar = [
    (1.55, 0.35, 0.28), (1.9, 1.35, 0.30), (1.6, 2.35, 0.26),
    (2.15, -0.55, 0.27), (1.75, -1.55, 0.30), (2.4, 2.9, 0.25),
    (1.3, -2.6, 0.24), (2.55, 1.0, 0.22),
]
OBSTACLES = np.array([
    (r * np.cos(theta), r * np.sin(theta), rad) for (r, theta, rad) in _obs_polar
])

EE_RADIUS = 0.12


def forward_kinematics(q):
    """Returns (positions, phi): positions is (N_JOINTS+1, 2) array of the
    base and every joint tip (positions[-1] is the end effector); phi is
    the cumulative link angle array used by the Jacobian."""
    phi = np.cumsum(q)
    dx = LINK_LENGTHS * np.cos(phi)
    dy = LINK_LENGTHS * np.sin(phi)
    xs = np.concatenate([[BASE[0]], BASE[0] + np.cumsum(dx)])
    ys = np.concatenate([[BASE[1]], BASE[1] + np.cumsum(dy)])
    positions = np.stack([xs, ys], axis=1)
    return positions, phi


def end_effector(q):
    positions, _ = forward_kinematics(q)
    return positions[-1]


def jacobian(q):
    """2 x N_JOINTS Jacobian of the end-effector position w.r.t. joint
    velocities, for the planar serial chain (standard closed form:
    column i sums the perpendicular contribution of every link k>=i)."""
    _, phi = forward_kinematics(q)
    n = N_JOINTS
    J = np.zeros((2, n))
    # suffix sums of link_k * (-sin phi_k, cos phi_k) for k = i..n-1
    sx = LINK_LENGTHS * (-np.sin(phi))
    sy = LINK_LENGTHS * (np.cos(phi))
    rev_cumsum_x = np.cumsum(sx[::-1])[::-1]
    rev_cumsum_y = np.cumsum(sy[::-1])[::-1]
    J[0, :] = rev_cumsum_x
    J[1, :] = rev_cumsum_y
    return J


def sample_goal(rng, max_tries=200):
    for _ in range(max_tries):
        r = rng.uniform(0.8 * REACH, 0.97 * REACH)
        theta = rng.uniform(-np.pi, np.pi)
        cand = np.array([r * np.cos(theta), r * np.sin(theta)])
        if is_task_free(cand, margin=0.15):
            return cand
    return np.array([REACH * 0.85, 0.0])


def is_task_free(p, margin=0.0):
    for (ox, oy, r) in OBSTACLES:
        if np.hypot(p[0] - ox, p[1] - oy) < r + EE_RADIUS + margin:
            return False
    return True
