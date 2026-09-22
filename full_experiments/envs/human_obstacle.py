"""
human_obstacle.py
-------------------
The moving "human" obstacle used for the moving-obstacle generalization
check. A single circular obstacle oscillating back and forth across
corridor S2 of the main maze (maze_layout.py; CORRIDORS[1] =
(-5,-1)-(-1,-1), half-width 0.8), so the robot's route through that
corridor genuinely has to negotiate a time-varying hazard rather than a
frozen one.

The trajectory is a deterministic, closed-form function of time --
o(t) = (HUMAN_X, HUMAN_Y_CENTER + HUMAN_Y_AMP*sin(omega t)) -- so its
velocity v(t) = do/dt is known in closed form as well (not estimated /
finite-differenced), matching the CBF derivation in the paper's moving-
obstacle extension (Sec. IX-I, "A moving obstacle"): for h(p,t) =
||p-o(t)||^2 - r_eff^2, the safety condition dh/dt + alpha h >= 0 expands
to

    2(p-o(t))^T u  >=  2(p-o(t))^T v(t) - alpha h(p,t)

i.e. A_row = 2(p-o(t)) (identical in form to the static case) and
b_row = 2(p-o(t))^T v(t) - alpha h(p,t) (gains one extra, exactly known
term). This module supplies o(t), v(t); the (A_row, b_row) construction
itself lives in moving_obstacle_env.py's _cbf_qp so it sits right next to
the static-obstacle rows it is built the same way as.
"""
import numpy as np

HUMAN_X = -3.0
HUMAN_Y_CENTER = -1.0
HUMAN_Y_AMP = 0.45
HUMAN_PERIOD = 6.0
HUMAN_RADIUS = 0.22

_OMEGA = 2.0 * np.pi / HUMAN_PERIOD


def human_state(t):
    """Returns (position, velocity), both shape (2,), at simulation time t
    (seconds). Closed-form, so velocity is EXACT, not estimated."""
    y = HUMAN_Y_CENTER + HUMAN_Y_AMP * np.sin(_OMEGA * t)
    vy = HUMAN_Y_AMP * _OMEGA * np.cos(_OMEGA * t)
    pos = np.array([HUMAN_X, y])
    vel = np.array([0.0, vy])
    return pos, vel
