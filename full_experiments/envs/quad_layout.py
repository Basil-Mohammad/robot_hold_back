"""
quad_layout.py
-----------------
Fixed layout for the third environment: a planar (2D, horizontal-plane)
quadrotor velocity-command navigation task through a field of circular
no-fly zones, distinct from the maze (open field, no corridors, a
different obstacle count/geometry/start-goal pair, and a larger robot
radius / control-authority regime typical of an aerial platform).

Same abstraction level as the maze environment: the CBF-QP safety filter
acts on the COMMANDED VELOCITY (the outer guidance loop of a standard
two-loop quadrotor controller: this outer loop commands a horizontal
velocity, and an inner attitude/thrust controller -- not modelled here,
exactly as an inner low-level tracking controller is not modelled by the
maze's single-integrator abstraction either -- realizes it). This keeps
the CBF-QP identical in structure (H=I, h(p)=||p-o||^2-r_eff^2) to the
maze and pilot, so the paper's certified closed-form counterfactual
formula applies unmodified; what changes is the task, the obstacle field,
and the trained policy.
"""
import numpy as np

ROBOT_RADIUS = 0.25          # larger effective safety radius (rotor span)
WORLD_XMIN, WORLD_XMAX = -6.0, 6.0
WORLD_YMIN, WORLD_YMAX = -6.0, 6.0

START = np.array([-5.0, -5.0])
GOAL = np.array([5.0, 5.0])
START_JITTER = 0.3
GOAL_JITTER = 0.3

# Fixed field of circular no-fly zones (cx, cy, r). Deliberately NOT a
# corridor/maze topology -- an open field with scattered restricted-
# airspace zones of varying size, forcing genuinely different local
# constraint geometries (multiple simultaneously-close zones, wide gaps)
# than the maze's pillars-in-corridor layout.
_rng = np.random.default_rng(42)
N_ZONES = 14
zones = []
for _ in range(N_ZONES):
    for _try in range(200):
        cx = _rng.uniform(WORLD_XMIN + 1.5, WORLD_XMAX - 1.5)
        cy = _rng.uniform(WORLD_YMIN + 1.5, WORLD_YMAX - 1.5)
        r = _rng.uniform(0.4, 0.9)
        if np.hypot(cx - START[0], cy - START[1]) < r + 1.2:
            continue
        if np.hypot(cx - GOAL[0], cy - GOAL[1]) < r + 1.2:
            continue
        ok = True
        for (ox, oy, orr) in zones:
            if np.hypot(cx - ox, cy - oy) < r + orr + 0.35:
                ok = False
                break
        if ok:
            zones.append((cx, cy, r))
            break
ZONES = np.array(zones, dtype=np.float64)


def all_obstacles():
    return ZONES.copy()


def is_free(p, robot_radius=ROBOT_RADIUS, margin=0.02):
    x, y = p
    if not (WORLD_XMIN <= x <= WORLD_XMAX and WORLD_YMIN <= y <= WORLD_YMAX):
        return False
    for (ox, oy, r) in ZONES:
        if np.hypot(x - ox, y - oy) < r + robot_radius + margin:
            return False
    return True
