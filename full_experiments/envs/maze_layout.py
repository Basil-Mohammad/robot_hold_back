"""
maze_layout.py
---------------
Geometric definition of the moderate-complexity 2D maze used for the full
experiments, built as a proper occupancy grid so the corridor structure is
FULLY ENCLOSED (no way to bypass the maze through open space outside the
designed path -- everything not within the path "tube" is solid wall).

Layout: a winding corridor (one right-angle turn sequence) with a single
dead-end branch and one open "junction room" containing four circular
pillars placed close enough together that a robot threading between them
faces two or more simultaneously active CBF constraints -- the mechanism
that produces shadowed / counter-acting constraints in real, policy-driven
trajectories (not just synthetic QPs).

Two obstacle representations are exported for the CBF safety filter:
  - PILLARS: genuine circular obstacles (used as-is; matches the paper's
    closed-form circular counterfactual in Sec. IV-C exactly).
  - wall_circles(): the solid maze structure (grid cells that are wall and
    border at least one free cell) approximated as small circles, so every
    CBF constraint in the whole system belongs to the same closed-form
    family. A robot running parallel to a wall "sees" several boundary
    circles simultaneously violated by its nominal policy, of which only
    one or two are ever actually intervention-necessary -- a realistic,
    dense source of shadowed constraints.
"""
import numpy as np

ROBOT_RADIUS = 0.15
CORRIDOR_HALF_WIDTH = 0.80      # free half-width of corridors/dead-end
GRID_RES = 0.25                 # occupancy grid cell size (m)
WALL_CIRCLE_R = GRID_RES * 0.62  # radius of each boundary wall circle

# ---- corridor centerline segments: (x1, y1, x2, y2) -----------------------
CORRIDORS = [
    (-5.0, -5.0, -5.0, -1.0),   # S1 vertical
    (-5.0, -1.0, -1.0, -1.0),   # S2 horizontal
    (-1.0, -1.0, -1.0,  2.0),   # S3 vertical
    (-1.0,  2.0, -4.0,  2.0),   # Sd dead-end branch (closed at far end)
    (-1.0,  2.0,  2.0,  2.0),   # S4 horizontal, leads into junction room
    ( 3.0,  4.5,  3.0,  7.0),   # S5 vertical, exits junction room
    ( 3.0,  7.0,  6.0,  7.0),   # S6 horizontal, to goal
]

ROOM_XMIN, ROOM_XMAX = 1.0, 5.0
ROOM_YMIN, ROOM_YMAX = 1.0, 4.5

PILLARS = [
    (2.15, 2.25, 0.40),
    (3.65, 2.15, 0.35),
    (2.60, 3.45, 0.40),
    (4.05, 3.55, 0.30),
]

START = np.array([-5.0, -5.0])
GOAL = np.array([6.0, 7.0])
START_JITTER = 0.25
GOAL_JITTER = 0.25

WORLD_XMIN, WORLD_XMAX = -6.5, 7.5
WORLD_YMIN, WORLD_YMAX = -6.5, 8.5


def _seg_dist(p, a, b):
    p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def on_path(p, half_width=CORRIDOR_HALF_WIDTH, pad=0.0):
    """True if point p lies within the free "tube" around the corridor
    centerlines or inside the junction room."""
    x, y = p
    if ROOM_XMIN - pad <= x <= ROOM_XMAX + pad and ROOM_YMIN - pad <= y <= ROOM_YMAX + pad:
        return True
    for (x1, y1, x2, y2) in CORRIDORS:
        if _seg_dist(p, (x1, y1), (x2, y2)) <= half_width + pad:
            return True
    return False


def is_free(p, robot_radius=ROBOT_RADIUS, margin=0.02):
    """True if a robot of given radius centered at p is fully inside the
    free tube/room and clear of pillars."""
    x, y = p
    if not (WORLD_XMIN <= x <= WORLD_XMAX and WORLD_YMIN <= y <= WORLD_YMAX):
        return False
    for (cx, cy, r) in PILLARS:
        if np.hypot(x - cx, y - cy) < r + robot_radius + margin:
            return False
    return on_path(p, pad=-(robot_radius + margin))


def _grid_coords():
    xs = np.arange(WORLD_XMIN, WORLD_XMAX + GRID_RES, GRID_RES)
    ys = np.arange(WORLD_YMIN, WORLD_YMAX + GRID_RES, GRID_RES)
    return xs, ys


def occupancy_grid():
    """Returns (xs, ys, free) where free[i,j] is True iff cell center
    (xs[i], ys[j]) lies on the path tube (wall thickness handled implicitly:
    a cell is 'wall' iff its center is NOT on the path)."""
    xs, ys = _grid_coords()
    free = np.zeros((len(xs), len(ys)), dtype=bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            free[i, j] = on_path((x, y))
    return xs, ys, free


def wall_boxes_merged():
    """Wall grid cells, run-length-merged along x within each row, as
    (cx, cy, hx, hy) boxes -- used for the MuJoCo XML geometry (keeps geom
    count reasonable)."""
    xs, ys, free = occupancy_grid()
    boxes = []
    half = GRID_RES / 2.0
    for j, y in enumerate(ys):
        i = 0
        n = len(xs)
        while i < n:
            if free[i, j]:
                i += 1
                continue
            i0 = i
            while i < n and not free[i, j]:
                i += 1
            x_lo = xs[i0] - half
            x_hi = xs[i - 1] + half
            cx = (x_lo + x_hi) / 2.0
            hx = (x_hi - x_lo) / 2.0
            boxes.append((cx, y, hx, half))
    return boxes


def wall_circles():
    """Wall cells that border at least one free cell -> small circle
    obstacles for the CBF filter (dense along every exposed wall surface)."""
    xs, ys, free = occupancy_grid()
    circles = []
    nx, ny = len(xs), len(ys)
    for i in range(nx):
        for j in range(ny):
            if free[i, j]:
                continue
            borders_free = False
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < ny and free[ni, nj]:
                    borders_free = True
                    break
            if borders_free:
                circles.append((float(xs[i]), float(ys[j]), WALL_CIRCLE_R))
    return circles


def all_obstacles():
    obs = [(x, y, r) for (x, y, r) in PILLARS]
    obs.extend(wall_circles())
    return obs


# ---------------------------------------------------------------------
# Reference route + arc-length progress metric (used for reward shaping
# and for evaluation-time "did it actually take the intended route"
# sanity checks). This is a *reward-shaping* device only -- it plays no
# role in the CBF safety filter or in any of the RQ1-RQ4 explanation
# machinery, which only ever see (H, A, b, u_pi, u*).
# ---------------------------------------------------------------------
REFERENCE_PATH = np.array([
    [-5.0, -5.0],
    [-5.0, -1.0],
    [-1.0, -1.0],
    [-1.0, 2.0],
    [2.0, 2.0],
    [3.3, 2.8],   # threads between the four pillars
    [3.0, 4.5],
    [3.0, 7.0],
    [6.0, 7.0],
])
_seg_vecs = np.diff(REFERENCE_PATH, axis=0)
_seg_lens = np.linalg.norm(_seg_vecs, axis=1)
_cum_len = np.concatenate([[0.0], np.cumsum(_seg_lens)])
TOTAL_PATH_LENGTH = float(_cum_len[-1])


def path_progress(p):
    """Arc-length progress of point p along REFERENCE_PATH (Frenet-style:
    projects onto every segment, uses the closest one)."""
    p = np.asarray(p, dtype=float)
    best_dist = np.inf
    best_progress = 0.0
    for i in range(len(REFERENCE_PATH) - 1):
        a = REFERENCE_PATH[i]
        seg = _seg_vecs[i]
        L2 = max(_seg_lens[i] ** 2, 1e-9)
        t = np.clip(np.dot(p - a, seg) / L2, 0.0, 1.0)
        proj = a + t * seg
        dist = np.linalg.norm(p - proj)
        if dist < best_dist:
            best_dist = dist
            best_progress = _cum_len[i] + t * _seg_lens[i]
    return best_progress


def point_at_arclength(s):
    """Point on REFERENCE_PATH at arc-length s (clipped to [0, total])."""
    s = float(np.clip(s, 0.0, TOTAL_PATH_LENGTH))
    i = np.searchsorted(_cum_len, s, side="right") - 1
    i = int(np.clip(i, 0, len(_seg_lens) - 1))
    t = (s - _cum_len[i]) / max(_seg_lens[i], 1e-9)
    t = np.clip(t, 0.0, 1.0)
    return REFERENCE_PATH[i] + t * _seg_vecs[i]


def lookahead_direction(p, lookahead=1.2):
    """Unit vector from p toward a point `lookahead` meters further along
    the reference route than p's own progress -- a local waypoint-
    following signal, much less ambiguous at corners than a straight line
    to the (possibly far away) final goal."""
    s = path_progress(p)
    target = point_at_arclength(s + lookahead)
    v = target - np.asarray(p, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros(2)
