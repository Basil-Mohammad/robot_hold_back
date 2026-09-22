"""
maze_layout2.py
----------------
A SECOND maze, topologically distinct from maze_layout.py, used for the
"second maze topology" generalization check (Sec. IX-I). Differences from
the main maze, all deliberate:

  - The main maze is a strict TREE (no cycles: exactly one route from start
    to goal plus one closed dead-end branch). This maze contains an actual
    LOOP -- two disjoint corridor routes connect the same two junction
    points around the pillar room -- so the robot has a genuine routing
    choice the main maze never presents.
  - Start and goal sit on the OPPOSITE diagonal of the world (top-left to
    bottom-right, vs. the main maze's bottom-left to top-right), with the
    path descending rather than ascending.
  - The junction room has THREE pillars in a triangular arrangement (vs.
    the main maze's four, clustered near-square), and the dead-end branch
    is placed at a different point in the route.

Same interface as maze_layout.py (all functions/constants below are used
identically by build_mjcf2.py / safe_maze_env2.py), so every downstream
CBF-QP, identity, attribution, counterfactual, and stability computation
is untouched -- only the geometry differs.
"""
import numpy as np

ROBOT_RADIUS = 0.15
CORRIDOR_HALF_WIDTH = 0.80
GRID_RES = 0.25
WALL_CIRCLE_R = GRID_RES * 0.62

# ---- corridor centerline segments: (x1, y1, x2, y2) -----------------------
CORRIDORS = [
    (-5.0,  7.0, -5.0,  3.0),   # S1 vertical, descending from start
    (-5.0,  3.0, -2.0,  3.0),   # S2 horizontal
    (-2.0,  3.0, -2.0,  0.0),   # S3 vertical, into the room from above
    (-2.0,  0.0,  1.5,  0.0),   # S4 horizontal, into the room (route A)
    ( 1.5,  0.0,  1.5, -3.2),   # Sd dead-end branch, downward (closed end)
    ( 1.5,  0.0,  4.5,  0.0),   # S5 horizontal, exits the room
    ( 4.5,  0.0,  4.5, -3.5),   # S6 vertical, descending toward goal
    ( 4.5, -3.5,  6.0, -3.5),   # S7 horizontal, to goal alcove
    (-2.0,  3.0,  1.5,  3.0),   # Sl1 loop branch: bypasses the room above
    ( 1.5,  3.0,  1.5,  0.0),   # Sl2 loop branch: rejoins the room from above
]

ROOM_XMIN, ROOM_XMAX = 0.5, 3.0
ROOM_YMIN, ROOM_YMAX = -1.6, 1.6

PILLARS = [
    (1.40,  0.55, 0.35),
    (2.35, -0.20, 0.30),
    (1.70, -1.05, 0.40),
]

START = np.array([-5.0, 7.0])
GOAL = np.array([6.0, -3.5])
START_JITTER = 0.25
GOAL_JITTER = 0.25

WORLD_XMIN, WORLD_XMAX = -6.5, 7.5
WORLD_YMIN, WORLD_YMAX = -5.0, 8.5


def _seg_dist(p, a, b):
    p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def on_path(p, half_width=CORRIDOR_HALF_WIDTH, pad=0.0):
    x, y = p
    if ROOM_XMIN - pad <= x <= ROOM_XMAX + pad and ROOM_YMIN - pad <= y <= ROOM_YMAX + pad:
        return True
    for (x1, y1, x2, y2) in CORRIDORS:
        if _seg_dist(p, (x1, y1), (x2, y2)) <= half_width + pad:
            return True
    return False


def is_free(p, robot_radius=ROBOT_RADIUS, margin=0.02):
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
    xs, ys = _grid_coords()
    free = np.zeros((len(xs), len(ys)), dtype=bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            free[i, j] = on_path((x, y))
    return xs, ys, free


def wall_boxes_merged():
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
# Reference route (route A through the room, not the loop bypass) + arc-
# length progress metric, reward-shaping only.
# ---------------------------------------------------------------------
REFERENCE_PATH = np.array([
    [-5.0,  7.0],
    [-5.0,  3.0],
    [-2.0,  3.0],
    [-2.0,  0.0],
    [ 0.75, 0.0],
    [ 0.75, 1.25],   # rises above pillar 1 to enter the room
    [ 2.77, 1.25],   # crosses the room above all three pillars
    [ 2.77, 0.5],    # descends back toward the S5 corridor centerline
    [ 3.5,  0.5],
    [ 3.5,  0.0],
    [ 4.5,  0.0],
    [ 4.5, -3.5],
    [ 6.0, -3.5],
])
_seg_vecs = np.diff(REFERENCE_PATH, axis=0)
_seg_lens = np.linalg.norm(_seg_vecs, axis=1)
_cum_len = np.concatenate([[0.0], np.cumsum(_seg_lens)])
TOTAL_PATH_LENGTH = float(_cum_len[-1])


def path_progress(p):
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
    s = float(np.clip(s, 0.0, TOTAL_PATH_LENGTH))
    i = np.searchsorted(_cum_len, s, side="right") - 1
    i = int(np.clip(i, 0, len(_seg_lens) - 1))
    t = (s - _cum_len[i]) / max(_seg_lens[i], 1e-9)
    t = np.clip(t, 0.0, 1.0)
    return REFERENCE_PATH[i] + t * _seg_vecs[i]


def lookahead_direction(p, lookahead=1.2):
    s = path_progress(p)
    target = point_at_arclength(s + lookahead)
    v = target - np.asarray(p, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros(2)
