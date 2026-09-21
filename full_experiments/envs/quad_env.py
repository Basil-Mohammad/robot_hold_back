"""
quad_env.py
-------------
Planar quadrotor velocity-command navigation environment, safeguarded by
the SAME CBF-QP machinery (qp_core.solve_qp) used throughout the paper.
Pure-NumPy single-integrator kinematics (no physics engine) -- this keeps
training fast (this is a third, supplementary environment added to
demonstrate the framework generalizes across tasks, not the paper's main
experimental platform) while remaining a genuinely distinct task from the
maze: an open field, a different obstacle layout, a larger robot radius,
and a higher commandable speed (typical of an aerial platform vs. a
ground robot).
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import quad_layout as ql
from qp_core import solve_qp

DT = 0.05
MAX_SPEED = 2.5
ALPHA = 2.5
K_NEAREST = 8
SAFETY_MARGIN = 0.05
GOAL_RADIUS = 0.4
MAX_STEPS = 400
N_OBS_IN_STATE = 5


class PlanarQuadrotorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, record_qp=False):
        super().__init__()
        self.record_qp = record_qp
        self._obstacles = ql.all_obstacles()
        self._robot_r = ql.ROBOT_RADIUS

        obs_dim = 2 + 2 + 1 + 3 * N_OBS_IN_STATE
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.pos = ql.START.copy()
        self.goal = ql.GOAL.copy()
        self.t = 0
        self.last_qp_log = None
        self.n_infeasible_fallbacks = 0
        self.np_random_local = np.random.default_rng(0)

    def _nearest_obstacles(self, pos, k):
        d = np.hypot(self._obstacles[:, 0] - pos[0], self._obstacles[:, 1] - pos[1])
        h = d - (self._obstacles[:, 2] + self._robot_r + SAFETY_MARGIN)
        idx = np.argsort(h)[:k]
        return self._obstacles[idx], h[idx]

    def _cbf_qp(self, pos, u_pi, k):
        obs_k, _ = self._nearest_obstacles(pos, k)
        p = obs_k.shape[0]
        A = np.zeros((p, 2))
        b = np.zeros(p)
        obstacles_eff = np.zeros((p, 3))
        for i in range(p):
            ox, oy, r = obs_k[i]
            d = pos - np.array([ox, oy])
            r_eff = r + self._robot_r + SAFETY_MARGIN
            h = d @ d - r_eff ** 2
            A[i] = 2 * d
            b[i] = -ALPHA * h
            obstacles_eff[i] = (ox, oy, r_eff)
        H = np.eye(2)
        res = solve_qp(H, A, b, u_pi)
        return res, A, b, H, obstacles_eff

    def _safety_filter(self, u_pi):
        res, A, b, H, obstacles_eff = self._cbf_qp(self.pos, u_pi, K_NEAREST)
        if not res["feasible"]:
            res, A, b, H, obstacles_eff = self._cbf_qp(
                self.pos, u_pi, min(K_NEAREST * 3, len(self._obstacles)))
            if not res["feasible"]:
                self.n_infeasible_fallbacks += 1
                u_star = np.zeros(2)
                res = dict(u_star=u_star, active_set=(), lam_full=np.zeros(A.shape[0]),
                           s=A @ self.pos - b, feasible=True, status="frozen_fallback")
        if self.record_qp:
            self.last_qp_log = dict(H=H.copy(), A=A.copy(), b=b.copy(), u_pi=u_pi.copy(),
                                     u_star=res["u_star"].copy(), pos=self.pos.copy(),
                                     obstacles_eff=obstacles_eff.copy(), alpha=ALPHA)
        return res["u_star"]

    def _get_obs(self):
        goal_vec = self.goal - self.pos
        dist = np.linalg.norm(goal_vec)
        obs_k, _ = self._nearest_obstacles(self.pos, N_OBS_IN_STATE)
        feats = []
        for i in range(N_OBS_IN_STATE):
            ox, oy, r = obs_k[i]
            feats.extend([(ox - self.pos[0]) / 4.0, (oy - self.pos[1]) / 4.0, r])
        obs = np.array(
            [self.pos[0] / 6.0, self.pos[1] / 6.0, goal_vec[0] / 12.0, goal_vec[1] / 12.0, dist / 16.0]
            + feats, dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        for _ in range(50):
            jitter = rng.uniform(-ql.START_JITTER, ql.START_JITTER, size=2)
            cand = ql.START + jitter
            if ql.is_free(cand, robot_radius=self._robot_r, margin=0.05):
                self.pos = cand
                break
        else:
            self.pos = ql.START.copy()
        for _ in range(50):
            jitter = rng.uniform(-ql.GOAL_JITTER, ql.GOAL_JITTER, size=2)
            cand = ql.GOAL + jitter
            if ql.is_free(cand, robot_radius=self._robot_r, margin=0.05):
                self.goal = cand
                break
        else:
            self.goal = ql.GOAL.copy()
        self.t = 0
        self._stagnation = 0
        return self._get_obs(), {}

    def _check_collision(self):
        for (ox, oy, r) in self._obstacles:
            if np.hypot(self.pos[0] - ox, self.pos[1] - oy) < r + self._robot_r - 1e-6:
                return True
        return False

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        u_pi = action * MAX_SPEED

        prev_dist = np.linalg.norm(self.goal - self.pos)
        u_star = self._safety_filter(u_pi)
        new_pos = self.pos + u_star * DT
        new_pos[0] = np.clip(new_pos[0], ql.WORLD_XMIN, ql.WORLD_XMAX)
        new_pos[1] = np.clip(new_pos[1], ql.WORLD_YMIN, ql.WORLD_YMAX)
        self.pos = new_pos

        collided = self._check_collision()
        new_dist = np.linalg.norm(self.goal - self.pos)
        intervention_energy = 0.5 * np.sum((u_star - u_pi) ** 2)

        progress = prev_dist - new_dist
        reward = 4.0 * progress - 0.01 * intervention_energy - 0.01
        self.t += 1

        reached = new_dist < GOAL_RADIUS
        if reached:
            reward += 20.0
        if collided:
            reward -= 5.0

        if progress < 1e-4:
            self._stagnation += 1
        else:
            self._stagnation = 0
        truncated = self.t >= MAX_STEPS or self._stagnation > 120
        terminated = bool(reached)

        info = dict(collided=collided, intervention_energy=intervention_energy,
                    dist_to_goal=new_dist,
                    filter_active=bool(np.linalg.norm(u_star - u_pi) > 1e-9))
        if self.record_qp and self.last_qp_log is not None:
            info["qp"] = self.last_qp_log
        return self._get_obs(), float(reward), terminated, truncated, info
