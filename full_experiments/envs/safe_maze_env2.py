"""
safe_maze_env2.py
-----------------
Gymnasium environment: a point-mass robot navigating the maze in
maze_layout.py, with the paper's own CBF-QP safety filter (H=I, single-
integrator dynamics dot p = u) sitting between the learned policy's
nominal action u_pi and the action actually executed, exactly matching the
architecture described in Sec. III of the manuscript. Physical embodiment,
collision ground-truth, and rendering are provided by MuJoCo; the robot is
advanced kinematically (qpos written directly from the filtered velocity)
so the control law is provably identical to the paper's continuous-time
single-integrator model rather than an approximation of it.

Every step, the safety filter is built from the K nearest obstacles (a
standard "local constraint set" construction for CBF-QP filters) and solved
with the EXACT active-set QP solver validated in the pilot study
(qp_core.py) -- so every downstream explanation computed on this env's
trajectories uses the same, already-verified machinery.
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco

import maze_layout2 as ml
from qp_core import solve_qp

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DT = 0.05
MAX_SPEED = 1.5
ALPHA = 2.5
K_NEAREST = 10
SAFETY_MARGIN = 0.03
GOAL_RADIUS = 0.35
MAX_STEPS = 550
N_OBS_IN_STATE = 5  # nearest obstacles reported in the observation vector


class SafeMazeEnv2(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, record_qp=False, xml_path=None, curriculum_p=0.0):
        super().__init__()
        self.record_qp = record_qp
        self.curriculum_p = curriculum_p
        xml_path = xml_path or os.path.join(THIS_DIR, "maze2.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self._obstacles = np.array(ml.all_obstacles(), dtype=np.float64)  # (N,3): cx,cy,r
        self._robot_r = ml.ROBOT_RADIUS

        self.geom_robot = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "robot_geom")

        obs_dim = 2 + 2 + 1 + 2 + 3 * N_OBS_IN_STATE
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.render_mode = render_mode
        self._renderer = None

        self.pos = ml.START.copy()
        self.goal = ml.GOAL.copy()
        self.t = 0
        self.last_qp_log = None
        self.n_infeasible_fallbacks = 0

    # ------------------------------------------------------------------ #
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
        obstacles_eff = np.zeros((p, 3))  # ox, oy, r_eff (for counterfactual analysis)
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
            # widen the local constraint set as a fallback; if still
            # infeasible, freeze (u*=0) and count the event for reporting
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

    # ------------------------------------------------------------------ #
    def _get_obs(self):
        goal_vec = self.goal - self.pos
        dist = np.linalg.norm(goal_vec)
        lookahead = ml.lookahead_direction(self.pos, lookahead=1.2)
        obs_k, h_k = self._nearest_obstacles(self.pos, N_OBS_IN_STATE)
        feats = []
        for i in range(N_OBS_IN_STATE):
            ox, oy, r = obs_k[i]
            feats.extend([(ox - self.pos[0]) / 3.0, (oy - self.pos[1]) / 3.0, r])
        obs = np.array(
            [self.pos[0] / 8.0, self.pos[1] / 8.0, goal_vec[0] / 8.0, goal_vec[1] / 8.0, dist / 12.0,
             lookahead[0], lookahead[1]]
            + feats, dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        use_curriculum = (self.curriculum_p > 0) and (rng.uniform() < self.curriculum_p)
        if use_curriculum:
            # Reverse/random-start curriculum: sample a point anywhere along
            # the reference route (not just the true start) so the policy
            # gets dense training signal from every part of the maze,
            # including the tight multi-pillar room -- a standard remedy
            # for hard-exploration bottlenecks in maze navigation, and
            # incidentally also gives a broader state-visitation
            # distribution for the later explanation-quality rollouts.
            s = rng.uniform(0.0, ml.TOTAL_PATH_LENGTH * 0.92)
            base = ml.point_at_arclength(s)
            placed = False
            for _ in range(50):
                jitter = rng.uniform(-0.35, 0.35, size=2)
                cand = base + jitter
                if ml.is_free(cand, robot_radius=self._robot_r, margin=0.05):
                    self.pos = cand
                    placed = True
                    break
            if not placed:
                self.pos = base
        else:
            for _ in range(50):
                jitter = rng.uniform(-ml.START_JITTER, ml.START_JITTER, size=2)
                cand = ml.START + jitter
                if ml.is_free(cand, robot_radius=self._robot_r, margin=0.05):
                    self.pos = cand
                    break
            else:
                self.pos = ml.START.copy()
        for _ in range(50):
            jitter = rng.uniform(-ml.GOAL_JITTER, ml.GOAL_JITTER, size=2)
            cand = ml.GOAL + jitter
            if ml.is_free(cand, robot_radius=self._robot_r, margin=0.05):
                self.goal = cand
                break
        else:
            self.goal = ml.GOAL.copy()

        self.t = 0
        self._stagnation = 0
        self._write_qpos()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def _write_qpos(self):
        self.data.qpos[0] = self.pos[0]
        self.data.qpos[1] = self.pos[1]
        self.data.qvel[:] = 0

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        u_pi = action * MAX_SPEED

        prev_dist = np.linalg.norm(self.goal - self.pos)
        prev_progress = ml.path_progress(self.pos)
        u_star = self._safety_filter(u_pi)
        new_pos = self.pos + u_star * DT
        new_pos[0] = np.clip(new_pos[0], ml.WORLD_XMIN, ml.WORLD_XMAX)
        new_pos[1] = np.clip(new_pos[1], ml.WORLD_YMIN, ml.WORLD_YMAX)
        self.pos = new_pos
        self._write_qpos()
        mujoco.mj_forward(self.model, self.data)

        collided = self._check_collision()
        new_dist = np.linalg.norm(self.goal - self.pos)
        new_progress = ml.path_progress(self.pos)
        intervention_energy = 0.5 * np.sum((u_star - u_pi) ** 2)

        # Arc-length progress along the maze's true reference route (not
        # raw Euclidean distance, which is topology-blind and can create
        # misleading flat/adversarial regions near corners). This reward
        # shaping choice affects only which policies are easy to learn; it
        # plays no role in the CBF filter or in any RQ1-RQ4 explanation
        # computation, which only ever see (H, A, b, u_pi, u*).
        progress_delta = new_progress - prev_progress
        if progress_delta < 0.01:
            self._stagnation += 1
        else:
            self._stagnation = 0
        stagnation_penalty = 0.0015 * min(self._stagnation, 100)

        reward = 4.0 * progress_delta - 0.006 - stagnation_penalty
        terminated = False
        truncated = False
        if new_dist < GOAL_RADIUS:
            reward += 20.0
            terminated = True
        if collided:
            reward -= 5.0
            terminated = True
        self.t += 1
        if self.t >= MAX_STEPS:
            truncated = True

        info = dict(collided=collided, intervention_energy=intervention_energy,
                    dist_to_goal=new_dist, filter_active=bool(np.linalg.norm(u_star - u_pi) > 1e-9))
        if self.record_qp and self.last_qp_log is not None:
            info["qp"] = self.last_qp_log
        return self._get_obs(), reward, terminated, truncated, info

    def _check_collision(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self.geom_robot or g2 == self.geom_robot:
                other = g2 if g1 == self.geom_robot else g1
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other)
                if name is not None and (name.startswith("wall") or name.startswith("pillar")):
                    if c.dist < -1e-4:  # actual penetration, not just touching
                        return True
        return False

    # ------------------------------------------------------------------ #
    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=700, width=620)
        cam = mujoco.MjvCamera()
        cam.lookat = [0.5, 1.0, 0]
        cam.distance = 17
        cam.azimuth = 90
        cam.elevation = -90
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()
