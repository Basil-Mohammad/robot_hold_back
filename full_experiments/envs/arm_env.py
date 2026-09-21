"""
arm_env.py
-----------
A genuine higher-dimensional (m=6) CBF-QP-safeguarded environment: a
6-joint planar manipulator commanded by joint-velocity actions u in R^6,
reaching for a task-space goal while its end effector avoids circular
keep-out regions and its joints respect box limits. Unlike
benchmark_overhead_highdim.py (Sec. IX-G), which times the exact solver
on synthetic random scenarios, this environment produces real
policy-driven (H,A,b,u_pi,u*) instances from an actually-trained PPO
policy, closing the gap the Limitations section flags: every identity,
attribution, counterfactual, and stability result can be checked at
m=6 on real trajectories, not just a timing benchmark.

Safety filter. At state q (joint angles), candidate constraints are
built from two families and pooled together, exactly as the maze/quad
environments pool nearby obstacles:
  (a) task-space obstacle avoidance for the end effector, using the
      first-order CBF condition  2(p_ee-o)^T J(q) u + alpha*h(q) >= 0
      with h(q) = ||p_ee(q)-o||^2 - r_eff^2 (identical in form to the
      maze/quad circular obstacles, but propagated through the 2x6
      manipulator Jacobian instead of the identity map);
  (b) joint-limit CBFs  u_i + alpha*(q_i-q_min) >= 0  and
      -u_i + alpha*(q_max-q_i) >= 0  (affine in u by construction).
The K_NEAREST candidates with smallest slack (closest to violation) are
kept, exactly as in safe_maze_env.py / quad_env.py, and the resulting
(H=I, A, b) is handed to the SAME exact solver (qp_core.solve_qp) used
throughout the paper -- no special-casing for m=6.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import arm_layout as al
from qp_core import solve_qp

DT = 0.05
MAX_QDOT = 2.0          # rad/s per joint
ALPHA = 2.5
K_NEAREST = 8            # pooled from obstacle + joint-limit candidates
SAFETY_MARGIN = 0.03
GOAL_RADIUS = 0.15
MAX_STEPS = 300
N_OBS_IN_STATE = 4


class PlanarArmEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, record_qp=False):
        super().__init__()
        self.record_qp = record_qp
        self.n = al.N_JOINTS
        obs_dim = self.n + 2 + 1 + 3 * N_OBS_IN_STATE
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n,), dtype=np.float32)

        self.q = al.Q_START.copy()
        self.goal = al.sample_goal(np.random.default_rng(0))
        self.t = 0
        self.last_qp_log = None
        self.n_infeasible_fallbacks = 0

    def _candidate_constraints(self, q):
        """Pools obstacle-avoidance and joint-limit CBF rows, ranked by
        slack (smallest/most-violated first), and returns the K_NEAREST."""
        p_ee = al.end_effector(q)
        J = al.jacobian(q)
        rows_A, rows_b, rank_h, meta = [], [], [], []

        for (ox, oy, r) in al.OBSTACLES:
            d = p_ee - np.array([ox, oy])
            r_eff = r + al.EE_RADIUS + SAFETY_MARGIN
            h = d @ d - r_eff ** 2
            a_row = 2 * d @ J  # (1x2)@(2xn) -> (n,)
            rows_A.append(a_row)
            rows_b.append(-ALPHA * h)
            rank_h.append(h)
            meta.append(("obs", ox, oy, r_eff))

        for i in range(self.n):
            h_lo = q[i] - al.Q_MIN[i]
            a_lo = np.zeros(self.n); a_lo[i] = 1.0
            rows_A.append(a_lo); rows_b.append(-ALPHA * h_lo)
            rank_h.append(h_lo); meta.append(("qlo", i, None, None))

            h_hi = al.Q_MAX[i] - q[i]
            a_hi = np.zeros(self.n); a_hi[i] = -1.0
            rows_A.append(a_hi); rows_b.append(-ALPHA * h_hi)
            rank_h.append(h_hi); meta.append(("qhi", i, None, None))

        rank_h = np.array(rank_h)
        idx = np.argsort(rank_h)[:K_NEAREST]
        A = np.array([rows_A[i] for i in idx])
        b = np.array([rows_b[i] for i in idx])
        meta_sel = [meta[i] for i in idx]
        return A, b, meta_sel

    def _cbf_qp(self, q, u_pi):
        A, b, meta = self._candidate_constraints(q)
        H = np.eye(self.n)
        res = solve_qp(H, A, b, u_pi)
        return res, A, b, H, meta

    def _safety_filter(self, u_pi):
        res, A, b, H, meta = self._cbf_qp(self.q, u_pi)
        if not res["feasible"]:
            self.n_infeasible_fallbacks += 1
            u_star = np.zeros(self.n)
            res = dict(u_star=u_star, active_set=(), lam_full=np.zeros(A.shape[0]),
                       s=np.zeros(A.shape[0]), feasible=True, status="frozen_fallback")
        if self.record_qp:
            self.last_qp_log = dict(H=H.copy(), A=A.copy(), b=b.copy(), u_pi=u_pi.copy(),
                                     u_star=res["u_star"].copy(), pos=self.q.copy(),
                                     meta=meta, alpha=ALPHA)
        return res["u_star"]

    def _get_obs(self):
        p_ee = al.end_effector(self.q)
        goal_vec = self.goal - p_ee
        dist = np.linalg.norm(goal_vec)
        d_obs = np.hypot(al.OBSTACLES[:, 0] - p_ee[0], al.OBSTACLES[:, 1] - p_ee[1]) - al.OBSTACLES[:, 2]
        idx = np.argsort(d_obs)[:N_OBS_IN_STATE]
        feats = []
        for i in idx:
            ox, oy, r = al.OBSTACLES[i]
            feats.extend([(ox - p_ee[0]) / 3.0, (oy - p_ee[1]) / 3.0, r])
        obs = np.concatenate([
            self.q / np.pi, goal_vec / 3.0, [dist / 3.0], feats
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        jitter = rng.uniform(-0.15, 0.15, size=self.n)
        self.q = np.clip(al.Q_START + jitter, al.Q_MIN, al.Q_MAX)
        rng_np = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        self.goal = al.sample_goal(rng_np)
        self.t = 0
        self._stagnation = 0
        return self._get_obs(), {}

    def _check_collision(self):
        p_ee = al.end_effector(self.q)
        return not al.is_task_free(p_ee, margin=0.0)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        u_pi = action * MAX_QDOT

        p_ee_prev = al.end_effector(self.q)
        prev_dist = np.linalg.norm(self.goal - p_ee_prev)

        u_star = self._safety_filter(u_pi)
        new_q = self.q + u_star * DT
        new_q = np.clip(new_q, al.Q_MIN, al.Q_MAX)
        self.q = new_q

        p_ee = al.end_effector(self.q)
        collided = self._check_collision()
        new_dist = np.linalg.norm(self.goal - p_ee)
        intervention_energy = 0.5 * np.sum((u_star - u_pi) ** 2)

        progress = prev_dist - new_dist
        reward = 4.0 * progress - 0.005 * intervention_energy - 0.01
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
        truncated = self.t >= MAX_STEPS or self._stagnation > 100
        terminated = bool(reached)

        info = dict(collided=collided, intervention_energy=intervention_energy,
                    dist_to_goal=new_dist,
                    filter_active=bool(np.linalg.norm(u_star - u_pi) > 1e-9))
        if self.record_qp and self.last_qp_log is not None:
            info["qp"] = self.last_qp_log
        return self._get_obs(), float(reward), terminated, truncated, info
