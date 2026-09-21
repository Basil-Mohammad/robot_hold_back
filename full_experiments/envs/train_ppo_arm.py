"""
train_ppo_arm.py
------------------
Train one PPO policy (stable-baselines3) on PlanarArmEnv (m=6 planar
6-DOF manipulator) for a given seed -- the higher-dimensional-action-space
generalization check (Sec. IX-H), mirroring train_ppo.py / train_ppo_quad.py
as closely as the different domain allows. The safety filter is in the
training loop, as in the main experiment.

Usage: python3 train_ppo_arm.py --seed 0 --timesteps 150000 --outdir ../models_arm
"""
import argparse
import json
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from arm_env import PlanarArmEnv


class EvalLogCallback(BaseCallback):
    def __init__(self, eval_every=15000, n_eval_episodes=10, verbose=0):
        super().__init__(verbose)
        self.eval_every = eval_every
        self.n_eval_episodes = n_eval_episodes
        self.history = []
        self._last_eval = 0

    def _on_step(self):
        if self.num_timesteps - self._last_eval >= self.eval_every:
            self._last_eval = self.num_timesteps
            self._run_eval()
        return True

    def _run_eval(self):
        env = PlanarArmEnv(record_qp=False)
        successes, returns, interventions, collisions = [], [], [], []
        for ep in range(self.n_eval_episodes):
            obs, _ = env.reset(seed=100000 + ep)
            done = False
            ep_ret = 0.0
            ep_interv = 0.0
            any_collision = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                ep_ret += r
                ep_interv += info["intervention_energy"]
                any_collision = any_collision or info["collided"]
                done = term or trunc
            successes.append(1.0 if (info["dist_to_goal"] < 0.2 and not any_collision) else 0.0)
            returns.append(ep_ret)
            interventions.append(ep_interv)
            collisions.append(1.0 if any_collision else 0.0)
        rec = dict(timesteps=int(self.num_timesteps),
                    success_rate=float(np.mean(successes)),
                    mean_return=float(np.mean(returns)),
                    mean_intervention_energy=float(np.mean(interventions)),
                    collision_rate=float(np.mean(collisions)))
        self.history.append(rec)
        if self.verbose:
            print(rec, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--timesteps", type=int, default=150000)
    ap.add_argument("--outdir", type=str, default="../models_arm")
    ap.add_argument("--eval_every", type=int, default=15000)
    ap.add_argument("--n_envs", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(args.seed)

    def make_env(rank):
        def _init():
            env = PlanarArmEnv(record_qp=False)
            env = Monitor(env)
            env.reset(seed=args.seed * 1000 + rank)
            return env
        return _init

    venv = DummyVecEnv([make_env(i) for i in range(args.n_envs)])

    model = PPO(
        "MlpPolicy", venv, verbose=0, seed=args.seed,
        n_steps=512, batch_size=256, n_epochs=10,
        gamma=0.995, gae_lambda=0.95, learning_rate=3e-4,
        ent_coef=0.01, clip_range=0.2,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        device="cpu",
    )

    cb = EvalLogCallback(eval_every=args.eval_every, n_eval_episodes=10, verbose=1)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=cb, progress_bar=False)
    elapsed = time.time() - t0

    model_path = os.path.join(args.outdir, f"ppo_arm_seed{args.seed}.zip")
    model.save(model_path)

    log_path = os.path.join(args.outdir, f"train_log_arm_seed{args.seed}.json")
    with open(log_path, "w") as f:
        json.dump(dict(seed=args.seed, timesteps=args.timesteps, elapsed_sec=elapsed,
                        history=cb.history), f, indent=2)

    print(f"[arm seed {args.seed}] done in {elapsed:.1f}s, final success_rate="
          f"{cb.history[-1]['success_rate'] if cb.history else float('nan')}")


if __name__ == "__main__":
    main()
