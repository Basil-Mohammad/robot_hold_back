"""
train_sac.py
-------------
Trains SafeMazeEnv with SAC (stable-baselines3) instead of PPO, to test
whether the paper's identity/attribution/counterfactual/stability results
are specific to the on-policy PPO algorithm used throughout the main
experiment or hold regardless of the underlying RL algorithm. Everything
else (environment, CBF-QP filter in the loop, reward) is identical to
train_ppo.py.

Usage: python3 train_sac.py --seed 0 --timesteps 100000 --outdir ../models_sac
"""
import argparse
import json
import os
import time

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from safe_maze_env import SafeMazeEnv


class EvalLogCallback(BaseCallback):
    def __init__(self, eval_every=20000, n_eval_episodes=10, verbose=0):
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
        env = SafeMazeEnv(record_qp=False)
        successes, returns, interventions = [], [], []
        for ep in range(self.n_eval_episodes):
            obs, _ = env.reset(seed=100000 + ep)
            done = False
            ep_ret = 0.0
            ep_interv = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                ep_ret += r
                ep_interv += info["intervention_energy"]
                done = term or trunc
            successes.append(1.0 if (info["dist_to_goal"] < 0.35 and not info["collided"]) else 0.0)
            returns.append(ep_ret)
            interventions.append(ep_interv)
        rec = dict(timesteps=int(self.num_timesteps),
                    success_rate=float(np.mean(successes)),
                    mean_return=float(np.mean(returns)),
                    mean_intervention_energy=float(np.mean(interventions)))
        self.history.append(rec)
        if self.verbose:
            print(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--outdir", type=str, default="../models_sac")
    ap.add_argument("--eval_every", type=int, default=20000)
    ap.add_argument("--curriculum_p", type=float, default=0.7)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(args.seed)

    def make_env():
        env = SafeMazeEnv(record_qp=False, curriculum_p=args.curriculum_p)
        env = Monitor(env)
        env.reset(seed=args.seed * 1000)
        return env

    venv = DummyVecEnv([make_env])

    model = SAC(
        "MlpPolicy", venv, verbose=0, seed=args.seed,
        learning_rate=3e-4, buffer_size=100_000, learning_starts=2000,
        batch_size=256, tau=0.005, gamma=0.995, train_freq=1,
        gradient_steps=1, ent_coef="auto",
        policy_kwargs=dict(net_arch=[64, 64]),
        device="cpu",
    )

    cb = EvalLogCallback(eval_every=args.eval_every, n_eval_episodes=10, verbose=1)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=cb, progress_bar=False)
    elapsed = time.time() - t0

    model_path = os.path.join(args.outdir, f"sac_seed{args.seed}.zip")
    model.save(model_path)

    log_path = os.path.join(args.outdir, f"train_log_seed{args.seed}.json")
    with open(log_path, "w") as f:
        json.dump(dict(seed=args.seed, timesteps=args.timesteps, elapsed_sec=elapsed,
                        history=cb.history), f, indent=2)

    print(f"[seed {args.seed}] done in {elapsed:.1f}s, final success_rate="
          f"{cb.history[-1]['success_rate'] if cb.history else float('nan')}")


if __name__ == "__main__":
    main()
