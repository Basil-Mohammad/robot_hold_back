"""
train_ppo.py
-------------
Train one PPO policy (stable-baselines3) on SafeMazeEnv for a given seed.
The safety filter is IN the training loop (the environment always executes
the filtered action u*, never the raw u_pi), matching the "filtering
actions during training" setup the manuscript cites (Cheng et al. 2019,
Yang et al. 2025, Thumm et al. 2023) -- so the policy learns to work well
*with* the filter rather than being penalized for triggering it via a
hand-designed collision cost.

Usage: python3 train_ppo.py --seed 0 --timesteps 600000 --outdir ../models
"""
import argparse
import json
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from safe_maze_env import SafeMazeEnv, MAX_STEPS


class EvalLogCallback(BaseCallback):
    """Periodically rolls out the current policy (deterministic) for a
    handful of episodes and logs success rate / mean return / mean
    intervention energy -- cheap, in-process eval (no separate vec env)."""

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
            steps = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                ep_ret += r
                ep_interv += info["intervention_energy"]
                steps += 1
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
    ap.add_argument("--timesteps", type=int, default=600000)
    ap.add_argument("--outdir", type=str, default="../models")
    ap.add_argument("--eval_every", type=int, default=25000)
    ap.add_argument("--n_envs", type=int, default=1)
    ap.add_argument("--curriculum_p", type=float, default=0.7)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(args.seed)

    def make_env(rank):
        def _init():
            env = SafeMazeEnv(record_qp=False, curriculum_p=args.curriculum_p)
            env = Monitor(env)
            env.reset(seed=args.seed * 1000 + rank)
            return env
        return _init

    if args.n_envs > 1:
        venv = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    else:
        venv = DummyVecEnv([make_env(0)])

    model = PPO(
        "MlpPolicy", venv, verbose=0, seed=args.seed,
        n_steps=1024, batch_size=256, n_epochs=10,
        gamma=0.995, gae_lambda=0.95, learning_rate=3e-4,
        ent_coef=0.01, clip_range=0.2,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        device="cpu",
    )

    cb = EvalLogCallback(eval_every=args.eval_every, n_eval_episodes=10, verbose=1)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=cb, progress_bar=False)
    elapsed = time.time() - t0

    model_path = os.path.join(args.outdir, f"ppo_seed{args.seed}.zip")
    model.save(model_path)

    log_path = os.path.join(args.outdir, f"train_log_seed{args.seed}.json")
    with open(log_path, "w") as f:
        json.dump(dict(seed=args.seed, timesteps=args.timesteps, elapsed_sec=elapsed,
                        history=cb.history), f, indent=2)

    print(f"[seed {args.seed}] done in {elapsed:.1f}s, final success_rate="
          f"{cb.history[-1]['success_rate'] if cb.history else float('nan')}")


if __name__ == "__main__":
    main()
