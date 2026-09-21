"""
collect_rollouts_quad.py
---------------------------
Same purpose as collect_rollouts.py, for the third (planar-quadrotor)
environment's 5 trained seeds.
"""
import argparse
import glob
import json
import os
import pickle
import time

from stable_baselines3 import PPO

from quad_env import PlanarQuadrotorEnv


def collect_for_seed(seed, model_path, n_episodes, max_steps_cap=400):
    model = PPO.load(model_path)
    env = PlanarQuadrotorEnv(record_qp=True)
    records = []
    n_success = 0
    n_collision = 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=20000 * seed + ep)
        for t in range(max_steps_cap):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            if info["filter_active"] and "qp" in info:
                qp = info["qp"]
                records.append(dict(
                    seed=seed, episode=ep, t=t,
                    H=qp["H"], A=qp["A"], b=qp["b"],
                    u_pi=qp["u_pi"], u_star=qp["u_star"], pos=qp["pos"],
                    obstacles_eff=qp["obstacles_eff"], alpha=qp["alpha"],
                ))
            if term or trunc:
                if info["dist_to_goal"] < 0.5 and not info["collided"]:
                    n_success += 1
                if info["collided"]:
                    n_collision += 1
                break
    return records, n_success, n_collision, env.n_infeasible_fallbacks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", type=str, default="../models_quad")
    ap.add_argument("--outdir", type=str, default="../results")
    ap.add_argument("--n_episodes_per_seed", type=int, default=60)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    model_paths = sorted(glob.glob(os.path.join(args.models_dir, "ppo_quad_seed*.zip")))
    print(f"found {len(model_paths)} trained quad seeds")

    all_records = []
    summary = []
    t0 = time.time()
    for mp in model_paths:
        seed = int(os.path.basename(mp).replace("ppo_quad_seed", "").replace(".zip", ""))
        recs, n_succ, n_coll, n_infeas = collect_for_seed(seed, mp, args.n_episodes_per_seed)
        all_records.extend(recs)
        summary.append(dict(seed=seed, n_qp_instances=len(recs),
                             n_episodes=args.n_episodes_per_seed,
                             success_rate=n_succ / args.n_episodes_per_seed,
                             collision_rate=n_coll / args.n_episodes_per_seed,
                             n_infeasible_fallbacks=n_infeas))
        print(summary[-1], flush=True)

    elapsed = time.time() - t0
    print(f"collected {len(all_records)} QP instances from {len(model_paths)} seeds in {elapsed:.1f}s")

    with open(os.path.join(args.outdir, "rollout_qp_dataset_quad.pkl"), "wb") as f:
        pickle.dump(all_records, f)

    with open(os.path.join(args.outdir, "rollout_summary_quad.json"), "w") as f:
        json.dump(dict(per_seed=summary, total_qp_instances=len(all_records),
                        elapsed_sec=elapsed), f, indent=2)


if __name__ == "__main__":
    main()
