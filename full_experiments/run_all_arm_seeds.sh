#!/bin/bash
set -e
cd "$(dirname "$0")/envs"
for seed in 0 1 2; do
    echo "=== arm seed $seed ==="
    python3 train_ppo_arm.py --seed $seed --timesteps 150000 --outdir ../models_arm --n_envs 2
done
