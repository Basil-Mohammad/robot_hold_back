#!/bin/bash
set -e
cd "$(dirname "$0")/envs"
for seed in 0 1 2; do
    echo "=== SAC seed $seed ==="
    python3 train_sac.py --seed $seed --timesteps 100000 --outdir ../models_sac
done
