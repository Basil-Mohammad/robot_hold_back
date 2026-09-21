#!/bin/bash
set -e
for seed in 0 1 2 3 4; do
  echo "=== training quad seed $seed ==="
  python3 train_ppo_quad.py --seed $seed --timesteps 200000 --outdir ../models_quad --eval_every 25000
done
