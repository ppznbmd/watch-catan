#!/usr/bin/env bash
# One pair of the saboteur study: the same seed played by {4 Plain} and by
# {3 Plain + Saboteur}, every seat on Luna at effort 'high'. The seed fixes the
# board and the seating, so the two tables differ in the fourth seat's goal
# alone until the first move that differs.
#
#   scripts/saboteur_pair.sh SEED
#
# Kept in the repo, not /tmp: a batch script in /tmp was lost to a reboot once.
set -euo pipefail
cd "$(dirname "$0")/.."
seed=${1:?usage: scripts/saboteur_pair.sh SEED}
log="runs/batch-saboteur-$(date +%Y%m%d-%H%M)-seed$seed.log"
for fourth in Plain Saboteur; do
  echo "=== seed $seed, fourth seat $fourth, $(date -Is)" >>"$log"
  .venv/bin/python -m arena.play --model gpt-5.6-luna --effort high --max-turns 200 \
    --seed "$seed" --seat Plain --seat Plain --seat Plain --seat "$fourth" --quiet \
    >>"$log" 2>&1
done
echo "=== pair done $(date -Is)" >>"$log"
