#!/usr/bin/env bash
# Runs the soft-label trainings one after another. Each line in RUNS is a set of overrides for configs/soft.yaml.
# Launch inside tmux: bash scripts/run_queue_soft.sh

cd "$(dirname "$0")/.."

RUNS=(
    "--identity_weights true"     # soft labels + identity weights
    "--identity_weights false"    # soft labels only
)

for overrides in "${RUNS[@]}"; do
    name="soft_$(echo "$overrides" | sed 's/--//g' | tr ' ' '_')"
    echo "=== $(date '+%H:%M') start: $overrides ==="
    # A failed run does not stop the queue
    python src/train_distilbert_soft.py $overrides 2>&1 | tee "results/logs/${name}.log"
    echo "=== $(date '+%H:%M') end: $overrides ==="
done
