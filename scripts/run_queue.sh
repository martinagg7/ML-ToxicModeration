#!/usr/bin/env bash
# Runs several trainings one after another. Each line in RUNS is a set of overrides for configs/base.yaml.
# Launch inside tmux: bash scripts/run_queue.sh

cd "$(dirname "$0")/.."

RUNS=(
    "--learning_rate 2e-5"
    "--learning_rate 3e-5"
)

for overrides in "${RUNS[@]}"; do
    name=$(echo "$overrides" | sed 's/--//g' | tr ' ' '_')
    echo "=== $(date '+%H:%M') start: $overrides ==="
    # A failed run does not stop the queue
    python src/train_distilbert.py $overrides 2>&1 | tee "results/logs/${name}.log"
    echo "=== $(date '+%H:%M') end: $overrides ==="
done
