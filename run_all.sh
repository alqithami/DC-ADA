#!/bin/bash

# DC-Ada Full Experiment Pipeline
# This script runs all experiments required to reproduce the results in the paper.

# Exit on error
set -e

echo "============================================================"
echo "DC-Ada Full Experiment Pipeline"
echo "============================================================"

# Create directories
mkdir -p checkpoints results figures

# Config file (override: CONFIG=configs/strong.yaml ./run_all.sh)
CONFIG=${CONFIG:-configs/default.yaml}

# You can override these from the shell:
#   PRETRAIN_EPISODES=2000 TOTAL_BUDGET=100000 ./run_all.sh
PRETRAIN_EPISODES=${PRETRAIN_EPISODES:-1000}

# Step 1: Pre-train policies for all environments
echo ""
echo "Step 1: Pre-training shared policies..."
echo "------------------------------------------------------------"

python scripts/pretrain_policy.py --config ${CONFIG} --env warehouse --output checkpoints/warehouse_policy.pth --episodes ${PRETRAIN_EPISODES}
python scripts/pretrain_policy.py --config ${CONFIG} --env search_rescue --output checkpoints/search_rescue_policy.pth --episodes ${PRETRAIN_EPISODES}
python scripts/pretrain_policy.py --config ${CONFIG} --env mapping --output checkpoints/mapping_policy.pth --episodes ${PRETRAIN_EPISODES}

# Step 2: Run main experiments from the YAML config.
# NOTE: run_experiment.py already supports multiple environments, heterogeneity levels,
# methods, and seeds in a single consolidated results file.
echo ""
echo "Step 2: Running main experiments (single consolidated run)..."
echo "------------------------------------------------------------"

python scripts/run_experiment.py --config ${CONFIG} --output results

# Step 3: Generate all figures
echo ""
echo "Step 3: Generating figures..."
echo "------------------------------------------------------------"

# Find the most recent results file (written by run_experiment.py)
RESULTS_FILE=$(ls -t results/results_*.json 2>/dev/null | head -1)
if [ -z "$RESULTS_FILE" ]; then
    echo "Warning: No results files found in results/"
else
    python scripts/generate_figures.py --results "$RESULTS_FILE" --output figures/
fi

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "Results saved to: results/"
echo "Figures saved to: figures/"
echo "============================================================"
