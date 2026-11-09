#!/bin/bash
# Quick check of an installation: every stage on a crop of each image with tiny training budgets.
# Takes a few minutes on one GPU. Its results are not the paper's.
#
# Usage (CSF3): mkdir -p logs; sbatch -p gpuL --export=ALL,VENV=<venv> pipelines/slurm/intestine_smoke.sbatch
#SBATCH --job-name=intestine_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_intestine_smoke}
source "$VENV/bin/activate"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n1) on $(hostname)"
mantpy-intestine all --smoke --n-trials "${N_TRIALS:-3}" --results "$RESULTS" --device cuda
# The new-search mode needs configurations beyond the three evaluated above.
mantpy-intestine search --smoke --search-mode nsga --n-trials 6 --results "$RESULTS" --device cuda
mantpy-intestine reproduce --smoke --results "$RESULTS" --device cuda
