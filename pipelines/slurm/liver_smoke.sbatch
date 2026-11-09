#!/bin/bash
# Quick check of an installation: every stage on a 200-pixel crop of each ROI with tiny training
# budgets (two new encoders in the search, three configurations per tuning fold, two relabellings
# in the permutation test). Its results are not the paper's.
#
# Usage (CSF3): mkdir -p logs; sbatch -p gpuL --export=ALL,VENV=<venv> pipelines/slurm/liver_smoke.sbatch
#SBATCH --job-name=liver_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_liver_smoke}
source "$VENV/bin/activate"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n1) on $(hostname)"
mantpy-liver all --smoke --results "$RESULTS" --device cuda
mantpy-liver reproduce --smoke --results "$RESULTS" --device cuda
