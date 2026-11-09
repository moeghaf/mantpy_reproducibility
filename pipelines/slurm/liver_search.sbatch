#!/bin/bash
# Label-free search of patch size, encoder settings and K. Each array task is one worker. All
# workers share the study in $RESULTS and stop when it holds N_ENCODERS new encoders (default 60).
# The paper's search used a single worker (--array=0). More workers finish sooner but propose
# different configurations. Resubmit the same command to continue an unfinished search.
#
# Usage (CSF3): mkdir -p logs; sbatch -p gpuL --export=ALL,VENV=<venv>,RESULTS=<dir> pipelines/slurm/liver_search.sbatch
# Then:         sbatch -p gpuL --dependency=afterok:<jobid> --export=ALL,VENV=<venv>,RESULTS=<dir> pipelines/slurm/liver_freeze.sbatch
#SBATCH --job-name=liver_search
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_liver_search}
source "$VENV/bin/activate"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n1) on $(hostname)"
mantpy-liver search --worker-index "$SLURM_ARRAY_TASK_ID" --n-encoders "${N_ENCODERS:-60}" --time-limit-hours 23 \
    --results "$RESULTS" --device cuda
