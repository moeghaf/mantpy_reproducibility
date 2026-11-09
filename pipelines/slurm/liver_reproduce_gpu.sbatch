#!/bin/bash
# Final fits that use a GPU: one task per array job (see `mantpy-liver tasks`).
# Tasks 0-4 are the five training seeds of the selected encoder; tasks 91-98 refit the tuned
# classifier of each held-out mouse with five seeds.
# SELECTION=paper (default) uses the paper's settings. SELECTION=search uses this directory's own
# selections and needs the 'freeze' and 'tune' stages to be finished.
#
# Usage (CSF3): mkdir -p logs; sbatch -p gpuL --export=ALL,VENV=<venv> pipelines/slurm/liver_reproduce_gpu.sbatch
#SBATCH --job-name=liver_final_gpu
#SBATCH --array=0-4,91-98
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_liver}
source "$VENV/bin/activate"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n1) on $(hostname)"
mantpy-liver final --task-index "$SLURM_ARRAY_TASK_ID" --selection "${SELECTION:-paper}" ${PARTITION:+--partition "$PARTITION"} \
    --results "$RESULTS" --device cuda
