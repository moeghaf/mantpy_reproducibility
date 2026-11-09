#!/bin/bash
# Final fits of the selected settings: one fit per array task.
# Tasks 0-21 are image 1 and 22-43 are image 2 (see `mantpy-intestine tasks`).
# SELECTION=paper (default) uses the paper's settings; SELECTION=search uses this directory's search result.
#
# Usage (CSF3): mkdir -p logs; sbatch -p gpuL --export=ALL,VENV=<venv> pipelines/slurm/intestine_reproduce.sbatch
# Then:         sbatch -p interactive --dependency=afterok:<jobid> --export=ALL,VENV=<venv> pipelines/slurm/intestine_report.sbatch
#SBATCH --job-name=intestine_final
#SBATCH --array=0-43
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_intestine}
source "$VENV/bin/activate"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n1) on $(hostname)"
IMAGES=(intestine1 intestine2)
IMAGE=${IMAGES[$((SLURM_ARRAY_TASK_ID / 22))]}
mantpy-intestine final --image "$IMAGE" --task-index $((SLURM_ARRAY_TASK_ID % 22))     --selection "${SELECTION:-paper}" --results "$RESULTS" --device cuda
