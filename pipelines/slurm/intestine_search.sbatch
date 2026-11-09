#!/bin/bash
# Label-free search: refit the configurations evaluated in the paper (197 and 200), three seeds each.
# Each array task takes one part of the list. IMAGES chooses the images (default: both), so the
# two images can run on different clusters. The largest graphs need a GPU with 40 GB or more.
# Resubmit the same command to continue an unfinished search.
# SEARCH_MODE=nsga runs a new NSGA-II search instead (N_TRIALS configurations per image).
#
# Usage (CSF3): mkdir -p logs; sbatch -p gpuL --export=ALL,VENV=<venv>,RESULTS=<dir> pipelines/slurm/intestine_search.sbatch
# Then:         sbatch -p gpuL --dependency=afterok:<jobid> --export=ALL,VENV=<venv>,RESULTS=<dir> pipelines/slurm/intestine_freeze.sbatch
#               sbatch -p gpuL --export=ALL,VENV=<venv>,RESULTS=<dir>,SELECTION=search pipelines/slurm/intestine_reproduce.sbatch
#SBATCH --job-name=intestine_search
#SBATCH --array=0-15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_intestine_search}
IMAGES=${IMAGES:-intestine1 intestine2}
source "$VENV/bin/activate"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n1) on $(hostname)"
# $IMAGES is deliberately unquoted: it may hold two image names.
mantpy-intestine search --image $IMAGES --search-mode "${SEARCH_MODE:-paper}" --n-trials "${N_TRIALS:-200}" --shard "$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT" --time-limit-hours 22 --results "$RESULTS" --device cuda
