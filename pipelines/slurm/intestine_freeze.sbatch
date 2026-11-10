#!/bin/bash
# Select one configuration per image from the finished search. This fits five seeds of every
# configuration within 0.01 of the most stable, so it needs a GPU. GraphMAE is trained again for
# the three seeds of the search, because the search keeps no embeddings.
# One image per task: for one image only, submit with --array=0 (image 1) or --array=1 (image 2).
# Set SEARCH_MODE and N_TRIALS to the values the search was submitted with.
#
# Usage (CSF3): mkdir -p logs; sbatch -p gpuL --dependency=afterok:<search job> --export=ALL,VENV=<venv>,RESULTS=<dir> pipelines/slurm/intestine_freeze.sbatch
#SBATCH --job-name=intestine_freeze
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_intestine_search}
source "$VENV/bin/activate"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n1) on $(hostname)"
IMAGES=(intestine1 intestine2)
mantpy-intestine freeze --image "${IMAGES[$SLURM_ARRAY_TASK_ID]}" --search-mode "${SEARCH_MODE:-paper}" --n-trials "${N_TRIALS:-200}" --results "$RESULTS" --device cuda
