#!/bin/bash
# Choose the encoder and K from the finished search. The encoders within 0.01 of the best
# three-seed stability at each K are refitted with two more seeds, so this job uses a GPU.
# Set N_ENCODERS to the value the search was submitted with (default 60).
# The command ends with an error when the search is not finished; submit liver_search.sbatch again first.
#
# Usage (CSF3): mkdir -p logs; sbatch -p gpuL --export=ALL,VENV=<venv>,RESULTS=<dir> pipelines/slurm/liver_freeze.sbatch
# Then:         sbatch -p gpuL --dependency=afterok:<jobid> --export=ALL,VENV=<venv>,RESULTS=<dir> pipelines/slurm/liver_tune.sbatch
#SBATCH --job-name=liver_freeze
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_liver_search}
source "$VENV/bin/activate"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n1) on $(hostname)"
mantpy-liver freeze --n-encoders "${N_ENCODERS:-60}" --results "$RESULTS" --device cuda
