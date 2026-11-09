#!/bin/bash
# Within-fold tuning of the classifier: one held-out mouse per array task (80 configurations, then
# the selected one refitted with five seeds). SELECTION=search (default here) tunes on the partition
# of this directory's own search. SELECTION=paper tunes on the partition deposited with the paper.
# A task that reaches the time limit continues from its finished configurations when resubmitted.
#
# Usage (CSF3): mkdir -p logs; sbatch -p gpuL --export=ALL,VENV=<venv>,RESULTS=<dir> pipelines/slurm/liver_tune.sbatch
# Then:         submit liver_reproduce_gpu.sbatch, liver_reproduce_cpu.sbatch and liver_report.sbatch
#               with RESULTS=<dir>,SELECTION=search
#SBATCH --job-name=liver_tune
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_liver_search}
source "$VENV/bin/activate"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n1) on $(hostname)"
mantpy-liver tune --task-index "$SLURM_ARRAY_TASK_ID" --selection "${SELECTION:-search}" --results "$RESULTS" --device cuda
