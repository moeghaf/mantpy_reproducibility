#!/bin/bash
# Nested Bayesian optimisation of the denoising settings: one study per array task. CPU only.
# Tasks 0-5 hold out each mouse (60 trials), 6-23 each ROI (150 trials) and 24-59 each ROI with the
# foreground and clusters refitted within every training set (150 trials); see `mantpy-lung tasks`.
# A finished study is skipped, so resubmit the same command to complete an interrupted array.
# Four CPUs are requested because the K-means of tasks 24-59 uses four threads; tasks 0-23
# use one thread.
#
# Usage (CSF3): mkdir -p logs; sbatch -p multicore --export=ALL,VENV=<venv>,RESULTS=<dir> pipelines/slurm/lung_search.sbatch
# Then:         mantpy-lung select --results <dir>
#               sbatch -p multicore --export=ALL,VENV=<venv>,RESULTS=<dir>,SELECTION=search pipelines/slurm/lung_reproduce.sbatch
#SBATCH --job-name=lung_search
#SBATCH --array=0-59
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_lung}
source "$VENV/bin/activate"
mantpy-lung search --task-index "$SLURM_ARRAY_TASK_ID" --results "$RESULTS" --device cpu
