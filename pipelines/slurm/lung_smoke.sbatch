#!/bin/bash
# Quick check of an installation: every stage with three trials per study, one corruption
# realisation, two random edge rules and one held-out ROI in the controls. CPU only.
# Its results are not the paper's.
# The first command searches and fits the settings it selects; the second fits the paper's
# settings. The report folder then holds the report of the second command.
#
# Usage (CSF3): mkdir -p logs; sbatch -p multicore --export=ALL,VENV=<venv> pipelines/slurm/lung_smoke.sbatch
#SBATCH --job-name=lung_smoke
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_lung_smoke}
source "$VENV/bin/activate"
mantpy-lung all --smoke --results "$RESULTS" --device cpu
mantpy-lung reproduce --smoke --results "$RESULTS" --device cpu
