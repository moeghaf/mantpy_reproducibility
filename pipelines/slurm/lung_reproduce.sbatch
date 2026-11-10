#!/bin/bash
# Graphs and association, then the final fits of the selected settings: 68 fits shared by one
# worker per CPU (mouse held out, single ROIs held out, refitted foreground and clusters). CPU only.
# SELECTION=paper (default) uses the paper's settings; SELECTION=search uses this directory's search result.
# The fits can also run one per array task: `mantpy-lung final --task-index N` (see `mantpy-lung tasks`).
#
# Usage (CSF3): mkdir -p logs; sbatch -p multicore --export=ALL,VENV=<venv> pipelines/slurm/lung_reproduce.sbatch
# Then:         sbatch -p interactive --dependency=afterany:<jobid> --export=ALL,VENV=<venv> pipelines/slurm/lung_report.sbatch
#               (afterany: the report also runs when fits failed, and names what is missing)
#SBATCH --job-name=lung_final
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_lung}
source "$VENV/bin/activate"
mantpy-lung describe --results "$RESULTS" --device cpu
mantpy-lung final --workers "$SLURM_CPUS_PER_TASK" --selection "${SELECTION:-paper}" --results "$RESULTS" --device cpu
