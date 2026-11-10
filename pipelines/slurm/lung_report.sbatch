#!/bin/bash
# Collect the finished results into tables and compare them with the paper. CPU only.
# Submit it after the fits with --dependency=afterany:<jobid>, so that it also runs when fits
# failed: the report then covers what exists and names what is missing, and the job ends with
# exit code 1.
# SELECTION=paper (default) or SELECTION=search, as given to lung_reproduce.sbatch.
#
# Usage (CSF3): mkdir -p logs; sbatch -p interactive --export=ALL,VENV=<venv> pipelines/slurm/lung_report.sbatch
#SBATCH --job-name=lung_report
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_lung}
source "$VENV/bin/activate"
mantpy-lung report --selection "${SELECTION:-paper}" --results "$RESULTS" --device cpu
