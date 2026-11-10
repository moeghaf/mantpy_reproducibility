#!/bin/bash
# Score the finished fits against the annotation and compare them with the paper. CPU only.
# The report is written from the results that exist. When final tasks have no result it names
# them and ends with exit code 1, so submit this job with --dependency=afterany.
# IMAGES=<image> reports one image only (default: both). SELECTION is that of the final fits.
#
# Usage (CSF3): mkdir -p logs; sbatch -p interactive --dependency=afterany:<job of the final fits> --export=ALL,VENV=<venv> pipelines/slurm/intestine_report.sbatch
#SBATCH --job-name=intestine_report
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_intestine}
source "$VENV/bin/activate"
# $IMAGES is deliberately unquoted: it may hold two image names.
mantpy-intestine report ${IMAGES:+--image $IMAGES} --selection "${SELECTION:-paper}" --results "$RESULTS" --device cpu
