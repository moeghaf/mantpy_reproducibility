#!/bin/bash
# Score the finished fits against the annotation and compare them with the paper. CPU only.
#
# Usage (CSF3): mkdir -p logs; sbatch -p interactive --export=ALL,VENV=<venv> pipelines/slurm/intestine_report.sbatch
#SBATCH --job-name=intestine_report
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_intestine}
source "$VENV/bin/activate"
mantpy-intestine report --selection "${SELECTION:-paper}" --results "$RESULTS" --device cpu
