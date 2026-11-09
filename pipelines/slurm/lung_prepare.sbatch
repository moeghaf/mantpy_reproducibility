#!/bin/bash
# Download the six lung ROIs, extract the ECM patches and compute the ECM partition once,
# before any array job starts. CPU only.
# The compute node needs internet access for the first download; otherwise run
#   mantpy-lung prepare --results "$RESULTS" --device cpu
# on the login node.
#
# Usage (CSF3): mkdir -p logs; sbatch -p interactive --export=ALL,VENV=<venv> pipelines/slurm/lung_prepare.sbatch
#SBATCH --job-name=lung_prepare
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_lung}
source "$VENV/bin/activate"
mantpy-lung prepare --results "$RESULTS" --device cpu
