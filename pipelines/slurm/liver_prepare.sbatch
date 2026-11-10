#!/bin/bash
# Download the 24 ROIs, normalise them and extract the patches once, before any array job starts. CPU only.
# A search needs this step in its own directory: give this job the RESULTS=<dir> of the search.
# The compute node needs internet access for the first download; otherwise run
#   mantpy-liver prepare --results "$RESULTS" --device cpu
# on the login node.
# The command keeps its default of 16 threads on the 4 CPUs requested, because the number of ECM
# signal patches kept by the background K-means changes with the thread count (see docs_liver.md).
#
# Usage (CSF3): mkdir -p logs; sbatch -p interactive --export=ALL,VENV=<venv> pipelines/slurm/liver_prepare.sbatch
#SBATCH --job-name=liver_prepare
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_liver}
source "$VENV/bin/activate"
mantpy-liver prepare --results "$RESULTS" --device cpu
