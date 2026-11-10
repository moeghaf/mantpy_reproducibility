#!/bin/bash
# Descriptive statistics, scores and the comparison with the paper. CPU only; no model is trained.
# Submit it with --dependency=afterany:<GPU job ID>:<CPU job ID> on the two arrays of final fits.
# When fits are missing, the report of the finished ones is written, the missing ones are named
# and the job ends with exit code 1.
# Give SELECTION and PARTITION as for the final fits. PARTITION sets the --partition option of the
# command (paper or refit). It is not the Slurm partition.
#
# Usage (CSF3): mkdir -p logs; sbatch -p interactive --export=ALL,VENV=<venv> pipelines/slurm/liver_report.sbatch
#SBATCH --job-name=liver_report
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_liver}
source "$VENV/bin/activate"
mantpy-liver report --selection "${SELECTION:-paper}" ${PARTITION:+--partition "$PARTITION"} --results "$RESULTS" --device cpu
