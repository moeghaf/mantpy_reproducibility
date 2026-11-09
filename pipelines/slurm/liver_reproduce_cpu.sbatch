#!/bin/bash
# Final fits on the CPU with the paper's 16 threads: one task per array job (see `mantpy-liver tasks`).
# Tasks 5-12: the classifier of Fig. 3g, one held-out mouse each. Tasks 13-20: the same classifier
# with the loss pooled over patches, and its integrated gradients. Task 21: logistic regression.
# Tasks 22-90: one relabelling of the permutation test each (40 fits).
# SELECTION=paper (default) analyses the partition deposited with the paper, so these jobs do not
# wait for the encoder. With SELECTION=search or PARTITION=refit they need the partition of the
# seed-0 encoder: submit them with --dependency=afterok:<jobid>_0 on liver_reproduce_gpu.sbatch.
#
# Usage (CSF3): mkdir -p logs; sbatch -p multicore --export=ALL,VENV=<venv> pipelines/slurm/liver_reproduce_cpu.sbatch
# Then:         sbatch -p interactive --dependency=afterok:<jobids> --export=ALL,VENV=<venv> pipelines/slurm/liver_report.sbatch
#SBATCH --job-name=liver_final_cpu
#SBATCH --array=5-90
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_liver}
source "$VENV/bin/activate"
mantpy-liver final --task-index "$SLURM_ARRAY_TASK_ID" --selection "${SELECTION:-paper}" ${PARTITION:+--partition "$PARTITION"} \
    --threads 16 --results "$RESULTS" --device cpu
