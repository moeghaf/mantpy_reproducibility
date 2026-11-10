#!/bin/bash
# Final fits on the CPU with the 16 threads of the paper's runs: each array task runs one task of
# `mantpy-liver tasks`.
# Tasks 5-12: the classifier of Fig. 3g, one held-out mouse each. Tasks 13-20: the same classifier
# with the loss pooled over patches, and its integrated gradients. Task 21: logistic regression.
# Tasks 22-90: one allocation of the permutation test each (40 fits).
# SELECTION=paper (default) analyses the partition deposited with the paper, so these tasks do not
# wait for the encoder. SELECTION=search needs the 'freeze' stage to be finished.
# PARTITION sets the --partition option of the command (paper or refit). It is not the Slurm partition.
# With PARTITION=refit and the paper's selection the tasks need the seed-0 encoder, which is task 0
# of liver_reproduce_gpu.sbatch; a task that starts before it is finished trains that encoder on
# its CPUs. Submit this file with --dependency=afterok:<GPU job ID>_0 in that case.
#
# Usage (CSF3): mkdir -p logs; sbatch -p multicore --export=ALL,VENV=<venv> pipelines/slurm/liver_reproduce_cpu.sbatch
# Then:         sbatch -p interactive --dependency=afterany:<GPU job ID>:<CPU job ID> --export=ALL,VENV=<venv> pipelines/slurm/liver_report.sbatch
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
