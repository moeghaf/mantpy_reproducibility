#!/bin/bash
# Label-free search: refit the configurations evaluated in the paper (197 and 200), three seeds each.
# Each array task takes one part of the list. SHARDS is the number of parts (default 16, the size
# of the array). To run only some tasks again, submit with --array=<indices>: the split stays the
# same. If you change the size of the array, set SHARDS to the same number.
# A task starts no further configuration after 22 hours (one limit for all its images) and then
# ends with exit code 3. Submit the same command again until every task ends with exit code 0.
# A freeze job that waits (afterok) for a search job with exit code 3 does not start: submit it
# again with the ID of the last search job.
# GPU memory: in the paper's runs (PyTorch 2.13.0) the largest fit allocated 17.8 GiB at its peak.
# A fit that runs out of GPU memory is recorded as "out of memory" and is fitted again when the
# search is submitted again, for example on a partition whose GPUs have more memory.
# SEARCH_MODE=nsga runs a new NSGA-II search instead (N_TRIALS configurations per image). Its
# tasks are workers that share one study file per image, and a configuration that ran out of
# memory is fitted again only if the search proposes it again.
# IMAGES chooses the images (default: both). For one image only, submit the freeze job with
# --array=0 (image 1) or --array=1 (image 2), the reproduce job with --array=0-21 (image 1) or
# --array=22-43 (image 2), and the report job with IMAGES=<image>.
#
# Usage (CSF3), with the same RESULTS=<dir> for every job:
#   mkdir -p logs
#   sbatch -p interactive --export=ALL,VENV=<venv>,RESULTS=<dir> pipelines/slurm/intestine_prepare.sbatch
#   sbatch -p gpuL --dependency=afterok:<prepare job> --export=ALL,VENV=<venv>,RESULTS=<dir> pipelines/slurm/intestine_search.sbatch
#   sbatch -p gpuL --dependency=afterok:<search job> --export=ALL,VENV=<venv>,RESULTS=<dir> pipelines/slurm/intestine_freeze.sbatch
#   sbatch -p gpuL --dependency=afterok:<freeze job> --export=ALL,VENV=<venv>,RESULTS=<dir>,SELECTION=search pipelines/slurm/intestine_reproduce.sbatch
#   sbatch -p interactive --dependency=afterany:<reproduce job> --export=ALL,VENV=<venv>,RESULTS=<dir>,SELECTION=search pipelines/slurm/intestine_report.sbatch
#SBATCH --job-name=intestine_search
#SBATCH --array=0-15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
: "${VENV:?set VENV to the Python environment that has 'pip install -e pipelines'}"
RESULTS=${RESULTS:-$HOME/scratch/mantpy_intestine_search}
IMAGES=${IMAGES:-intestine1 intestine2}
SHARDS=${SHARDS:-16}
source "$VENV/bin/activate"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n1) on $(hostname)"
# $IMAGES is deliberately unquoted: it may hold two image names.
mantpy-intestine search --image $IMAGES --search-mode "${SEARCH_MODE:-paper}" --n-trials "${N_TRIALS:-200}" --shard "$SLURM_ARRAY_TASK_ID/$SHARDS" --time-limit-hours 22 --results "$RESULTS" --device cuda
