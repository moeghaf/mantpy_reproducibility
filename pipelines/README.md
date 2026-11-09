# Full end-to-end experiment scripts

Command-line scripts that rerun the paper's experiments end to end, including the
label-free optimisation of settings, on a workstation with a GPU or a cluster. For quick,
interactive reproduction of the figures use the notebooks in [`../notebooks`](../notebooks).

| Command | Experiment | Details |
|---|---|---|
| `mantpy-intestine` | Small intestine: Figure 2, Supplementary Fig. 1a, Supplementary Table 3 | this page |
| `mantpy-liver` | Liver: Figure 3, Supplementary Fig. 2, Supplementary Tables 4 and 5 | [docs_liver.md](docs_liver.md) |
| `mantpy-lung` | Lung: Figure 4, Supplementary Fig. 3, Supplementary Table 6 | [docs_lung.md](docs_lung.md) |

## Install

Python 3.11 to 3.13. Intestine and liver need a CUDA GPU; lung runs on CPUs.

```bash
pip install -e pipelines
```

This installs Mantpy (1.1.0 or newer), Optuna and the other dependencies.

## Reproduce the paper's results

```bash
mantpy-intestine reproduce --results results/intestine
```

This uses the settings selected in the paper and, for both images:

1. downloads the image and annotation and extracts the ECM patches;
2. fits the selected configuration for five training seeds (CNN, ECM graph, GraphMAE);
3. refits GraphMAE with shuffled edges, with no edges, and with a GIN encoder;
4. computes the intensity baselines and CNN Ward clustering;
5. repeats the clustering at K = 2 to 8 to choose K by seed stability;
6. scores everything against the manual annotation and prints each number beside the paper's.

Tables are written to `results/intestine/report/`:

| File | Content |
|---|---|
| `scores_per_seed.csv` | ARI and macro-F1 of every method and seed |
| `scores_summary.csv` | mean and s.d. over seeds |
| `k_sweep.csv` | seed stability and silhouette at each K |
| `comparison_with_paper.csv` | each reported number, the paper's value and the difference |

## Run the optimisation as well

```bash
mantpy-intestine all --results results/intestine_search
```

`all` runs `prepare`, `search`, `freeze`, `final` and `report`, and the final fits use the
configuration that this search selects. The stages can also be run one at a time:

| Stage | What it does |
|---|---|
| `prepare` | Extracts 16-, 32- and 64-pixel patches with the frozen foreground thresholds and computes the common anchors and coverage. |
| `search` | Refits the configurations evaluated in the paper's NSGA-II search of the 28 settings (197 and 200), three seeds each. Objectives: seed stability on the common anchors, and coverage. A configuration is feasible when every cluster holds at least 1% of the anchors in every seed. `--search-mode nsga` runs a new NSGA-II search (Optuna) instead. |
| `freeze` | Among feasible configurations with at least the coverage of 32-pixel patches, takes the most stable; refits those within 0.01 with two more seeds; selects the most stable over five seeds. |
| `final` | The fits listed under "Reproduce the paper's results". |
| `report` | Scores and tables. |

The search does not read the annotation. A new NSGA-II search is stochastic: it evaluates
different configurations from the paper's and can select a different one. `--selection paper`
(the default of `final` and `report`) always uses the paper's selection.

## Resuming and running in parallel

Every fit (one CNN, or one GraphMAE for one image, configuration and seed) is saved to its
own folder when it finishes. Rerunning any command skips what is complete, so an interrupted
run continues where it stopped. A lock file stops two processes from fitting the same thing,
so any number of processes can share one `--results` directory.

On one machine:

```bash
mantpy-intestine search --workers 2 --results results/intestine_search
```

On a cluster, submit the job files in [`slurm/`](slurm). They run the search as an array of
workers that share one study file, and the final fits as an array with one fit per job.
The partition is given at submission and the Python environment through `VENV`:

```bash
mkdir -p logs
V=$HOME/venvs/mantpy
prepare=$(sbatch --parsable -p interactive --export=ALL,VENV=$V pipelines/slurm/intestine_prepare.sbatch)
fits=$(sbatch --parsable -p gpuL --dependency=afterok:$prepare --export=ALL,VENV=$V pipelines/slurm/intestine_reproduce.sbatch)
sbatch -p interactive --dependency=afterany:$fits --export=ALL,VENV=$V pipelines/slurm/intestine_report.sbatch
```

The report job runs even if some fits failed and lists what is missing.

The partition names are those of the University of Manchester CSF3; use your cluster's.
Results go to `$RESULTS` (default `~/scratch/mantpy_intestine`).

## Check an installation quickly

```bash
mantpy-intestine all --smoke --n-trials 3 --results results/smoke
```

`--smoke` analyses a central crop of each image with tiny training budgets. It exercises
every stage in a few minutes. Its results are not the paper's.

## Notes on exact reproduction

- **Random starts.** Every model is trained from scratch. For the same seed, the CNN's starting
  weights depend on whether the code creates the network once or twice before training. The
  paper's runs used both, and `--cnn-init paper` (the default) uses the same one as the paper
  for each configuration and seed. `--cnn-init once` uses Mantpy's normal behaviour throughout.
- **Numerical settings.** TF32 is switched off and K-means uses the thread counts of the
  paper's runs, because both change results in the last digits.
- **Hardware.** GraphMAE scores can differ by about 0.01 ARI between GPUs and library versions.

## Run times

Measured on one NVIDIA RTX 4070 SUPER (12 GB):

| Step | Time |
|---|---|
| `prepare`, both images | under 1 minute |
| one fit of the selected configuration (CNN and GraphMAE) | about 75 s |

`reproduce` makes 44 fits for the two images.

The paper's search evaluated about 200 configurations per image with three seeds each.
Most fits take one to three minutes; the largest (16-pixel patches, wide and deep GraphMAE)
took over 20 minutes each.

## Tests

```bash
pip install -e "pipelines[test]"
pytest pipelines/tests -m "not gpu"     # seconds
pytest pipelines/tests -m gpu           # trains seed 0 of image 1 and compares it with the paper
```
