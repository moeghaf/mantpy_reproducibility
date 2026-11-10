# Command-line pipelines for the paper's experiments

Command-line scripts that rerun the paper's experiments end to end, including the
label-free optimisation of settings, on a workstation with a GPU or a cluster. For quick,
interactive reproduction of the figures use the notebooks in [`../notebooks`](../notebooks).

| Command | Experiment | Details |
|---|---|---|
| `mantpy-intestine` | Small intestine: ARI values of Figure 2a, b and d, intensity baselines of Figure 2c, Supplementary Fig. 1a (seed stability and silhouette), Supplementary Table 3 | this page |
| `mantpy-liver` | Liver: Figure 3, Supplementary Fig. 2, Supplementary Tables 4 and 5 | [docs_liver.md](docs_liver.md) |
| `mantpy-lung` | Lung: Figure 4, Supplementary Fig. 3, Supplementary Table 6 | [docs_lung.md](docs_lung.md) |

## Install

Python 3.11 to 3.13. Intestine and liver need a CUDA GPU; lung runs on CPUs. When no GPU is
visible, `--device auto` (the default) trains on the CPU and prints a line that says so;
`--device cuda` stops with an error instead.

```bash
pip install -e pipelines
```

This installs Mantpy (1.3.0 or newer), PyTorch (2.13 or newer), Optuna and the other
dependencies. [`constraints.txt`](constraints.txt) lists the versions the pipelines were run
with; `pip install -e pipelines -c pipelines/constraints.txt` installs these versions.

PyTorch must be a CUDA build for the intestine and liver commands. On Windows the default
download is CPU-only, so install PyTorch first:
`pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130`.

## Reproduce the paper's results

```bash
mantpy-intestine reproduce --results results/intestine
```

This uses the settings selected in the paper and, for both images:

1. downloads the image and annotation and extracts the ECM patches (image 1 through
   `mantpy.datasets.coliv_intestine`, image 2 from Zenodo record 23740311; both are kept in the
   dataset cache: `--cache-dir`, or `MANTPY_CACHE`, or `~/.cache/mantpy`);
2. fits the selected configuration for five training seeds (CNN, ECM graph, GraphMAE);
3. refits GraphMAE with shuffled edges, with no edges, and with a GIN encoder;
4. computes the intensity baselines and CNN Ward clustering;
5. repeats the clustering of seeds 0 to 2 at K = 2 to 8 and chooses K by seed stability (the
   smallest K within 0.01 of the highest stability);
6. scores everything against the manual annotation, writes each compared number beside the
   paper's value, and prints the means, the single runs and the choice of K.

Not rerun by this command: the method comparison of Figure 2c (BANKSY, UTAG, CellCharter, and
the CNN and GraphMAE tuned for that comparison), Supplementary Fig. 1b to d, the ARI at each K
of Supplementary Fig. 1a, and the maps of Figure 2. The notebook
[`mantpy_intestine_fig2.ipynb`](../notebooks/mantpy_intestine_fig2.ipynb) refits the method
comparison with the selected settings and draws the maps. The GIN encoder of step 3 is an
ablation of the selected configuration, not the "GraphMAE (GIN)" entry of Figure 2c.

Tables are written to `results/intestine/report/`:

| File | Content |
|---|---|
| `scores_per_seed.csv` | ARI and macro-F1 of every method and seed |
| `scores_summary.csv` | mean and s.d. of the ARI over seeds, and the number of seeds |
| `k_sweep.csv` | seed stability and silhouette at each K, and the label-free choice of K |
| `comparison_with_paper.csv` | the 35 numbers per image that are compared with the paper: this run, the paper's value, the difference, the tolerance and a status (`within` or `outside` the tolerance, or `not run`) |

The GIN ablation and CNN Ward are scored but not compared with the paper.
`comparison_with_paper.csv` is written only when the selected configuration is the paper's
and the run is not a `--smoke` run. When final tasks have no result, `report` writes the tables
from the results that exist, names the missing tasks and ends with exit code 1.

## Run the optimisation as well

```bash
mantpy-intestine all --results results/intestine_search
```

`all` runs `prepare`, `search`, `freeze`, `final` and `report`, and the final fits use the
configuration that this search selects. The stages can also be run one at a time:

| Stage | What it does |
|---|---|
| `prepare` | Extracts 16-, 32- and 64-pixel patches with the foreground thresholds of the paper's runs (stored values) and computes the common anchors and coverage. |
| `search` | Refits the configurations evaluated in the paper's search of the 28 settings (197 for image 1, 200 for image 2), three seeds each. Objectives: seed stability on the common anchors, and coverage. A configuration is feasible when every cluster holds at least 1% of the anchors in every seed. `--search-mode nsga` runs a new NSGA-II search (Optuna) instead. |
| `freeze` | Among feasible configurations with at least the coverage of 32-pixel patches, takes the most stable over three seeds and every configuration within 0.01 of it; fits these with five seeds; selects the most stable over five seeds that is still feasible. Exact ties go to the shorter mean GraphMAE training time, then to the configuration ID. |
| `final` | The fits listed under "Reproduce the paper's results". |
| `report` | Scores and tables. |

The search does not read the annotation.

The paper's list holds, for each image, 78 configurations that NSGA-II proposed and the
configurations fitted before it (119 for image 1, 122 for image 2), from which the paper's
search started. The earlier configurations were recorded with two to seven seeds; `search`
fits every configuration of the list with three seeds. Seven configurations of the list (six
for image 1, one for image 2) have 2.9 to 4.1 million estimated directed edges. This is more
than the limit of 2.5 million above which a new search skips a graph (`--max-edges`). The
paper's runs fitted them before that limit was set, and `search` fits them like the others.

A new NSGA-II search (`--search-mode nsga`) is stochastic and starts with no earlier results.
It evaluates different configurations from the paper's and can select a different one.

For its five-seed fits, `freeze` fits two further seeds and trains GraphMAE again for the three
seeds of the search, because the search keeps no embeddings. The CNN features are reused.
`freeze` stops with an error when the search is not finished; after a search with
`--search-mode nsga` or `--n-trials`, give `freeze` the same options. The selection is made
from the stabilities of this run, so it can differ from the paper's also when the search
refits the paper's list. `freeze` prints whether the configuration it selects is the paper's.

`--selection paper` (the default of `final` and `report`) always uses the paper's selection.
`--selection search` (the default of `all`) uses the selection of `freeze` in the same results
directory. When that is not the paper's configuration, `report` writes the scores without
`comparison_with_paper.csv`.

## Resuming and running in parallel

Every fit (one CNN, or one GraphMAE for one image, configuration and seed) is saved to its
own folder when it finishes. Rerunning any command skips what is complete, so an interrupted
run continues where it stopped. A lock file stops two processes from fitting the same thing,
so any number of processes can share one `--results` directory. On a cluster, the directory
must be on a file system whose file locks hold between nodes (Lustre on the two clusters used
here).

On one machine:

```bash
mantpy-intestine search --workers 2 --results results/intestine_search
```

`--workers` applies to the `search` and `final` stages. `--shard I/N` runs part I of N of one
of these stages; the search job file uses it for its array. `--time-limit-hours` makes
`search` start no further configuration after the given time; it is one limit for all images
of the command. `mantpy-intestine tasks` prints the number of final tasks of each image (22);
`final --image <image> --task-index N` runs one of them.

The command returns 0 when it has finished and 1 after an error. `report` also returns 1 when
final tasks have no result: it writes the report of what exists and names what is missing.
`search` returns 3 when it stopped at `--time-limit-hours` before it was finished; running the
same command again continues it.

A fit of the search that runs out of GPU memory is recorded as "out of memory", and the search
continues. Such a configuration cannot be selected. Running `search` again fits it again; a
new NSGA-II search fits it again only if it proposes the configuration again.

### On a cluster

Submit the job files in [`slurm/`](slurm). The partition is given at submission and the Python
environment through `VENV`. The final stage runs as an array with one task (a GraphMAE fit,
the baselines or the K sweep) per array task:

```bash
mkdir -p logs
V=$HOME/venvs/mantpy
prepare=$(sbatch --parsable -p interactive --export=ALL,VENV=$V pipelines/slurm/intestine_prepare.sbatch)
fits=$(sbatch --parsable -p gpuL --dependency=afterok:$prepare --export=ALL,VENV=$V pipelines/slurm/intestine_reproduce.sbatch)
sbatch -p interactive --dependency=afterany:$fits --export=ALL,VENV=$V pipelines/slurm/intestine_report.sbatch
```

The report job follows the fits with `afterany`, so it runs even if some fits failed. It
then names the tasks without a result and ends with exit code 1.

For the optimisation, the search runs as a job array in which each task takes one part of the
list of the paper's configurations. With `SEARCH_MODE=nsga` the tasks are workers that share
one study file per image. Every job of this chain needs the same `RESULTS`:

```bash
R=$HOME/scratch/mantpy_intestine_search
prepare=$(sbatch --parsable -p interactive --export=ALL,VENV=$V,RESULTS=$R pipelines/slurm/intestine_prepare.sbatch)
search=$(sbatch --parsable -p gpuL --dependency=afterok:$prepare --export=ALL,VENV=$V,RESULTS=$R pipelines/slurm/intestine_search.sbatch)
freeze=$(sbatch --parsable -p gpuL --dependency=afterok:$search --export=ALL,VENV=$V,RESULTS=$R pipelines/slurm/intestine_freeze.sbatch)
fits=$(sbatch --parsable -p gpuL --dependency=afterok:$freeze --export=ALL,VENV=$V,RESULTS=$R,SELECTION=search pipelines/slurm/intestine_reproduce.sbatch)
sbatch -p interactive --dependency=afterany:$fits --export=ALL,VENV=$V,RESULTS=$R,SELECTION=search pipelines/slurm/intestine_report.sbatch
```

A search task starts no further configuration after 22 hours and then ends with exit code 3.
The jobs that wait for it with `afterok` then do not start: cancel them, submit
`intestine_search.sbatch` again until every task ends with exit code 0, and submit the later
jobs again.

For one image only, set `IMAGES=intestine1` or `IMAGES=intestine2` for the search and the
report jobs, and submit the freeze job with `--array=0` (image 1) or `--array=1` (image 2)
and the final fits with `--array=0-21` (image 1) or `--array=22-43` (image 2).

The images are kept in the dataset cache (`$MANTPY_CACHE`, default `~/.cache/mantpy`). The
later jobs read them again, so the compute nodes must see the cache; to move it, export
`MANTPY_CACHE` before every `sbatch` command. When the compute nodes have no internet access,
run `mantpy-intestine prepare --results <dir> --device cpu` on the login node instead of the
prepare job.

The partition names are those of the University of Manchester CSF3; use your cluster's.
Results go to `$RESULTS`. The defaults are `~/scratch/mantpy_intestine` for the prepare, final
and report jobs, `~/scratch/mantpy_intestine_search` for the search and freeze jobs, and
`~/scratch/mantpy_intestine_smoke` for `intestine_smoke.sbatch`.

## Check an installation quickly

```bash
mantpy-intestine all --smoke --n-trials 3 --results results/smoke
```

`--smoke` analyses a central crop of each image with tiny training budgets. It exercises
every stage in a few minutes. Its results are not the paper's: the search refits only the
first `--n-trials` configurations of the paper's list, the selection does not apply the 1%
condition on cluster sizes, and no comparison with the paper is written. A results directory
keeps the mode (full or smoke) of its first run, so a smoke run and a full run need separate
directories. On a cluster:

```bash
sbatch -p gpuL --export=ALL,VENV=$V pipelines/slurm/intestine_smoke.sbatch
```

## Notes on exact reproduction

- **Random starts.** Every model is trained from scratch. For the same seed, the CNN's starting
  weights depend on whether the code creates the network once or twice before training. The
  paper's runs used both, and `--cnn-init paper` (the default) uses the same one as the paper
  for each configuration and seed. `--cnn-init once` creates the network once for every fit,
  which is Mantpy's normal behaviour; `--cnn-init rebuilt` creates it twice for every fit. Use
  a separate results directory for each `--cnn-init`.
- **K-means seed.** The K-means that turns the features into clusters uses seed 0 in every
  fit, so the training seed changes only the CNN and GraphMAE. In the K sweep, K-means is
  seeded with the training seed.
- **Foreground thresholds.** `prepare` uses the thresholds stored from the paper's runs and
  does not fit them again. A new two-means fit with Mantpy 1.3.0 gives the same thresholds for
  image 2. For image 1 the new fit depends on the random seed: seed 0 gives 13,359 patches of
  32 pixels, and the paper has 13,376.
- **Numerical settings.** TF32 is switched off and K-means uses the thread counts of the
  paper's runs, because each of these settings changes the numerical results.
- **GPU model.** On one GPU model the results repeat: two full runs on an NVIDIA L40S agreed in
  all 70 compared numbers, and the command gives the same scores as the notebook. Between GPU
  models the scores differ. The paper's values were computed on an RTX 4070 SUPER. With PyTorch
  2.13.0 on an L40S and on an H100, the five-seed mean ARI of GraphMAE is 0.890 and 0.895 for
  image 1 (paper 0.897) and 0.380 and 0.379 for image 2 (paper 0.418). Single fits differ more:
  GraphMAE by up to 0.18 (seed 2 of image 2), CNN K-means by up to 0.015 and the seed stability
  at one K by up to 0.18. The label-free choice of K also changed: for image 1 it is 2 in the
  paper, 7 on the L40S and 2 on the H100, and for image 2 it is 4, 3 and 5. The tolerances of
  `comparison_with_paper.csv` are set for the paper's GPU model, so on another model some rows
  are marked `outside` (18 of 70 on the L40S, 16 on the H100).
- **scikit-learn version.** Patch-mean K-means and Multi-Otsu equal the paper on every machine
  tested. Pixel K-means changes with the scikit-learn version: for image 1 it gives an ARI of
  0.196 with 1.7.2 (the version in `constraints.txt`) and 0.203, the paper's value, with 1.9.1.

## Run times

Measured on one NVIDIA RTX 4070 SUPER (12 GB) with PyTorch 2.13.0:

| Step | Time |
|---|---|
| `prepare`, both images | under 1 minute |
| one fit of the selected configuration (CNN and GraphMAE) | about 75 s |

`reproduce` runs 44 tasks for the two images: 40 GraphMAE fits, which share 10 CNN fits, and
the baselines and the K sweep of each image.

The search fits 197 and 200 configurations with three seeds each. Most fits take one to three
minutes; the largest (16-pixel patches, wide and deep GraphMAE) took over 20 minutes each.

GPU memory, as the peak allocated by PyTorch 2.13.0: about 6 GiB for a fit of the selected
configuration, and 17.8 GiB for the largest fit of the search (image 1, 16-pixel patches,
40 neighbours, GraphMAE width 256 and depth 5). Three configurations of the search (two for
image 1, one for image 2) allocated more than 12 GiB. An older PyTorch can need much more
memory (see [`intestine_requirements.txt`](../notebooks/intestine_requirements.txt)).

## Tests

```bash
pip install -e "pipelines[test]"
pytest pipelines/tests/test_intestine.py -m "not gpu"   # a few seconds, on the CPU
pytest pipelines/tests/test_intestine.py -m gpu         # trains seed 0 of image 1 and compares it with the paper
```

The tests of the other two commands are described in [docs_liver.md](docs_liver.md) and
[docs_lung.md](docs_lung.md). `pytest pipelines/tests -m "not gpu"` runs the fast tests of
all three commands on the CPU; the tests marked `gpu` refit models on the real data.
