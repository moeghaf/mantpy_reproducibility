## `mantpy-lung`: healthy mouse lung cell-ECM analysis

Figure 4, Supplementary Fig. 3 and Supplementary Table 6. Everything runs on CPUs; no GPU
is used. Install as described in the [README](README.md#install).

### Reproduce the paper's results

```bash
mantpy-lung reproduce --results results/lung
```

This uses the ECM partition and the settings selected in the paper and:

1. downloads the six ROIs (Zenodo 10.5281/zenodo.21538382), extracts the 10 x 10-pixel ECM
   patches, scans K-means with three to seven components and computes the ECM partition;
2. builds the graph of Fig. 4 (Squidpy cell graph within 15 pixels, eight-connected patch grid,
   five nearest patches within 15 pixels) and counts its edges;
3. computes the cell-type by ECM-cluster association, the marker profile of ECM 1 patches
   with and without an airway-epithelial-cell edge, and the association under 48 random
   cell-ECM edge rules;
4. refits the selected denoising configuration of every fold and context on the pristine
   training ROIs and scores each held-out ROI on mixed, block and isolated corruption, five
   realisations each. The folds hold out each mouse (main result), each ROI (control), and
   each ROI with the foreground and clusters refitted within the training set (control).
   The label-prior baseline is scored on the same folds;
5. writes the tables and the comparison with the paper, and prints the headline numbers
   beside the paper's.

Tables are written to `results/lung/report/`. No figures are drawn; the notebooks in
[`../notebooks`](../notebooks) draw the panels.

| File | Content |
|---|---|
| `component_scan.csv` | silhouette, cross-seed ARI and background size for three to seven components |
| `partition.csv` | every patch with its ECM cluster (-1 = background) and marker means |
| `graph_counts.csv` | nodes and edges of the Fig. 4 graph in each ROI |
| `association.csv`, `association_per_roi.csv` | observed and expected cell-ECM edges and log2(observed/expected) |
| `aec_marker_contrast.csv` | ECM 1 marker differences per ROI |
| `edge_rule_sensitivity.csv` | airway-epithelial association under the Fig. 4 rule and 48 random rules |
| `selected_configurations.csv` | selected cell-context settings in the layout of Supplementary Table 6 |
| `denoising_scores.csv` | every fold, context, corruption type, realisation and held-out ROI |
| `denoising_per_roi.csv`, `denoising_summary.csv` | means over realisations within ROI, then over ROIs |
| `class_recovery.csv` | fraction of replaced labels restored, per true cluster |
| `refit_control_scores.csv`, `refit_control_per_roi.csv`, `refit_control_summary.csv` | the refit control |
| `comparison_with_paper.csv` | each reported number, the paper's value, the difference and the tolerance |

`component_scan.csv` holds one K-means fit (seed 0) for each number of components. The
dimmest component is taken as background, the silhouette is computed on a seeded sample of
at most 8,000 of the other patches, and the cross-seed ARI is the mean adjusted Rand index
against three fits with other seeds. These are the values of Supplementary Fig. 3a. The
notebook prints a second scan, in which K-means is refitted on the signal patches only.
Both select three ECM clusters besides the background.

A number agrees with the paper when it is within 0.01 of the paper's value. Counts must be
equal. For numbers that the paper gives only rounded in its text, half of the last printed
digit is added to the tolerance.

### Run the optimisation as well

```bash
mantpy-lung all --results results/lung
```

`all` runs `prepare`, `describe`, `search`, `select`, `final` and `report`, and the final fits
use the configurations that this search selects. The search and the fits use the paper's ECM
partition unless `--partition fresh` is given. The stages can also be run one at a time:

| Stage | What it does |
|---|---|
| `prepare` | Patches, component scan and ECM partition; also stores the cells and the raw marker statistics, so later stages need neither the dataset nor internet access. |
| `describe` | Steps 2 and 3 above. |
| `search` | Nested Bayesian optimisation (Optuna tree-structured Parzen estimator), one study per outer fold and context: 6 studies with a mouse held out (60 trials), 18 with an ROI held out (150 trials) and 36 for the refit control (150 trials; two marker scalings x six held-out ROIs x three contexts). A trial samples graph rules, features, regularisation and cleaning parameters; its value is the accuracy after cleaning, averaged over an inner loop that holds out each training ROI and over the two tuning realisations (seeds 100 and 101). Held-out ROIs and the evaluation realisations (seeds 0 to 4) are never read. |
| `select` | For each study, among trials within 1e-4 of the best inner accuracy, takes the one with the fewest graph edges in the first training ROI, then the earliest. Writes `selection/search_<partition>.json`. Every study must be finished. |
| `final` | Step 4 above. |
| `report` | Step 5 above. |

`--selection paper` (the default of `final` and `report`) always uses the paper's
selection; `--selection search` uses the result of `select`. Fits of the two selections are
stored separately, so one results directory can hold both. The `report` folder holds the
last report written.

`--partition paper` (the default) uses the ECM partition of the paper's runs, which is
packaged with the pipeline (from the source data of Supplementary Fig. 3). `--partition fresh`
uses the K-means partition computed by `prepare`. Its clusters are numbered by size, which
for this dataset is the paper's numbering. `prepare` always computes the fresh partition and
the report states how many patches it labels differently.

### Resuming and running in parallel

Every study of the search and every final fit is saved to its own folder when it finishes.
Rerunning any command skips what is complete. A lock file stops two processes from running
the same study or fit, so any number of processes can share one `--results` directory. A
study that was interrupted starts again from its seed; it is not continued half-way, so a
finished study is always the result of one seeded run.

On one machine:

```bash
mantpy-lung search --workers 8 --results results/lung
mantpy-lung final --workers 8 --results results/lung
```

`--workers` and `--task-index` apply to `search` and `final` only. Each worker uses one
thread for its fits; the K-means of the refit control uses four. `mantpy-lung tasks` prints
the number of independent tasks (60 studies, 68 final fits); `--task-index N` runs one of
them (0 to 59 for `search`, 0 to 67 for `final`). An index outside this range ends the
command with a message.

The command returns 0 when it has finished and 1 after an error or when a worker failed.
`report` also returns 1 when results are missing: it writes the report of what exists and
names what is missing. `reproduce` and `all` end with `report` and return its code.

On a cluster, submit the job files in [`slurm/`](slurm):

```bash
mkdir -p logs
V=$HOME/venvs/mantpy
prepare=$(sbatch --parsable -p interactive --export=ALL,VENV=$V pipelines/slurm/lung_prepare.sbatch)
fits=$(sbatch --parsable -p multicore --dependency=afterok:$prepare --export=ALL,VENV=$V pipelines/slurm/lung_reproduce.sbatch)
sbatch -p interactive --dependency=afterany:$fits --export=ALL,VENV=$V pipelines/slurm/lung_report.sbatch
```

The report job follows the fits with `afterany`, so it runs even if some fits failed. It
then names what is missing and ends with exit code 1.

For the optimisation, submit `lung_search.sbatch` (an array with one study per task) after
`lung_prepare.sbatch`, then run `mantpy-lung select --results <dir>` and submit
`lung_reproduce.sbatch` and `lung_report.sbatch` with `SELECTION=search`. Results go to
`$RESULTS` (default `~/scratch/mantpy_lung`).

### Check an installation quickly

```bash
mantpy-lung all --smoke --results results/lung_smoke
```

`--smoke` runs every stage on the six ROIs with three trials per study, one corruption
realisation, two random edge rules and one held-out ROI in the two controls. Its results are
not the paper's and no comparison with the paper is written. A results directory keeps the
mode (full or smoke) of its first run, so a smoke run and a full run need separate
directories. On a cluster:

```bash
sbatch -p multicore --export=ALL,VENV=$V pipelines/slurm/lung_smoke.sbatch
```

### Notes on exact reproduction

- **ECM partition.** K-means with several OpenMP threads does not repeat with a fixed seed:
  the order of its sums depends on thread timing, and a few patches next to the background
  boundary can change cluster between runs. `prepare` therefore fits the fresh partition
  with one thread, as in the paper's run and in the notebook; this gives the paper's 11,435
  signal patches. The default, `--partition paper`, reads the paper's partition from the
  packaged file, so the results do not depend on the installed K-means.
- **Component scan.** The fits with five and seven components change with the library
  versions and the platform. On Linux with scikit-learn 1.7.2 the seven-component fit gave a
  silhouette of 0.284 and a cross-seed ARI of 0.872, against 0.302 and 0.986 in the paper, and
  the background at five and seven components differed by 5 and 16 patches. The values at
  three, four and six components agreed with the paper's.
- **Patch table.** The paper's runs used Mantpy 1.0.1, which left out, in every ROI, the one
  patch with the lowest mean standardised intensity. Mantpy 1.1 and later keep all 4,554
  tiles of each ROI, so the pipeline applies that rule itself before clustering. The refit
  control uses all tiles, as it did in the paper.
- **Graphs.** All graphs are built with Mantpy. The paper's search ran with an equivalent
  SciPy implementation whose cell-ECM Delaunay edges carried doubled lengths inside the
  triangulation; this changes distance weights for that rule only. In the authors' runs the
  single-ROI scores recomputed with the Mantpy graphs agreed with those of the search to
  within 0.00002 ROC-AUC.
- **Logistic regression.** Fits with an L1 or elastic-net penalty use SAGA stopped after 200
  iterations, as in the paper, and are not converged. Their scores can differ between
  library versions and platforms, by about 0.01 ROC-AUC for a single ROI and realisation.
- **Threads.** A fit in the pipeline runs with BLAS and OpenMP limited to one thread (four
  for the K-means of the refit control). The notebook limits OpenMP to one thread during
  K-means only.
- **Search.** A study is seeded like the paper's. With Optuna 5.0 the search space gives the
  configurations of the paper's random start-up trials in the 24 mouse and single-ROI
  studies. Later trials depend on the Optuna version and on the last digits of earlier
  objective values, so a new search does not retrace the paper's and can select other
  configurations.
- **Refit control.** The markers are scaled in two ways: with the pixel mean and standard
  deviation of the training ROIs (`training_cohort_z`) or of each image (`per_roi_z`). For
  each scaling, K-means is fitted on the tiles of the training ROIs and the cluster with the
  lowest mean centre is background. The number of signal clusters (two to six) is the
  smallest within 1e-4 of the best mean silhouette over three seeds, among the numbers for
  which every signal cluster holds at least 1% of the signal tiles in every seed. The two
  scalings are corrupted at the same positions, on the tiles that are foreground in both.
  With `--selection paper` the number of clusters of each training set is the paper's
  choice (two when ROI 002 is held out, three otherwise) and the centres are refitted; their
  distance from the paper's centres is stored in the `completion.json` of each unit under
  `refit/targets/`. With `--selection search` the number of clusters is selected on each
  training set. This K-means runs with four threads, as in the paper's run, so its
  partition may not repeat exactly between runs.

### Run times

In the authors' runs, replaying the selected settings took about 7 minutes on eight threads
(partition 3 s, graphs and association 8 s, edge rules 3 minutes, mouse folds 24 s,
single-ROI folds 76 s, refit control 2 minutes). In the paper's search a mouse study took
1 to 2 minutes and a single-ROI study 3 to 10 minutes on one thread with the faster SciPy
graphs; a study of the refit control took 17 to 40 minutes on four threads. Expect 11 to 27
hours or more for `mantpy-lung search` with one worker, most of it in the 36 studies of the
refit control; the time divides over workers.

### Tests

```bash
pytest pipelines/tests/test_lung.py -m "not gpu"     # under a minute, synthetic data only
pytest pipelines/tests/test_lung.py -m gpu           # refits the two cell-context mouse folds on the dataset (CPU)
```
