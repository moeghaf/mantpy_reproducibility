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
   realisations each: each mouse held out (main result), each ROI held out (control), and
   each ROI held out with the foreground and clusters refitted within the training set
   (control); the label-prior baseline is scored on the same folds;
5. writes the tables and prints every recomputed number beside the paper's.

Tables are written to `results/lung/report/`:

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

A number agrees with the paper when it is within 0.01 of the paper's value. Counts must be
equal. For numbers that the paper gives only rounded in its text, half of the last printed
digit is added to the tolerance.

### Run the optimisation as well

```bash
mantpy-lung all --results results/lung
```

`all` runs `prepare`, `describe`, `search`, `select`, `final` and `report`, and the final fits
use the configurations that this search selects. The stages can also be run one at a time:

| Stage | What it does |
|---|---|
| `prepare` | Patches, component scan and ECM partition; also stores the cells and the raw marker statistics, so later stages need neither the dataset nor internet access. |
| `describe` | Steps 2 and 3 above. |
| `search` | Nested Bayesian optimisation (Optuna tree-structured Parzen estimator), one study per outer fold and context: 6 studies with a mouse held out (60 trials), 18 with an ROI held out (150 trials) and 36 for the refit control (150 trials). A trial samples graph rules, features, regularisation and cleaning parameters; its value is the accuracy after cleaning, averaged over an inner loop that holds out each training ROI and over the two tuning realisations (seeds 100 and 101). Held-out ROIs and the evaluation realisations (seeds 0 to 4) are never read. |
| `select` | For each study, among trials within 1e-4 of the best inner accuracy, takes the one with the fewest graph edges, then the earliest. Writes `selection/search_<partition>.json`. |
| `final` | Step 4 above. |
| `report` | Step 5 above. |

`--selection paper` (the default of `final` and `report`) always uses the paper's
selection; `--selection search` uses the result of `select`. Fits of the two selections are
stored separately, so one results directory can hold both.

`--partition paper` (the default) uses the ECM partition of the paper's runs, which is part
of the published source data. `--partition fresh` uses the K-means partition computed by
`prepare`. `prepare` always computes the fresh partition and the report states how many
patches it labels differently.

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

Each worker uses one thread for its fits. `mantpy-lung tasks` prints the number of
independent tasks (60 studies, 68 final fits); `--task-index N` runs one of them.

On a cluster, submit the job files in [`slurm/`](slurm):

```bash
mkdir -p logs
V=$HOME/venvs/mantpy
prepare=$(sbatch --parsable -p interactive --export=ALL,VENV=$V pipelines/slurm/lung_prepare.sbatch)
fits=$(sbatch --parsable -p multicore --dependency=afterok:$prepare --export=ALL,VENV=$V pipelines/slurm/lung_reproduce.sbatch)
sbatch -p interactive --dependency=afterany:$fits --export=ALL,VENV=$V pipelines/slurm/lung_report.sbatch
```

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
not the paper's, and a smoke results directory cannot be reused for a full run. On a cluster,
submit `slurm/lung_smoke.sbatch`.

### Notes on exact reproduction

- **ECM partition.** K-means assigns a few patches next to the background boundary
  differently on different platforms: the paper's partition has 11,435 signal patches, and
  the published notebook, on another machine, obtained 11,450 (16 patches changed cluster).
  The denoising scores depend on these patches in the third decimal place (in the notebook:
  cell-context ROC-AUC 0.966 instead of 0.967, corrected labels 69% instead of 70%). The
  default therefore reads the paper's partition from the source data of Supplementary Fig. 3.
  The fresh partition is fitted with one thread, as in the paper's run.
- **Patch table.** The paper's runs used Mantpy 1.0.1, which left out, in every ROI, the one
  patch with the lowest mean standardised intensity. Mantpy 1.1 keeps all 4,554 tiles, so the
  pipeline applies that rule itself before clustering. The refit control uses all tiles, as
  it did in the paper.
- **Graphs.** All graphs are built with Mantpy. The paper's search ran with an equivalent
  SciPy implementation whose cell-ECM Delaunay edges carried doubled lengths inside the
  triangulation; this changes distance weights for that rule only. In the paper's records
  the single-ROI scores recomputed with the Mantpy graphs agree with those of the search
  run to within 0.00002 ROC-AUC.
- **Logistic regression.** Fits with an L1 or elastic-net penalty use SAGA stopped after 200
  iterations, as in the paper, and are not converged. Their scores can differ between
  library versions and platforms, by up to 0.01 ROC-AUC for a single ROI and realisation.
- **Search.** A study is seeded like the paper's. With Optuna 5.0 the ported search space
  reproduces the configurations of the paper's random start-up trials (checked against the
  archived trials of all 24 studies without fitting anything). Later trials depend on the
  Optuna version and on the last digits of earlier objective values, so a new search does
  not retrace the paper's and can select other configurations.
- **Refit control.** With `--selection paper` the number of clusters of each training set is
  the paper's choice (two when ROI 002 is held out, three otherwise) and the centres are
  refitted; the report records how far they are from the paper's. With `--selection search`
  the number of clusters is selected on each training set.

### Run times

From the paper's records: the notebook
that replays the selected settings takes about 8 minutes on eight threads (partition 3 s,
graphs and association 8 s, edge rules 3 minutes, mouse folds 24 s, single-ROI folds 76 s,
refit control 2 minutes). In the paper's search a mouse study took 1 to 2 minutes and a
single-ROI study 3 to 10 minutes on one thread with the faster SciPy graphs; a study of the
refit control took 17 to 40 minutes on four threads. Expect several CPU hours for
`mantpy-lung search` in total; it divides over workers.

### Tests

```bash
pytest pipelines/tests/test_lung.py -m "not gpu"     # under a minute, synthetic data only
pytest pipelines/tests/test_lung.py -m gpu           # refits the two cell-context mouse folds on the dataset (CPU)
```
