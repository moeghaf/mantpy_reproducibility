## `mantpy-liver`: Schistosoma-infected mouse liver

Figure 3, Supplementary Fig. 2 and Supplementary Tables 4 and 5. The encoder and the tuned
classifiers need a CUDA GPU in practice; the fixed classifier, the logistic regression and the
permutation test run on CPUs. Install as described in the [README](README.md#install).

### Reproduce the paper's results

```bash
mantpy-liver reproduce --results results/liver
```

This uses the settings selected in the paper and:

1. downloads the 24 ROIs (Zenodo 10.5281/zenodo.21538382), applies the arcsinh and
   99.5th-percentile transform, extracts 5-, 10-, 15- and 20-pixel patches, removes the
   background component and computes the coverage of each patch size;
2. trains the selected contrastive encoder (10-pixel patches) with five seeds and clusters each
   embedding at K = 3 to 12, which gives the five-seed stability at K = 6 and the same-seed
   refit of the partition;
3. fits the fixed GraphSAGE classifier with the ROI-mean loss (the class-weighted loss is averaged
   within each ROI and then over the ROIs), each mouse held out in turn, five seeds per fold
   (40 fits);
4. fits the same classifier with the loss pooled over patches (the class-weighted loss is pooled
   over all patches of the training ROIs; 40 fits) and computes integrated gradients for every
   held-out ROI. These models are also the untuned column of Supplementary Table 5, and the
   report calls them untuned;
5. fits the composition-only logistic regression, choosing C inside each fold;
6. refits the classifier of step 3 for the other 69 allocations (2,760 fits) for the
   permutation test. An allocation is one way of labelling four of the eight mice as infected;
7. refits the tuned classifier of each fold (the eight configurations of Supplementary Table 5)
   with five seeds and the loss pooled over patches, which is also the loss of the tuning;
8. computes composition, the infection-associated cluster, co-occurrence and the subgraphs,
   scores the classifiers and compares the numbers listed in `comparison_with_paper.csv` with
   the paper's values.

**Which clusters are analysed.** The paper froze its partition from the search's seed-0 encoder,
and encoder training on a GPU is not exactly repeatable (a same-seed refit reproduced the
partition with an adjusted Rand index of 0.96). By default steps 3 to 8 therefore analyse the
partition deposited with the paper, and the report shows how closely the encoder refitted in
step 2 reproduces it. To carry this run's own seed-0 partition through steps 3 to 8, add
`--partition refit`. Cluster numbers then follow this run's cluster sizes, so numbers that name
clusters (cluster sizes, per-cluster fractions, co-occurrence pairs) are reported but not
compared with the paper. The infection-associated cluster is chosen by the paper's ratio rule.
If no cluster makes up at least 1% of the ECM patches of every infected mouse, the cluster with
the largest difference between the infected and the naive mice is taken, and the report says so.

The paper's selection also uses the paper's 107,740 ECM signal patches, which are deposited
with its partition (see [Notes on exact reproduction](#notes-on-exact-reproduction-liver)).

Tables are written to `results/liver/report/`:

| File | Content |
|---|---|
| `patches.csv` | patches, recomputed signal patches and coverage at each patch size |
| `encoder_stability.csv` | three- and five-seed stability and spread guard (defined under `search` below) of the selected encoder at each K |
| `composition_per_mouse.csv`, `composition_by_group.csv` | cluster fractions |
| `marker_profiles.csv` | cluster sizes and mean marker intensities |
| `cooccurrence_pairs.csv`, `cooccurrence_per_roi.csv` | log2(observed / expected) per infected mouse, KO - WT, and per ROI |
| `subgraphs.csv` | subgraphs of the infection-associated cluster: patches, edges, mean degree |
| `classifier_per_seed.csv`, `classifier_roi.csv`, `classifier_mouse.csv` | held-out probabilities of the fixed classifier |
| `composition_lr_roi.csv`, `composition_lr_mouse.csv` | logistic regression and the C chosen in each fold |
| `untuned_mouse.csv`, `tuned_mouse.csv`, `tuned_settings.csv` | Supplementary Table 5: the untuned classifier of step 4 and the tuned classifiers of step 7. The tuned settings are the paper's, refitted in this run; with `--selection search` they are the settings chosen by the `tune` stage of this directory |
| `tuning_mouse.csv`, `tuning_settings.csv` | the classifiers with the settings chosen by the `tune` stage of this directory, when `tune` was run with the paper's selection |
| `integrated_gradients.csv.gz`, `ig_localisation.csv` | patch importance and its correlation with cluster membership |
| `permutation_allocations.csv` | mouse-level ROC-AUC of every allocation |
| `classifier_summary.csv` | mice and ROIs correct and ROC-AUC of every model |
| `comparison_with_paper.csv` | each compared number: the value of this run, the paper's value, the difference and the tolerance |

A number agrees with the paper when it is within the tolerance of its kind: counts and
permutation counts must be equal; the recomputed count of 10-pixel signal patches within 100,
that of 5-pixel signal patches within 4,000 and coverage within 0.005; bounds, fractions,
co-occurrence and subgraph statistics within 0.0001; logistic-regression probabilities within
0.005; classifier probabilities within 0.05; adjusted Rand indices and integrated-gradient
correlations within 0.02.

`report` trains no model and can be run at any time. It computes the files of `prepare` when
they are missing. Analyses whose fits are not finished are named and left out, and the command
then returns 1. `comparison_with_paper.csv` is written only when the selected configuration is
the paper's and the run is not a `--smoke` run; otherwise a file left by an earlier report is
removed. Each report writes over the tables of the one before it in `report/`, also when
`--selection` or `--partition` differ. A table of an analysis that is not finished is not
rewritten and stays from the earlier report.

### Run the optimisation as well

```bash
mantpy-liver all --results results/liver_search
```

`all` runs `prepare`, `search`, `freeze`, `tune`, `final` and `report`, and the later stages use
the encoder, K and tuned classifiers that this run selects. The stages can also be run one at a
time:

| Stage | What it does |
|---|---|
| `prepare` | Step 1 above. No label is used. |
| `search` | NSGA-II (Optuna) over the 15 settings of Supplementary Table 4. The study starts from 25 fixed starting configurations (a two-block CNN on 5-pixel patches and 24 scrambled-Sobol points, the same as in the paper's runs) and stops when NSGA-II has proposed 60 new encoders. Every fitted encoder is trained with three seeds and clustered at every K. Objectives: seed stability on the 5-pixel anchors, coverage, and K. A configuration meets the spread guard when every cluster has at least ceil(1000 / s^2) patches (s is the patch size in pixels) in at least three ROIs in every seed. A configuration that needs more than 60,000 optimiser steps is recorded but not fitted, and still counts as one of the 60. The same limit applies to the starting configurations: the first one is over it with the paper's images and is not fitted. |
| `freeze` | Among pairs of encoder and K with at least the coverage of 10-pixel patches that meet the spread guard: refits those within 0.01 of the best three-seed stability at each K with two more seeds; K is the largest value at which a refitted pair meets the spread guard on five seeds and reaches a five-seed stability of 0.80, and the most stable encoder at that K is selected. Writes `selection/selection.json` and the tables behind Supplementary Fig. 2a. It ends with an error when the search is not finished. Give it the `--n-encoders` of the search when the search was given one. |
| `tune` | Within-fold tuning of the classifier: for each held-out mouse, 80 configurations (TPE) scored on the seven training mice by leaving each out in turn with two seeds; the best is refitted with five seeds and the held-out mouse is predicted once. Without `--selection search` it tunes on the paper's partition, which repeats the procedure behind Supplementary Table 5; `report` then shows this tuning beside the paper's refitted settings. |
| `final` | Steps 2 to 7 above. With `--selection search`, step 7 refits the configurations chosen by `tune`. |
| `report` | Step 8. |

The search and the tuning are stochastic: a new run evaluates different configurations from the
paper's and can select differently. `--selection paper` (the default of `tune`, `final` and
`report`) always uses the paper's selections. Results of a search are compared with the paper
only if it selects the paper's encoder and K.

### Resuming and running in parallel

Every fit is saved to its own folder when it finishes: one encoder and seed (with its labels at
every K), one classifier for one allocation, held-out mouse and seed, one proposed tuning
configuration (14 fits), the logistic regression, and the graphs of one graph type. Rerunning any
command skips what is complete, and a lock file stops two processes from fitting the same thing,
so any number of processes can share one `--results` directory. A stored fit is reused only for
the same settings. The settings of a classifier fit include the device and, on the CPU, the
thread count, so a rerun with another `--device` or `--threads` ends with an error for these
fits instead of mixing results. An encoder fit does not record the device, and a tuning
configuration records the device but not the thread count.

An interrupted `search` continues when the same command is run again: the finished encoder fits
are kept, and a starting configuration that was being fitted is queued again. `freeze` ends with
an error until the search is finished.

On one machine:

```bash
mantpy-liver final --workers 2 --results results/liver
```

`--workers` applies to the `search`, `tune` and `final` stages. Each worker uses `--threads` CPU
threads for a classifier fit (16 by default, as in the paper's runs). `--time-limit-hours` makes
`search` start no further configuration after the given time.

`mantpy-liver tasks` lists the independent tasks of `final` (99: encoder seeds 0-4, classifier
folds 5-12, explanation folds 13-20, logistic regression 21, allocations 22-90, tuned folds
91-98) and of `tune` (8, one held-out mouse each); `--task-index` runs one of them. An index
outside the range ends the command with a message.

The command returns 0 when it has finished and 1 after an error or when a worker failed.
`report` also returns 1 when fits are missing: it writes the report of what exists and names
what is missing. `reproduce` and `all` end with `report` and return its code. `search` returns 3
when it stopped at `--time-limit-hours` before it was finished; running the same command again
continues it.

On a cluster, submit the job files in [`slurm/`](slurm):

```bash
mkdir -p logs
V=$HOME/venvs/mantpy
prepare=$(sbatch --parsable -p interactive --export=ALL,VENV=$V pipelines/slurm/liver_prepare.sbatch)
gpu=$(sbatch --parsable -p gpuL --dependency=afterok:$prepare --export=ALL,VENV=$V pipelines/slurm/liver_reproduce_gpu.sbatch)
cpu=$(sbatch --parsable -p multicore --dependency=afterok:$prepare --export=ALL,VENV=$V pipelines/slurm/liver_reproduce_cpu.sbatch)
sbatch -p interactive --dependency=afterany:$gpu:$cpu --export=ALL,VENV=$V pipelines/slurm/liver_report.sbatch
```

The report job follows the fits with `afterany`, so it runs even if some fits failed. It
then names what is missing and ends with exit code 1.

Add `PARTITION=refit` to `--export` of the three later jobs to give them the `--partition refit`
option of the command (it is not the Slurm partition). The CPU tasks then need the seed-0
encoder, which is task 0 of the GPU array: submit the CPU array with
`--dependency=afterok:${gpu}_0`. A CPU task that starts before that encoder is finished trains
it on its CPUs.

For the optimisation, every job of the chain needs the same `RESULTS`:

```bash
R=$HOME/scratch/mantpy_liver_search
prepare=$(sbatch --parsable -p interactive --export=ALL,VENV=$V,RESULTS=$R pipelines/slurm/liver_prepare.sbatch)
search=$(sbatch --parsable -p gpuL --dependency=afterok:$prepare --export=ALL,VENV=$V,RESULTS=$R pipelines/slurm/liver_search.sbatch)
freeze=$(sbatch --parsable -p gpuL --dependency=afterok:$search --export=ALL,VENV=$V,RESULTS=$R pipelines/slurm/liver_freeze.sbatch)
tune=$(sbatch --parsable -p gpuL --dependency=afterok:$freeze --export=ALL,VENV=$V,RESULTS=$R pipelines/slurm/liver_tune.sbatch)
gpu=$(sbatch --parsable -p gpuL --dependency=afterok:$tune --export=ALL,VENV=$V,RESULTS=$R,SELECTION=search pipelines/slurm/liver_reproduce_gpu.sbatch)
cpu=$(sbatch --parsable -p multicore --dependency=afterok:$freeze --export=ALL,VENV=$V,RESULTS=$R,SELECTION=search pipelines/slurm/liver_reproduce_cpu.sbatch)
sbatch -p interactive --dependency=afterany:$gpu:$cpu --export=ALL,VENV=$V,RESULTS=$R,SELECTION=search pipelines/slurm/liver_report.sbatch
```

The CPU array waits for `freeze` only: the seed-0 encoder of the selected configuration, which
defines the partition, is fitted by the search. A search task starts no further configuration
after 23 hours and then ends with exit code 3. The jobs that wait for it with `afterok` then do
not start: cancel them, submit `liver_search.sbatch` again until every task ends with exit
code 0, and submit the later jobs again.

To repeat the tuning on the partition deposited with the paper, submit `liver_tune.sbatch` with
`SELECTION=paper` and the `RESULTS` of the reproduction; `report` reads this tuning there.

The partition names are those of the University of Manchester CSF3; use your cluster's.
Results go to `$RESULTS`. The defaults are `~/scratch/mantpy_liver` for the prepare, final and
report jobs, `~/scratch/mantpy_liver_search` for the search, freeze and tune jobs, and
`~/scratch/mantpy_liver_smoke` for `liver_smoke.sbatch`.

### Check an installation quickly

```bash
mantpy-liver all --smoke --results results/liver_smoke
```

`--smoke` analyses a central 200-pixel crop of every ROI with tiny budgets: a few encoder and
classifier epochs, three starting configurations and two new encoders in the search, three
configurations per tuning fold and two allocations of the permutation test. It exercises every
stage. Its results are not the paper's: no comparison with the paper is written, and no
permutation P value is printed, because that needs all 70 allocations. A results directory keeps
the mode (full or smoke) of its first run, so a smoke run and a full run need separate
directories. On a cluster:

```bash
sbatch -p gpuL --export=ALL,VENV=$V pipelines/slurm/liver_smoke.sbatch
```

The job runs `all --smoke` and then `reproduce --smoke`, which repeats the final fits and the
report with the paper's settings.

<a id="notes-on-exact-reproduction-liver"></a>
### Notes on exact reproduction

- **ECM signal patches.** The two-component background split is a K-means run whose result
  changes with the scikit-learn version, the number of threads and the platform. `prepare` runs
  it with `--threads` OpenMP threads. In our tests the split kept 107,732 or 107,733 of the
  240,000 10-pixel patches with 1, 2, 8 and 16 threads, and 108,618 with 4 threads. The paper's
  count is 107,740. The 5-pixel signal patches, on which seed stability and coverage are
  measured, varied more: 413,145, 413,672 and 414,541 with 1, 4 and 16 threads on Windows and
  416,338 with 16 threads on Linux (scikit-learn 1.7.2), against 413,416 in the paper.
  `--selection paper` therefore trains the encoder on, and analyses, the paper's deposited
  signal patches, and the report shows the recomputed counts beside the paper's. A search uses
  the split recomputed by `prepare`, which is stored once per results directory.
- **Encoder.** Training on CUDA is not exactly repeatable (max-pooling has no deterministic
  backward pass), so a refitted partition differs slightly from the paper's and a new search
  can select differently. On another GPU model the difference is larger: the seed-0 refit of
  the paper's configuration had an adjusted Rand index of 0.86 with the paper's partition on an
  NVIDIA H100 and on an L40S, against 0.96 in the paper (NVIDIA RTX 4070 SUPER). For each seed
  the network takes the same random starting weights as in the paper's search code. The
  notebook [`mantpy_liver_fig3.ipynb`](../notebooks/mantpy_liver_fig3.ipynb) creates the network
  once and so starts from other weights for the same seed. TF32 is off and the encoder and
  K-means use two CPU threads, as in the paper's runs.
- **Classifier.** The paper's fits ran on CPUs with 16 threads (`--threads 16`, the default),
  where they are exactly repeatable on one platform. Another thread count or platform changes
  probabilities in the second or third decimal (up to 0.024 was seen on Windows). The notebook
  builds the graphs with `mt.gr.build_ecm_graphs`, which lists the same edges in another order,
  so its classifier probabilities are not identical to this pipeline's on the same partition.
  `--threads` also sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
  `NUMEXPR_NUM_THREADS` for every stage, whatever the environment holds.
  `MKL_THREADING_LAYER=SEQUENTIAL` is set unless the variable is already set in the environment.
- **Tuned classifiers.** The paper tuned and refitted them on GPUs, so they are not exactly
  repeatable either; they run on `--device`. In our runs on H100 and L40S GPUs the mean
  probability of two mice differed from the paper's by 0.05 and 0.06, and every mouse was
  classified as in the paper. A tuning fold that is interrupted replays its finished
  configurations into a new sampler when it is rerun, as in the paper's code, so the
  configurations proposed after that point differ from an uninterrupted run. The tuning code
  samples 19 named settings; the paper states 22.
- **Graphs.** Patch centres form a regular lattice, on which the Delaunay triangulation is not
  unique. The paper's triangulation is SciPy's with default options and no jitter, and so is
  this pipeline's; the co-occurrence statistics of the deposited partition reproduce the
  paper's to 1e-15 with SciPy 1.16.
- **Permutation test.** The true labelling is the first of the 70 allocations; its 40 fits are the
  classifier of Fig. 3g and are not repeated.

### Run times

Measured during the paper's runs: one encoder fit took
about 5 minutes on an NVIDIA RTX 4070 SUPER, plus about 1 minute for K-means at the ten values
of K; one classifier fit took about 14 s with 16 threads on a Xeon Platinum 8488C (36 to 47 s
with 4 threads on a desktop CPU), so the permutation test (2,760 further fits) is about 11
hours on one such node and much less as a job array; the search took 5 h 40 min plus 30 min of
selection refits on one GPU; tuning took 2 to 6 GPU-hours per held-out mouse.

### Tests

```bash
pytest pipelines/tests/test_liver.py -m "not gpu"   # about two minutes; trains tiny models on a synthetic cohort
pytest pipelines/tests/test_liver.py -m gpu         # real data: descriptive statistics, and seed 0 of the encoder against the paper
```
