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
3. fits the fixed GraphSAGE classifier with the ROI-mean loss, each mouse held out in turn, five
   seeds per fold (40 fits);
4. fits the same classifier with the loss pooled over patches (40 fits) and computes integrated
   gradients for every held-out ROI. These models are also the untuned column of Supplementary
   Table 5;
5. fits the composition-only logistic regression, choosing C inside each fold;
6. refits the classifier of step 3 for the other 69 ways of labelling four mice as infected
   (2,760 fits) for the permutation test;
7. refits the tuned classifier of each fold (the eight configurations of Supplementary Table 5)
   with five seeds;
8. computes composition, the infection-associated cluster, co-occurrence and the subgraphs,
   scores the classifiers and prints every number beside the paper's.

**Which clusters are analysed.** The paper froze its partition from the search's seed-0 encoder,
and encoder training on a GPU is not exactly repeatable (a same-seed refit reproduced the
partition with an adjusted Rand index of 0.96). By default steps 3 to 8 therefore analyse the
partition deposited with the paper, and step 2 reports how closely the refitted encoder
reproduces it. To carry this run's own seed-0 partition through steps 3 to 8, add
`--partition refit`. Cluster numbers then follow this run's cluster sizes, so numbers that name
clusters (cluster sizes, per-cluster fractions, co-occurrence pairs) are reported but not
compared with the paper. The infection-associated cluster is always chosen by the paper's rule.

The paper's selection also uses the paper's 107,740 ECM signal patches, which are deposited
with its partition (see [Notes on exact reproduction](#notes-on-exact-reproduction-liver)).

Tables are written to `results/liver/report/`:

| File | Content |
|---|---|
| `patches.csv` | patches, recomputed signal patches and coverage at each patch size |
| `encoder_stability.csv` | three- and five-seed stability and spread guard of the selected encoder at each K |
| `composition_per_mouse.csv`, `composition_by_group.csv` | cluster fractions |
| `marker_profiles.csv` | cluster sizes and mean marker intensities |
| `cooccurrence_pairs.csv`, `cooccurrence_per_roi.csv` | log2(observed / expected) per infected mouse, KO - WT, and per ROI |
| `subgraphs.csv` | subgraphs of the infection-associated cluster: patches, edges, mean degree |
| `classifier_per_seed.csv`, `classifier_roi.csv`, `classifier_mouse.csv` | held-out probabilities of the fixed classifier |
| `composition_lr_roi.csv`, `composition_lr_mouse.csv` | logistic regression and the C chosen in each fold |
| `untuned_mouse.csv`, `tuned_mouse.csv`, `tuned_settings.csv` | Supplementary Table 5 |
| `tuning_mouse.csv`, `tuning_settings.csv` | the same for this run's own tuning, when `tune` was run with the paper's selection |
| `integrated_gradients.csv.gz`, `ig_localisation.csv` | patch importance and its correlation with cluster membership |
| `permutation_allocations.csv` | mouse-level ROC-AUC of every relabelling |
| `classifier_summary.csv` | mice and ROIs correct and ROC-AUC of every model |
| `comparison_with_paper.csv` | each reported number, the paper's value, the difference and the tolerance |

A number agrees with the paper when it is within the tolerance of its kind: counts and
permutation counts must be equal; recomputed signal-patch counts within 100 patches and coverage
within 0.005; bounds, fractions, co-occurrence and subgraph statistics
within 0.0001; logistic-regression probabilities within 0.005; classifier probabilities within
0.05; adjusted Rand indices and integrated-gradient correlations within 0.02. `report` fits
nothing and can be run at any time; analyses whose fits are not finished are named and left out.

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
| `search` | NSGA-II (Optuna) over the 15 settings of Supplementary Table 4. The study starts from 25 fixed configurations (one starting configuration and 24 scrambled-Sobol points, the same as in the paper) and stops after 60 new encoders. Each encoder is trained with three seeds and clustered at every K. Objectives: seed stability on the 5-pixel anchors, coverage, and K. A configuration is feasible when every cluster has at least ceil(1000 / s^2) patches in at least three ROIs in every seed. Configurations needing more than 60,000 optimiser steps are not fitted. |
| `freeze` | Among pairs of encoder and K with at least the coverage of 10-pixel patches that meet the feasibility rule: refits those within 0.01 of the best three-seed stability at each K with two more seeds; K is the largest value at which a refitted pair reaches a five-seed stability of 0.80, and the most stable encoder at that K is selected. Writes `selection/selection.json` and the tables behind Supplementary Fig. 2a. |
| `tune` | Within-fold tuning of the classifier: for each held-out mouse, 80 configurations (TPE) scored on the seven training mice by leaving each out in turn with two seeds; the best is refitted with five seeds and the held-out mouse is predicted once. Without `--selection search` it tunes on the paper's partition, which repeats the procedure behind Supplementary Table 5; `report` then shows this tuning beside the paper's refitted settings. |
| `final` | Steps 2 to 7 above. With `--selection search`, step 7 refits the configurations chosen by `tune`. |
| `report` | Step 8. |

The search and the tuning are stochastic: a new run evaluates different configurations from the
paper's and can select differently. `--selection paper` (the default of `tune`, `final` and
`report`) always uses the paper's selections. Results of a search are compared with the paper
only if it selects the paper's encoder and K.

### Resuming and running in parallel

Every fit is saved to its own folder when it finishes: one encoder and seed (with its labels at
every K), one classifier for one labelling, held-out mouse and seed, one proposed tuning
configuration (14 fits), the logistic regression, and the graphs of one edge rule. Rerunning any
command skips what is complete, and a lock file stops two processes from fitting the same thing,
so any number of processes can share one `--results` directory. A stored fit is reused only for
the same settings; a different thread count or device for the same classifier is refused rather
than mixed in.

On one machine:

```bash
mantpy-liver final --workers 2 --results results/liver
```

Each worker uses `--threads` CPU threads for a classifier fit (16 by default, as in the paper).

`mantpy-liver tasks` lists the independent tasks of `final` (99: encoder seeds 0-4, classifier
folds 5-12, explanation folds 13-20, logistic regression 21, relabellings 22-90, tuned folds
91-98) and of `tune` (8, one held-out mouse each); `--task-index` runs one of them. On a cluster,
submit the job files in [`slurm/`](slurm):

```bash
mkdir -p logs
sbatch -p interactive --export=ALL,VENV=$HOME/venvs/mantpy pipelines/slurm/liver_prepare.sbatch
sbatch -p gpuL        --export=ALL,VENV=$HOME/venvs/mantpy pipelines/slurm/liver_reproduce_gpu.sbatch
sbatch -p multicore   --export=ALL,VENV=$HOME/venvs/mantpy pipelines/slurm/liver_reproduce_cpu.sbatch
sbatch -p interactive --export=ALL,VENV=$HOME/venvs/mantpy pipelines/slurm/liver_report.sbatch
```

For the optimisation, submit `liver_search.sbatch`, then `liver_freeze.sbatch`, then
`liver_tune.sbatch`, then the three job files above with `RESULTS=<dir>,SELECTION=search`; the
CPU array must then wait for task 0 of the GPU array, whose encoder defines the partition.
Results go to `$RESULTS` (default `~/scratch/mantpy_liver`, or `~/scratch/mantpy_liver_search`
for the optimisation).

### Check an installation quickly

```bash
mantpy-liver all --smoke --results results/liver_smoke
```

`--smoke` analyses a central 200-pixel crop of every ROI with tiny budgets: a few encoder and
classifier epochs, two new encoders in the search, three configurations per tuning fold and two
relabellings. It exercises every stage. Its results are not the paper's and are never compared
with them.

<a id="notes-on-exact-reproduction-liver"></a>
### Notes on exact reproduction

- **ECM signal patches.** The two-component background split is a K-means run whose result
  changes in a few patches with the scikit-learn version and the number of threads: on the
  test workstation it kept 107,732 or 107,733 of the 240,000 10-pixel patches at 1, 2, 8 and 16
  threads and 108,618 at 4 threads, against the paper's 107,740. `--selection paper` therefore
  trains the encoder on, and analyses, the paper's deposited signal patches, and the report
  shows the recomputed count beside the paper's. A search uses the split recomputed by
  `prepare`, which is stored once per results directory.
- **Encoder.** Training on CUDA is not exactly repeatable (max-pooling has no deterministic
  backward pass), so a refitted partition differs slightly from the paper's and a new search
  can select differently. For each seed the network takes the same random starting weights as
  in the paper's search code. TF32 is off and the encoder and K-means use two CPU threads, as
  in the paper's runs.
- **Classifier.** The paper's fits ran on CPUs with 16 threads (`--threads 16`, the default),
  where they are exactly repeatable on one platform. Another thread count or platform changes
  probabilities in the second or third decimal (up to 0.024 was seen on Windows).
  `MKL_THREADING_LAYER=SEQUENTIAL` and the thread variables are set as in the paper's runs
  unless they are already set in the environment.
- **Tuned classifiers.** The paper tuned and refitted them on GPUs, so they are not exactly
  repeatable either; they run on `--device`. A tuning fold that is interrupted replays its
  finished configurations into a new sampler when it is rerun, as in the paper's code, so the
  configurations proposed after that point differ from an uninterrupted run. The tuning code,
  ported unchanged, samples 19 named settings; the paper's count of 22 is that of the original
  search-space plan.
- **Graphs.** Patch centres form a regular lattice, on which the Delaunay triangulation is not
  unique. The paper's triangulation is SciPy's with default options and no jitter, and so is
  this pipeline's; the co-occurrence statistics of the deposited partition reproduce the
  paper's to 1e-15 with SciPy 1.16.
- **Permutation test.** The true labelling is the first of the 70; its 40 fits are the
  classifier of Fig. 3g and are not repeated.

### Run times

From the paper's run records: one encoder fit took
about 5 minutes on an NVIDIA RTX 4070 SUPER, plus about 1 minute for K-means at the ten values
of K; one classifier fit took about 14 s with 16 threads on a Xeon Platinum 8488C (36 to 47 s
with 4 threads on a workstation), so the permutation test (2,760 further fits) is about 11
hours on one such node and much less as a job array; the search took 5 h 40 min plus 30 min of
selection refits on one GPU; tuning took 2 to 6 GPU-hours per held-out mouse.

### Tests

```bash
pytest pipelines/tests/test_liver.py -m "not gpu"   # about a minute; trains tiny models on a synthetic cohort
pytest pipelines/tests/test_liver.py -m gpu         # real data: descriptive statistics, and seed 0 of the encoder against the paper
```
