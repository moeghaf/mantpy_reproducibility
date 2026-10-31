<p align="center">
  <img src="../../mantpy_logo.png" alt="Mantpy" width="260"/>
</p>

# Mantpy full experiments

Recompute the paper experiments with the published settings, without rerunning
optimisation. The [short demos](../demos/README.md) remain available separately.

These notebooks use Mantpy 1.1.0. The full lung notebook includes outputs from
a fresh local CPU run. Intestine and liver timings below are recorded prior
full runs; their helper calculations passed focused checks against the original
recipes. The three [local demos](../demos/README.md) were also freshly executed.

| Notebook | Analysis | Runtime and execution status | Colab |
|---|---|---|---|
| [Intestine](mantpy_intestine_full.ipynb) | Both images, method comparisons, training seeds and graph controls | 83.1 min, Colab T4 (prior full run) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_intestine_full.ipynb) |
| [Liver](mantpy_liver_full.ipynb) | All ROIs, mouse-held-out prediction and integrated gradients | 48 min 47 s, local GPU/CPU (prior full run) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_liver_full.ipynb) |
| [Lung](mantpy_lung_full.ipynb) | Cell–ECM associations, both held-out mice and repeated label reconstruction | 7 min 53 s, local CPU (fresh full run) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_lung_full.ipynb) |
| [Liver controls](mantpy_liver_controls.ipynb) | Fold-specific selected classifiers, encoder stability and representation controls | Several hours or more (estimate) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_liver_controls.ipynb) |
| [Liver permutation test](mantpy_liver_permutation.ipynb) | All 70 label allocations with full ensemble refitting | Hours to days (estimate); resumable | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_liver_permutation.ipynb) |

Installation is excluded; first downloads and hardware can affect elapsed time.
Liver used an RTX 4070 SUPER for the CNN and CPU for the GNNs; allow 1–2.5 h on
Colab T4/CPU. The complete liver controls and permutation test have not been
completed for this update; their runtimes remain estimates.

To run locally (Python 3.11/3.12/3.13):

```bash
cd notebooks/experiments
python -m pip install -r lung_requirements.txt jupyterlab
jupyter lab
```

Use `intestine_requirements.txt` or `liver_requirements.txt` for those studies.
On Colab, use a button above and keep the default Latest runtime. Run setup,
restart when requested, then run all cells; companion files download automatically.
Setup installs Mantpy 1.1.0 and verifies companions from the immutable
`mantpy-api-v1.1.0` repository tag.

Models are fitted afresh using the saved selections for each training fold.
Completed stages can resume from verified checkpoints in `results/`.
For long Colab runs, set `RESULTS` to a Drive folder to retain checkpoints.
Fresh results and deposited paper references are labelled separately.

[Mantpy package](https://github.com/moeghaf/Mantpy) ·
[Source data](../../source_data/) ·
[Paper figure code](../../figures/)

Ghafoor et al. (2026).
*[Mantpy: a framework for extracellular matrix analysis in spatial proteomics](https://doi.org/10.1101/2025.06.04.657781).*
bioRxiv.

## Optional local package override

For package development, set `MANTPY_PACKAGE_SOURCE` to a matching checkout
or `MANTPY_PACKAGE_WHEEL` to its wheel. Setup also detects an adjacent checkout
or a single uploaded wheel before selecting Mantpy 1.1.0.
