<p align="center">
  <img src="../../mantpy_logo.png" alt="Mantpy" width="260"/>
</p>

# Mantpy full experiments

Recompute the paper experiments with the published settings, without rerunning
optimisation. The [short demos](../demos/README.md) remain available separately.

This checkout contains the development API migration. Install its matching
package before running; see [development setup](#development-api-inspection).
The Colab buttons and measured times below refer to the published reference.

| Notebook | Analysis | Runtime | Colab |
|---|---|---|---|
| [Intestine](mantpy_intestine_full.ipynb) | Both images, method comparisons, training seeds and graph controls | 83.1 min, Colab T4 (measured) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_intestine_full.ipynb) |
| [Liver](mantpy_liver_full.ipynb) | All ROIs, mouse-held-out prediction and integrated gradients | 48 min 47 s, local GPU/CPU (measured) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_liver_full.ipynb) |
| [Lung](mantpy_lung_full.ipynb) | Cell–ECM associations, both held-out mice and repeated label reconstruction | 12 min 38 s, Colab CPU (measured) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_lung_full.ipynb) |
| [Liver controls](mantpy_liver_controls.ipynb) | Fold-specific selected classifiers, encoder stability and representation controls | Several hours or more (estimate) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_liver_controls.ipynb) |
| [Liver permutation test](mantpy_liver_permutation.ipynb) | All 70 label allocations with full ensemble refitting | Hours to days (estimate); resumable | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_liver_permutation.ipynb) |

Installation is excluded; first downloads and hardware can affect elapsed time.
Liver used an RTX 4070 SUPER for the CNN and CPU for the GNNs; allow 1–2.5 h on
Colab T4/CPU. Full liver control and permutation runtimes have not yet been measured.

To run locally, first install the matching Mantpy development wheel or source
checkout, then:

```bash
cd notebooks/experiments
python -m pip install -r lung_requirements.txt jupyterlab
jupyter lab
```

Use `intestine_requirements.txt` or `liver_requirements.txt` for those studies.
For the published reference on Colab, use a button above and keep the default Latest runtime. Run setup,
restart when requested, then run all cells; companion files download automatically.

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

## Development API inspection

These notebooks target Mantpy 1.1.0 and checksum-pinned companions at the
mantpy-api-v1.1.0 repository tag. The automatic Colab path requires both that
tag and the matching PyPI release; use the local candidate until publication.

For local inspection, install the matching 1.1.0 candidate wheel or package
checkout with patch and gnn extras, then the study requirements. Setup prefers
MANTPY_PACKAGE_SOURCE or MANTPY_PACKAGE_WHEEL, and detects an adjacent
checkout or a single uploaded wheel before selecting the public release.

For Colab inspection before publication, upload the matching notebook, setup
file, companions and 1.1.0 candidate wheel. Full experiment companions may be
supplied as mantpy_experiment_companions.zip. Setup verifies all checksums;
restart when requested. Candidate execution status is recorded in the inspection
gallery. The complete liver controls and 70-allocation permutation test have
not been rerun for this migration.
