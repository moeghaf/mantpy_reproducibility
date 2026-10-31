<p align="center">
  <img src="mantpy_logo.png" alt="Mantpy" width="260"/>
</p>

# Mantpy notebook tutorials

Short tutorials and full experiments demonstrate ECM analysis from image loading
to spatial graphs and interpretation.

This checkout contains the development API migration. Install its matching
package before running; see [development setup](#development-api-inspection).
The Colab buttons and measured times below refer to the published reference.

## Short tutorials

| Notebook | Analysis | Measured runtime | Colab |
|---|---|---|---|
| [Intestine: Figure 2, Image 1](notebooks/demos/mantpy_fig2_image1_demo.ipynb) | Pixel and patch K-means, one shared CNN, GraphMAE, UTAG, BANKSY and CellCharter | 3 min 43 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/demos/mantpy_fig2_image1_demo.ipynb) |
| [Liver: composition and spatial organisation](notebooks/demos/mantpy_liver_demo.ipynb) | Six-marker cohort clustering, patch graphs, connected regions, held-out GraphSAGE prediction and integrated gradients | 7 min 37 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/demos/mantpy_liver_demo.ipynb) |
| [Lung: cell–ECM interactions](notebooks/demos/mantpy_lung_demo.ipynb) | Cell–ECM contacts, airway epithelial cell associations and label reconstruction in a held-out mouse | 32 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/demos/mantpy_lung_demo.ipynb) |

Demo times exclude installation and downloads; intestine and liver used an RTX
4070 SUPER, and lung used CPU only. All demos use fixed settings and seed 0.

## Full experiments

Fit models afresh using the paper's selected settings, folds and repeats,
without rerunning optimisation.

| Notebook | Analysis | Measured runtime | Colab |
|---|---|---|---|
| [Intestine](notebooks/experiments/mantpy_intestine_full.ipynb) | Both images, spatial methods, seeds and graph controls | 83.1 min, Colab T4 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_intestine_full.ipynb) |
| [Liver](notebooks/experiments/mantpy_liver_full.ipynb) | All 24 ROIs, held-out mouse prediction and integrated gradients | 48 min 47 s, local GPU/CPU | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_liver_full.ipynb) |
| [Lung](notebooks/experiments/mantpy_lung_full.ipynb) | All six ROIs, cell–ECM associations and held-out mouse label reconstruction | 12 min 38 s, Colab CPU | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_lung_full.ipynb) |

Installation is excluded; first downloads and hardware can affect elapsed time.
Liver used an RTX 4070 SUPER for the CNN and CPU for the GNNs; allow 1–2.5 h on Colab.
[Liver supplementary controls](notebooks/experiments/mantpy_liver_controls.ipynb)
and the [exact permutation test](notebooks/experiments/mantpy_liver_permutation.ipynb)
are separate, longer runs. See [full experiment setup](notebooks/experiments/README.md).

## Run

**To run locally** (Python 3.11/3.12/3.13):

Install the matching Mantpy development wheel or source checkout first.

```bash
cd notebooks/demos
python -m pip install -r fig2_demo_requirements.txt jupyterlab
jupyter lab
```

Use `liver_demo_requirements.txt` or `lung_demo_requirements.txt` for the other
tutorials. For full experiments, use the requirements in `notebooks/experiments/`.
Data download automatically.

**Published reference on Colab:** keep the default **Latest** runtime, with a GPU for intestine/liver
and CPU for lung. Run setup, restart when requested, then run all cells.

[Mantpy package](https://github.com/moeghaf/Mantpy) ·
[Source data](source_data/) ·
[Paper figure code](figures/)

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
