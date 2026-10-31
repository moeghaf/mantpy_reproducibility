<p align="center">
  <img src="mantpy_logo.png" alt="Mantpy" width="260"/>
</p>

# Mantpy notebook tutorials

Short tutorials and full experiments demonstrate ECM analysis from image loading
to spatial graphs and interpretation.

The three demos and full lung notebook include outputs from fresh local runs
with Mantpy 1.1.0. Calculations used by the other full notebooks passed focused
checks; their complete runs are recorded separately.

## Short tutorials

| Notebook | Analysis | Measured runtime | Colab |
|---|---|---|---|
| [Intestine: Figure 2, Image 1](notebooks/demos/mantpy_fig2_image1_demo.ipynb) | Pixel and patch K-means, one shared CNN, GraphMAE, UTAG, BANKSY and CellCharter | 2 min 52 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/demos/mantpy_fig2_image1_demo.ipynb) |
| [Liver: composition and spatial organisation](notebooks/demos/mantpy_liver_demo.ipynb) | Six-marker cohort clustering, patch graphs, connected regions, held-out GraphSAGE prediction and integrated gradients | 6 min 15 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/demos/mantpy_liver_demo.ipynb) |
| [Lung: cell–ECM interactions](notebooks/demos/mantpy_lung_demo.ipynb) | Cell–ECM contacts, airway epithelial cell associations and label reconstruction in a held-out mouse | 37 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/demos/mantpy_lung_demo.ipynb) |

Demo times exclude installation and downloads; intestine and liver used an RTX
4070 SUPER, and lung used CPU only. All demos use fixed settings and seed 0.

## Full experiments

Fit models afresh using the paper's selected settings, folds and repeats,
without rerunning optimisation.

| Notebook | Analysis | Runtime and execution status | Colab |
|---|---|---|---|
| [Intestine](notebooks/experiments/mantpy_intestine_full.ipynb) | Both images, spatial methods, seeds and graph controls | 83.1 min, Colab T4 (prior full run) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_intestine_full.ipynb) |
| [Liver](notebooks/experiments/mantpy_liver_full.ipynb) | All 24 ROIs, held-out mouse prediction and integrated gradients | 48 min 47 s, local GPU/CPU (prior full run) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_liver_full.ipynb) |
| [Lung](notebooks/experiments/mantpy_lung_full.ipynb) | All six ROIs, cell–ECM associations and held-out mouse label reconstruction | 7 min 53 s, local CPU (fresh full run) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_lung_full.ipynb) |

Installation is excluded; first downloads and hardware can affect elapsed time.
Liver used an RTX 4070 SUPER for the CNN and CPU for the GNNs; allow 1–2.5 h on Colab.
[Liver supplementary controls](notebooks/experiments/mantpy_liver_controls.ipynb)
and the [exact permutation test](notebooks/experiments/mantpy_liver_permutation.ipynb)
are separate, longer runs that have not been completed for this update.
See [full experiment setup](notebooks/experiments/README.md).

## Run

**To run locally** (Python 3.11/3.12/3.13):

```bash
cd notebooks/demos
python -m pip install -r fig2_demo_requirements.txt jupyterlab
jupyter lab
```

Use `liver_demo_requirements.txt` or `lung_demo_requirements.txt` for the other
tutorials. For full experiments, use the requirements in `notebooks/experiments/`.
Data download automatically.

**On Colab:** keep the default **Latest** runtime, with a GPU for intestine/liver
and CPU for lung. Run setup, restart when requested, then run all cells.
Setup installs Mantpy 1.1.0 and verifies companions from the immutable
`mantpy-api-v1.1.0` repository tag.

[Mantpy package](https://github.com/moeghaf/Mantpy) ·
[Source data](source_data/) ·
[Paper figure code](figures/)

Ghafoor et al. (2026).
*[Mantpy: a framework for extracellular matrix analysis in spatial proteomics](https://doi.org/10.1101/2025.06.04.657781).*
bioRxiv.

## Optional local package override

For package development, set `MANTPY_PACKAGE_SOURCE` to a matching checkout
or `MANTPY_PACKAGE_WHEEL` to its wheel. Setup also detects an adjacent checkout
or a single uploaded wheel before selecting Mantpy 1.1.0.
