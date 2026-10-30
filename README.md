<p align="center">
  <img src="mantpy_logo.png" alt="Mantpy" width="260"/>
</p>

# Mantpy notebook tutorials

Short tutorials and full experiments demonstrate ECM analysis from image loading
to spatial graphs and interpretation.

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

Times exclude installation and downloads. Liver used an RTX 4070 SUPER for the
CNN and CPU for the GNNs; allow 1–2.5 h on Colab.
[Liver supplementary controls](notebooks/experiments/mantpy_liver_controls.ipynb)
and the [exact permutation test](notebooks/experiments/mantpy_liver_permutation.ipynb)
are separate, longer runs. See [full experiment setup](notebooks/experiments/README.md).

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

[Mantpy package](https://github.com/moeghaf/Mantpy) ·
[Source data](source_data/) ·
[Paper figure code](figures/)

Ghafoor et al. (2026).
*[Mantpy: a framework for extracellular matrix analysis in spatial proteomics](https://doi.org/10.1101/2025.06.04.657781).*
bioRxiv.
