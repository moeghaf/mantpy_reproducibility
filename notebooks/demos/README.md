<p align="center">
  <img src="../../mantpy_logo.png" alt="Mantpy" width="260"/>
</p>

# Mantpy notebook tutorials

Three short tutorials demonstrate ECM analysis from image loading to spatial
graphs and interpretation.

All three notebooks include outputs from fresh local runs with Mantpy 1.1.0.

| Notebook | Analysis | Measured runtime | Colab |
|---|---|---|---|
| [Intestine: Figure 2, Image 1](mantpy_fig2_image1_demo.ipynb) | Pixel and patch K-means, one shared CNN, GraphMAE, UTAG, BANKSY and CellCharter | 2 min 52 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/demos/mantpy_fig2_image1_demo.ipynb) |
| [Liver: composition and spatial organisation](mantpy_liver_demo.ipynb) | Six-marker cohort clustering, patch graphs, connected regions, held-out GraphSAGE prediction and integrated gradients | 6 min 15 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/demos/mantpy_liver_demo.ipynb) |
| [Lung: cell–ECM interactions](mantpy_lung_demo.ipynb) | Cell–ECM contacts, airway epithelial cell associations and label reconstruction in a held-out mouse | 37 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/demos/mantpy_lung_demo.ipynb) |

Times exclude installation and downloads; intestine and liver used an RTX
4070 SUPER, and lung used CPU only. All tutorials use fixed settings and seed 0.

**To run locally** (Python 3.11/3.12/3.13):

```bash
cd notebooks/demos
python -m pip install -r fig2_demo_requirements.txt jupyterlab
jupyter lab
```

Use `liver_demo_requirements.txt` or `lung_demo_requirements.txt` for the other
tutorials. Data download automatically; results are saved in `demo_results/`.

**On Colab:** keep the default **Latest** runtime, with a GPU for intestine/liver
and CPU for lung. Run setup, restart when requested, then run all cells.
Setup installs Mantpy 1.1.0 and verifies companions from the immutable
`mantpy-api-v1.1.0` repository tag.

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
