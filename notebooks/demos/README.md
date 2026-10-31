<p align="center">
  <img src="../../mantpy_logo.png" alt="Mantpy" width="260"/>
</p>

# Mantpy notebook tutorials

Three short tutorials demonstrate ECM analysis from image loading to spatial
graphs and interpretation.

This checkout contains the development API migration. Install its matching
package before running; see [development setup](#development-api-inspection).
The Colab buttons and measured times below refer to the published reference.

| Notebook | Analysis | Measured runtime | Colab |
|---|---|---|---|
| [Intestine: Figure 2, Image 1](mantpy_fig2_image1_demo.ipynb) | Pixel and patch K-means, one shared CNN, GraphMAE, UTAG, BANKSY and CellCharter | 3 min 43 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/demos/mantpy_fig2_image1_demo.ipynb) |
| [Liver: composition and spatial organisation](mantpy_liver_demo.ipynb) | Six-marker cohort clustering, patch graphs, connected regions, held-out GraphSAGE prediction and integrated gradients | 7 min 37 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/demos/mantpy_liver_demo.ipynb) |
| [Lung: cell–ECM interactions](mantpy_lung_demo.ipynb) | Cell–ECM contacts, airway epithelial cell associations and label reconstruction in a held-out mouse | 32 s | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/demos/mantpy_lung_demo.ipynb) |

Times exclude installation and downloads; intestine and liver used an RTX
4070 SUPER, and lung used CPU only. All tutorials use fixed settings and seed 0.

**To run locally** (Python 3.11/3.12/3.13):

Install the matching Mantpy development wheel or source checkout first.

```bash
cd notebooks/demos
python -m pip install -r fig2_demo_requirements.txt jupyterlab
jupyter lab
```

Use `liver_demo_requirements.txt` or `lung_demo_requirements.txt` for the other
tutorials. Data download automatically; results are saved in `demo_results/`.

**Published reference on Colab:** keep the default **Latest** runtime, with a GPU for intestine/liver
and CPU for lung. Run setup, restart when requested, then run all cells.

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
