<p align="center">
  <img src="../../mantpy_logo.png" alt="Mantpy" width="260"/>
</p>

# Mantpy full experiments

Recompute the paper experiments with the published settings, without rerunning
optimisation. The [short demos](../demos/README.md) remain available separately.

| Notebook | Analysis | Estimated Colab runtime | Colab |
|---|---|---|---|
| [Intestine](mantpy_intestine_full.ipynb) | Both images, method comparisons, training seeds and graph controls | 1–3 h, T4 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_intestine_full.ipynb) |
| [Liver](mantpy_liver_full.ipynb) | All ROIs, mouse-held-out prediction and integrated gradients | 1–2.5 h, T4/CPU | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_liver_full.ipynb) |
| [Lung](mantpy_lung_full.ipynb) | Cell–ECM associations, both held-out mice and repeated label reconstruction | 5–15 min with controls, CPU | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_lung_full.ipynb) |
| [Liver controls](mantpy_liver_controls.ipynb) | Fold-specific selected classifiers, encoder stability and representation controls | Longer; time each stage | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_liver_controls.ipynb) |
| [Liver permutation test](mantpy_liver_permutation.ipynb) | All 70 label allocations with full ensemble refitting | Many hours; resumable | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/experiments/mantpy_liver_permutation.ipynb) |

Estimates exclude installation and downloads. Runtime depends on hardware.

To run:

```bash
cd notebooks/experiments
python -m pip install -r lung_requirements.txt jupyterlab
jupyter lab
```

Use `intestine_requirements.txt` or `liver_requirements.txt` for those studies.
On Colab, use a button above and keep the default Latest runtime. Run setup,
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
