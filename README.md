<p align="center">
  <img src="mantpy_logo.png" alt="Mantpy" width="260"/>
</p>

# Mantpy: reproducing the paper

Code and data to reproduce every figure of
*[Mantpy: a framework for extracellular matrix analysis in spatial proteomics](https://doi.org/10.1101/2025.06.04.657781)*
(Ghafoor et al., 2026). There are three ways, from fastest to most complete.

| | What it does | Time |
|---|---|---|
| [1. Notebooks](#1-notebooks) | Refit each figure from the raw images with the paper's settings | 1 to 30 min |
| [2. Full scripts](#2-full-scripts) | Rerun each experiment end to end, including the search for the settings | hours, GPU or cluster |
| [3. Redraw the figures](#3-redraw-the-figures) | Draw the published figures from the saved numbers | seconds |

## 1. Notebooks

One notebook per figure. Each refits the analysis from the raw images, draws every panel and prints
each value beside the paper's. The search for the settings is not rerun.

| Figure | Notebook | Runtime | Hardware | Colab |
|---|---|---|---|---|
| 2 · Small intestine: tissue layers from collagen IV | [mantpy_intestine_fig2.ipynb](notebooks/mantpy_intestine_fig2.ipynb) | 27 min | GPU | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/mantpy_intestine_fig2.ipynb) |
| 3 · Liver: ECM composition and organisation in infection | [mantpy_liver_fig3.ipynb](notebooks/mantpy_liver_fig3.ipynb) | 6 min | GPU | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/mantpy_liver_fig3.ipynb) |
| 4 · Lung: cell–matrix association and label recovery | [mantpy_lung_fig4.ipynb](notebooks/mantpy_lung_fig4.ipynb) | 52 s | CPU | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/mantpy_lung_fig4.ipynb) |

Runtimes exclude installation and downloads and were measured on desktop computers (intestine with
an NVIDIA RTX 4070 SUPER, liver with an RTX 3080 Ti, lung on the CPU). The notebooks are stored with their outputs, so the results can be read without running
anything. Results are close to the paper's but not identical, because every model is refitted.

## 2. Full scripts

Command-line scripts rerun each experiment from the raw data, including the search for the settings.

```bash
pip install -e pipelines
```

```bash
mantpy-intestine reproduce --results results/intestine
```

| Command | Experiment |
|---|---|
| `mantpy-intestine` | Small intestine (Figure 2, Supplementary Fig. 1, Supplementary Table 3) |
| `mantpy-liver` | Liver (Figure 3, Supplementary Fig. 2, Supplementary Tables 4 and 5) |
| `mantpy-lung` | Lung (Figure 4, Supplementary Fig. 3, Supplementary Table 6) |

`reproduce` uses the paper's selected settings and prints each result beside the paper's; `all`
also reruns the search. Stages, outputs and cluster job files are described in
[pipelines](pipelines/README.md).

## 3. Redraw the figures

[source_data](source_data/) holds the numbers behind every figure and table, and
[figures](figures/) the plotting code that produced the published figures.

```python
from paper_figures import plot_figure4   # run from the figures/ folder
plot_figure4().savefig("fig4.pdf")
```

## Requirements

- Python 3.11 to 3.13 and Mantpy 1.2.0 or newer (`pip install "mantpy[patch,gnn]"`). The packages for
  each notebook are listed in its `*_requirements.txt` file; on Colab the first cell installs them.
- Intestine and liver need a CUDA GPU; lung runs on a CPU.
- Data download automatically from Zenodo and the HuBMAP Data Portal on first use.
- Tested on Windows 11.

## Links

[Mantpy package](https://github.com/moeghaf/Mantpy) ·
[Paper (bioRxiv)](https://doi.org/10.1101/2025.06.04.657781) ·
[Licence (MIT)](LICENSE)
