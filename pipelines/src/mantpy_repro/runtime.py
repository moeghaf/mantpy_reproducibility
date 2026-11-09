"""Seeding, numerical settings and thread limits shared by every stage."""

from __future__ import annotations

from contextlib import contextmanager
import gc
import importlib.metadata
import os
import random

import numpy as np


def resolve_device(requested: str = "auto") -> str:
    """Return ``"cuda"`` or ``"cpu"``. ``"auto"`` uses a GPU when one is visible."""
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but no CUDA device is visible.")
    return requested


def configure_numerics(threads: int = 2) -> None:
    """Numerical settings of the paper's runs: full float32 precision and deterministic kernels.

    TF32 is switched off because it changes convolution results on recent NVIDIA GPUs
    enough to alter the trained CNN.
    """
    import torch

    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_everything(seed: int, threads: int = 2) -> None:
    """Seed Python, NumPy and PyTorch, and apply :func:`configure_numerics`."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configure_numerics(threads)


@contextmanager
def calculation_threads(n: int):
    """Limit BLAS/OpenMP threads for a block (K-means results depend on the thread count)."""
    from threadpoolctl import threadpool_limits

    previous = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = str(n)
    try:
        with threadpool_limits(limits=n):
            yield
    finally:
        if previous is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = previous


def release_cuda() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def environment() -> dict:
    """Software and hardware recorded with every result."""
    import torch

    versions = {}
    for name in ("mantpy", "torch", "torch-geometric", "pytorch-lightning", "numpy", "scipy", "scikit-learn",
                 "anndata", "optuna", "pybanksy", "utag", "cellcharter", "squidpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        import mantpy

        versions["mantpy"] = versions["mantpy"] or getattr(mantpy, "__version__", None)
    except ImportError:
        pass
    return {
        "versions": versions,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
