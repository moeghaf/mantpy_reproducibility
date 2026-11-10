"""Seeding, numerical settings and thread limits shared by every stage."""

from __future__ import annotations

from contextlib import contextmanager
import gc
import importlib.metadata
import os
import platform
import random
import signal
import subprocess
import sys

import numpy as np

UNFINISHED = 3     # exit code of a search that stopped at its time limit; the same command continues it


def run_workers(commands: list[list[str]]) -> int:
    """Run the commands side by side and wait for all of them.

    Returns 0 when every worker succeeded, ``UNFINISHED`` when the only other outcome was a
    search that stopped at its time limit, and 1 otherwise (also for a worker killed by a signal).
    """
    processes = [subprocess.Popen(command, env=os.environ.copy()) for command in commands]
    codes = [process.wait() for process in processes]
    for index, code in enumerate(codes):
        if code != 0:
            print(f"worker {index} exited with code {code}", file=sys.stderr)
    if all(code == 0 for code in codes):
        return 0
    return UNFINISHED if all(code in (0, UNFINISHED) for code in codes) else 1


@contextmanager
def exit_on_sigterm():
    """Within the block, a termination request (a cancelled or timed-out cluster job) raises ``SystemExit``.

    The running code can then close what it has open before the process ends.
    """
    def handler(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def resolve_device(requested: str = "auto") -> str:
    """Return ``"cuda"`` or ``"cpu"``. ``"auto"`` uses a GPU when one is visible."""
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        print("No CUDA device is visible: running on the CPU.", file=sys.stderr)
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but no CUDA device is visible.")
    return requested


def configure_numerics(threads: int = 2) -> None:
    """Numerical settings of the paper's runs: full float32 precision and deterministic kernels
    where PyTorch has them (an operation without one runs with a warning).

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
    """Package versions, Python, platform, CUDA version and the visible GPU, stored with fitted units."""
    import torch

    versions = {}
    for name in ("mantpy", "torch", "torch-geometric", "pytorch-lightning", "numpy", "scipy", "scikit-learn",
                 "scikit-image", "pandas", "anndata", "optuna", "squidpy", "filelock", "threadpoolctl"):
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
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
