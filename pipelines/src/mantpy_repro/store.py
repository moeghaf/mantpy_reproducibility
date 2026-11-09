"""Resumable, parallel-safe storage of fitted results.

Every unit of work (for example one CNN for one image and seed) owns one folder.
A folder is complete only when ``completion.json`` exists; it is written last, after
the arrays, so an interrupted job leaves no half-finished result that looks complete.
A lock file stops two workers from fitting the same unit at the same time, which
makes the pipeline safe to run as many cluster jobs sharing one results directory.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Callable

from filelock import FileLock
import numpy as np


def canonical(value: Any) -> str:
    """Stable JSON text for hashing configurations."""

    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=convert)


def digest(value: Any, length: int = 12) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()[:length]


def unique_suffix() -> str:
    """Name part that differs between processes, also on different nodes of a cluster."""
    return f"{socket.gethostname()}.{os.getpid()}"


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{unique_suffix()}.part")
    temporary.write_text(json.dumps(json.loads(canonical(value)), indent=1), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Unit:
    """One resumable unit of work stored in ``folder``."""

    def __init__(self, folder: Path, request: dict):
        self.folder = Path(folder)
        self.request = request

    @property
    def marker(self) -> Path:
        return self.folder / "completion.json"

    def done(self) -> bool:
        if not self.marker.exists():
            return False
        saved = read_json(self.marker)["request"]
        if canonical(saved) != canonical(self.request):
            raise RuntimeError(
                f"{self.folder} holds a result for different settings. Use a new --results directory "
                "or delete that folder."
            )
        return True

    def load(self) -> dict[str, np.ndarray]:
        with np.load(self.folder / "arrays.npz", allow_pickle=False) as saved:
            return {name: saved[name] for name in saved.files}

    def metadata(self) -> dict:
        return read_json(self.marker)

    def save(self, seconds: float, metadata: dict | None = None, **arrays: np.ndarray) -> dict[str, np.ndarray]:
        for name, value in arrays.items():
            value = np.asarray(value)
            if value.dtype.kind in "fc" and not np.isfinite(value).all():
                raise ValueError(f"Non-finite values in {name!r}; not saving {self.folder}.")
        self.folder.mkdir(parents=True, exist_ok=True)
        temporary = self.folder / f"arrays.{unique_suffix()}.part.npz"
        np.savez_compressed(temporary, **arrays)
        temporary.replace(self.folder / "arrays.npz")
        write_json(self.marker, {"state": "complete", "request": self.request, "seconds": seconds,
                                 "metadata": metadata or {}})
        return {name: np.asarray(value) for name, value in arrays.items()}

    @contextmanager
    def lock(self, timeout: float = -1):
        self.folder.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.folder / ".lock"), timeout=timeout):
            yield


class Store:
    """Root of a results directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def unit(self, *parts: str, request: dict) -> Unit:
        for part in parts:
            if not part or any(ch in part for ch in '\\/:*?"<>|'):
                raise ValueError(f"Invalid path component {part!r}.")
        return Unit(self.root.joinpath(*parts), request)

    def compute(self, unit: Unit, fit: Callable[[], tuple[dict, dict]], *, label: str,
                log: Callable[[str], None] = print, require: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
        """Return the unit's arrays, fitting it under a lock only when it is not complete.

        ``require`` names arrays that must be present; a complete unit saved without them
        (for example without its embedding) is fitted again.
        """

        def ready():
            if not unit.done():
                return None
            arrays = unit.load()
            return arrays if all(name in arrays for name in require) else None

        arrays = ready()
        if arrays is not None:
            return arrays
        with unit.lock():
            arrays = ready()         # another worker may have finished it while we waited
            if arrays is not None:
                return arrays
            started = time.perf_counter()
            arrays, metadata = fit()
            saved = unit.save(time.perf_counter() - started, metadata, **arrays)
            log(f"done  {label}  ({time.perf_counter() - started:.0f} s)")
            return saved
