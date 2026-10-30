"""Verify experiment companions and prepare their environment in Colab."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from urllib.request import urlopen
import zipfile


BASE_URL = (
    "https://raw.githubusercontent.com/moeghaf/mantpy_reproducibility/"
    "main/notebooks/experiments/"
)
RESTART = "Select Runtime > Restart session, then run all cells again."


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_path(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Invalid experiment companion path: {name!r}")
    target = root.joinpath(*path.parts).resolve()
    if not target.is_relative_to(root.resolve()) or target == root.resolve():
        raise ValueError(f"Companion must be a file within {root}: {name!r}")
    return target


def _download(url, target, expected_sha256=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=90) as response:
        data = response.read()
    if expected_sha256 and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise RuntimeError(f"Checksum mismatch downloading {target.name}")
    temporary = target.with_name(target.name + ".download")
    temporary.write_bytes(data)
    temporary.replace(target)


def _extract_uploaded_bundle(study, root):
    """Read only the selected study's verified files from a companion bundle."""
    bundle = root / "mantpy_experiment_companions.zip"
    if not bundle.exists():
        return
    manifest_name = f"{study}_assets.json"
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate paths in experiment companion bundle")
        for name in names:
            _relative_path(root, name)
        if manifest_name not in names:
            raise ValueError(f"The uploaded bundle has no {study} manifest")
        manifest_data = archive.read(manifest_name)
        manifest = json.loads(manifest_data)
        if manifest.get("schema_version") != 1 or manifest.get("study") != study:
            raise ValueError(f"Invalid {study} asset manifest")
        files = manifest["files"]
        required = {f"{study}_helpers.py", f"{study}_requirements.txt",
                    f"{study}_selected.json"}
        records = [item["path"] for item in files]
        if len(records) != len(set(records)) or not required.issubset(records):
            raise ValueError("Missing or duplicate experiment companion records")
        pending = [(root / manifest_name, manifest_data)]
        for item in files:
            target = _relative_path(root, item["path"])
            expected = item["sha256"]
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError(f"Invalid checksum for {item['path']}")
            data = archive.read(item["path"])
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f"Checksum mismatch in uploaded bundle: {item['path']}")
            pending.append((target, data))
        # Validate the whole study before extracting; never overwrite local edits.
        for target, data in pending:
            if target.exists() and target.read_bytes() != data:
                raise RuntimeError(f"{target.name} differs from the uploaded experiment bundle.")
        for target, data in pending:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)


def verify_assets(study, root=None, base_url=None):
    """Fetch missing companions and verify their committed SHA-256 identities."""
    if study not in ("intestine", "liver", "lung"):
        raise ValueError("Choose intestine, liver or lung.")
    root = Path.cwd() if root is None else Path(root)
    root = root.resolve()
    base_url = BASE_URL if base_url is None else base_url.rstrip("/") + "/"
    manifest_path = root / f"{study}_assets.json"
    try:
        _extract_uploaded_bundle(study, root)
        if not manifest_path.exists():
            _download(base_url + manifest_path.name, manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("study") != study:
            raise ValueError(f"Invalid {study} asset manifest")
        required = {f"{study}_helpers.py", f"{study}_requirements.txt",
                    f"{study}_selected.json"}
        files = manifest["files"]
        if not isinstance(files, list) or not files:
            raise ValueError("The experiment manifest contains no files")
        names = [item["path"] for item in files]
        if len(names) != len(set(names)) or not required.issubset(names):
            raise ValueError("Missing or duplicate experiment companion records")
        for item in files:
            target = _relative_path(root, item["path"])
            expected = item["sha256"]
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError(f"Invalid checksum for {item['path']}")
            if not target.exists():
                _download(base_url + item["path"], target, expected)
            if _sha256(target) != expected:
                raise RuntimeError(
                    f"{item['path']} differs from the selected experiment files. "
                    "Restore the companion bundle before running."
                )
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Experiment companions are unavailable. Extract the experiment "
            "bundle beside this notebook, or use the published Colab link."
        ) from error
    return manifest


def prepare_experiment(study, root=None, base_url=None):
    """Verify inputs; install pinned Colab packages before analysis imports."""
    if sys.version_info[:2] not in ((3, 11), (3, 12), (3, 13)):
        raise RuntimeError("Use Python 3.11, 3.12 or 3.13 for these experiments.")
    root = Path.cwd() if root is None else Path(root)
    root = root.resolve()
    manifest = verify_assets(study, root, base_url)
    requirements = root / f"{study}_requirements.txt"
    on_colab = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
    if not on_colab:
        print(f"{study.capitalize()} experiment files verified.")
        return manifest

    fingerprint = _sha256(requirements)
    pins = re.findall(
        r"^([A-Za-z0-9_-]+)(?:\[[^\]]+\])?==([^\s#]+)",
        requirements.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    marker = Path(tempfile.gettempdir()) / f"mantpy-experiment-{study}.json"
    try:
        state = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = {}
    try:
        installed = all(
            importlib.metadata.version(name) == version for name, version in pins
        )
    except importlib.metadata.PackageNotFoundError:
        installed = False
    if state.get("requirements") != fingerprint or not installed:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)]
        )
        marker.write_text(
            json.dumps({"requirements": fingerprint, "pid": os.getpid()}),
            encoding="utf-8",
        )
        raise RuntimeError("Installation complete. " + RESTART)
    if state.get("pid") == os.getpid():
        raise RuntimeError(RESTART)
    for name, version in pins:
        module_name = "sklearn" if name == "scikit-learn" else name.replace("-", "_")
        loaded = sys.modules.get(module_name)
        if loaded is not None and getattr(loaded, "__version__", version) != version:
            raise RuntimeError(RESTART)
    print(f"{study.capitalize()} Colab environment ready.")
    return manifest
