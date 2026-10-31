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
    "mantpy-api-v1.1.0/notebooks/experiments/"
)
PACKAGE_RELEASE = "1.1.0"
MANIFEST_SHA256 = {'intestine': '3a590ff0b87ed8f663b80d655b9aa500d7d30d417b4656e65924f179bdf4e691',
 'liver': 'f140bb9a5627e8b6776ff511079847f003f3d8d7e5f9d9d775d65dcd58aececa',
 'lung': '011eaff54ae7ff55238c0aa83acf0dd7ae10eb0c6a0fbb2a0f979f6c0f74d26a'}
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
        if hashlib.sha256(manifest_data).hexdigest() != MANIFEST_SHA256[study]:
            raise RuntimeError(f"{manifest_name} differs from this notebook's release manifest.")
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
    base_url = base_url or os.environ.get("MANTPY_NOTEBOOK_ASSET_BASE") or BASE_URL
    base_url = base_url.rstrip("/") + "/"
    manifest_path = root / f"{study}_assets.json"
    try:
        _extract_uploaded_bundle(study, root)
        if not manifest_path.exists():
            _download(base_url + manifest_path.name, manifest_path, MANIFEST_SHA256[study])
        if _sha256(manifest_path) != MANIFEST_SHA256[study]:
            raise RuntimeError(f"{manifest_path.name} differs from this notebook's release manifest.")
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
            "bundle beside this notebook."
        ) from error
    return manifest

def _package_requirement(package_source=None, package_wheel=None, package_release=None):
    """Prefer an explicit source/wheel, otherwise use the matching public release."""
    source = package_source or os.environ.get("MANTPY_PACKAGE_SOURCE")
    wheel = package_wheel or os.environ.get("MANTPY_PACKAGE_WHEEL")
    release = package_release or os.environ.get("MANTPY_PACKAGE_RELEASE")
    if not source and not wheel and not release:
        wheels = list(Path.cwd().glob("mantpy*.whl"))
        if len(wheels) == 1:
            wheel = str(wheels[0])
        if not wheel:
            local = next((parent / "mantpy_api_refactor"
                          for parent in Path(__file__).resolve().parents
                          if (parent / "mantpy_api_refactor" / "src" / "mantpy").is_dir()), None)
            if local is not None:
                source = str(local)
    if source:
        directory = Path(source).expanduser().resolve()
        module_root = directory / "src" if (directory / "src" / "mantpy").is_dir() else directory
        if not (module_root / "mantpy").is_dir():
            raise RuntimeError("MANTPY_PACKAGE_SOURCE must contain src/mantpy or mantpy.")
        sys.path.insert(0, str(module_root))
        return str(directory) + "[patch,gnn]" if (directory / "pyproject.toml").exists() else None
    if wheel:
        path = Path(wheel).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError("MANTPY_PACKAGE_WHEEL does not exist.")
        return str(path) + "[patch,gnn]"
    return "mantpy[patch,gnn]==" + str(release or PACKAGE_RELEASE)


def _package_fingerprint(package):
    """Include development wheel/source bytes so rebuilt code is reinstalled."""
    digest = hashlib.sha256(str(package).encode())
    if package:
        path = Path(str(package).removesuffix("[patch,gnn]"))
        if path.is_file():
            digest.update(path.read_bytes())
        elif path.is_dir():
            paths = sorted((path / "src" / "mantpy").rglob("*.py"))
            paths += [path / "pyproject.toml"]
            for source in paths:
                if source.is_file():
                    digest.update(source.relative_to(path).as_posix().encode())
                    digest.update(source.read_bytes())
    return digest.digest()


def _require_analysis_apis():
    import mantpy as mt
    required = {"tl": ("neighbourhood_features", "contact_marker_contrast", "component_summary", "reconstruct_labels"),
                "pp": ("ImageNormalizer",), "gr": ("joint_graph_from_connectivities",),
                "pl": ("patch_map",), "nn": ("train_node_classifier", "node_integrated_gradients")}
    missing = [f"mt.{namespace}.{name}" for namespace, names in required.items()
               for name in names if not hasattr(getattr(mt, namespace), name)]
    if missing or not hasattr(mt.nn.PatchEncoder, "model"):
        raise RuntimeError(
            "These notebooks need Mantpy 1.1.0 or the matching source/wheel. "
            "Set MANTPY_PACKAGE_SOURCE or upload its wheel/set MANTPY_PACKAGE_WHEEL, "
            "then restart the kernel. Missing: " + ", ".join(missing))


def prepare_experiment(study, root=None, base_url=None, *, package_source=None, package_wheel=None, package_release=None):
    """Verify release companions and select a matching release or local package."""
    if sys.version_info[:2] not in ((3, 11), (3, 12), (3, 13)):
        raise RuntimeError("Use Python 3.11, 3.12 or 3.13.")
    root = (Path.cwd() if root is None else Path(root)).resolve()
    manifest = verify_assets(study, root, base_url)
    package = _package_requirement(package_source, package_wheel, package_release)
    on_colab = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
    if not on_colab:
        _require_analysis_apis()
        print(f"{study.capitalize()} files and matching Mantpy APIs verified.")
        return manifest
    requirements = root / f"{study}_requirements.txt"
    fingerprint = hashlib.sha256(requirements.read_bytes() + _package_fingerprint(package)).hexdigest()
    pins = re.findall(r"^([A-Za-z0-9_-]+)(?:\[[^\]]+\])?==([^\s#]+)", requirements.read_text(), flags=re.MULTILINE)
    marker = Path(tempfile.gettempdir()) / f"mantpy-experiment-{study}.json"
    try:
        state = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        state = {}
    try:
        installed = all(importlib.metadata.version(name) == version for name, version in pins)
    except importlib.metadata.PackageNotFoundError:
        installed = False
    if state.get("requirements") != fingerprint or not installed:
        if package is None:
            raise RuntimeError("Use an installable Mantpy source checkout or its wheel in Colab.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements), package])
        marker.write_text(json.dumps({"requirements": fingerprint, "pid": os.getpid()}))
        raise RuntimeError("Installation complete. " + RESTART)
    if state.get("pid") == os.getpid():
        raise RuntimeError(RESTART)
    _require_analysis_apis()
    print(f"{study.capitalize()} Colab environment ready.")
    return manifest
