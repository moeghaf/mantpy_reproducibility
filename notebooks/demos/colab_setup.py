"""Install tutorial dependencies before importing analysis libraries in Colab."""

import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen


BASE_URL = (
    "https://raw.githubusercontent.com/moeghaf/mantpy_reproducibility/"
    "mantpy-api-v1.1.0/notebooks/demos/"
)
PACKAGE_RELEASE = "1.1.0"
EXPECTED_COMPANIONS = {'fig2': {'fig2_demo_helpers.py': 'd15a0fcdf047fa980da7d07f450a1c89954698cb1c03e6a7da4fca4f3b65ed3e',
          'fig2_demo_requirements.txt': '366511ff815e60b3a023f26d997f9a88a91fa759ceb8801e3bc29039e1bb324b'},
 'liver': {'liver_demo_helpers.py': 'c575d31ad99cc1d1de9130005cf23fc64084cc39af7bccaaaa4815aaa3f65c21',
           'liver_demo_requirements.txt': '8ef52970143d1e4a02ecaee0cad2b1a95d79c14706f2006f39f97a111282388d'},
 'lung': {'lung_demo_helpers.py': '9ad401be6c09b8c4fbb2a39e29ddbee8ed16477befae551ce240f23a39b3c287',
          'lung_demo_requirements.txt': '6a1b23f625ce5e3ddd4bd4ab2bfda3d039f56a0c9ce202fa427676cd2e3cae0f'}}
RESTART = "Select Runtime > Restart session, then run all cells again."

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


def verify_companions(tutorial, root=None, base_url=None):
    """Verify local companions or fetch checksum-pinned files from the release tag."""
    if tutorial not in ("fig2", "liver", "lung"):
        raise ValueError("Choose fig2, liver or lung.")
    root = (Path.cwd() if root is None else Path(root)).resolve()
    base_url = base_url or os.environ.get("MANTPY_NOTEBOOK_ASSET_BASE") or BASE_URL
    for name, expected in EXPECTED_COMPANIONS[tutorial].items():
        target = root / name
        if not target.exists():
            with urlopen(base_url.rstrip("/") + "/" + name, timeout=90) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f"Checksum mismatch downloading {name}")
            temporary = target.with_name(target.name + ".download")
            temporary.write_bytes(data)
            temporary.replace(target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"{name} differs from this notebook's release companions.")


def prepare_colab(tutorial, *, package_source=None, package_wheel=None, package_release=None, base_url=None):
    """Prepare the matching release, preserving explicit local source/wheel overrides."""
    if sys.version_info[:2] not in ((3, 11), (3, 12), (3, 13)):
        raise RuntimeError("Use Python 3.11, 3.12 or 3.13.")
    verify_companions(tutorial, base_url=base_url)
    package = _package_requirement(package_source, package_wheel, package_release)
    on_colab = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
    if not on_colab:
        _require_analysis_apis()
        print("Matching Mantpy APIs and companions verified; local dependencies are unchanged.")
        return
    requirements = Path(f"{tutorial}_demo_requirements.txt")
    fingerprint = hashlib.sha256(requirements.read_bytes() + _package_fingerprint(package)).hexdigest()
    pins = re.findall(r"^([A-Za-z0-9_-]+)(?:\[[^\]]+\])?==([^\s#]+)", requirements.read_text(), flags=re.MULTILINE)
    marker = Path(tempfile.gettempdir()) / f"mantpy-colab-{tutorial}.json"
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
    print("Colab setup ready.")
