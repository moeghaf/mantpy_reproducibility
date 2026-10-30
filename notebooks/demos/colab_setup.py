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
from urllib.request import urlretrieve


BASE_URL = (
    "https://raw.githubusercontent.com/moeghaf/mantpy_reproducibility/"
    "main/notebooks/demos/"
)
RESTART = "Select Runtime > Restart session, then run all cells again."


def prepare_colab(tutorial):
    """Download missing companions, install the pinned environment and restart."""
    if tutorial not in ("fig2", "liver", "lung"):
        raise ValueError("Choose fig2, liver or lung.")
    if sys.version_info[:2] not in ((3, 11), (3, 12), (3, 13)):
        raise RuntimeError(
            "These requirements support Python 3.11, 3.12 and 3.13. "
            "Choose a Colab runtime with one of these Python versions."
        )
    for suffix in ("helpers.py", "requirements.txt"):
        name = f"{tutorial}_demo_{suffix}"
        if not Path(name).exists():
            try:
                urlretrieve(BASE_URL + name, name)
            except Exception as error:
                raise RuntimeError(
                    "The tutorial companions are not available from GitHub. "
                    "Upload the helper and requirements files beside the notebook."
                ) from error
    requirements = Path(f"{tutorial}_demo_requirements.txt")
    fingerprint = hashlib.sha256(requirements.read_bytes()).hexdigest()
    pins = re.findall(
        r"^([A-Za-z0-9_-]+)(?:\[[^\]]+\])?==([^\s#]+)",
        requirements.read_text(),
        flags=re.MULTILINE,
    )
    marker = Path(tempfile.gettempdir()) / f"mantpy-colab-{tutorial}.json"
    state = json.loads(marker.read_text()) if marker.exists() else {}
    try:
        installed = all(importlib.metadata.version(name) == version for name, version in pins)
    except importlib.metadata.PackageNotFoundError:
        installed = False
    if state.get("requirements") != fingerprint or not installed:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)]
        )
        marker.write_text(json.dumps({"requirements": fingerprint, "pid": os.getpid()}))
        raise RuntimeError("Installation complete. " + RESTART)
    if state.get("pid") == os.getpid():
        raise RuntimeError(RESTART)
    for name, version in pins:
        module_name = "sklearn" if name == "scikit-learn" else name.replace("-", "_")
        loaded = sys.modules.get(module_name)
        if loaded is not None and getattr(loaded, "__version__", version) != version:
            raise RuntimeError(RESTART)
    print("Colab setup ready.")
