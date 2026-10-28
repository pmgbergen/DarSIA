"""Ordinary, disposable per-machine caches, independent of job ownership.

This module is deliberately stdlib-only: workers select these directories
before importing DarSIA, Numba or Matplotlib. There is no service lease or
qualification receipt in folder mode. Queue ownership stays in the shared job.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PureWindowsPath
import re
import sys
import tempfile
import uuid

from .policy import CachePolicyError

FOLDER_ENV = "DARSIA_FOLDER_CACHE_CONTEXT"
_CURRENT = None


def _contains_directory(directory, path):
    """Recognize mapped-drive/UNC aliases before treating a cache as disposable."""
    directory, path = Path(directory).absolute(), Path(path).absolute()
    if path.is_relative_to(directory):
        return True
    if path.anchor.casefold() == directory.anchor.casefold():
        return False
    try:
        expected = directory.stat()
    except FileNotFoundError:
        return False
    for candidate in (path, *path.parents):
        try:
            observed = candidate.stat()
        except FileNotFoundError:
            continue
        if (observed.st_dev, observed.st_ino) == (expected.st_dev, expected.st_ino):
            return True
        if observed.st_dev != expected.st_dev:
            return False
    return False


def _series_key(value):
    key = str(value).strip().casefold().replace("_", "").replace("-", "")
    if not re.fullmatch(r"[a-z][a-z0-9]*", key):
        raise CachePolicyError("Invalid cache series ID")
    return key


def folder_roots(document, series_id):
    """A per-series cache alias explicitly selects ordinary folder mode."""
    blocks = document.get("stores", {})
    selected = [roots for key, roots in blocks.items()
                if _series_key(key) == _series_key(series_id)]
    if not selected:
        return None
    if len(selected) != 1 or not isinstance(selected[0], dict):
        raise CachePolicyError("Choose the series folders in Start > Series folders")
    roots = selected[0]
    if "cache" not in roots:
        return None
    value = roots["cache"]
    if (not isinstance(value, str) or not value.strip() or "\x00" in value
            or ".." in PureWindowsPath(value).parts
            or not (Path(value).is_absolute() or PureWindowsPath(value).is_absolute())
            or value.replace("/", "\\").startswith(("\\\\?\\", "\\\\.\\"))):
        raise CachePolicyError("Choose an absolute Cache folder in Start > Series folders")
    # A cache may be a child of a broad results directory, but it cannot
    # contain the durable result/input roots, nor equal one of them.
    windows = PureWindowsPath(value)
    cache = windows if windows.drive or "\\" in value else Path(value)
    for alias, locator in roots.items():
        if alias == "cache":
            continue
        other = PureWindowsPath(locator)
        durable = other if other.drive or "\\" in locator else Path(locator)
        if type(durable) is type(cache) and durable.is_relative_to(cache):
            raise CachePolicyError("Cache folder must not contain the " + alias + " folder")
    return dict(roots)


def folder_mode(stores, series_id):
    return folder_roots(stores.document, series_id) is not None


def require_durable_path(stores, series_id, path):
    roots = folder_roots(stores.document, series_id)
    if roots is not None and _contains_directory(roots["cache"], path):
        raise CachePolicyError("Job inputs, work and results must be outside the disposable Cache folder")


@dataclass(frozen=True)
class FolderContext:
    source: dict
    machine: str
    series_id: str
    job_root: str
    root: Path
    pid: int


def current_folder_context():
    return _CURRENT


def folder_environment(document, source, series_id, *, job_root=None, parent_environment):
    roots = folder_roots(document, series_id)
    if roots is None:
        raise CachePolicyError("Choose a Cache folder in Start > Series folders before starting a worker")
    for alias, locator in roots.items():
        if alias != "cache" and _contains_directory(roots["cache"], locator):
            raise CachePolicyError("Cache folder must not contain the " + alias + " folder")
    machine = document.get("machine", {}).get("id", "")
    if not isinstance(machine, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", machine) is None:
        raise CachePolicyError("Cache needs this computer's configured machine ID")
    root = Path(roots["cache"]).expanduser().resolve() / machine.casefold()
    if job_root is not None and _contains_directory(roots["cache"], job_root):
        raise CachePolicyError("The shared job must be outside the disposable Cache folder")
    result = dict(parent_environment)
    # Managed service state must never leak into an independently selected
    # folder worker, including when Start is opened from a managed GUI.
    for key in ("DARSIA_CACHE_BOOTSTRAP_RECEIPT", "DARSIA_CACHE_NATIVE_LAUNCH", "DARSIA_CACHE_POLICY"):
        result.pop(key, None)
    directories = {key: root / subdir for key, subdir in {
        "NUMBA_CACHE_DIR": "numba", "MPLCONFIGDIR": "matplotlib",
        "PYTHONPYCACHEPREFIX": "pycache", "CUPY_CACHE_DIR": "cupy",
        "XDG_CACHE_HOME": "xdg-cache", "JOBLIB_TEMP_FOLDER": "joblib",
        "DARSIA_PHASE_DISPLAY_CACHE": "display"}.items()}
    temp = root / "tmp" / (str(os.getpid()) + "-" + uuid.uuid4().hex)
    directories.update(TEMP=temp, TMP=temp, TMPDIR=temp)
    for key, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        result[key] = str(path)
    result["DARSIA_FLUIDFLOWER_CACHE"] = str(root)
    result["NUMBA_CACHE_LOCATOR_CLASSES"] = "UserProvidedCacheLocator"
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    result[FOLDER_ENV] = json.dumps(dict(source=dict(source), series_id=series_id,
        job_root=str(Path(job_root).resolve()) if job_root is not None else None), sort_keys=True)
    return result


def activate_folder(document, source, series_id, *, job_root=None):
    global _CURRENT
    environment = folder_environment(document, source, series_id,
        job_root=job_root, parent_environment=os.environ)
    for key in ("DARSIA_CACHE_BOOTSTRAP_RECEIPT", "DARSIA_CACHE_NATIVE_LAUNCH", "DARSIA_CACHE_POLICY"):
        os.environ.pop(key, None)
    os.environ.update(environment)
    sys.pycache_prefix = environment["PYTHONPYCACHEPREFIX"]
    tempfile.tempdir = None  # Re-evaluate TEMP even if a stdlib consumer used it earlier.
    _CURRENT = FolderContext(dict(source), document["machine"]["id"], series_id,
        str(Path(job_root).resolve()) if job_root is not None else None,
        Path(environment["DARSIA_FLUIDFLOWER_CACHE"]), os.getpid())
    return _CURRENT


def worker_selection(arguments):
    """Read only the named job's bounded metadata to select its series cache."""
    values = []
    for index, value in enumerate(arguments or ()):
        if value == "--job" and index + 1 < len(arguments):
            values.append(arguments[index + 1])
        elif value.startswith("--job="):
            values.append(value.split("=", 1)[1])
    if len(values) != 1:
        raise CachePolicyError("Worker needs one --job to select its Cache folder")
    root = Path(values[0]).expanduser().resolve()
    with (root / "job.json").open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise CachePolicyError("Worker job metadata exceeds one MiB")
    return json.loads(raw)["series_id"], root
