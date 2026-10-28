"""Published datasets opened as series: where each one's results go, per user.

A dataset folder is read-only, so DarSIA asks for a results location outside
it when the dataset is added on Start. The pair is kept in a per-user file next
to the series registry (``published_datasets.toml``), never in the dataset::

    schema_version = 1

    [[dataset]]
    root = "<the dataset folder>"
    results = "<the results location>"
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import tomllib

from .layout import MANIFEST

DATASETS_FILE = "published_datasets.toml"


def datasets_path() -> Path:
    """This user's file of dataset results locations (it may not exist yet)."""
    from ..series_paths import series_registry_path

    return series_registry_path().parent / DATASETS_FILE


def _key(path) -> str:
    """A comparable form of a folder path."""
    return os.path.normcase(str(Path(path).resolve()))


def registered_datasets() -> list[dict]:
    """The registered datasets: {root, results} rows."""
    path = datasets_path()
    if not path.is_file():
        return []
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    if document.get("schema_version") != 1:
        raise ValueError(f"Unsupported published-dataset settings: {path}")
    return [{"root": str(row["root"]), "results": str(row["results"])} for row in document.get("dataset", [])]


def dataset_results(root) -> Path | None:
    """The results location registered for a dataset folder, or None."""
    wanted = _key(root)
    for row in registered_datasets():
        if _key(row["root"]) == wanted:
            return Path(row["results"])
    return None


def results_location_problems(root, results) -> list[str]:
    """Why a results location cannot serve a dataset: missing, or inside the dataset."""
    if not str(results or "").strip():
        return ["Choose a results location outside the dataset"]
    dataset, chosen = Path(root).resolve(), Path(results).resolve()
    if chosen.is_relative_to(dataset) or dataset.is_relative_to(chosen):
        return ["The results location must lie outside the dataset folder"]
    return []


def fresh_results_folder(results) -> Path:
    """The first ``analysis-<n>`` folder in a results location that is missing or empty.

    Each new analysis of a dataset gets its own folder, so an earlier report is
    never overwritten.
    """
    results = Path(results)
    number = 1
    while True:
        folder = results / f"analysis-{number}"
        if not folder.exists() or (folder.is_dir() and not any(folder.iterdir())):
            return folder
        number += 1


def register_dataset(root, results) -> Path:
    """Record (or update) a dataset's results location in the per-user file."""
    if not (Path(root) / MANIFEST).is_file():
        raise ValueError(f"No {MANIFEST} in {root}")
    problems = results_location_problems(root, results)
    if problems:
        raise ValueError(problems[0])
    rows = [row for row in registered_datasets() if _key(row["root"]) != _key(root)]
    rows.append({"root": str(Path(root).resolve()), "results": str(Path(results).resolve())})
    text = "schema_version = 1\n" + "".join(
        f"\n[[dataset]]\nroot = {json.dumps(row['root'])}\nresults = {json.dumps(row['results'])}\n" for row in rows)
    path = datasets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".published_datasets-",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def protect_datasets(roots) -> list[Path]:
    """Protect each of ``roots`` that is a published dataset folder (see read_only.protect)."""
    from .read_only import protect

    protected = []
    for root in map(Path, roots):
        if (root / MANIFEST).is_file():
            protect(root)
            protected.append(root)
    return protected
