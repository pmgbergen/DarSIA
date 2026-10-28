"""Resolve installed code separately from series-owned configuration and results.

The ``series.toml`` file lives inside each series' DarSIA directory. Relative
manifest paths are resolved there, independently of the process working directory.
Native workflow TOMLs themselves retain DarSIA's own configuration schema.
An optional [paths] ``phase_model`` folder collects everything used to train a
phase model; series without it keep every historical location.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path, PureWindowsPath
import re
import json
import stat
import tempfile
import tomllib


_PATHS = {
    "config": "config", "common_config": "config/common.toml",
    "run_configs": "config/runs", "merged_configs": "config/merged",
    "protocols": "protocols", "data": "data", "models": "models",
    "results": "results", "reviews": "reviews", "training": "training",
    "project": "..", "jobs": "jobs", "image_selection": "image_selection",
    "calibration": "Calibration",
    "review_collections": "phase_review_collections",
    "geometry_facies": "geometry_facies",
    # Fixed phase-model folders of series without [paths] phase_model.
    "evidence_actions": "reviews/evidence_actions",
    "evidence_audits": "reviews/evidence_audits",
    "phase_model_library": "models/phase",
}
# Without [paths] phase_model, these keys name another key's folder.
_SAME_FOLDER = {"phase_annotations": "reviews", "phase_models": "models"}
# With [paths] phase_model, each folder used to train a phase model defaults to
# one of its two parts: Annotations (what was reviewed and on which images; it
# is also the review-session folder) and Model (the trained versions and their
# library index). Explicit keys win.
_PHASE_MODEL_FOLDERS = {
    "phase_annotations": ("phase_model", "Annotations"),
    "phase_models": ("phase_model", "Model"),
    "reviews": ("phase_annotations", ""),
    "review_collections": ("phase_annotations", "review_collections"),
    "evidence_actions": ("phase_annotations", "evidence_actions"),
    "evidence_audits": ("phase_annotations", "evidence_audits"),
    "training_enrollments": ("phase_annotations", "training_enrollments"),
    "phase_model_library": ("phase_models", "library"),
}
PHASE_MODEL_FOLDER = "ExpertKnowledge/PhaseModels"
PHASE_TRAINING_INPUT_FOLDER = "TrainingInput"
PHASE_MODELS_FOLDER = "Models"
_CANONICAL_PHASE_MODEL_FOLDERS = dict(_PHASE_MODEL_FOLDERS,
    phase_annotations=("phase_model", PHASE_TRAINING_INPUT_FOLDER),
    phase_models=("phase_model", PHASE_MODELS_FOLDER))
_CALIBRATION_FOLDERS = {
    "calibration_plans": "plans", "calibration_reviews": "reviews",
    "indicator_response_models": "indicator_response/models",
    "calibration_refit": "refit", "calibration_jobs": "concentration_mass/jobs",
    "mass_calibration_library": "mass_calibration",
    "colour_calibration_reports": "mass_calibration_optuna",
}
_LEGACY_CALIBRATION_FOLDERS = {
    "calibration": "calibration", "calibration_plans": "results/calibration/plans",
    "calibration_reviews": "results/calibration/reviews",
    "indicator_response_models": "results/indicator_response/models",
    "calibration_refit": "calibration/refit", "calibration_jobs": "calibration/concentration_mass/jobs",
    "mass_calibration_library": "models/mass_calibration",
}


def _configured_path(paths, key: str, *, phase_model: bool = True):
    """The [paths] value of ``key``: explicit, inside phase_model, or historical."""
    if key in paths:
        return paths[key]
    if key in _CALIBRATION_FOLDERS:
        return str(Path(_configured_path(paths, "calibration")) / _CALIBRATION_FOLDERS[key])
    if phase_model and paths.get("phase_model") and key in _PHASE_MODEL_FOLDERS:
        # Old manifests specified only their PhaseModel/custom root. Keep its
        # implied Annotations/Model paths while new PhaseModels roots use the
        # two canonical folders. Explicit child paths always remain authoritative.
        folders = (_CANONICAL_PHASE_MODEL_FOLDERS
            if Path(paths["phase_model"]).name.casefold() == Path(PHASE_MODEL_FOLDER).name.casefold()
            else _PHASE_MODEL_FOLDERS)
        parent, name = folders[key]
        folder = _configured_path(paths, parent)
        return str(Path(str(folder)) / name) if name else folder
    if key in _SAME_FOLDER:
        return _configured_path(paths, _SAME_FOLDER[key], phase_model=phase_model)
    if key in _PATHS:
        return _PATHS[key]
    raise KeyError(f"Unknown series path: {key}")


@lru_cache(maxsize=64)
def _cached_manifest_paths(path: str, _modified: int, _size: int) -> dict:
    """[paths] of one series.toml revision (keyed by its modification stamp)."""
    with open(path, "rb") as stream:
        return dict(tomllib.load(stream).get("paths") or {})


def _manifest_paths(root: Path) -> dict:
    """[paths] of the series.toml in ``root``; empty without that file."""
    manifest = Path(root) / "series.toml"
    try:
        info = manifest.stat()
    except OSError:
        return {}
    if not stat.S_ISREG(info.st_mode):
        return {}
    return dict(_cached_manifest_paths(str(manifest), info.st_mtime_ns, info.st_size))


def series_folder_path(root: str | Path, key: str, *, before_phase_model: bool = False) -> Path:
    """Resolve one [paths] key for a series folder, registered or not.

    ``root`` holds series.toml; without that file every key keeps its
    historical default. The result is lexical (no filesystem resolution), as
    callers holding NAS paths expect. ``before_phase_model`` ignores [paths]
    phase_model and returns the folder the series used before that layout.
    """
    from .storage_paths import resolve_storage_path
    root = Path(root).expanduser()
    raw = _configured_path(_manifest_paths(root), key, phase_model=not before_phase_model)
    path = Path(raw).expanduser()
    joined = path if path.is_absolute() or PureWindowsPath(str(path)).is_absolute() else root / path
    return resolve_storage_path(joined, series_roots=(root.absolute(),))


def phase_model_root(root: str | Path) -> Path | None:
    """The [paths] phase_model folder of a series folder, or None when unset."""
    if not _manifest_paths(Path(root).expanduser()).get("phase_model"):
        return None
    return series_folder_path(root, "phase_model")


def calibration_read_folders(root: str | Path, key: str) -> tuple[Path, ...]:
    """Current calibration storage followed by its retained historical location."""
    folders = [series_folder_path(root, key)]
    legacy = _LEGACY_CALIBRATION_FOLDERS.get(key)
    if legacy is not None:
        from .storage_paths import resolve_storage_path
        earlier = resolve_storage_path(Path(root).expanduser() / legacy, series_roots=(Path(root).absolute(),))
        if earlier not in folders:
            folders.append(earlier)
    return tuple(folders)


def calibration_artifact_path(root: str | Path, key: str, *parts: str) -> Path:
    """Read a named artifact at its current or original location, without moving it."""
    paths = [folder.joinpath(*parts) for folder in calibration_read_folders(root, key)]
    return next((path for path in paths if path.is_file()), paths[0])


def model_versions_folder(root: str | Path) -> Path:
    """Where a series folder keeps trained phase-model versions.

    [paths] phase_models when the series sets [paths] phase_model; otherwise
    the fixed ``models`` folder that callers holding only a folder always used.
    """
    if phase_model_root(root) is not None:
        return series_folder_path(root, "phase_models")
    return Path(root).expanduser() / "models"


def series_key(dataset: str) -> str:
    key = str(dataset).strip().casefold().replace("_", "").replace("-", "")
    if not re.fullmatch(r"[a-z][a-z0-9]*", key):
        raise ValueError(f"Invalid series ID: {dataset!r}")
    return key


def series_registry_path() -> Path:
    """User-owned project index; no experiment path is stored in the code tree."""
    explicit = os.environ.get("DARSIA_SERIES_REGISTRY")
    if explicit:
        return Path(os.path.abspath(os.path.expanduser(explicit)))
    config_root = os.environ.get("APPDATA") if os.name == "nt" else os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_root) if config_root else Path.home() / ("AppData/Roaming" if os.name == "nt" else ".config")
    return Path(os.path.abspath(root / "DarSIA" / "series_registry.toml"))


def _registry_series() -> list[tuple[str, str, Path]]:
    from .storage_paths import resolve_storage_path
    path = series_registry_path()
    if not path.is_file():
        return []
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported DarSIA series registry: {path}")
    rows: list[tuple[str, str, Path]] = []
    for entry in data.get("series", []):
        key = series_key(entry["id"])
        raw_root = resolve_storage_path(entry["root"])
        root = (raw_root if raw_root.is_absolute() else path.parent / raw_root).resolve()
        try:
            root = _project_directory(root)
        except ValueError:
            # Keep an unavailable registered path visible, for example while a
            # removable drive or network share is temporarily disconnected.
            pass
        rows.append((key, str(entry.get("name") or entry["id"]), root))
    return rows


def _manifest_series(root: Path) -> tuple[str, str, Path] | None:
    manifest = root / "series.toml"
    if not manifest.is_file():
        return None
    with manifest.open("rb") as stream:
        data = tomllib.load(stream)
    raw_id = str(data["series_id"])
    return series_key(raw_id), str(data.get("name") or raw_id), root.resolve()


def _project_directory(selection: Path) -> Path:
    """Accept either a series root or its DarSIA project directory."""
    from .storage_paths import resolve_storage_path
    selection = resolve_storage_path(selection)
    root = selection.expanduser().resolve()
    if (root / "series.toml").is_file():
        return root
    nested = root / "DarSIA"
    if (nested / "series.toml").is_file():
        return nested.resolve()
    raise ValueError(
        f"No series.toml found in {root} or its DarSIA subfolder"
    )


def configured_series() -> tuple[tuple[str, str, Path], ...]:
    """List registered projects, whether their folders are local or remote.

    Explicit project-root environment variables and an optional parent-folder
    discovery remain supported for older installations. Neither requires a
    particular experiment name or storage host.
    """
    from .storage_paths import resolve_storage_path
    rows: dict[str, tuple[str, str, Path]] = {
        key: (key, name, root) for key, name, root in _registry_series()
    }
    project_env = re.compile(r"^DARSIA_([A-Z][A-Z0-9_]*)_PROJECT_ROOT$")
    for variable, value in sorted(os.environ.items()):
        match = project_env.fullmatch(variable)
        if match is None or not value:
            continue
        key = series_key(match.group(1))
        selection = resolve_storage_path(value).resolve()
        try:
            root = _project_directory(selection)
        except ValueError:
            root = selection
        entry = _manifest_series(root)
        rows[key] = (key, entry[1] if entry else match.group(1), root)
    parent = os.environ.get("DARSIA_FLUIDFLOWER_DATA_ROOT")
    if parent:
        parent_path = resolve_storage_path(parent)
        if parent_path.is_dir():
            for child in sorted(parent_path.iterdir(), key=lambda path: path.name.casefold()):
                if not child.is_dir():
                    continue
                for candidate in (child / "DarSIA", child):
                    entry = _manifest_series(candidate)
                    if entry is not None:
                        rows.setdefault(entry[0], entry)
                        break
    return tuple(rows.values())


def studio_series_keys() -> tuple[str, ...]:
    """Keep the original Studio choices and append explicitly configured series."""
    return tuple(dict.fromkeys(("ffum", "ffac", "ffbm",
        *(series_key(key) for key, _name, _root in configured_series()))))


def register_series(root: str | Path) -> tuple[str, str, Path]:
    """Register a series root or its DarSIA subfolder in the user index."""
    root = _project_directory(Path(root))
    entry = _manifest_series(root)
    assert entry is not None
    key, name, root = entry
    path = series_registry_path()
    existing = _registry_series()
    for existing_key, _existing_name, existing_root in existing:
        if existing_key == key:
            if existing_root == root:
                return entry
            raise ValueError(f"Series {key!r} is already registered at {existing_root}")
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = path.read_text(encoding="utf-8") if path.exists() else "schema_version = 1\n"
    addition = ("\n[[series]]\n"
                f"id = {json.dumps(key)}\n"
                f"name = {json.dumps(name)}\n"
                f"root = {json.dumps(str(root))}\n")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".series_registry-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(previous.rstrip() + "\n" + addition)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return entry


def series_root(dataset: str) -> Path:
    from .storage_paths import resolve_storage_path
    key = series_key(dataset)
    explicit = os.environ.get(f"DARSIA_{key.upper()}_PROJECT_ROOT")
    if explicit:
        selection = resolve_storage_path(explicit).resolve()
        try:
            return _project_directory(selection)
        except ValueError:
            # Existing callers also use an explicit path while creating the
            # manifest; retain that behavior until the file is written.
            return selection
    for configured_key, _name, root in configured_series():
        if configured_key == key:
            return root
    raise ValueError(f"Series {dataset!r} is not registered. Add its series.toml folder to {series_registry_path()}")


def series_manifest(dataset: str) -> dict:
    root = series_root(dataset)
    with (root / "series.toml").open("rb") as stream:
        value = tomllib.load(stream)
    if value.get("schema_version") != 1 or series_key(value["series_id"]) != series_key(dataset):
        raise ValueError(f"Wrong series manifest: {root / 'series.toml'}")
    if "cycle_runs" in value:
        from .series_cycles import parse_cycle_runs
        parse_cycle_runs(value)
    return value


def series_path(dataset: str, key: str) -> Path:
    from .storage_paths import resolve_storage_path
    root = series_root(dataset)
    # Existing projects remain discoverable during the create-only migration.
    manifest = series_manifest(dataset) if (root / "series.toml").is_file() else {}
    raw = _configured_path(manifest.get("paths", {}), key)
    path = Path(raw).expanduser()
    joined = path if path.is_absolute() or PureWindowsPath(str(path)).is_absolute() else root / path
    return resolve_storage_path(joined, series_roots=(root,)).resolve()


def config_files(dataset: str, run: str) -> list[Path]:
    from .storage_paths import resolve_storage_path
    run = str(run).strip().casefold()
    manifest = series_manifest(dataset)
    if run not in manifest["runs"]["ids"]:
        raise ValueError(f"Unknown {dataset} experiment: {run}")
    root = series_root(dataset)
    run_path = Path(manifest["runs"]["config_files"][run]).expanduser()
    if not (run_path.is_absolute() or PureWindowsPath(str(run_path)).is_absolute()):
        run_path = root / run_path
    files = [series_path(dataset, "common_config"),
             resolve_storage_path(run_path, series_roots=(root,)).resolve()]
    state = files[1].parent / ".color_state" / f"{run}.txt"
    if state.is_file() and state.read_text(encoding="utf-8-sig").strip().casefold() == "on":
        files.append(series_path(dataset, "config") / "coloron.toml")
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    return files


def local_cache_root() -> Path:
    from .storage_paths import resolve_storage_path
    configured = os.environ.get("DARSIA_FLUIDFLOWER_CACHE")
    registry_parent: Path | None = None
    if not configured:
        registry = series_registry_path()
        if registry.is_file():
            with registry.open("rb") as stream:
                configured = tomllib.load(stream).get("storage", {}).get("cache_root")
            registry_parent = registry.parent
    if configured:
        root = resolve_storage_path(configured)
        if registry_parent is not None and not root.is_absolute():
            root = registry_parent / root
    else:
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "DarSIA" / "cache"
    # Configuration/path resolution does not allocate. A named admitted writer
    # owns allocation; importing a module's default paths must never mkdir.
    return root.absolute()


def package_root() -> Path:
    return Path(__file__).resolve().parent


def darsia_source_checkout() -> Path | None:
    """Return the Git working tree that holds the imported DarSIA code, if any.

    A series project folder may be a Git repository of its own; only this
    checkout must never receive series-owned or per-user files. An installed
    copy (``site-packages``) has no source checkout, even inside a repository.
    """
    package = package_root().parent
    for folder in package.parents:
        if (folder / ".git").exists():
            between = {part.casefold() for part in package.relative_to(folder).parts}
            return None if between & {"site-packages", "dist-packages"} else folder
    return None


def tool_path(name: str) -> Path:
    stem = name[:-3] if name.endswith(".py") else name
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", stem):
        raise ValueError(f"Invalid packaged tool name: {name!r}")
    path = package_root() / "tools" / f"{stem}.py"
    if not path.is_file():
        raise FileNotFoundError(f"Packaged FluidFlower tool is unavailable: {stem}")
    return path


def tool_arguments(path: str | Path) -> list[str]:
    path = Path(path).resolve()
    if path.parent == package_root() / "tools":
        return ["-m", f"darsia.fluidflower.tools.{path.stem}"]
    # Explicit external executables are supported for user-supplied integrations.
    return [str(path)]
