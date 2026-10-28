"""Find a staged analysis plan's inputs where the series keeps them.

New analysis prepares a plan from files that the series' producers save in the
series' analysis-inputs folder ([paths] analysis_inputs; by default
analysis/inputs in the series' results folder):

- runs/<run>.json: one complete run document per registered run;
- geometry_index.json: the measured run geometry, with frame rows for every run;
- pressure_index.json: the running pressure index, with every run's series;
- phase_two_inputs.json: the phase-one completions that mass from finished
  phase maps uses, together with the mass calibration chosen on the page.

Every file is read through the original metadata readers. A file, or a file it
binds, outside the series folder and its results folder (a local disk or an
archive folder, for example) counts as missing. Finding inputs writes, adopts
and starts nothing; Prepare repeats all original checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import ntpath
import os
from pathlib import Path

INPUTS_KEY = "analysis_inputs"
PLANS_KEY = "analysis_plans"
RUN_DOCUMENTS = "runs"
GEOMETRY_INDEX = "geometry_index.json"
PRESSURE_INDEX = "pressure_index.json"
PHASE_TWO_INPUTS = "phase_two_inputs.json"
PRESSURE_FORMATS = ("darsia-pressure-dataset-index-v1", "darsia-running-gfi-pressure-index-v1")
PHASE_TWO_FORMAT = "darsia-analysis-phase2-inputs-v1"
SHOWN_RUNS = 6


@dataclass(frozen=True)
class Source:
    """One plan input: what it is, where the series keeps it, and its state.

    ``missing`` says why the input cannot be used and is empty when it was
    found; ``found`` then summarises it and ``bound`` holds what Prepare
    passes on (the run documents' descriptors, in run order, for run documents).
    """

    label: str
    expected: str
    missing: str = ""
    found: str = ""
    bound: object = None


def _folder(series_root, key, *default) -> Path:
    """The series' [paths] ``key``, else ``default`` inside its results folder."""
    from .series_paths import series_folder_path
    try:
        return series_folder_path(series_root, key)
    except KeyError:
        return series_folder_path(series_root, "results").joinpath(*default)


def analysis_inputs_folder(series_root) -> Path:
    """The folder where the series' producers save the analysis inputs."""
    return _folder(series_root, INPUTS_KEY, "analysis", "inputs")


def analysis_plans_folder(series_root) -> Path:
    """The folder where New analysis prepares and adopts plans."""
    return _folder(series_root, PLANS_KEY, "analysis", "plans")


def _canonical(path) -> str:
    """One spelling per location: a mapped drive as its share, then the storage aliases."""
    from .storage_paths import resolve_storage_path
    text = str(path)
    try:
        text = os.path.realpath(text)
    except (OSError, ValueError):
        pass
    return ntpath.normcase(ntpath.normpath(str(resolve_storage_path(text))))


def series_folders(series_root) -> tuple[str, ...]:
    """The series folder and its results folder: where plan inputs must live."""
    from .series_paths import series_folder_path
    return tuple(dict.fromkeys(_canonical(folder) for folder in (
        Path(series_root), series_folder_path(series_root, "results"))))


def outside_series(path, folders) -> bool:
    """True when ``path`` lies in none of ``folders`` (see series_folders)."""
    key = _canonical(path)
    return not any(key == folder or key.startswith(folder.rstrip("\\") + "\\") for folder in folders)


def _runs(series_root):
    """The registered series id and run order, read like plan preparation reads them."""
    from .pipeline.alias_job import _manifest
    from .pipeline.job_queue import descriptor
    _, registered, _ = _manifest(descriptor(Path(series_root) / "series.toml"))
    runs = (registered.get("runs") or {}).get("ids")
    if (not isinstance(runs, list) or not runs or len(set(runs)) != len(runs)
            or any(not isinstance(run, str) or not run for run in runs)):
        raise ValueError("series.toml [runs] ids must list the distinct registered runs in order")
    return str(registered.get("series_id") or ""), runs


def _read(path):
    """Bind and read one metadata file through the original readers; None when absent."""
    from .analysis_preparation import source_metadata_descriptor
    from .jobs_metadata import JobSnapshotReader
    if not Path(path).is_file():
        return None, None
    bound = source_metadata_descriptor(path)
    return bound, JobSnapshotReader().read(Path(bound["path"]), bound)


def _named(runs) -> str:
    shown = ", ".join(runs[:SHOWN_RUNS])
    more = len(runs) - SHOWN_RUNS
    return shown + (f" and {more} more" if more > 0 else "")


def _path(value):
    return value.get("path") if isinstance(value, dict) else None


def _problems(groups, total):
    """One sentence per kind of problem, naming its runs."""
    return " ".join(f"{text} for {len(runs)} of {total} runs: {_named(runs)}." for text, runs in groups if runs)


def _run_documents(folder, runs, folders) -> Source:
    label, expected = "Run documents", str(folder / RUN_DOCUMENTS / "<run>.json")
    absent, outside, unreadable, bound = [], [], [], []
    for run in runs:
        path = folder / RUN_DOCUMENTS / (run + ".json")
        try:
            descriptor, value = _read(path)
        except (OSError, ValueError) as error:
            unreadable.append(f"{run} ({error})")
            continue
        if value is None:
            absent.append(run)
            continue
        if not isinstance(value, dict) or value.get("run") != run:
            unreadable.append(f"{run} (another run's document)")
            continue
        # Preparation also reads the method file that the run document binds.
        method = _path(value["mass"].get("calibration")) if isinstance(value.get("mass"), dict) else None
        if outside_series(descriptor["path"], folders) or (method and outside_series(method, folders)):
            outside.append(run)
        else:
            bound.append(descriptor)
    missing = _problems((("Not saved yet", absent),
                         ("Outside the series folders (or binding a method file there)", outside),
                         ("Not readable", unreadable)), len(runs))
    if missing:
        return Source(label, expected, missing)
    return Source(label, expected, found=f"{len(runs)} runs in {folder / RUN_DOCUMENTS}", bound=bound)


def _index(label, path, folders, formats):
    """Read one index file, or a missing Source saying why it cannot be used."""
    try:
        bound, value = _read(path)
    except (OSError, ValueError) as error:
        return None, None, f"Not readable: {error}"
    if value is None:
        return None, None, "Not saved yet."
    if outside_series(bound["path"], folders):
        return None, None, f"Outside the series folders: {bound['path']}"
    if not isinstance(value, dict) or value.get("format") not in formats:
        return None, None, "Not a " + label.lower() + " file (" + ", ".join(formats) + ")."
    return bound, value, ""


def _geometry(folder, runs, folders) -> Source:
    from .map_placement import GEOMETRY_FORMAT
    label, path = "Measured frame geometry", folder / GEOMETRY_INDEX
    expected = f"{path} ({GEOMETRY_FORMAT}, with frame rows for every run)"
    bound, value, problem = _index("measured run geometry", path, folders, (GEOMETRY_FORMAT,))
    if problem:
        return Source(label, expected, problem)
    entries = value.get("runs") if isinstance(value.get("runs"), dict) else {}
    no_model, no_rows, outside = [], [], []
    for run in runs:
        entry = entries.get(run) if isinstance(entries.get(run), dict) else {}
        model, rows = _path(entry.get("reference_model")), _path(entry.get("frame_data"))
        if not model:
            no_model.append(run)
        elif not rows:
            no_rows.append(run)
        elif outside_series(model, folders) or outside_series(rows, folders):
            outside.append(run)
    problems = _problems((("No reference model", no_model), ("Frame rows not saved yet", no_rows),
                          ("Model or frame rows outside the series folders", outside)), len(runs))
    if value.get("reference_models_complete") is not True:
        problems = ("The index does not mark its reference models complete. " + problems).strip()
    if problems:
        return Source(label, expected, problems)
    return Source(label, expected, found=str(bound["path"]), bound=bound)


def _pressure(folder, series_id, runs, folders) -> Source:
    label, path = "Running pressure", folder / PRESSURE_INDEX
    expected = f"{path} ({' or '.join(PRESSURE_FORMATS)}, with every run's series)"
    bound, value, problem = _index("running pressure index", path, folders, PRESSURE_FORMATS)
    if problem:
        return Source(label, expected, problem)
    table = (value.get("runs") or {}).get(series_id) if isinstance(value.get("runs"), dict) else None
    table = table if isinstance(table, dict) else {}
    absent, outside = [], []
    for run in runs:
        series = _path((table.get(run) or {}).get("atmosphere_series") if isinstance(table.get(run), dict) else None)
        if not series:
            absent.append(run)
        elif outside_series(series, folders):
            outside.append(run)
    problems = _problems((("No pressure series", absent),
                          ("Pressure series outside the series folders", outside)), len(runs))
    if problems:
        return Source(label, expected, problems)
    return Source(label, expected, found=str(bound["path"]), bound=bound)


def phase_one_sources(series_root) -> list[Source]:
    """Phase maps and statistics: run documents, frame geometry and running pressure."""
    series_id, runs = _runs(series_root)
    folder, folders = analysis_inputs_folder(series_root), series_folders(series_root)
    return [_run_documents(folder, runs, folders), _geometry(folder, runs, folders),
            _pressure(folder, series_id, runs, folders)]


def phase_two_sources(series_root, calibration=None) -> list[Source]:
    """Mass from finished phase maps: phase-one completions and the fitted calibration.

    ``calibration`` is the completed mass calibration job chosen on the page
    (production_selection.read_named_calibration), or None.
    """
    from .series_paths import series_folder_path
    folder, folders = analysis_inputs_folder(series_root), series_folders(series_root)
    path = folder / PHASE_TWO_INPUTS
    expected = f"{path} ({PHASE_TWO_FORMAT}, saved after phase one has run)"
    bound, value, problem = _index("phase-one completion", path, folders, (PHASE_TWO_FORMAT,))
    completions = (Source("Phase-one completions", expected, problem) if problem else
                   Source("Phase-one completions", expected,
                          found=f"{len(value.get('runs') or ())} runs in {bound['path']}", bound=bound))
    jobs = series_folder_path(series_root, "calibration_jobs")
    label, expected = "Fitted mass calibration", f"A completed mass calibration job in {jobs}"
    fitted = _path((calibration or {}).get("calibration"))
    if not fitted:
        fitted_source = Source(label, expected, "Choose a completed mass calibration job on this page.")
    elif outside_series(fitted, folders):
        fitted_source = Source(label, expected, f"Outside the series folders: {fitted}")
    else:
        fitted_source = Source(label, expected, found=f"{calibration.get('name', '')} · {fitted}",
                               bound=calibration["calibration"])
    return [completions, fitted_source]


def sources_text(sources) -> str:
    """Plain text for the page: found inputs, and where each missing one is expected."""
    lines = []
    for source in sources:
        if source.missing:
            lines.append(f"Missing: {source.label}. {source.missing}\n    Expected: {source.expected}")
        else:
            lines.append(f"Found: {source.label}: {source.found}")
    return "\n".join(lines)


def fresh_plan_paths(series_root, name) -> tuple[Path, Path]:
    """A new plan folder in the series' plans folder and its source-input manifest beside it."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = analysis_plans_folder(series_root) / f"{name}-{stamp}"
    return folder, folder.with_name(folder.name + "-source-inputs.json")


def prepare_phase_one_plan(series_root, sources, *, folder, manifest, gas_probability, repository, job_id):
    """Prepare, never adopt, a phase-one plan from the found ``sources``.

    The original source collection writes the manifest; the original
    preparation then writes the inert plan into the fresh ``folder``. Both
    repeat every check and bind the exact files that were found.
    """
    from . import analysis_preparation as preparation
    missing = [f"{source.label}: {source.missing}" for source in sources if source.missing]
    if missing:
        raise ValueError("Analysis inputs are missing; no plan was prepared.\n" + "\n".join(missing))
    runs, geometry, pressure = (source.bound for source in sources)
    job = preparation.job_identity(repository)
    folder, manifest = Path(folder), Path(manifest)
    if folder.exists() or manifest.exists():
        raise FileExistsError("A plan of this name and time exists; prepare again.")
    folder.parent.mkdir(parents=True, exist_ok=True)
    collected = preparation.collect_phase1_source_inputs(runs, manifest, series_root=series_root,
        geometry_index=geometry, pressure_index=pressure, gas_probability=gas_probability)
    return preparation.prepare_phase1_inputs(collected["inputs"], folder, series_root=series_root,
        geometry_index=geometry, pressure_index=pressure, gas_probability=gas_probability,
        job_id=job_id, job=job)
