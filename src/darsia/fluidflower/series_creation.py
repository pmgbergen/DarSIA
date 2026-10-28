"""Create a native series folder: series.toml and the run configurations.

A series folder is the ``DarSIA`` folder next to the experiments' image
folders. It holds ``series.toml`` (identity, rig class, runs), the shared
``config/common.toml``, one ``config/runs/<run>.toml`` per run with its image
folders, baseline photograph, results folder and protocol paths. Shared Results
default to the project's ``Results`` folder beside ``DarSIA``. Phase setup,
calibration and job products follow that selected Results folder. Images are
never copied or changed.

Native Setup's protocol step then writes the imaging protocols (from EXIF) and
templates for the injection and pressure-temperature protocols at the paths
given in the run files.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from darsia.fluidflower.series_paths import PHASE_MODEL_FOLDER, series_key
from darsia.fluidflower.series_cycles import CycleRun, cycle_runs_table
from darsia.fluidflower.storage_paths import machine_local_root, machine_stored_text
from darsia.fluidflower.templates import RIG_CLASSES

# Image formats of DarSIA's [data] section, by file suffix.
IMAGE_FORMATS: dict[str, str] = {
    ".jpg": "JPG",
    ".jpeg": "JPG",
    ".png": "PNG",
    ".tif": "TIF",
    ".tiff": "TIFF",
    ".bmp": "BMP",
}
SERIES_FOLDER_NAME = "DarSIA"
# Folders named after injections are ticked by default, unless the name marks a
# test, a blue-water (tracer) injection or flushing.
INJECTION_WORD = "inj"
NOT_A_RUN_WORDS = ("test", "blue", "blå", "skyll", "flush", "clean")
RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
GENERATOR = "darsia.fluidflower.series_creation"
# Every folder used to train the series' phase model: review sessions, review
# collections, the evidence ledger and audits under ExpertKnowledge/PhaseModels/TrainingInput,
# trained versions and their library index under ExpertKnowledge/PhaseModels/Models.
_RESULT_FOLDERS = (
    ("models", "calibration/" + PHASE_MODEL_FOLDER + "/Models"),
    ("phase_model", "calibration/" + PHASE_MODEL_FOLDER),
    ("geometry_facies", "setup/Geometry"),
    ("data", "setup/inputs"),
    ("calibration", "calibration"),
    ("image_selection", "setup/image_selection"),
    ("jobs", "jobs"),
)
PATHS = (
    ("root", "."),
    ("config", "config"),
    ("common_config", "config/common.toml"),
    ("run_configs", "config/runs"),
    ("merged_configs", "config/merged"),
    ("protocols", "protocols"),
    *((key, "../Results/" + folder) for key, folder in _RESULT_FOLDERS),
    ("project", ".."),
    ("results", "../Results"),
)


def _natural_key(text: str) -> list[object]:
    """Numbers compare as numbers; spaces, _ and - count alike."""

    text = re.sub(r"[\s_-]+", " ", text)
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", text)
    ]


def image_files(folder: Path) -> list[Path]:
    """Images directly in a folder, in natural order."""

    # os.scandir lists file types with the names (no stat per file), which
    # matters for folders of thousands of images on a network share.
    files: list[Path] = []
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                name = entry.name
                if (
                    not name.startswith(".")
                    and os.path.splitext(name)[1].casefold() in IMAGE_FORMATS
                    and entry.is_file()
                ):
                    files.append(Path(folder) / name)
    except OSError:
        return []
    return sorted(files, key=lambda path: _natural_key(path.name))


def run_id_from_name(name: str) -> str:
    """A run ID from a folder name: lower case letters, digits, _ and -."""

    text = re.sub(r"[^a-z0-9_-]+", "_", str(name).strip().casefold()).strip("_-")
    return text or "run"


@dataclass
class FolderInfo:
    """A folder with images of a run's format."""

    path: Path
    count: int
    first_image: str
    # Time of the first image (EXIF, else file time), for ordering folders.
    first_time: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.path.name} ({self.count} images)"


@dataclass
class RunDraft:
    """One run (experiment) of a new series."""

    run_id: str
    folders: list[Path]
    baseline: str = ""
    image_format: str = "JPG"
    comment: str = ""
    image_count: int = 0
    # Folders with images of the run's format, in time order.
    candidates: list[FolderInfo] = field(default_factory=list)
    # The experiment folder the run was found in.
    source: Path | None = None
    # Optional provenance only; ordinary configuration/baseline selection stays explicit.
    cycle: CycleRun | None = None

    def folder_label(self, path: Path) -> str:
        """A folder relative to the experiment folder."""

        path = Path(path)
        if self.source is not None and path != self.source:
            try:
                return path.relative_to(self.source).as_posix()
            except ValueError:
                pass
        return path.name

    def baseline_path(self) -> Path | None:
        for folder in self.folders:
            candidate = Path(folder) / self.baseline
            if self.baseline and candidate.is_file():
                return candidate
        return None

    def set_folders(self, folders: Sequence[Path]) -> None:
        """Use these image folders (kept in candidate order) and update the
        image count; the baseline moves to the first folder when it is no
        longer in one of them."""

        chosen = {Path(path) for path in folders}
        known = {info.path: info for info in self.candidates}
        ordered = [info.path for info in self.candidates if info.path in chosen]
        ordered += [Path(path) for path in folders if Path(path) not in known]
        self.folders = ordered
        self.image_count = sum(known[path].count for path in ordered if path in known)
        if self.folders and self.baseline_path() is None:
            first = known.get(self.folders[0])
            if first is not None:
                self.baseline = first.first_image
            else:
                images = _images_of(self.folders[0], self.image_format)
                self.baseline = images[0].name if images else ""


@dataclass
class SeriesDraft:
    """Everything the new series folder is written from."""

    series_id: str
    name: str
    rig_class: str
    darsia_folder: Path
    runs: list[RunDraft] = field(default_factory=list)
    created_by: str = ""
    rig_profile: str = ""
    project_folder: Path | None = None
    results_folder: Path | None = None
    jobs_folder: Path | None = None


def project_folder(draft: SeriesDraft) -> Path:
    """The shared project container, separate from its DarSIA settings folder."""
    return Path(getattr(draft, "project_folder", None) or Path(getattr(draft, "darsia_folder", ".")).parent)


def results_folder(draft: SeriesDraft) -> Path:
    """An explicitly chosen Results folder, or the project's shared default."""
    return Path(getattr(draft, "results_folder", None) or project_folder(draft) / "Results")


def jobs_folder(draft: SeriesDraft) -> Path:
    """The shared queue selected during Setup; each job gets its own child."""
    return Path(getattr(draft, "jobs_folder", None) or results_folder(draft) / "jobs")


def _relative_folder(folder: Path, root: Path) -> str:
    """Use a portable relative declaration when both folders share a volume.

    On a computer with a machine path map, a folder that is not below the same
    shared root as ``root`` is declared by its absolute shared path, which
    every computer reads (see ``storage_paths.machine_stored_text``).
    """
    if machine_local_root(folder) != machine_local_root(root):
        return machine_stored_text(folder)
    try:
        return Path(os.path.relpath(folder, root)).as_posix()
    except ValueError:
        return machine_stored_text(folder)


def _images_of(folder: Path, image_format: str) -> list[Path]:
    return [
        path
        for path in image_files(folder)
        if IMAGE_FORMATS[path.suffix.casefold()] == image_format
    ]


def _image_time(path: Path) -> float:
    """Capture time of an image (EXIF), else its modification time."""

    from darsia.fluidflower.photograph_io import is_camera_raw, raw_capture_time

    if is_camera_raw(path):
        taken = raw_capture_time(path)
        if taken is not None:
            return taken.timestamp()
    try:
        from PIL import Image

        with Image.open(path) as image:
            exif = image.getexif()
            raw = exif.get_ifd(0x8769).get(36867) or exif.get(306)
        if raw:
            return dt.datetime.strptime(
                str(raw).strip(), "%Y:%m:%d %H:%M:%S"
            ).timestamp()
    except (OSError, ValueError, SyntaxError):
        raw = None  # unreadable or without a capture time: use the file time
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _image_folders(folder: Path, depth: int) -> list[Path]:
    """The folder and its subfolders (to ``depth`` levels) that hold images."""

    found = [folder] if image_files(folder) else []
    if depth <= 0:
        return found
    try:
        with os.scandir(folder) as entries:
            children = sorted(
                (
                    Path(folder) / entry.name
                    for entry in entries
                    if entry.is_dir()
                    and not entry.name.startswith(".")
                    and entry.name.casefold() != SERIES_FOLDER_NAME.casefold()
                ),
                key=lambda path: _natural_key(path.name),
            )
    except OSError:
        return found
    for child in children:
        found.extend(_image_folders(child, depth - 1))
    return found


def _is_injection(name: str) -> bool:
    text = name.casefold()
    return INJECTION_WORD in text and not any(word in text for word in NOT_A_RUN_WORDS)


def scan_run_folder(
    folder: Path, run_id: str = "", *, depth: int = 2, min_images: int = 10
) -> RunDraft:
    """A run from an experiment folder: the folders below it with images.

    The image format is the most common one. Candidate folders are ordered by
    the capture time of their first image. Folders whose name contains "inj"
    (Injeksjon, injection) but no word of ``NOT_A_RUN_WORDS`` (tests,
    blue-water injections, flushing) and that hold at least ``min_images``
    images are used; without such folders, every folder with ``min_images``
    images (or, failing that, every candidate). The baseline is the first image
    of the first folder used.
    """

    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a folder.")
    folders = _image_folders(folder, depth)
    counts: Counter[str] = Counter()
    for path in folders:
        counts.update(
            IMAGE_FORMATS[image.suffix.casefold()] for image in image_files(path)
        )
    if not counts:
        raise ValueError(f"No images in {folder} or its subfolders.")
    image_format = counts.most_common(1)[0][0]
    candidates: list[FolderInfo] = []
    for path in folders:
        images = _images_of(path, image_format)
        if images:
            candidates.append(
                FolderInfo(path, len(images), images[0].name, _image_time(images[0]))
            )
    candidates.sort(key=lambda info: (info.first_time, _natural_key(str(info.path))))
    large = [info for info in candidates if info.count >= min_images]
    injections = [info for info in large if _is_injection(info.path.name)]
    chosen = injections or large or candidates
    run = RunDraft(
        run_id=run_id or run_id_from_name(folder.name),
        folders=[],
        image_format=image_format,
        candidates=candidates,
        source=folder,
    )
    run.set_folders([info.path for info in chosen])
    return run


def _subfolders(folder: Path, skip: Sequence[str] = ()) -> list[Path]:
    skipped = {name.casefold() for name in skip}
    try:
        with os.scandir(folder) as entries:
            children = [
                Path(folder) / entry.name
                for entry in entries
                if entry.is_dir()
                and not entry.name.startswith(".")
                and entry.name.casefold() not in skipped
            ]
    except OSError:
        return []
    return sorted(children, key=lambda path: _natural_key(path.name))


def _holds_injection_folders(folder: Path, depth: int = 2) -> bool:
    """Whether a folder has injection folders below it (by name only)."""

    if depth <= 0:
        return False
    for child in _subfolders(folder):
        if _is_injection(child.name) or _holds_injection_folders(child, depth - 1):
            return True
    return False


def _runs_in(folder: Path, levels: int) -> list[RunDraft]:
    """One run per experiment folder; a folder of experiments is opened."""

    if levels > 0:
        children = _subfolders(folder, (SERIES_FOLDER_NAME,))
        experiments = [child for child in children if _holds_injection_folders(child)]
        if len(experiments) >= 2:
            runs: list[RunDraft] = []
            for child in children:
                runs.extend(_runs_in(child, levels - 1))
            return runs
    try:
        return [scan_run_folder(folder)]
    except ValueError:
        return []


def find_runs(
    folder: Path, *, skip: Sequence[str] = (SERIES_FOLDER_NAME,)
) -> list[RunDraft]:
    """One run per experiment below a folder.

    Each subfolder with images is an experiment, unless it holds several
    experiments (subfolders that each contain injection folders); those are
    listed one by one, up to two levels down.
    """

    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a folder.")
    runs: list[RunDraft] = []
    used: set[str] = set()
    for child in _subfolders(folder, skip):
        for run in _runs_in(child, 2):
            base = run.run_id
            number = 2
            while run.run_id in used:
                run.run_id = f"{base}_{number}"
                number += 1
            used.add(run.run_id)
            runs.append(run)
    return runs


def check_runs(runs: Sequence[RunDraft]) -> list[str]:
    """Reasons why the runs cannot be written (empty when they can)."""

    problems: list[str] = []
    if not runs:
        problems.append("Add at least one experiment.")
    seen: set[str] = set()
    for run in runs:
        label = run.run_id or "an experiment"
        if not RUN_ID_PATTERN.match(run.run_id or ""):
            problems.append(
                f"Run ID {run.run_id!r}: use lower case letters, digits, _ or -, "
                "starting with a letter or digit."
            )
        elif run.run_id in seen:
            problems.append(f"Run ID {run.run_id} is used twice.")
        seen.add(run.run_id)
        if not run.folders:
            problems.append(f"{label}: add an image folder.")
        missing = [str(path) for path in run.folders if not Path(path).is_dir()]
        if missing:
            problems.append(f"{label}: missing folder {missing[0]}.")
        if run.image_format not in set(IMAGE_FORMATS.values()):
            problems.append(f"{label}: unknown image format {run.image_format!r}.")
        if run.folders and not missing and run.baseline_path() is None:
            problems.append(
                f"{label}: the baseline photograph {run.baseline or '(none)'} is not "
                "in its image folders."
            )
    if any(getattr(run, "cycle", None) is not None for run in runs):
        try:
            cycle_runs_table([getattr(run, "cycle", None) for run in runs], [run.run_id for run in runs])
        except ValueError as error:
            problems.append(str(error))
    return problems


def check_series_draft(
    draft: SeriesDraft, *, registered_ids: Sequence[str] = ()
) -> list[str]:
    """Reasons why the series cannot be created (empty when it can)."""

    problems: list[str] = []
    if not str(draft.name).strip():
        problems.append("Enter a name for the series.")
    try:
        key = series_key(draft.series_id)
    except ValueError:
        problems.append(
            "The series ID must start with a letter and contain letters, digits, "
            "_ or - only (for example FF_XY)."
        )
        key = ""
    if key and key in {series_key(value) for value in registered_ids if value}:
        problems.append(
            f"A series with the ID {draft.series_id} is already registered."
        )
    if draft.rig_class not in RIG_CLASSES:
        problems.append("Choose the rig class.")
    folder = Path(draft.darsia_folder) if str(draft.darsia_folder) else None
    if folder is None:
        problems.append("Choose the experiment folder.")
    elif (folder / "series.toml").exists():
        problems.append(
            f"{folder} already holds a series; add it on Start with 'Add existing "
            "series folder'."
        )
    elif not folder.parent.is_dir():
        problems.append(f"The folder {folder.parent} does not exist.")
    problems.extend(check_runs(draft.runs))
    return problems


def _toml_string(value: object) -> str:
    """A TOML basic string (JSON escapes are valid TOML escapes)."""

    return json.dumps(str(value), ensure_ascii=False)


def _path_text(path: Path | str) -> str:
    """The stored text of an absolute run path, in its shared spelling when mapped."""
    return machine_stored_text(Path(path))


def _series_toml(draft: SeriesDraft, created: str) -> str:
    runs = [run.run_id for run in draft.runs]
    rig_profile = getattr(draft, "rig_profile", "")
    root = Path(getattr(draft, "darsia_folder", "."))
    paths = dict(PATHS)
    if hasattr(draft, "darsia_folder"):
        paths.update(project=_relative_folder(project_folder(draft), root),
                     results=_relative_folder(results_folder(draft), root))
    paths.update((key, (Path(paths["results"]) / folder).as_posix())
                 for key, folder in _RESULT_FOLDERS)
    paths["jobs"] = _relative_folder(jobs_folder(draft), root)
    image_roots = list(dict.fromkeys(str(folder) for run in draft.runs for folder in getattr(run, "folders", [])))
    roots = {"series": ".", "results": paths["results"], "jobs": paths["jobs"]}
    known = {Path(os.path.abspath(root)), Path(os.path.abspath(root / paths["results"]))}
    project = Path(os.path.abspath(root / paths["project"]))
    if project not in known:
        roots["project"] = paths["project"]
        known.add(project)
    for index, folder in enumerate(image_roots, 1):
        source = Path(os.path.abspath(folder))
        if source not in known:
            roots[f"images_{index}"] = _relative_folder(source, root)
            known.add(source)
    lines = [
        "# Paths are relative to this manifest unless absolute.",
        f"# Created {created[:10]} by the New series wizard of DarSIA-dev.",
        "",
        "schema_version = 1",
        f"series_id = {_toml_string(draft.series_id)}",
        f"name = {_toml_string(draft.name.strip())}",
        f"rig_class = {_toml_string(draft.rig_class)}",
        *([f"rig_profile = {_toml_string(rig_profile)}"] if rig_profile else []),
        'config_format = "darsia-dev-fluidflower"',
        "",
        "[method]",
        'active = "phase_model"',
        "",
        "[paths]",
        *(f"{key} = {_toml_string(value)}" for key, value in paths.items()),
        "",
        "[roots]",
        *(f"{key} = {_toml_string(value)}" for key, value in roots.items()),
        "",
        "[runs]",
        "ids = [" + ", ".join(_toml_string(run) for run in runs) + "]",
        f"expected_count = {len(runs)}",
        "",
        "[runs.config_files]",
        *(f"{run} = {_toml_string(f'config/runs/{run}.toml')}" for run in runs),
        "",
        "[provenance]",
        f"generator = {_toml_string(GENERATOR)}",
        f"created_utc = {_toml_string(created)}",
        f"created_by = {_toml_string(draft.created_by)}",
        "",
    ]
    if any(getattr(run, "cycle", None) is not None for run in draft.runs):
        table = cycle_runs_table([getattr(run, "cycle", None) for run in draft.runs], runs)
        lines += ["[cycle_runs]", f"format = {_toml_string(table['format'])}", ""]
        for run_id, record in table["runs"].items():
            lines += [f"[cycle_runs.runs.{_toml_string(run_id)}]",
                      f"experiment = {_toml_string(record['experiment'])}",
                      f"number = {record['number']}"]
            if "previous_run_id" in record:
                lines.append(f"previous_run_id = {_toml_string(record['previous_run_id'])}")
            lines.append("")
    return "\n".join(lines)


def _common_toml(draft: SeriesDraft, created: str) -> str:
    rig = RIG_CLASSES[draft.rig_class]
    lines = [
        f"# Settings shared by all runs of {draft.name.strip()}; the run files in",
        "# config/runs override them.",
        f"# Created {created[:10]} by the New series wizard of DarSIA-dev.",
        "#",
        "# Next steps:",
        "# 1. Setup > Protocols writes the imaging protocols (from EXIF) and",
        "#    templates for the injection and pressure-temperature protocols.",
        "# 2. Geometry and facies: rig frame, injection points and sand layers. Its",
        "#    export gives the [labeling] and [facies] sections for this file.",
        "# 3. Native Setup: depth ([depth]), labels, facies and rig.",
        "",
        "[rig]",
        "dim = 2",
    ]
    if rig.width_m is not None and rig.height_m is not None:
        lines += [
            (
                f"width = {rig.width_m:g}  # nominal inner width of a {rig.label}; "
                "replace with measured values"
            ),
            f"height = {rig.height_m:g}  # nominal inner height",
        ]
    else:
        lines += [
            "# width = 0.0  # inner width in metres",
            "# height = 0.0  # inner height in metres",
        ]
    lines.append("")
    return "\n".join(lines)


def _run_toml(draft: SeriesDraft, run: RunDraft, root: Path, created: str) -> str:
    protocols = root / "protocols" / run.run_id
    lines = [
        f"# Run {run.run_id} of {draft.name.strip()}.",
        f"# Created {created[:10]} by the New series wizard of DarSIA-dev. Setup >",
        "# Protocols writes the imaging protocols and the injection and",
        "# pressure-temperature templates at the paths below.",
        "",
        "[id]",
        f"run_id = {_toml_string(f'{draft.series_id} {run.run_id.upper()}')}",
        f"comment = {_toml_string(run.comment)}",
        "",
        "[data]",
        "folders = [",
        *(f"  {_toml_string(_path_text(folder))}," for folder in run.folders),
        "]",
        f"format = {_toml_string(run.image_format)}",
        f"baseline = {_toml_string(run.baseline)}",
        f"results = {_toml_string(_path_text(results_folder(draft) / run.run_id))}",
        "",
        "[protocols]",
        "injection = " + _toml_string(_path_text(protocols / "injection_protocol.csv")),
        "pressure_temperature = "
        + _toml_string(_path_text(protocols / "pressure_temperature_protocol.csv")),
        'imaging_mode = "exif"',
        "",
        "[protocols.imaging]",
    ]
    for folder in run.folders:
        name = run_id_from_name(Path(folder).name)
        lines.append(
            f"{_toml_string(_path_text(folder))} = "
            + _toml_string(_path_text(protocols / f"imaging_protocol_{name}.csv"))
        )
    lines.append("")
    return "\n".join(lines)


def _readme(draft: SeriesDraft, created: str) -> str:
    rig = RIG_CLASSES[draft.rig_class]
    runs = ", ".join(run.run_id for run in draft.runs)
    return "\n".join(
        [
            f"# {draft.name.strip()}",
            "",
            (
                f"DarSIA series folder created {created[:10]} with the New series "
                "wizard of DarSIA-dev."
            ),
            "",
            f"* Series ID: {draft.series_id}",
            f"* Rig class: {rig.title()}",
            f"* Runs: {runs}",
            "",
            (
                "The image folders stay where they are; each run's file in "
                "`config/runs` lists its folders, baseline photograph, results "
                "folder and protocols."
            ),
            "",
            "Next steps:",
            "",
            (
                "1. Setup > Protocols writes the imaging protocols (from EXIF) and "
                "templates for the injection and pressure-temperature protocols. "
                "Fill in the injection times, rate and port position."
            ),
            (
                "2. Geometry and facies: rig frame, injection points and sand "
                "layers. Its export gives the `[labeling]` and `[facies]` sections "
                "for `config/common.toml`."
            ),
            "3. Native Setup: depth (`[depth]`), labels, facies and rig.",
            "",
        ]
    )


def series_files(draft: SeriesDraft, *, created: str = "") -> dict[str, str]:
    """Relative path and text of every file the series folder starts with."""

    created = created or dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    root = Path(draft.darsia_folder)
    files = {
        "series.toml": _series_toml(draft, created),
        "config/common.toml": _common_toml(draft, created),
        "README.md": _readme(draft, created),
    }
    for run in draft.runs:
        files[f"config/runs/{run.run_id}.toml"] = _run_toml(draft, run, root, created)
    return files


def create_series_folder(
    draft: SeriesDraft, *, registered_ids: Sequence[str] = ()
) -> Path:
    """Write the series folder; nothing existing is overwritten."""

    problems = check_series_draft(draft, registered_ids=registered_ids)
    if problems:
        raise ValueError(" ".join(problems))
    root = Path(draft.darsia_folder)
    files = series_files(draft)
    clashes = [name for name in files if (root / name).exists()]
    if clashes:
        raise ValueError(
            f"{root} already contains {clashes[0]}; choose an empty series folder."
        )
    root.mkdir(parents=True, exist_ok=True)
    for folder in ("config/runs", "protocols"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    results_folder(draft).mkdir(parents=True, exist_ok=True)
    jobs_folder(draft).mkdir(parents=True, exist_ok=True)
    for run in draft.runs:
        (root / "protocols" / run.run_id).mkdir(parents=True, exist_ok=True)
    # series.toml last, so that an interrupted creation is not a series yet.
    for name in sorted(files, key=lambda name: name == "series.toml"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(files[name], encoding="utf-8")
        os.replace(temporary, path)
    return root


__all__ = [
    "IMAGE_FORMATS",
    "SERIES_FOLDER_NAME",
    "RunDraft",
    "SeriesDraft",
    "check_runs",
    "check_series_draft",
    "create_series_folder",
    "find_runs",
    "image_files",
    "run_id_from_name",
    "scan_run_folder",
    "series_files",
]
