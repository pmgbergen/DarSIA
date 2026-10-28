"""Choose the native reference configuration using small saved metadata only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import stat
import tomllib

MAX_METADATA_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ReferenceChoice:
    run_id: str
    config_path: str
    source: str
    summary: str
    issues: tuple[str, ...] = ()


def _document(path: Path) -> Mapping:
    with path.open("rb") as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError(
            f"Reference metadata exceeds {MAX_METADATA_BYTES} bytes: {path}"
        )
    text = raw.decode("utf-8-sig")
    value = (
        tomllib.loads(text) if path.suffix.casefold() == ".toml" else json.loads(text)
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"Reference metadata is not an object: {path}")
    return value


def reference_choice(
    series_root: str | Path,
    choices: Sequence[tuple[str, Path | None, str]],
) -> ReferenceChoice:
    """Saved native geometry reference wins; otherwise first manifest row.

    ``choices`` comes from the existing native ``run_configurations`` function.
    An invalid saved reference, unreadable metadata or missing reference config
    stops this choice. It never skips to a different usable run, builds a
    merged file, creates a geometry project, or reads any image/NPZ content.
    """
    root = Path(series_root)
    source = "series manifest"
    run = ""
    try:
        manifest = _document(root / "series.toml")
        runs = manifest.get("runs")
        ids = runs.get("ids") if isinstance(runs, Mapping) else None
        if (
            manifest.get("schema_version") != 1
            or not isinstance(ids, list)
            or not ids
            or not all(isinstance(item, str) and item for item in ids)
        ):
            raise ValueError("The selected series has no valid ordered run IDs.")
        paths = manifest.get("paths")
        raw = paths.get("geometry_facies") if isinstance(paths, Mapping) else None
        package = Path(str(raw or "geometry_facies"))
        if not package.is_absolute():
            package = root / package
        project_path = package / "series.json"
        # A normal absent geometry package has no saved reference yet. Errors
        # reading an existing package are retained, never converted to fallback.
        try:
            saved_project = project_path.stat()
        except FileNotFoundError:
            saved_project = None
        if saved_project is not None:
            if not stat.S_ISREG(saved_project.st_mode):
                raise ValueError(
                    f"Saved geometry metadata is not a regular file: {project_path}"
                )
            project = _document(project_path)
            tools = project.get("tools")
            link = tools.get("native_series") if isinstance(tools, Mapping) else None
            if link is not None and not isinstance(link, Mapping):
                raise ValueError(
                    "Saved native-series reference metadata is not an object."
                )
            saved = link.get("reference_run") if isinstance(link, Mapping) else None
            if saved is not None and saved != "":
                source = "saved native geometry reference"
                if not isinstance(saved, str):
                    raise ValueError(
                        "Saved reference_run must be a string, null or absent."
                    )
                run = saved
                if run not in ids:
                    raise ValueError(
                        f"Saved reference run {run!r} is not registered in this series."
                    )
        if not run:
            run = ids[0]
            source = "first run in series manifest order (no saved reference)"
        row = next((item for item in choices if item[0] == run), None)
        if row is None or row[1] is None:
            detail = (
                row[2]
                if row is not None
                else "No single-file configuration is registered for the reference run."
            )
            return ReferenceChoice(
                run,
                "",
                source,
                f"Reference {run.upper()} could not be opened.",
                (detail,),
            )
        path = Path(row[1])
        if not path.is_file():
            return ReferenceChoice(
                run,
                "",
                source,
                f"Reference {run.upper()} could not be opened.",
                (f"Reference configuration does not exist: {path}",),
            )
        return ReferenceChoice(
            run, str(path), source, f"Reference {run.upper()} from {source}: {path}"
        )
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        return ReferenceChoice(
            run,
            "",
            source,
            "Reference configuration could not be selected.",
            (str(error),),
        )
