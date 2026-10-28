"""Image choices from a series' time grids ([[data_time]] in config/common.toml).

A grid is a named list of times after the first injection start, with a
tolerance. Grids whose names differ only in their last part, for example
``comparison_grid_v1_injection`` and ``comparison_grid_v1_rest``, form one
grid family. A proposal matches each time of the chosen grids to a run's
camera images with the existing camera-grid selector,
:func:`darsia.fluidflower.image_selection_plan.resolve_quality_time_grid`: the
nearest image within 15 s up to 3 h and within 150 s after, and a rejected
image replaced by an ok one within 60 s or 600 s. As every saved choice
must, a proposal also holds the last usable image before injection (or, when
there is none, the first usable image). Image quality comes from the run's
saved quality receipt where it has the image; other images count as ok and
are flagged. Nothing is decoded, measured or written here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
import tomllib
from typing import Any

_CLOCK = re.compile(r"(-)?(\d+):([0-5]\d):([0-5]\d(?:\.\d+)?)")


def clock_hours(text: str) -> float:
    """Hours of a time written as [-]HH:MM:SS[.s], for example "02:30:00" -> 2.5."""

    match = _CLOCK.fullmatch(str(text).strip())
    if match is None:
        raise ValueError(f"Not a time of the form HH:MM:SS: {text!r}")
    sign, hours, minutes, seconds = match.groups()
    value = int(hours) + int(minutes) / 60.0 + float(seconds) / 3600.0
    return -value if sign else value


def read_time_grids(series_root: Path | str) -> dict[str, dict[str, Any]]:
    """The named time grids of a series, in file order.

    Each grid is {"times_h": [...], "tolerance_s": seconds}. They come from
    the ``[[data_time]]`` tables of ``config/common.toml``; a table without a
    name or times is left out. A missing file gives no grids.
    """

    path = Path(series_root) / "config" / "common.toml"
    if not path.is_file():
        return {}
    document = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    grids: dict[str, dict[str, Any]] = {}
    for table in document.get("data_time") or []:
        if not isinstance(table, Mapping):
            continue
        name, times = table.get("name"), table.get("times")
        if not (isinstance(name, str) and name and isinstance(times, list) and times):
            continue
        tolerance = table.get("tol")
        grids[name] = {
            "times_h": [clock_hours(value) for value in times],
            "tolerance_s": 3600.0 * clock_hours(tolerance) if tolerance else None,
        }
    return grids


def grid_families(grids: Mapping[str, Any]) -> dict[str, list[str]]:
    """Grid families: names that differ only in their last part, in file order.

    A grid that shares its stem with no other grid is a family of its own.
    """

    stems: dict[str, list[str]] = {}
    for name in grids:
        stem = name.rsplit("_", 1)[0] if "_" in name else name
        stems.setdefault(stem, []).append(name)
    families: dict[str, list[str]] = {}
    for stem, names in stems.items():
        if len(names) > 1:
            families[stem] = names
        else:
            families[names[0]] = names
    return families


def grid_target_hours(
    grids: Mapping[str, Mapping[str, Any]], names: Sequence[str]
) -> list[float]:
    """The times of the named grids together, ascending, each once."""

    if not names:
        raise ValueError("Choose at least one time grid.")
    times: set[float] = set()
    for name in names:
        if name not in grids:
            raise ValueError(f"No time grid named {name!r} in config/common.toml.")
        times.update(round(float(value), 9) for value in grids[name]["times_h"])
    return sorted(times)


def propose_from_grid(
    metadata: Any,
    grids: Mapping[str, Mapping[str, Any]],
    names: Sequence[str],
    *,
    quality: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Match a run's camera images to the times of the named grids.

    ``metadata`` is the run's protocol metadata (its ``frames`` with
    ``source_path`` and ``elapsed_h``, and ``injection_start_h``);
    ``quality`` is its saved quality receipt as the image selection editor
    reads it, or None. Returns the selector's result with ``grids``,
    ``target_hours``, ``baseline_path`` (the last usable image before
    injection, else the first usable image), ``selected_paths`` (the
    baseline first, then in time order), ``missing_hours`` and
    ``unrecorded_quality`` (images chosen without a saved class) added.
    """

    from darsia.fluidflower.image_selection_plan import resolve_quality_time_grid
    from darsia.fluidflower.image_selection_proposal import eligible_frames

    targets = grid_target_hours(grids, names)
    classes = dict((quality or {}).get("classes") or {})
    unrecorded: set[str] = set()

    def classify(frame: Mapping[str, Any]) -> dict[str, Any]:
        key = str(frame.get("source_path", "")).replace("\\", "/").casefold()
        saved = classes.get(key)
        if saved is None:
            unrecorded.add(key)
            return {"class": "ok", "recorded": False}
        return dict(saved)

    result = resolve_quality_time_grid(list(metadata.frames), targets, quality=classify)
    def saved_class(row: Mapping[str, Any]) -> str | None:
        key = str(row["source_path"]).replace("\\", "/").casefold()
        return (classes.get(key) or {}).get("class")

    usable = [row for row in eligible_frames(metadata) if saved_class(row) != "reject"]
    if not usable:
        raise ValueError("The saved quality marks every eligible image reject.")
    before = [row for row in usable if row["elapsed_h"] < metadata.injection_start_h]
    baseline = max(before, key=lambda row: row["elapsed_h"]) if before else usable[0]
    paths = [row["source_path"] for row in result["selected"]]
    key = str(baseline["source_path"]).replace("\\", "/").casefold()
    if all(str(path).replace("\\", "/").casefold() != key for path in paths):
        paths.insert(0, baseline["source_path"])
        classify(baseline)
    chosen = {str(path).replace("\\", "/").casefold() for path in paths}
    result.update(
        grids=list(names),
        target_hours=targets,
        baseline_path=baseline["source_path"],
        selected_paths=paths,
        missing_hours=[
            row["target_hour"] for row in result["target_matches"] if not row["selected"]
        ],
        unrecorded_quality=len(unrecorded & chosen),
    )
    return result


def proposal_summary(result: Mapping[str, Any]) -> str:
    """One line for a grid proposal: images chosen, times without an image,
    replacements and images without a saved quality class."""

    replaced = sum(1 for row in result["target_matches"] if row.get("replacement"))
    missing = result["missing_hours"]
    parts = [
        f"{result['selected_frame_count']} images for {len(result['target_hours'])} times "
        f"of {', '.join(result['grids'])}, and the image before injection"
    ]
    if missing:
        shown = ", ".join(f"{value:g}" for value in missing[:6])
        parts.append(
            f"{len(missing)} time(s) without an image within the tolerance ({shown}"
            + (", …" if len(missing) > 6 else "")
            + " h)"
        )
    if replaced:
        parts.append(f"{replaced} rejected image(s) replaced")
    if result["unrecorded_quality"]:
        count = result["unrecorded_quality"]
        parts.append(f"{count} image(s) without a saved quality class")
    return "; ".join(parts) + "."


__all__ = [
    "clock_hours",
    "grid_families",
    "grid_target_hours",
    "proposal_summary",
    "propose_from_grid",
    "read_time_grids",
]
