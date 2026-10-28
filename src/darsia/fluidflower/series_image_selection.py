"""Series-owned image choices; metadata reads and explicit saves only.

The existing target-hour resolver owns nearest-frame, blacklist, duplicate and
late-coverage semantics. This module never changes that resolver or numerical
phase/mass code. Protocol readers are the original DarSIA classes.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tomllib
import uuid

import toml

from darsia.fluidflower.image_selection_plan import _eligible_frames, resolve_selection
from darsia.fluidflower.series_method import ManifestChangedError, _same_values
from darsia.fluidflower.series_paths import series_key
from darsia.fluidflower.storage_paths import machine_stored_text, storage_path_remapper
from darsia.fluidflower.image_selection_proposal import (
    default20_proposal,
    first_usable_rule,
    load_saved_quality,
    load_stored_native_plan,
)
from darsia.presets.workflows.config.utils import _is_relative_path


FORMAT = "fluidflower-series-image-selection-v1"
# Explicit draft interpretation. Root must obtain the count clarification before
# enabling the suggestion action; there is deliberately no implicit default.
DRAFT_EARLY_BLOCK = "eight_includes_preinjection"


def _finite(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number.")
    return float(value)


def _count(value, label):
    if type(value) is not int or value < 2:
        raise ValueError(f"{label} must be an integer of at least 2.")
    return value


def _key(path):
    return str(path).replace("\\", "/").casefold()


# Saved fields that name files: descriptors, frame identities and provenance.
_PATH_FIELDS = frozenset({"path", "source_path", "protocol_source_path"})


def _located(value, root, remap):
    """Return a saved or configured path as this machine reads it.

    A path stored relative to the series root (new saves) is joined to
    ``root``. An absolute path passes through ``remap`` (a
    ``storage_path_remapper()``); without a matching alias its text is
    unchanged, so an old absolute selection reads exactly as before.
    """
    text = str(value)
    if _is_relative_path(text):
        return remap(str(Path(root) / text))
    return remap(text)


def _stored(value, root):
    """Return the portable text a new save writes for a path.

    A path inside the series root is stored relative to it with ``/``
    separators, so a moved or cloned project still validates; any other path
    is stored absolute, as given, or in its shared spelling on a computer with
    a machine path map (``storage_paths.machine_stored_text``).
    """
    text = str(value)
    if _is_relative_path(text):
        return text
    try:
        relative = Path(text).relative_to(Path(root))
    except ValueError:
        return machine_stored_text(text)
    return relative.as_posix() if relative.parts else machine_stored_text(text)


def _map_paths(value, convert):
    """Return a copy of a saved structure with ``convert`` applied to path fields."""
    if isinstance(value, dict):
        return {
            key: (
                convert(item)
                if key in _PATH_FIELDS and isinstance(item, str)
                else _map_paths(item, convert)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_map_paths(item, convert) for item in value]
    return value


def _comparable(value, root, remap):
    """Return a saved structure with path fields reduced to location keys.

    Saved and freshly read metadata then compare equal whether a path was
    stored relative, absolute or under another aliased spelling.
    """
    return _map_paths(value, lambda path: _key(_located(path, root, remap)))


def _descriptor(path):
    raw = Path(path).read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


@dataclass(frozen=True)
class RunImageMetadata:
    run_id: str
    frames: tuple[dict, ...]
    injection_start_h: float
    injection_end_h: float
    sources: tuple[dict, ...] = ()
    notes: tuple[str, ...] = ()
    series_root: str = ""


def _deep_merge(base, update):
    # Match native config layering without loading raw data or unrelated sections.
    from darsia.presets.workflows.config.utils import _deep_merge as original_merge

    return original_merge(base, update)


def _path(value, root, remap):
    # Join series-relative locators before applying the shared storage aliases.
    path = Path(str(value))
    return Path(remap(str(path if path.is_absolute() else Path(root) / path)))


def _protocol_path(value, root, remap):
    if isinstance(value, tuple):
        return (_path(value[0], root, remap), value[1])
    return _path(value, root, remap)


def _source_file(value):
    return value[0] if isinstance(value, tuple) else value


def load_run_protocol_metadata(series_root, run_id, *, config_paths=None):
    """Read exactly this run's declared TOMLs/CSV/XLS metadata; no file scan.

    ``config_paths`` is useful for the native adapter's actual selected config
    chain and for isolated tests. Without it, the original series config_files
    ordering and colour-state branch are used.
    """
    from darsia import ImagingProtocol, InjectionProtocol
    from darsia.fluidflower.series_paths import (
        config_files,
        series_root as registered_series_root,
    )
    from darsia.presets.workflows.config.protocols import ProtocolsConfig

    root = Path(series_root).resolve()
    manifest_path = root / "series.toml"
    manifest = tomllib.loads(manifest_path.read_bytes().decode("utf-8-sig"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Image selection requires a version-1 series manifest.")
    run = str(run_id).strip().casefold()
    if run not in manifest.get("runs", {}).get("ids", []):
        raise ValueError(f"Run {run!r} is not in this series manifest.")
    if (
        config_paths is None
        and registered_series_root(manifest["series_id"]).resolve() != root
    ):
        raise ValueError(
            "The registered series folder differs from the selected manifest."
        )
    paths = (
        list(config_paths)
        if config_paths is not None
        else config_files(manifest["series_id"], run)
    )
    if not paths:
        raise ValueError("No saved configuration is supplied for this run.")
    merged = {}
    sources = [_descriptor(manifest_path)]
    for path in paths:
        path = Path(path)
        sources.append(_descriptor(path))
        _deep_merge(merged, tomllib.loads(path.read_bytes().decode("utf-8-sig")))
    data = merged.get("data") or {}
    folders = []
    if value := data.get("folder"):
        folders.append(value)
    for value in data.get("folders") or []:
        if value not in folders:
            folders.append(value)
    if not folders:
        raise ValueError("The run has no saved data folders.")
    protocols = ProtocolsConfig()
    protocols.load_imaging(merged.get("protocols") or {})
    protocols.load_injection(merged.get("protocols") or {})
    if not protocols.imaging or {_key(p) for p in protocols.imaging} != {
        _key(p) for p in folders
    }:
        raise ValueError(
            "Imaging protocols must cover every saved data folder exactly."
        )
    if protocols.injection is None:
        raise ValueError(
            "Choose the saved injection protocol before suggesting image times."
        )
    remap = storage_path_remapper()  # The registry is read once per load.
    injection_source = _protocol_path(protocols.injection, root, remap)
    sources.append(_descriptor(_source_file(injection_source)))
    injection = InjectionProtocol(injection_source)
    active = injection.df[injection.df["rate_kg_s"] > 0]
    if active.empty:
        raise ValueError("The saved injection protocol has no positive-rate interval.")
    start, end = active["start"].min(), active["end"].max()
    if not end > start:
        raise ValueError("The saved injection end must be later than its start.")
    notes = []
    intervals = sorted(zip(active["start"], active["end"]))
    covered_end = intervals[0][1]
    for left, right in intervals[1:]:
        if left > covered_end:
            notes.append(
                "Injection contains separate windows; the suggestion spans first start to final end."
            )
        covered_end = max(covered_end, right)
    frames = []
    for folder_value in folders:
        folder = _path(folder_value, root, remap)
        configured_folder = next(
            p for p in protocols.imaging if _key(p) == _key(folder_value)
        )
        source = _protocol_path(protocols.imaging[configured_folder], root, remap)
        blacklist = protocols.blacklist
        if isinstance(blacklist, dict):
            blacklist = next(
                (v for p, v in blacklist.items() if _key(p) == _key(folder_value)), None
            )
        blacklist = (
            _protocol_path(blacklist, root, remap) if blacklist is not None else None
        )
        sources.append(_descriptor(_source_file(source)))
        if blacklist is not None:
            sources.append(_descriptor(_source_file(blacklist)))
        imaging = ImagingProtocol(source, blacklist=blacklist)
        for row in imaging.df.itertuples(index=False):
            row_key = imaging._normalize_protocol_path(str(row.path))
            if row_key.lower() in {"nan", "none", ""}:
                # Same missing-path omission as the original reader's lookup.
                continue
            folder_key = imaging._normalize_protocol_path(str(folder)).rstrip("/")
            # Original ImagingProtocol normalizes leading slashes in its keys.
            # Recover a declared full UNC source through this declared folder,
            # rather than treating the normalized host name as a relative file.
            path = (
                folder / row_key[len(folder_key) + 1 :]
                if row_key.casefold().startswith(folder_key.casefold() + "/")
                else _path(row.path, folder, remap)
            )
            # Datetime conversion/blacklist parsing stay with original readers.
            elapsed_h = (row.datetime - start).total_seconds() / 3600.0
            if not math.isfinite(elapsed_h):
                raise ValueError("An imaging-protocol timestamp is not finite.")
            frames.append(
                {
                    "source_path": str(path),
                    "image_id": path.name,
                    "frame_index": len(frames),
                    "elapsed_h": elapsed_h,
                    "blacklisted": bool(imaging.is_blacklisted(path)),
                    "camera_time": row.datetime.isoformat(),
                }
            )
    eligible, omitted = _eligible_frames(frames, check_files=False, source_root=None)
    if any(
        row["reason"] == "duplicate_source_path_conflicting_time" for row in omitted
    ):
        raise ValueError(
            "The same source image has conflicting protocol acquisition times."
        )
    if not eligible:
        raise ValueError(
            "No eligible images are declared by the saved imaging protocol."
        )
    sources = list({_key(s["path"]): s for s in sources}.values())
    for source in sources:
        if _descriptor(source["path"]) != source:
            raise ManifestChangedError(
                "Source metadata changed while it was read; load the run again."
            )
    return RunImageMetadata(
        run,
        tuple(frames),
        0.0,
        (end - start).total_seconds() / 3600.0,
        tuple(sources),
        tuple(dict.fromkeys(notes)),
        str(root),
    )


def suggest_target_hours(metadata, *, early_count=8, rest_count=11, interpretation=""):
    """Q6 proposal targets only; never selects/applies the historical Studio20."""
    from darsia.fluidflower.image_selection_proposal import RULE

    if (
        type(early_count) is not int
        or early_count != 8
        or type(rest_count) is not int
        or rest_count != 11
    ):
        raise ValueError("Default20 has fixed counts 1 + 8 + 11.")
    if interpretation not in {"", RULE}:
        raise ValueError(
            "The historical count interpretation is not the Q6 Default20 rule."
        )
    return [
        row["target_time_h"] for row in default20_proposal(metadata)["target_matches"]
    ]


def preview_targets(metadata, targets, *, tolerance_seconds):
    """Delegate target matching/late coverage unchanged to the existing resolver."""
    targets = [_finite(t, "Target hour") for t in targets]
    receipt = resolve_selection(
        {"run": metadata.run_id, "frames": list(metadata.frames)},
        {
            "schema_version": 1,
            "purposes": {
                "review": {
                    "mode": "target_hours",
                    "target_hours": targets,
                    "tolerance_seconds": tolerance_seconds,
                    "on_unmatched_target": "error",
                }
            },
        },
        "review",
    )
    receipt["requested_frame_count"] = len(targets)
    receipt["distinct_selected_frame_count"] = receipt["runs"][0][
        "selected_frame_count"
    ]
    receipt["selection_notes"] = list(metadata.notes)
    if receipt["distinct_selected_frame_count"] != len(targets):
        receipt["selection_notes"].append(
            "Some targets do not give distinct images; edit targets or add/remove actual frames."
        )
    return receipt


def selected_frame_rows(metadata, source_paths):
    """Validate manual choices against eligible protocol identities, then order."""
    if not isinstance(source_paths, (list, tuple)):
        raise ValueError("Selected source paths must be a list.")
    eligible, omitted = _eligible_frames(
        list(metadata.frames), check_files=False, source_root=None
    )
    if any(
        row["reason"] == "duplicate_source_path_conflicting_time" for row in omitted
    ):
        raise ValueError("The protocol contains conflicting duplicate source times.")
    available = {_key(row["source_path"]): row for row in eligible}
    selected = set()
    for path in source_paths:
        if not isinstance(path, str) or _key(path) not in available:
            raise ValueError(
                f"Selected image is not an eligible protocol source: {path!r}"
            )
        if _key(path) in selected:
            raise ValueError("The same source image cannot be selected twice.")
        selected.add(_key(path))
    return [deepcopy(row) for row in eligible if _key(row["source_path"]) in selected]


def _validate_saved_table(table, allowed_runs):
    if (
        not isinstance(table, dict)
        or table.get("format") != FORMAT
        or type(table.get("schema_version")) is not int
        or table["schema_version"] != 1
    ):
        raise ValueError("The saved image-selection table has an unsupported format.")
    runs = table.get("runs")
    if not isinstance(runs, dict):
        raise ValueError("The saved image-selection runs must be a table.")
    for run, value in runs.items():
        if run not in allowed_runs or not isinstance(value, dict):
            raise ValueError("An image-selection run is not registered in this series.")
        rows = value.get("selected")
        if not isinstance(rows, list):
            raise ValueError("Saved selected images must be an array of tables.")
        seen = set()
        for row in rows:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("source_path"), str)
                or not row["source_path"]
            ):
                raise ValueError("Every selected image needs its source path.")
            if (
                not isinstance(row.get("image_id"), str)
                or not row["image_id"]
                or type(row.get("frame_index")) is not int
                or row["frame_index"] < 0
            ):
                raise ValueError(
                    "Every saved image needs its exact protocol image/index identity."
                )
            _finite(row.get("elapsed_h"), "Saved image time")
            if _key(row["source_path"]) in seen:
                raise ValueError("Duplicate source image in saved selection.")
            seen.add(_key(row["source_path"]))
    return table


def read_series_image_selection(root):
    """Return the saved choices with every path as this machine reads it.

    Paths stored relative to the series root are joined to it and absolute
    ones follow the storage aliases; ``raw`` stays the exact file bytes.
    """
    path = Path(root) / "series.toml"
    raw = path.read_bytes()
    document = tomllib.loads(raw.decode("utf-8-sig"))
    run_section = document.get("runs")
    allowed = run_section.get("ids") if isinstance(run_section, dict) else None
    if (
        type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
        or not isinstance(document.get("series_id"), str)
        or not document["series_id"].strip()
        or not isinstance(allowed, list)
        or not allowed
        or any(not isinstance(run, str) or not run for run in allowed)
        or len(allowed) != len(set(allowed))
    ):
        raise ValueError(
            "Image selection requires a version-1 series with distinct registered runs."
        )
    table = document.get("image_selection")
    if table is None:
        table = {"format": FORMAT, "schema_version": 1, "runs": {}}
    base = Path(root).resolve()
    remap = storage_path_remapper()
    table = _map_paths(table, lambda value: _located(value, base, remap))
    return deepcopy(_validate_saved_table(table, allowed)), raw


def _proposal_background(metadata, quality, value):
    """Keep an applied proposal's original target history bound to its sources."""
    if value is None:
        return None
    keys = ("rule", "parameters", "target_matches", "no_preinjection_image")
    root, remap = metadata.series_root, storage_path_remapper()
    try:
        fresh = default20_proposal(
            metadata, d1_h=value["parameters"]["d1_h"], quality=quality
        )
        if any(
            _comparable(value[key], root, remap) != _comparable(fresh[key], root, remap)
            for key in keys
        ):
            raise ManifestChangedError(
                "The applied Default20 proposal no longer matches current source metadata."
            )
    except (KeyError, TypeError) as error:
        raise ManifestChangedError(
            "The applied Default20 proposal is incomplete."
        ) from error
    return {key: deepcopy(fresh[key]) for key in keys}


def _selection_frame_records(metadata, selected, quality, proposal):
    """Separate the saved camera times from the targets that selected them."""
    declared = {_key(row["source_path"]): row for row in metadata.frames}
    matches = {}
    for match in (proposal or {}).get("target_matches", []):
        if not match.get("selected"):
            continue
        identity = match["chosen_image"]
        key = _key(identity["source_path"])
        if key in matches:
            raise ManifestChangedError(
                "An applied proposal assigns one image more than once."
            )
        matches[key] = match
    records = []
    for row in selected:
        key = _key(row["source_path"])
        identity = {
            field: row[field] for field in ("source_path", "image_id", "frame_index")
        }
        record = {
            "target_time_h": row["elapsed_h"],
            "chosen_image": identity,
            "camera_time": str(declared[key].get("camera_time") or "Not recorded"),
            "camera_elapsed_h": row["elapsed_h"],
            "data_time_h": row["elapsed_h"],
            "offset_seconds": 0.0,
            "frame_list_class": str(
                (quality or {}).get("classes", {}).get(key, {}).get("class")
                or declared[key].get("frame_list_class")
                or "Not recorded"
            ),
            "selection_origin": "manual",
            "replacement": "No automatic replacement; explicit manual choice",
            "flags": [],
        }
        match = matches.get(key)
        if match is not None:
            if (
                match["chosen_image"] != identity
                or match["camera_elapsed_h"] != row["elapsed_h"]
            ):
                raise ManifestChangedError(
                    "The applied proposal image identity/time differs from the selected image."
                )
            record.update(
                target_time_h=match["target_time_h"],
                offset_seconds=match["offset_seconds"],
                frame_list_class=match["frame_list_class"],
                selection_origin="default20_proposal",
                replacement="; ".join(
                    flag
                    for flag in match["flags"]
                    if flag.startswith("duplicate target:")
                    or flag.startswith("reject replaced")
                )
                or "No replacement; applied Default20 camera choice",
                flags=deepcopy(match["flags"]),
            )
            for field in (
                "target_index",
                "tolerance_seconds",
                "replacement_tolerance_seconds",
                "rejected_image",
            ):
                if field in match:
                    record[field] = deepcopy(match[field])
        records.append(record)
    return records


def _stored_plan_background(value):
    if value is None:
        return None
    return {
        key: deepcopy(value[key])
        for key in ("source", "time_basis", "native_source_identities")
    }


def resolve_saved_selection(root, run, protocols, *, check_files=False):
    """Validate saved explicit choices for a future review/job UI consumer.

    This does not launch a job, alter native manifests, select a scientific
    calibration, or make any existing consumer use this table automatically.
    ``protocols`` is the current ``RunImageMetadata`` from the original readers.
    """
    root = Path(root).resolve()
    if not protocols.series_root or Path(protocols.series_root).resolve() != root:
        raise ValueError("Image metadata belongs to another series folder.")
    run = str(run).strip().casefold()
    if protocols.run_id != run:
        raise ValueError("Loaded protocol metadata belongs to another run.")
    remap = storage_path_remapper()
    table, raw = read_series_image_selection(root)
    saved = table["runs"].get(run)
    if not saved or not saved.get("selected"):
        raise ValueError("No explicit saved image choices exist for this run.")
    sources = saved.get("source_metadata")
    current = {
        _key(s["path"]): s
        for s in protocols.sources
        if Path(s["path"]).resolve() != (root / "series.toml").resolve()
    }
    if not isinstance(sources, list) or not sources or not current:
        raise ValueError(
            "Saved image choices lack their protocol/config source binding."
        )
    if (
        any(
            not isinstance(source, dict) or not isinstance(source.get("path"), str)
            for source in sources
        )
        or len({_key(source["path"]) for source in sources}) != len(sources)
        or len(sources) != len(current)
        or any(
            _comparable(current.get(_key(source["path"])), root, remap)
            != _comparable(source, root, remap)
            for source in sources
        )
    ):
        raise ManifestChangedError(
            "Saved choices were made from different protocol/config metadata; edit and save them again."
        )
    for source in sources:
        if _descriptor(source["path"]) != source:
            raise ManifestChangedError(
                "Source metadata changed after this run was loaded."
            )
    selected = selected_frame_rows(
        protocols, [r["source_path"] for r in saved["selected"]]
    )
    original = {_key(r["source_path"]): r for r in saved["selected"]}
    for row in selected:
        saved_row = original[_key(row["source_path"])]
        if any(
            row[key] != saved_row[key]
            for key in ("elapsed_h", "image_id", "frame_index")
        ):
            raise ManifestChangedError(
                "A selected image has a different protocol identity/time; edit and save again."
            )
        if check_files and not Path(row["source_path"]).is_file():
            raise FileNotFoundError(row["source_path"])
    supplemental = saved.get("supplemental_source_metadata", [])
    if not isinstance(supplemental, list):
        raise ValueError("Imported selection metadata bindings must be a list.")
    for source in supplemental:
        if _descriptor(source["path"]) != source:
            raise ManifestChangedError("Imported selection/quality metadata changed.")
    quality = (
        load_saved_quality(
            saved["quality_source"]["path"], protocols, expected=saved["quality_source"]
        )
        if saved.get("quality_source")
        else None
    )
    stored_plan = None
    if saved.get("stored_plan_source"):
        stored_plan = load_stored_native_plan(
            saved["stored_plan_source"]["path"],
            protocols,
            series_id=tomllib.loads(raw.decode("utf-8-sig"))["series_id"],
            expected=saved["stored_plan_source"],
        )
    no_pre = first_usable_rule(protocols, selected, quality)
    if bool(saved.get("no_preinjection_image", False)) != no_pre:
        raise ManifestChangedError(
            "The saved pre-injection availability flag differs from current metadata."
        )
    bound = saved.get("selection_manifest")
    if not isinstance(bound, dict) or not isinstance(bound.get("path"), str):
        raise ValueError("Saved image choices lack their per-run selection manifest.")
    from .series_paths import series_folder_path
    if (
        Path(bound["path"]).resolve().parent
        != (series_folder_path(root, "image_selection") / "manifests" / run).resolve()
    ):
        raise ValueError(
            "Selection manifest is outside the current run manifest folder."
        )
    manifest_raw = Path(bound["path"]).read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != bound.get("sha256") or len(
        manifest_raw
    ) != bound.get("size_bytes"):
        raise ManifestChangedError("The per-run selection manifest changed.")
    choice_manifest = json.loads(manifest_raw)
    if (
        not isinstance(choice_manifest, dict)
        or choice_manifest.get("format")
        != "fluidflower-explicit-image-selection-manifest-v1"
    ):
        raise ValueError("Unsupported per-run selection manifest.")
    background = _proposal_background(
        protocols, quality, saved.get("default20_proposal")
    )
    # The manifest keeps the path text of its save (relative inside the series
    # root, absolute outside); compare locations, not spellings.
    if _comparable(choice_manifest.get("proposal"), root, remap) != _comparable(
        background, root, remap
    ):
        raise ManifestChangedError(
            "The per-run proposal provenance differs from the saved applied proposal."
        )
    stored_background = _comparable(_stored_plan_background(stored_plan), root, remap)
    if (
        _comparable(choice_manifest.get("stored_plan"), root, remap)
        != stored_background
        or _comparable(saved.get("stored_plan_provenance"), root, remap)
        != stored_background
    ):
        raise ManifestChangedError(
            "The stored native/protocol clock provenance differs from its bound source identities."
        )
    document = tomllib.loads(raw.decode("utf-8-sig"))
    names = saved.get("data_time_names")
    entries = document.get("data_time")
    if (
        not isinstance(names, list)
        or not names
        or len(set(names)) != len(names)
        or not isinstance(entries, list)
    ):
        raise ValueError("Saved choices lack their named data_time sections.")
    selected_sections = []
    for section_name in names:
        matches = [
            e for e in entries if isinstance(e, dict) and e.get("name") == section_name
        ]
        if len(matches) != 1:
            raise ManifestChangedError(
                "A named image-selection time section is missing or duplicated."
            )
        section = matches[0]
        times = section.get("times")
        if not isinstance(times, list) or not times:
            raise ValueError("Selected data_time section has no explicit times.")
        expected_tolerance = "00:00:15" if all(t <= 3.0 for t in times) else "00:02:30"
        if section.get("tol") != expected_tolerance:
            raise ManifestChangedError("Image-selection time tolerance changed.")
        selected_sections.extend(times)
    if selected_sections != [r["elapsed_h"] for r in selected]:
        raise ManifestChangedError(
            "Named image-selection times differ from the chosen image times."
        )
    if (
        choice_manifest.get("series_id")
        != tomllib.loads(raw.decode("utf-8-sig"))["series_id"]
        or choice_manifest.get("run_id") != run
        or _comparable(choice_manifest.get("source_metadata"), root, remap)
        != _comparable(sources, root, remap)
        or _comparable(
            choice_manifest.get("supplemental_source_metadata", []), root, remap
        )
        != _comparable(supplemental, root, remap)
        or bool(choice_manifest.get("no_preinjection_image", False)) != no_pre
        or _comparable(choice_manifest.get("frames"), root, remap)
        != _comparable(
            _selection_frame_records(protocols, selected, quality, background),
            root,
            remap,
        )
    ):
        raise ManifestChangedError(
            "The per-run selection manifest belongs to different choices."
        )
    manifest_path = root / "series.toml"
    if manifest_path.read_bytes() != raw:
        raise ManifestChangedError(
            "Series manifest changed while resolving saved image choices."
        )
    document = tomllib.loads(raw.decode("utf-8-sig"))
    manifest_binding = {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }
    return {
        "format": "fluidflower-resolved-series-image-selection-v1",
        "series_id": document["series_id"],
        "run_id": run,
        "manifest": manifest_binding,
        "source_metadata": deepcopy(sources + supplemental),
        "selected": selected,
        "selected_frame_count": len(selected),
        "check_files": bool(check_files),
        "scope": "Explicit metadata selection only; no review/analysis job was started.",
    }


def save_run_image_selection(
    root,
    metadata,
    source_paths,
    *,
    expected_bytes,
    target_hours=(),
    tolerance_seconds=300.0,
    rule="manual",
    proposal=None,
    stored_plan=None,
    quality=None,
    manual_edits_after_apply=False,
    default20_d1_h=0.25,
):
    """Explicit atomic series.toml save; other runs/method/unknown fields retained.

    Stored choices are exact source identities, not evidence of computed maps.
    The source metadata binding is checked again before publication.
    """
    selected = selected_frame_rows(metadata, source_paths)
    if (
        not metadata.series_root
        or Path(metadata.series_root).resolve() != Path(root).resolve()
    ):
        raise ValueError("Image metadata belongs to another series folder.")
    if not selected:
        raise ValueError("Choose at least one eligible image before saving.")
    if _finite(tolerance_seconds, "Tolerance") < 0:
        raise ValueError("Tolerance must be nonnegative.")
    if _finite(default20_d1_h, "d1") <= 0:
        raise ValueError("d1 must be greater than zero.")
    path = Path(root) / "series.toml"
    original = path.read_bytes()
    if original != expected_bytes:
        raise ManifestChangedError(
            "Series manifest changed; reload before saving image selection."
        )
    document = tomllib.loads(original.decode("utf-8-sig"))
    supplemental = []
    if stored_plan is not None:
        stored_plan = load_stored_native_plan(
            stored_plan["source"]["path"],
            metadata,
            series_id=document["series_id"],
            expected=stored_plan["source"],
        )
        supplemental.append(deepcopy(stored_plan["source"]))
    if quality is not None:
        load_saved_quality(
            quality["source"]["path"], metadata, expected=quality["source"]
        )
        supplemental.append(deepcopy(quality["source"]))
    if proposal is not None:
        if (
            proposal.get("run_id") != metadata.run_id
            or Path(proposal.get("series_root", "")).resolve() != Path(root).resolve()
        ):
            raise ValueError("The applied proposal belongs to another series/run.")
    allowed = document.get("runs", {}).get("ids", [])
    if metadata.run_id not in allowed:
        raise ValueError("The selected run does not belong to this series.")
    for source in metadata.sources:
        if Path(source["path"]).resolve() == path.resolve():
            continue
        if _descriptor(source["path"]) != source:
            raise ManifestChangedError(
                "Source protocol/config changed; reload before saving image selection."
            )
    updated = deepcopy(document)
    table = updated.setdefault(
        "image_selection", {"format": FORMAT, "schema_version": 1, "runs": {}}
    )
    _validate_saved_table(table, allowed)
    base = Path(root).resolve()
    run = deepcopy(table["runs"].get(metadata.run_id, {}))
    # Earlier choices are read where this machine finds their files.
    remap = storage_path_remapper()
    previous = _map_paths(run, lambda value: _located(value, base, remap))
    if stored_plan is None and previous.get("stored_plan_source"):
        stored_plan = load_stored_native_plan(
            previous["stored_plan_source"]["path"],
            metadata,
            series_id=document["series_id"],
            expected=previous["stored_plan_source"],
        )
    if quality is None and previous.get("quality_source"):
        quality = load_saved_quality(
            previous["quality_source"]["path"],
            metadata,
            expected=previous["quality_source"],
        )
    supplemental = [
        deepcopy(value["source"])
        for value in (stored_plan, quality)
        if value is not None
    ]
    no_pre = first_usable_rule(metadata, selected, quality)
    proposal = _proposal_background(
        metadata,
        quality,
        proposal if proposal is not None else previous.get("default20_proposal"),
    )
    # Every explicit save adds named data_time sections in the adopted format.
    # Numeric hours preserve sub-second and negative pre-injection times exactly.
    entries = updated.setdefault("data_time", [])
    if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
        raise ValueError("Existing data_time must be an array of named tables.")
    revision = uuid.uuid4().hex
    names = []
    for part, tolerance, rows in (
        ("early", "00:00:15", [r for r in selected if r["elapsed_h"] <= 3.0]),
        ("rest", "00:02:30", [r for r in selected if r["elapsed_h"] > 3.0]),
    ):
        if rows:
            section_name = f"studio_images_{metadata.run_id}_{revision}_{part}"
            if any(e.get("name") == section_name for e in entries):
                raise ValueError("The fresh selection section name already exists.")
            entries.append(
                {
                    "name": section_name,
                    "times": [r["elapsed_h"] for r in rows],
                    "tol": tolerance,
                }
            )
            names.append(section_name)
    selection_manifest = {
        "format": "fluidflower-explicit-image-selection-manifest-v1",
        "series_id": document["series_id"],
        "run_id": metadata.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "explicit_selected_camera_times",
        "requested_target_hours": [_finite(t, "Target hour") for t in target_hours],
        "no_preinjection_image": no_pre,
        "supplemental_source_metadata": supplemental,
        "time_basis": "saved imaging protocol; no new camera-clock correction",
        "source_metadata": [
            deepcopy(x)
            for x in metadata.sources
            if Path(x["path"]).resolve() != path.resolve()
        ],
        "frames": _selection_frame_records(metadata, selected, quality, proposal),
        "scientific_computation": False,
        "execution_plan_created": False,
    }
    from .series_paths import series_folder_path
    manifest_dir = series_folder_path(base, "image_selection") / "manifests" / metadata.run_id
    selection_manifest["rule"] = str(rule)
    if proposal is not None:
        selection_manifest["proposal"] = {
            key: deepcopy(proposal[key])
            for key in ("rule", "parameters", "target_matches", "no_preinjection_image")
        }
    if stored_plan is not None:
        selection_manifest["stored_plan"] = _stored_plan_background(stored_plan)
    # New saves keep paths inside the series root relative to it (portable).
    selection_manifest = _map_paths(
        selection_manifest, lambda value: _stored(value, base)
    )
    manifest_path = manifest_dir / (revision + ".json")
    manifest_raw = (
        json.dumps(selection_manifest, ensure_ascii=False, allow_nan=False, indent=2)
        + "\n"
    ).encode("utf-8")
    run.update(
        data_time_names=names,
        selection_manifest={
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "size_bytes": len(manifest_raw),
        },
    )
    run.update(
        selected=[
            {k: row[k] for k in ("source_path", "image_id", "frame_index", "elapsed_h")}
            for row in selected
        ],
        rule=str(rule),
        target_hours=[r["elapsed_h"] for r in selected],
        requested_target_hours=[_finite(t, "Target hour") for t in target_hours],
        tolerance_seconds=_finite(tolerance_seconds, "Tolerance"),
        saved_utc=datetime.now(timezone.utc).isoformat(),
        source_metadata=[
            deepcopy(s)
            for s in metadata.sources
            if Path(s["path"]).resolve() != path.resolve()
        ],
        no_preinjection_image=no_pre,
        default20_d1_h=float(default20_d1_h),
        supplemental_source_metadata=supplemental,
    )
    if stored_plan is not None:
        run["stored_plan_source"] = deepcopy(stored_plan["source"])
        run["stored_plan_provenance"] = _stored_plan_background(stored_plan)
    if quality is not None:
        run["quality_source"] = deepcopy(quality["source"])
    if proposal is not None:
        run["default20_proposal"] = {
            key: deepcopy(proposal[key])
            for key in ("rule", "parameters", "target_matches", "no_preinjection_image")
        }
        proposed_paths = {
            _key(match["chosen_image"]["source_path"])
            for match in proposal["target_matches"]
            if match.get("selected")
        }
        run["manual_edits_after_apply"] = bool(
            manual_edits_after_apply
            or {_key(row["source_path"]) for row in selected} != proposed_paths
        )
    elif previous.get("default20_proposal") and _comparable(
        run["selected"], base, remap
    ) != _comparable(previous.get("selected"), base, remap):
        run["manual_edits_after_apply"] = True
    table["runs"][metadata.run_id] = _map_paths(run, lambda value: _stored(value, base))
    _validate_saved_table(table, allowed)
    text = toml.dumps(updated)
    if b"\r\n" in original:
        text = text.replace("\n", "\r\n")
    encoded = (
        b"\xef\xbb\xbf" if original.startswith(b"\xef\xbb\xbf") else b""
    ) + text.encode("utf-8")
    if not _same_values(updated, tomllib.loads(encoded.decode("utf-8-sig"))):
        raise ValueError("Saving image choices would change other TOML values.")
    if path.read_bytes() != original:
        raise ManifestChangedError(
            "Series manifest changed before publishing image choices."
        )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("xb") as stream:
        stream.write(manifest_raw)
    # A later CAS failure preserves this immutable manifest as well as all old data.
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
        if path.read_bytes() != original:
            raise ManifestChangedError(
                "Series manifest changed during the image-selection save."
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _map_paths(table, lambda value: _located(value, base, remap)), encoded
