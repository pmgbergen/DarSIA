"""A shared production time grid, separate from saved calibration choices."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tomllib
import uuid

FORMAT = "fluidflower-production-image-grid-v1"
KEY = "production_image_selection"


def _log_times(first, last, count):
    if count == 1:
        return [float(last)]
    return [round(math.exp(math.log(first) + index * math.log(last / first) / (count - 1)), 9)
            for index in range(count)]


def propose_production_grid(metadata, *, main_count=80, through_h=48., late_count=10,
                            first_h=.25, quality=None):
    """Match one series-wide clock; a short run never gets a stretched grid.

    Counts refer to post-injection targets. The baseline is additional. Late
    targets span the longest recorded run; other runs retain explicit missing
    targets once their own photographs end. Existing camera tolerances and
    saved quality replacements remain authoritative.
    """
    from .image_selection_grid import propose_from_grid
    from .image_selection_proposal import eligible_frames
    from .image_selection_plan import resolve_quality_time_grid
    if not metadata or type(main_count) is not int or main_count < 2:
        raise ValueError("Choose runs and at least two production targets")
    if type(late_count) is not int or late_count < 0:
        raise ValueError("The late-image limit must be a nonnegative integer")
    if (not math.isfinite(first_h) or not math.isfinite(through_h)
            or not 0 < first_h < through_h):
        raise ValueError("Production times require 0 < first time < end of the main interval")
    available = {run: eligible_frames(value) for run, value in metadata.items()}
    if any(not rows for rows in available.values()):
        raise ValueError("Every production run needs protocol-declared eligible photographs")
    latest = max(row["elapsed_h"] for rows in available.values() for row in rows)
    main = _log_times(first_h, through_h, main_count)
    late = _log_times(through_h, latest, late_count + 1)[1:] if latest > through_h and late_count else []
    grids = {"production_main": dict(times_h=main), "production_late": dict(times_h=late)}
    results = {}
    for run, value in metadata.items():
        if str(value.run_id) != run:
            raise ValueError("Production metadata belongs to another run")
        saved_quality = (quality or {}).get(run)
        result = propose_from_grid(value, grids, list(grids), quality=saved_quality)
        classes = (saved_quality or {}).get("classes", {})
        def classify(frame):
            key = str(frame["source_path"]).replace("\\", "/").casefold()
            return dict(classes[key]) if key in classes else {"class": "ok", "recorded": False}
        # Camera tolerance must not move a main target to an image after the
        # main interval and accidentally exceed the separate late-image cap.
        blocks = []
        for targets, accept in ((main, lambda time: 0 <= time <= through_h),
                                (late, lambda time: time > through_h)):
            if targets:
                blocks.append(resolve_quality_time_grid(
                    [frame for frame in value.frames if accept(frame["elapsed_h"])],
                    targets, quality=classify))
        result["target_matches"] = [row for block in blocks for row in block["target_matches"]]
        combined = {row["source_path"]: row for block in blocks for row in block["selected"]}
        result["selected"] = sorted(combined.values(), key=lambda row: (row["elapsed_h"], row["input_index"]))
        original_indices = {row["source_path"]: row["input_index"] for row in available[run]}
        for row in result["selected"]:
            row["input_index"] = original_indices[row["source_path"]]
        end = max(row["elapsed_h"] for row in available[run])
        for match in result["target_matches"]:
            if match["target_hour"] > end + 1e-9:
                match.update(selected=False, reason="after_end_of_run")
        keys = {row["source_path"] for row in result["target_matches"] if row["selected"]}
        keys.add(result["baseline_path"])
        result["selected"] = [row for row in result["selected"] if row["source_path"] in keys]
        positions = {row["source_path"]: index for index, row in enumerate(result["selected"])}
        for row in result["selected"]:
            row["selection_index"] = positions[row["source_path"]]
        for match in result["target_matches"]:
            if match.get("source_path") in original_indices:
                match["input_index"] = original_indices[match["source_path"]]
            match["selection_index"] = positions.get(match.get("source_path")) if match["selected"] else None
        result["selected_paths"] = [result["baseline_path"], *[row["source_path"]
            for row in result["selected"] if row["source_path"] != result["baseline_path"]]]
        result["missing_hours"] = [row["target_hour"] for row in result["target_matches"] if not row["selected"]]
        result["unrecorded_quality"] = sum(str(path).replace("\\", "/").casefold() not in classes
                                          for path in result["selected_paths"])
        result["counts"] = dict(baselines=1,
            main=len({row["source_path"] for row in result["target_matches"]
                      if row["selected"] and row["target_hour"] <= through_h}),
            late=len({row["source_path"] for row in result["target_matches"]
                      if row["selected"] and row["target_hour"] > through_h}),
            selected=len(result["selected_paths"]), missing=len(result["missing_hours"]))
        results[run] = result
    return dict(format=FORMAT, purpose="production", main_target_hours=main, late_target_hours=late,
        parameters=dict(main_count=main_count, through_h=through_h, late_count=late_count, first_h=first_h),
        spacing="one shared logarithmic clock; no per-run stretching", runs=results)


def save_production_grid(root, proposal, metadata, *, expected_bytes, actor="", quality=None):
    """Save one immutable production choice; calibration rows/configs stay intact."""
    from .series_image_selection import _descriptor, _map_paths, _stored, selected_frame_rows
    from .series_method import _write_manifest, ManifestChangedError
    from .series_paths import series_folder_path
    from .storage_paths import machine_stored_text
    root = Path(root).resolve()
    path = root / "series.toml"
    if path.read_bytes() != expected_bytes:
        raise ManifestChangedError("Series choices changed; propose the production grid again")
    document = tomllib.loads(expected_bytes.decode("utf-8-sig"))
    if proposal.get("format") != FORMAT or set(metadata) != set(proposal["runs"]):
        raise ValueError("A complete inspected production-grid proposal is required")
    if propose_production_grid(metadata, quality=quality, **proposal["parameters"]) != proposal:
        raise ManifestChangedError("The production proposal changed; inspect it again before saving")
    if set(metadata) != set(document["runs"]["ids"]):
        raise ValueError("The production grid must cover the registered runs")
    records, verified_sources = {}, set()
    for run, value in metadata.items():
        if Path(value.series_root).resolve() != root:
            raise ValueError("Production metadata belongs to another series")
        sources = [dict(pin) for pin in value.sources if Path(pin["path"]).resolve() != path]
        if not sources:
            raise ManifestChangedError("Production protocol metadata changed; propose again")
        for pin in sources:
            key = (pin["path"], pin["sha256"], pin["size_bytes"])
            if key not in verified_sources:
                if _descriptor(pin["path"]) != pin:
                    raise ManifestChangedError("Production protocol metadata changed; propose again")
                verified_sources.add(key)
        selected = selected_frame_rows(value, proposal["runs"][run]["selected_paths"])
        records[run] = dict(selected=selected, source_metadata=sources,
            quality_sources=[(quality or {})[run]["source"]]
                if (quality or {}).get(run, {}).get("source") else [],
            baseline_path=machine_stored_text(proposal["runs"][run]["baseline_path"]),
            target_matches=proposal["runs"][run]["target_matches"], counts=proposal["runs"][run]["counts"])
    record = dict(proposal, runs=records, series_id=document["series_id"],
        created_utc=datetime.now(timezone.utc).isoformat(), created_by=actor,
        calibration_choices_changed=False, execution_plan_created=False)
    record = _map_paths(record, lambda value: _stored(value, root))
    raw = json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    folder = series_folder_path(root, "image_selection") / "production"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / (uuid.uuid4().hex + ".json")
    with output.open("xb") as stream:
        stream.write(raw)
    bound = dict(path=_stored(str(output), root), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    updated = deepcopy(document)
    updated[KEY] = dict(format=FORMAT, selection_manifest=bound, parameters=deepcopy(proposal["parameters"]))
    _write_manifest(path, expected_bytes, updated)
    return dict(manifest=bound, series_bytes=path.read_bytes(), proposal=record)


def resolve_production_selection(root, run, protocols, *, check_files=False):
    """Resolve the production purpose without replacing the calibration resolver."""
    from .series_image_selection import (_descriptor, _map_paths, _located,
        selected_frame_rows, _comparable)
    from .storage_paths import storage_path_remapper
    from .series_method import ManifestChangedError
    root = Path(root).resolve()
    remap = storage_path_remapper()
    raw = (root / "series.toml").read_bytes()
    document = tomllib.loads(raw.decode("utf-8-sig"))
    choice = document.get(KEY)
    if not choice:
        raise ValueError("Save the production image grid in Image selection first")
    pin = choice["selection_manifest"]
    path = Path(_located(pin["path"], root, remap))
    actual = _descriptor(path)
    if any(actual[key] != pin.get(key) for key in ("sha256", "size_bytes")):
        raise ManifestChangedError("The saved production grid changed")
    saved = _map_paths(json.loads(path.read_text(encoding="utf-8-sig")),
        lambda value: _located(value, root, remap))
    if (saved.get("format") != FORMAT or saved.get("series_id") != document["series_id"]
            or run not in document["runs"]["ids"] or protocols.run_id != run
            or Path(protocols.series_root).resolve() != root):
        raise ValueError("Production choices belong to another series or run")
    row = saved["runs"][run]
    if any(_descriptor(pin["path"]) != pin for pin in row.get("quality_sources", [])):
        raise ManifestChangedError("The production image-quality receipt changed")
    sources = [dict(value) for value in protocols.sources
               if Path(value["path"]).resolve() != root / "series.toml"]
    if (_comparable(sources, root, remap) != _comparable(row["source_metadata"], root, remap)
            or any(_descriptor(value["path"]) != value for value in sources)):
        raise ManifestChangedError("Production protocol metadata changed; propose the grid again")
    selected = selected_frame_rows(protocols, [value["source_path"] for value in row["selected"]])
    if _comparable(selected, root, remap) != _comparable(row["selected"], root, remap):
        raise ManifestChangedError("Production image identities or times changed")
    if check_files:
        for value in selected:
            if not Path(value["source_path"]).is_file():
                raise FileNotFoundError(value["source_path"])
    return dict(format="fluidflower-resolved-series-image-selection-v1", purpose="production",
        series_id=document["series_id"], run_id=run, manifest=_descriptor(root / "series.toml"),
        production_manifest=dict(pin, path=str(path)), source_metadata=sources,
        selected=selected, selected_frame_count=len(selected), check_files=bool(check_files))


def production_grid_targets(root, resolved):
    """The saved grid's baseline and target matches for one resolved run.

    ``resolved`` is resolve_production_selection's result. The grid file it
    verified is read again and must still have its pinned bytes. Returns the
    run's ``baseline_path`` and ``target_matches`` (each target time with its
    matched image), with paths as this machine reads them.
    """
    from .series_image_selection import _descriptor, _map_paths, _located
    from .storage_paths import storage_path_remapper
    from .series_method import ManifestChangedError
    root = Path(root).resolve()
    remap = storage_path_remapper()
    pin = resolved["production_manifest"]
    if _descriptor(pin["path"]) != pin:
        raise ManifestChangedError("The saved production grid changed")
    saved = _map_paths(json.loads(Path(pin["path"]).read_text(encoding="utf-8-sig")),
        lambda value: _located(value, root, remap))
    row = saved["runs"][resolved["run_id"]]
    return dict(baseline_path=_located(row["baseline_path"], root, remap),
                target_matches=deepcopy(row["target_matches"]))


def validate_production_phase_inputs(root, inputs, *, reader=None, metadata_reader=None):
    """Give early feedback when selected phase maps use another production clock.

    Original job creation still verifies full acquisition identities, geometry,
    phase completion and calibration. No old completion is modified or reused
    for newly requested photographs.
    """
    path = Path(root) / "series.toml"
    if not path.is_file() or KEY not in tomllib.loads(path.read_text(encoding="utf-8-sig")):
        return None
    from .pipeline.current_inputs import current_input_reader
    from .series_image_selection import load_run_protocol_metadata
    reader = current_input_reader(reader)
    metadata_reader = metadata_reader or load_run_protocol_metadata
    document = reader.json(inputs)
    for row in document["runs"]:
        run = row["run"]
        expected = resolve_production_selection(root, run, metadata_reader(root, run))
        config = reader.json(row["configuration_bound"])
        times = [value["elapsed_h"] for value in config["frames"]]
        selected = [value["elapsed_h"] for value in expected["selected"]]
        if times not in (selected, selected[1:]):
            raise ValueError("The saved phase maps use another image selection for " + run
                + ". Prepare phase maps for the saved production grid before preparing production.")
    return True
