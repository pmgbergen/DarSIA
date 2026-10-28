"""Read retained optical and chemical inputs for current Build statistics.

Only metadata is decoded here. Current physical fields, geometry and the
photographic coordinate binding belong to the caller's actual Build products.
"""
from __future__ import annotations

from copy import deepcopy
import math
import ntpath
from pathlib import Path
import re
import tomllib

from ..mass.n1_pipeline import DataReader, asset
from .staged_data import STATISTICS_FORMAT


def _key(path):
    return ntpath.normcase(ntpath.normpath(str(path)))


def _same_binding(first, second):
    return (_key(first["path"]) == _key(second["path"])
            and first["sha256"] == second["sha256"])


def _run_row(rows, run, description):
    rows = list(rows.values()) if isinstance(rows, dict) else rows
    found = [row for row in rows if row.get("run") == run]
    if len(found) != 1:
        raise ValueError(f"Exactly one retained {description} is required for {run}")
    return deepcopy(found[0])


def _finite(value, description, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (positive and value <= 0)):
        raise ValueError(f"Retained {description} must be finite")
    return value


def _sized(binding, reader):
    """Add a byte count without decoding or rehashing a scientific array.

    Its digest remains the original declared digest. The ordinary data reader
    verifies that digest when the worker actually consumes the payload.
    """
    result = deepcopy(binding)
    if hasattr(reader, "physical_descriptor"):
        result = reader.physical_descriptor(result)
    if "size_bytes" not in result:
        physical = result
        io_binding = getattr(reader, "io_binding", None)
        if io_binding is not None:
            physical = io_binding.physical_descriptor(result)
        else:
            physical = getattr(reader, "aliases", {}).get(_key(result["path"]), result)
        if physical["sha256"] != result["sha256"]:
            raise ValueError("A retained input alias changes its source digest")
        result["size_bytes"] = Path(physical["path"]).stat().st_size
    return result


def _optical_inputs(freeze, reader):
    root = _key(freeze["path_folder"])
    result = {"color_paths": {}, "signals": {}}
    for path, value in freeze["calibration_files"].items():
        relative = ntpath.relpath(_key(path), root)
        if relative.startswith(".."):
            continue
        bound = dict(value, path=path) if isinstance(value, dict) else dict(path=path, sha256=value)
        if relative == "metadata.json":
            result["metadata"] = _sized(bound, reader)
        elif relative == "flash\\flash.json":
            result["flash"] = _sized(bound, reader)
        else:
            match = re.fullmatch(r"color_path_interpretation\\color_path_interpretation_(\d+)\.json", relative)
            group = "color_paths"
            if match is None:
                match = re.fullmatch(r"signal_model\\signal_model_(\d+)\.csv", relative)
                group = "signals"
            if match is not None:
                label = match.group(1)
                if label in result[group]:
                    raise ValueError("The retained optical model repeats a label")
                result[group][label] = _sized(bound, reader)
    if (not {"metadata", "flash"} <= result.keys() or not result["color_paths"]
            or result["color_paths"].keys() != result["signals"].keys()):
        raise ValueError("The retained optical model lacks matching colour and signal paths")
    return result


def _flash_bounds(freeze, reader, sources):
    """Read only the original scalar flash rule, not its colour-path payloads."""
    root = _key(freeze["path_folder"])
    matches = [(path, value) for path, value in freeze["calibration_files"].items()
               if ntpath.relpath(_key(path), root) == "flash\\flash.json"]
    if len(matches) != 1:
        raise ValueError("The retained method needs one explicit flash rule")
    path, value = matches[0]
    pin = _sized(dict(value, path=path) if isinstance(value, dict)
                 else dict(path=path, sha256=value), reader)
    data = reader.json(pin)
    sources.append(pin)
    bounds = {name: _finite(data[name], name) for name in
              ("min_value_aq", "max_value_aq", "min_value_g", "max_value_g")}
    if bounds["max_value_aq"] <= bounds["min_value_aq"] or bounds["max_value_g"] <= bounds["min_value_g"]:
        raise ValueError("The retained flash rule has invalid signal windows")
    return bounds


def declared_setup_clearwater(authority, run, *, reader=None):
    """Use the exact reviewed GUI candidate, including shared wash references.

    Only explicitly declared project roots and their small manifests are read.
    A folder is never rescanned to silently select another clear-water image.
    """
    from .current_inputs import current_input_reader
    from ..series_geometry import geometry_package_root
    from ..series_paths import series_key
    reader = current_input_reader(reader)
    path = reader.path(authority)
    document = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    roots = [geometry_package_root(path.parent)]
    project_root = document.get("paths", {}).get("project")
    if project_root:
        project_root = Path(project_root)
        roots.append(reader.resolve_path(project_root if project_root.is_absolute() else path.parent / project_root))
    roots.append(path.parent)
    seen = set()
    for root in roots:
        project_path = Path(root) / "series.json"
        key = _key(project_path)
        if key in seen or not project_path.is_file():
            continue
        seen.add(key)
        project_pin = asset(project_path)
        project = reader.json(project_pin)
        if (document.get("series_id") and project.get("series_id") and
                series_key(document["series_id"]) != series_key(project["series_id"])):
            continue
        mapping = project.get("tools", {}).get("source_mapping", {})
        candidates = [value.get("source_mapping", {}) for value in project.get("runs", [])
                      if str(value.get("run_id", "")).casefold() == run.casefold()]
        candidates.extend(value for value in mapping.get("rows", [])
                          if str(value.get("run_id", "")).casefold() == run.casefold())
        candidates = [value for value in candidates if value.get("clearwater_candidate")]
        if not candidates:
            continue
        locators = {str(value["clearwater_candidate"]) for value in candidates}
        if len(locators) != 1:
            raise ValueError("The GUI contains conflicting clear-water candidates: " + run)
        selected = candidates[0]
        if (selected.get("candidate_status") not in {"approved", "user_confirmed"}
                and mapping.get("baseline_and_clearwater_candidates_approved") is not True):
            raise ValueError("Review and approve the GUI clear-water candidate before Finish setup: " + run)
        raw_path = Path(selected["clearwater_candidate"])
        raw_path = reader.resolve_path(raw_path if raw_path.is_absolute() else Path(root) / raw_path)
        source_path = Path(root) / "source_manifest.json"
        if not source_path.is_file():
            raise ValueError("The GUI clear-water candidate has no recorded source identity: " + run)
        source_pin = asset(source_path)
        source = reader.json(source_pin)
        matches = [value for value in source.get("sources", [])
            if value.get("kind") in {"clearwater_image", "clearwater_images"}
            and str(value.get("run_id") or "").casefold() in {"", run.casefold()}
            and _key(reader.resolve_path(value.get("path", ""))) == _key(raw_path)
            and value.get("sha256") and type(value.get("size_bytes")) is int]
        if not matches or len({(value["sha256"], value["size_bytes"]) for value in matches}) != 1:
            raise ValueError("The reviewed GUI clear-water image lacks one exact source SHA: " + run)
        chosen = matches[0]
        if raw_path.stat().st_size != chosen["size_bytes"]:
            raise ValueError("The GUI clear-water source size changed: " + run)
        return dict(clearwater_source=dict(run=run, raw=dict(path=str(raw_path),
            sha256=chosen["sha256"], size_bytes=chosen["size_bytes"]),
            selection_rule="Reviewed GUI clear-water candidate (" + str(selected.get("clearwater_selector") or "manual") + ")",
            project_manifest=project_pin, source_manifest=source_pin), sources=[project_pin, source_pin])
    return None


def setup_optical_sources(authority, run, *, reader=None):
    """Read P1's exact selected clear-water RAW and scalar flash convention.

    Setup needs these existing optical inputs, without loading phase heads,
    chemistry fits, physical rasters or calibration observation cohorts.
    """
    from .current_inputs import current_input_reader
    from ..series_method import calibration_ancestry_reference
    reader = current_input_reader(reader)
    path = reader.path(authority)
    series = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    declared = declared_setup_clearwater(authority, run, reader=reader)
    method = series["method"][series["method"]["active"]]
    reference = calibration_ancestry_reference(method)
    if not reference:
        if declared is None:
            raise ValueError("Select and review the GUI clear-water source before finishing Setup: " + run)
        from ..mass.optical_initialization import initial_flash_bounds
        return dict(clearwater_source=declared["clearwater_source"], flash_bounds=initial_flash_bounds(),
            flash_convention="original DarSIA starting windows; unfitted and not mass-calibrated",
            sources=[deepcopy(authority), *declared["sources"]])
    source = Path(reference)
    source = reader.resolve_path(source if source.is_absolute() else path.parent / source)
    sources = [deepcopy(authority)]
    def document(pin):
        sources.append(deepcopy(pin))
        return reader.json(pin)
    calibration = document(asset(source))
    selected = document(calibration["selected_input_plan"])
    parts = document(selected["saved_parts_plan"])
    second = document(parts["r2_input_plan"])
    freeze = document(parts["r1_freeze"])
    bounds = _flash_bounds(freeze, reader, sources)
    if declared is not None:
        return dict(clearwater_source=declared["clearwater_source"], flash_bounds=bounds,
            flash_convention="retained imported optical windows; no new mass fit",
            sources=[*sources, *declared["sources"]])
    transport_pin = second["clearwater_transports"].get(run)
    if transport_pin is None:
        raise ValueError("No selected measured clear-water reference is recorded for " + run)
    transport = document(transport_pin)
    clear_pin = dict(path=transport["clearwater_receipt"], sha256=transport["clearwater_receipt_sha256"])
    clear = document(clear_pin)
    processing_pin = dict(path=clear["correction_receipt_path"], sha256=clear["correction_receipt_sha256"])
    processing = document(processing_pin)
    if (clear.get("run") != run or transport.get("run") != run or processing.get("run") != run
            or clear.get("encoding") != "float32-camera-linear-unit-rgb"
            or processing.get("raw_source_sha256") != clear.get("raw_source_sha256")
            or not processing.get("selected_rule")):
        raise ValueError("The selected clear-water RAW lacks its original optical source identity: " + run)
    raw = _sized(dict(path=processing["raw_source_path"], sha256=processing["raw_source_sha256"]), reader)
    return dict(clearwater_source=dict(run=run, raw=raw, selection_rule=processing["selected_rule"],
        receipt=clear_pin, processing_receipt=processing_pin, transport=deepcopy(transport_pin)),
        flash_bounds=bounds, flash_convention="retained imported optical windows; no new mass fit", sources=sources)


def retained_statistics_recipe(selection, run, *, reader=None, fresh_images=False):
    """Follow the selected method's existing ancestry without changing physics.

    The returned statistics template deliberately lacks Build-owned physical
    fields and frame geometry. The caller injects those before admission.
    Current protocol observations are pinned with the saved image selection.
    """
    from .current_inputs import current_input_reader
    reader = current_input_reader(reader)
    selected_run = _run_row(selection["runs"], run, "selected run")
    sources = []

    def document(binding):
        sources.append(deepcopy(binding))
        return reader.json(binding)

    current = selection["input_authority"]["current_series"]
    series_path = reader.path(current)
    sources.append(deepcopy(current))
    series = tomllib.loads(series_path.read_text(encoding="utf-8-sig"))
    method = series["method"][series["method"]["active"]]
    from ..series_method import calibration_ancestry_reference
    reference = calibration_ancestry_reference(method)
    if reference is None:
        raise ValueError("The saved image selection has no retained calibration ancestry")
    calibration_path = Path(reference)
    if not calibration_path.is_absolute():
        calibration_path = series_path.parent / calibration_path
    calibration = document(asset(reader.resolve_path(calibration_path)))
    if not _same_binding(calibration["calibration_files"]["global"],
                         selection["input_authority"]["global_fit"]):
        raise ValueError("The selected RAW cohort belongs to another mass calibration")
    selected_plan = document(calibration["selected_input_plan"])
    parts = document(selected_plan["saved_parts_plan"])
    second = document(parts["r2_input_plan"])
    base = document(second["base_input_plan"])
    freeze = document(parts["r1_freeze"])
    optics = None if fresh_images else _optical_inputs(freeze, reader)
    flash_bounds = _flash_bounds(freeze, reader, sources) if fresh_images else None

    recipes = document(base["recipes"])
    recipe = _run_row(recipes["runs"], run, "chemical recipe")
    chemical = {key: _finite(recipe[key], key, positive=True)
                for key in ("alkalinity_M", "btb_M")}
    from .current_run_protocols import read_frozen_protocols
    protocols = read_frozen_protocols(selected_run, reader=reader)
    sources.extend(deepcopy(protocols["sources"]))
    atmosphere = protocols["atmosphere"]
    rows = atmosphere["rows"]
    if not rows:
        raise ValueError("The retained atmosphere has no observed time series")
    previous = None
    for row in rows:
        time = _finite(row["time_hours"], "atmosphere time")
        if previous is not None and time <= previous:
            raise ValueError("The retained atmosphere must have increasing observation times")
        previous = time
        for key in ("pressure_bar", "temperature_c", "pressure_gradient_bar_per_m",
                    "temperature_gradient_c_per_m"):
            _finite(row[key], key)
    dose = protocols["dose"]

    transport_binding = second["clearwater_transports"][run]
    transport = document(transport_binding)
    clear_binding = dict(path=transport["clearwater_receipt"],
                         sha256=transport["clearwater_receipt_sha256"])
    clear = document(clear_binding)
    if (clear.get("run") != run or transport.get("run") != run
            or clear.get("encoding") != "float32-camera-linear-unit-rgb"):
        raise ValueError("The retained scientific clear-water reference belongs to another run or encoding")
    clearwater_source = None
    if fresh_images:
        processing_pin = dict(path=clear["correction_receipt_path"],
                              sha256=clear["correction_receipt_sha256"])
        processing = document(processing_pin)
        if (processing.get("run") != run
                or processing.get("raw_source_sha256") != clear.get("raw_source_sha256")
                or not isinstance(processing.get("selected_rule"), str)
                or not processing["selected_rule"].strip()):
            raise ValueError("The selected clear-water RAW source lacks its original run and selection rule")
        raw = _sized(dict(path=processing["raw_source_path"],
                          sha256=processing["raw_source_sha256"]), reader)
        clearwater_source = dict(run=run, raw=raw, selection_rule=processing["selected_rule"],
            receipt=deepcopy(clear_binding), processing_receipt=processing_pin,
            transport=deepcopy(transport_binding))
        clearwater = None
    else:
        clearwater = dict(
            rgb=_sized(dict(path=clear["rgb_path"], sha256=clear["rgb_sha256"]), reader),
            valid=_sized(dict(path=clear["valid_mask_path"], sha256=clear["valid_mask_sha256"]), reader),
            H_reference_to_grid=deepcopy(transport["H_ac28_to_clearwater_grid"]))

    epochs = document(second["light_epochs"])
    if isinstance(epochs.get("runs"), dict):
        light_epoch = epochs["runs"][run]
    else:
        light_epoch = _run_row(epochs["runs"], run, "light epoch")
    if isinstance(light_epoch, dict):
        light_epoch = light_epoch["epoch"]
    if not isinstance(light_epoch, str) or not light_epoch:
        raise ValueError("The retained light epoch must be explicit")

    limits = second["physical_limits"]
    statistics = dict(format=STATISTICS_FORMAT,
        shared=dict(optics=optics, clearwater=clearwater,
                    swi_by_facies=deepcopy(second["swi_by_facies"])),
        recipe=chemical,
        atmosphere_series=dict(rows=deepcopy(rows)),
        summary=dict(baseline_gain=24.253536351793787, intervals=32768,
                     max_column_distance_m=0.020, anchor_target_h=2.25,
                     max_depth_m=_finite(limits["max_depth_m"], "capacity depth", positive=True),
                     max_porosity=_finite(limits["max_porosity"], "capacity porosity", positive=True)),
        bindings={key: calibration[key] for key in ("round4_spec_sha256", "amendment_sha256")},
        provenance=dict(retained_statistics_inputs=sources,
            current_protocol_inputs=deepcopy(protocols["sources"]),
            current_protocol_time_origin=protocols.get("time_origin")))
    for source in sources:
        reader.path(source)
    result = dict(statistics=statistics, atmosphere=atmosphere, dose=dose,
                  light_epoch=deepcopy(light_epoch))
    if fresh_images:
        result.update(clearwater_source=clearwater_source, flash_bounds=flash_bounds)
    return result
