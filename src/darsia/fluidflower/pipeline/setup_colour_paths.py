"""Finish Setup with corrected, facies-wise colour paths before phase modelling.

Only the selected optical photographs are decoded. Full corrected frames stay
in memory one at a time; shared job files retain sampled triples and the real
path package. No phase classifier, training labels or mass fit is consumed.
"""
from copy import deepcopy
from dataclasses import replace
from collections import OrderedDict
import json
from pathlib import Path
import tomllib

import cv2
import numpy as np

from ..mass import pooled_optical_paths as optical
from ..mass.n1_pipeline import asset, _write_json
from .current_inputs import CurrentInputReader, current_input_reader
from .current_optical_preparation import (DEFAULT_SAMPLING, colour_path_image_count,
    optical_frame_selection, save_samples, load_samples)

LEGACY_FORMAT = "darsia-setup-colour-paths-v1"
FORMAT = "darsia-setup-colour-paths-v2"
COLOUR_PATH_SCOPE = "per-run"
PREPARED_RUN_FORMAT = "darsia-prepared-optical-run-v1"
PARTIAL_REUSE_FORMAT = "darsia-retired-setup-colour-path-partials-v1"
SIGNAL_AXIS_CONTRACT = "darsia-per-run-facies-common-distance-axis-v1"
REPRESENTATION = "corrected-clearwater-normalized-relative-rgb-v1"


class _OpticalMetadataReader(CurrentInputReader):
    """Share finite decoded ancestry metadata across runs, never image payloads."""
    def __init__(self, reader, maximum_bytes=16 * 1024**2):
        super().__init__(reader)
        self._metadata, self._cost, self._limit = OrderedDict(), 0, maximum_bytes

    def json(self, bound):
        from ..mass.n1_pipeline import _stamp
        path = self.path(bound)
        stamp = _stamp(path)
        key = str(path), bound["sha256"], stamp
        if key in self._metadata:
            self._metadata.move_to_end(key)
            return deepcopy(self._metadata[key][0])
        value = super().json(bound)
        size = path.stat().st_size
        if size <= self._limit:
            self._metadata[key] = deepcopy(value), size
            self._cost += size
            while self._cost > self._limit:
                _old, (_value, cost) = self._metadata.popitem(last=False)
                self._cost -= cost
        return value


def _selection(document, operations, root, reader):
    rows = []
    for run in operations:
        saved = document.get("image_selection", {}).get("runs", {}).get(run, {})
        if not saved.get("selected"):
            raise ValueError("Save Setup image choices before colour paths: " + run)
        if saved.get("selection_manifest"):
            pin = deepcopy(saved["selection_manifest"])
            path = Path(pin["path"])
            pin["path"] = str(reader.resolve_path(path if path.is_absolute() else root / path))
            reader.path(pin)
        frames, seen = [], set()
        for index, row in enumerate(saved["selected"]):
            path = Path(row["source_path"])
            path = reader.resolve_path(path if path.is_absolute() else root / path)
            key = str(path).replace("\\", "/").casefold()
            if key in seen:
                raise ValueError("Setup image choices repeat an acquisition: " + run)
            seen.add(key)
            frames.append(dict(raw_path=str(path), current_frame_index=row.get("frame_index", index),
                               elapsed_h=row["elapsed_h"]))
        rows.append(dict(run=run, selected_frames=frames))
    return optical_frame_selection(dict(runs=rows), colour_path_image_count(document))


def _roles(context, reader):
    roles, adopted = _role_metadata(context.provenance["adopted_revision"], reader)
    enabled = [row["label_id"] for row in adopted["configuration"]["regions"] if row.get("enabled", True)]
    return roles, np.isin(context.labels, enabled)


def _role_metadata(adopted_revision, reader):
    """Read material roles without reopening the completed Build's rasters."""
    from ..geometry_raster_sources import _full_csv
    adopted = reader.json(adopted_revision)
    pin = adopted["artifacts"]["source_mode_files"]["facies_properties"]
    columns, records, ids = _full_csv(reader.path(pin), strict=True)
    names = {name.strip().casefold(): name for name in columns}
    roles = [dict(facies_id=key, name=row[names["name"]], analysis_role=row[names["analysis_role"]])
             for key, row in zip(ids, records)]
    return roles, adopted


def _recipe(runs, roles, adopted, build, native, flash_bounds):
    return optical.recipe_from_document(dict(format=optical.FORMAT, method=optical.SETUP_METHOD,
        fit_runs=list(runs), excluded_runs=[], roles=deepcopy(roles), runs=deepcopy(runs),
        adopted_revision=deepcopy(adopted), build_context=deepcopy(build), phase_configuration=None,
        native_configuration=deepcopy(native), sampling=dict(DEFAULT_SAMPLING,
            min_runs=min(DEFAULT_SAMPLING["min_runs"], len(runs))),
        flash_bounds=deepcopy(flash_bounds), path_name_prefix="Prepared image basis · facies "))


def _rgb(record):
    from ..mass.production_precision import PixelMath
    return np.moveaxis(PixelMath("float32").decode_log_rgb(record["log_linear_rgb"]), 0, -1)


def _frame(bridge, current, baseline, rgb_baseline, clear, rgb_clear, known, run, chosen, recipe):
    shape = current["valid_mask"].shape
    height, width = shape
    transform = np.asarray(current["geometry"]["baseline_to_native_h"], np.float64)
    build_h = np.asarray(current["build_to_frame_native_h"], np.float64)
    baseline_rgb = cv2.warpPerspective(np.where(baseline["valid_mask"][..., None], rgb_baseline, 0.),
        transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    baseline_valid = cv2.warpPerspective(baseline["valid_mask"].astype(np.float32), transform,
        (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) >= 1 - 1e-6
    domain = cv2.warpPerspective(known.astype(np.uint8), build_h, (width, height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    facies = cv2.warpPerspective(bridge.context.facies.astype(np.float64), build_h, (width, height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.int32)
    support = current["valid_mask"] & baseline_valid
    clear_h = (np.asarray(current["geometry"]["reference_to_native_h"], np.float64)
               @ np.linalg.inv(np.asarray(clear["geometry"]["reference_to_native_h"], np.float64)))
    clear_rgb = cv2.warpPerspective(np.where(clear["valid_mask"][..., None], rgb_clear, 0.),
        clear_h, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    clear_valid = cv2.warpPerspective(clear["valid_mask"].astype(np.float32), clear_h, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) >= 1 - 1e-6
    contract = optical._source_contract(recipe.document, run, chosen)
    return optical.PreparedFaciesFrame(run, chosen["frame_index"], chosen["elapsed_h"],
        _rgb(current), baseline_rgb, clear_rgb, clear_valid, facies, domain, support, dict(optical_context_verified=True,
            optical_source_contract=contract, input_representation=REPRESENTATION,
            geometry=current["geometry"], correction=current["correction"],
            build_to_frame_native_h=current["build_to_frame_native_h"], bridge=deepcopy(bridge.proof),
            full_corrected_arrays_retained=False, phase_inference_performed=False))


def _prepare_run(arguments):
    run, operation, basis, selected, output, aliases, optical_sources = arguments[:7]
    resumed = arguments[7] if len(arguments) == 8 else None
    from ..mass.build_native_bridge import BuildNativeBridge
    from .setup_image_preparation import image_basis_reader
    reader = image_basis_reader(current_input_reader(), dict(source_aliases=aliases))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    bridge = BuildNativeBridge(operation, basis["source_binding"], root / "bridge", reader=reader)
    roles, known = _roles(bridge.context, reader)
    baseline_source = deepcopy(bridge.context.provenance["raw_acquisition"])
    baseline = bridge.corrected_frame(baseline_source, prepared_baseline=basis["baseline"])
    baseline_rgb = _rgb(baseline)
    clear_source = optical_sources["clearwater_source"]
    clear = bridge.corrected_frame(clear_source["raw"], source_purpose="clearwater")
    clear_rgb = _rgb(clear)
    if resumed is None:
        frames = _selected_source_frames(run, selected, reader)
        row = dict(baseline=baseline_source, clearwater=deepcopy(clear_source["raw"]), profile=deepcopy(bridge.binding), frames=frames)
        recipe = _recipe({run: row}, roles, bridge.context.provenance["adopted_revision"],
                         operation, bridge.context.provenance["native_configuration"], optical_sources["flash_bounds"])
        retained = {}
    else:
        recipe = optical.recipe_from_document(resumed["recipe"])
        row = recipe.document["runs"][run]
        # The old descriptor stays truthful: the retry independently generated
        # the same numerical RAW source binding at a fresh owned location.
        if (reader.json(row["profile"]) != reader.json(bridge.binding)
                or row["baseline"] != baseline_source or row["clearwater"] != clear_source["raw"]
                or recipe.document["roles"] != roles
                or recipe.document["adopted_revision"] != bridge.context.provenance["adopted_revision"]
                or recipe.document["build_context"] != operation
                or recipe.document["native_configuration"] != bridge.context.provenance["native_configuration"]
                or recipe.document["flash_bounds"] != optical_sources["flash_bounds"]):
            raise ValueError("Retained optical samples differ from the retry's measured image basis: " + run)
        frames, retained = row["frames"], resumed["samples_by_index"]
    # The recipe is available even if this run is interrupted between frames.
    _write_json(root / "sampling_recipe.json", dict(recipe=recipe.document,
        input=_prepared_identity(arguments), resumed_from=deepcopy(resumed)))
    samples = []
    for index, chosen in enumerate(frames):
        if str(index) in retained:
            samples.append(deepcopy(retained[str(index)]))
            continue
        # Correction uses the very same saved crop, light and colour producer
        # as later Calibration. No full-frame NPZ copy is published here.
        current = bridge.corrected_frame(chosen["source"])
        frame = _frame(bridge, current, baseline, baseline_rgb, clear, clear_rgb, known, run, chosen, recipe)
        frame.provenance["clearwater_selection"] = deepcopy(clear_source)
        if resumed is not None:
            frame.provenance["retained_sampling_profile"] = dict(
                original=deepcopy(row["profile"]), current=deepcopy(bridge.binding),
                identical_binding_bytes=True)
        samples.append(save_samples(root / (str(index).zfill(4) + "-samples"),
            optical.sample_frame_for_common_paths(frame, recipe, case_id=chosen["case_id"])))
        del current, frame
    bridge._revalidate()
    result = dict(run=run, recipe=recipe.document, samples=samples,
        operation=deepcopy(operation), image_basis=deepcopy(basis),
        phase_inference_performed=False, model_training_performed=False)
    # A future ordinary retry can reuse the completed numerical samples without
    # decoding the same photographs again. Incomplete runs publish no checkpoint.
    _write_json(root / "prepared_run.json", dict(format=PREPARED_RUN_FORMAT,
        complete=True, input=dict(operation=operation, image_basis=basis,
            selection=selected, optical_sources=optical_sources), result=result))
    return result


def _selected_source_frames(run, selected, reader, retained=None):
    """Pin only sources not already recorded in a verified saved sample."""
    frames = []
    for index, target in enumerate(selected):
        original = target["frame"]
        raw = (deepcopy(retained[index]) if retained and index in retained
               else asset(reader.resolve_path(original["raw_path"])))
        if (not isinstance(raw, dict)
                or reader.resolve_path(raw.get("path", "")) != reader.resolve_path(original["raw_path"])):
            raise ValueError("Retained optical sample names another selected photograph: " + run)
        frames.append(dict(case_id=run + "-" + raw["sha256"][:20],
            frame_index=original["current_frame_index"], elapsed_h=original["elapsed_h"],
            target_h=target["target_h"], source=raw))
    return frames


def _pooled_recipe(rows, folder):
    Path(folder).mkdir(parents=True, exist_ok=True)
    first = rows[0]["recipe"]
    if any(row["recipe"]["roles"] != first["roles"] or row["recipe"]["flash_bounds"] != first["flash_bounds"] or
           row["recipe"]["adopted_revision"] != first["adopted_revision"] for row in rows):
        raise ValueError("Setup colour paths require the same adopted facies table")
    def bundle(role):
        return _write_json(Path(folder) / (role + ".json"), dict(
            format="darsia-pooled-optical-run-inputs-v1", role=role,
            runs={row["run"]: row["recipe"][role] for row in rows}))
    return _recipe({row["run"]: row["recipe"]["runs"][row["run"]] for row in rows},
        first["roles"], first["adopted_revision"], bundle("build_context"), bundle("native_configuration"), first["flash_bounds"])


def _rebind_sample(batch, row, recipe, reader):
    """Join exact per-run samples into one pooled recipe without changing RGB."""
    original = optical.recipe_from_document(row["recipe"])
    if batch.recipe_sha256 != original.sha256 or batch.run != row["run"]:
        raise ValueError("Setup samples belong to another corrected run")
    before = original.document
    after = recipe.document
    if (before["method"] != after["method"] or before["roles"] != after["roles"]
            or before["sampling"]["stride_px"] != after["sampling"]["stride_px"]
            or before["adopted_revision"] != after["adopted_revision"]
            or before["runs"][batch.run] != after["runs"][batch.run]):
        raise ValueError("Pooled Setup sampling changes a source, grid or material role")
    for role in ("build_context", "native_configuration"):
        combined = reader.json(after[role])
        if combined["runs"][batch.run] != before[role]:
            raise ValueError("Pooled Setup sampling changes a corrected source basis")
    selected = next(value for value in after["runs"][batch.run]["frames"] if value["case_id"] == batch.case_id)
    provenance = json.loads(batch.frame_provenance_json)
    if provenance["optical_source_contract"] != optical._source_contract(before, batch.run, selected):
        raise ValueError("Setup sample provenance differs from its actual corrected source")
    provenance.update(optical_source_contract=optical._source_contract(after, batch.run, selected),
                      original_run_recipe_sha256=original.sha256)
    encoded = optical._json(provenance)
    audit = dict(batch.audit, source_binding=provenance["optical_source_contract"], frame_provenance=provenance)
    return replace(batch, recipe_sha256=recipe.sha256, audit=audit, frame_provenance_json=encoded)


def preflight_setup_colour_paths(choice_authority, runs, *, reader=None):
    """Validate saved selections and original optical sources before long work."""
    from .current_statistics_recipe import setup_optical_sources
    reader = _OpticalMetadataReader(current_input_reader(reader))
    path = reader.path(choice_authority)
    document = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    if not runs or len(set(runs)) != len(runs):
        raise ValueError("Finish Setup requires its distinct included run order")
    selected = _selection(document, dict.fromkeys(runs), path.parent, reader)
    sources = {run: setup_optical_sources(choice_authority, run, reader=reader) for run in runs}
    if len({optical._json(value["flash_bounds"]) for value in sources.values()}) != 1:
        raise ValueError("Setup colour paths need one unchanged optical flash convention")
    return dict(frame_selection=selected, optical_sources=sources)


def colour_path_packages(receipt, reader=None, *, require_per_run=False, run=None):
    """Read the selected native package without mixing another run's paths.

    Old shared receipts remain readable as historical products. A caller that
    requests per-run paths must explicitly reject those historical shared fits.
    ``run`` keeps the ordinary per-run consumer from reopening every package.
    """
    reader = current_input_reader(reader)
    runs = receipt.get("runs", [])
    if (receipt.get("complete") is not True or not isinstance(runs, list) or not runs
            or len(set(runs)) != len(runs) or run is not None and run not in runs):
        raise ValueError("Complete colour paths require their exact included run selection")
    if receipt.get("format") == FORMAT:
        packages = receipt.get("packages_by_run")
        if (receipt.get("scope") != COLOUR_PATH_SCOPE or not isinstance(packages, dict)
                or set(packages) != set(runs) or "package" in receipt):
            raise ValueError("Per-run colour paths need exactly one own package per included run")
        axis = receipt.get("signal_axis", {})
        scales = axis.get("scales_by_facies", {})
        if (axis.get("contract") != SIGNAL_AXIS_CONTRACT
                or axis.get("source_cohort") != runs or axis.get("frozen_for_production") is not True
                or axis.get("units") != "corrected-clearwater-relative-rgb-euclidean-distance"
                or axis.get("scale_rule") != "maximum-supported-path-length-per-facies"
                or not isinstance(scales, dict) or not scales
                or any(type(value) not in (int, float) or not np.isfinite(value) or value <= 0
                       for value in scales.values())):
            raise ValueError("Per-run colour paths require their frozen shared facies signal scale")
    elif receipt.get("format") == LEGACY_FORMAT and not require_per_run:
        packages = {key: receipt["package"] for key in runs}
    else:
        raise ValueError("Generate colour paths separately for each run before calibration")
    selected = runs if run is None else [run]
    result, observed_scales = {}, {}
    for key in selected:
        package = packages[key]
        if not isinstance(package, dict) or set(package) != {"metadata", "flash", "color_paths", "signals"}:
            raise ValueError("Colour paths require their actual native package files")
        metadata = reader.json(package["metadata"])
        recipe = metadata.get("recipe", {})
        if (recipe.get("method") != optical.SETUP_METHOD
                or receipt.get("format") == FORMAT and
                (recipe.get("fit_runs") != [key] or set(recipe.get("runs", {})) != {key}
                 or set(metadata.get("source_bindings", {})) != {key})):
            raise ValueError("A run's colour-path package contains different optical sources")
        if receipt["format"] == FORMAT and metadata.get("signal_axis") != axis:
            raise ValueError("A run's colour paths use a different shared signal scale")
        for group in (package["color_paths"], package["signals"]):
            for pin in group.values():
                reader.path(pin)
        reader.path(package["flash"])
        if receipt["format"] == FORMAT:
            for facies, audit in metadata.get("path_audit", {}).items():
                if audit.get("status") != "supported":
                    continue
                path = reader.json(package["color_paths"][facies])
                colours = np.asarray(path["color_path"]["colors"], dtype=np.float64)
                expected = np.r_[0., np.cumsum(np.linalg.norm(np.diff(colours, axis=0), axis=1))]
                scale = scales.get(facies)
                if (scale is None or expected[-1] > scale * (1 + 1e-12)
                        or not np.allclose(np.asarray(path["values"]), expected / scale, rtol=1e-12, atol=0)):
                    raise ValueError("A native colour-path coordinate differs from its frozen signal scale")
                observed_scales[facies] = max(observed_scales.get(facies, 0.), float(expected[-1]))
        result[key] = package
    if receipt["format"] == FORMAT and run is None and (set(scales) != set(observed_scales)
            or any(not np.isclose(scales[key], length, rtol=1e-12, atol=0)
                   for key, length in observed_scales.items())):
        raise ValueError("The shared facies scale differs from the complete fitted run set")
    return result


def _prepared_identity(arguments):
    run, operation, basis, selected, _output, _aliases, sources = arguments[:7]
    return dict(operation=operation, image_basis=basis, selection=selected,
                optical_sources=sources)


def _prepared_checkpoint(arguments, row, *, reader):
    """Bind the real completed run samples, including reused numerical files."""
    path = Path(arguments[4]) / "prepared_run.json"
    value = dict(format=PREPARED_RUN_FORMAT, complete=True,
                 input=_prepared_identity(arguments), result=row)
    if path.is_file():
        if reader.json(asset(path)) != value:
            raise ValueError("Prepared optical run checkpoint changed")
        return asset(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return _write_json(path, value)


def _check_selection(row, selected, *, reader):
    recipe = optical.recipe_from_document(row["recipe"])
    run = row["run"]
    if recipe.document["fit_runs"] != [run] or set(recipe.document["runs"]) != {run}:
        raise ValueError("Prepared optical samples must belong only to their own run")
    frames = recipe.document["runs"][run]["frames"]
    if len(frames) != len(selected) or len(row["samples"]) != len(frames):
        raise ValueError("Prepared optical samples differ from the selected image count")
    for frame, choice in zip(frames, selected, strict=True):
        original = choice["frame"]
        if (frame["frame_index"] != original["current_frame_index"]
                or frame["elapsed_h"] != original["elapsed_h"]
                or frame["target_h"] != choice["target_h"]
                or reader.resolve_path(frame["source"]["path"]) != reader.resolve_path(original["raw_path"])):
            raise ValueError("Prepared optical samples belong to a different image selection")
    return recipe


def _reused_runs(bound, identity, *, reader):
    """Reuse real completed optical samples, never re-label a shared fitted path.

    The legacy producer did not publish a per-run checkpoint. Its immutable
    shared package nevertheless records the original per-frame audit and the
    exact recipe sources. Reconstruct each original one-run recipe and require
    both its SHA and its rebound audit to match those actual completed inputs.
    Existing sample NPZ descriptors and numerical identities remain the source
    of sample validation; no prior sidecar SHA anchoring is invented here.
    """
    if bound is None:
        return {}, None
    previous = reader.json(bound)
    old_input = previous.get("input", {})
    if (previous.get("complete") is not True or any(old_input.get(key) != identity[key]
            for key in ("operations", "image_basis", "frame_selection", "optical_sources"))):
        raise ValueError("Reusable colour-path samples belong to another image basis or selection")
    packages = colour_path_packages(previous, reader)
    result = {}
    if previous["format"] == FORMAT:
        checkpoints = previous.get("prepared_runs", {})
        if set(checkpoints) != set(previous["runs"]):
            raise ValueError("Reusable per-run colour paths lack their actual sample checkpoints")
        for run, pin in checkpoints.items():
            saved = reader.json(pin)
            if (saved.get("format") != PREPARED_RUN_FORMAT or saved.get("complete") is not True
                    or saved["input"] != dict(operation=identity["operations"][run],
                        image_basis=identity["image_basis"][run],
                        selection=identity["frame_selection"]["runs"][run],
                        optical_sources=identity["optical_sources"][run])):
                raise ValueError("Reusable optical run checkpoint changed its actual inputs")
            row = saved["result"]
            _check_selection(row, identity["frame_selection"]["runs"][run], reader=reader)
            result[run] = row
        return result, deepcopy(bound)
    package = packages[previous["runs"][0]]
    metadata = reader.json(package["metadata"])
    pooled = optical.recipe_from_document(metadata["recipe"])
    value = pooled.document
    if set(value["runs"]) != set(identity["operations"]):
        raise ValueError("Legacy optical recipe belongs to another included cohort")
    bundles = {}
    for role in ("build_context", "native_configuration"):
        bundle = reader.json(value[role])
        if (bundle.get("format") != "darsia-pooled-optical-run-inputs-v1"
                or bundle.get("role") != role or set(bundle.get("runs", {})) != set(value["runs"])):
            raise ValueError("Legacy optical recipe lacks its exact per-run source bundle")
        bundles[role] = bundle["runs"]
    folder = reader.path(bound).parent
    for run in previous["runs"]:
        recipe = _recipe({run: value["runs"][run]}, value["roles"], value["adopted_revision"],
            bundles["build_context"][run], bundles["native_configuration"][run], value["flash_bounds"])
        samples = [asset(folder / run / (str(index).zfill(4) + "-samples") / "samples.json")
                   for index in range(len(value["runs"][run]["frames"]))]
        row = dict(run=run, recipe=recipe.document, samples=samples,
            operation=deepcopy(identity["operations"][run]), image_basis=deepcopy(identity["image_basis"][run]),
            phase_inference_performed=False, model_training_performed=False)
        _check_selection(row, identity["frame_selection"]["runs"][run], reader=reader)
        audits = metadata.get("source_bindings", {}).get(run, [])
        if len(audits) != len(samples):
            raise ValueError("Legacy optical samples lack the completed package's frame audits")
        for pin, audit in zip(samples, audits, strict=True):
            batch = load_samples(pin, reader)
            if optical._json(_rebind_sample(batch, row, pooled, reader).audit) != optical._json(audit):
                raise ValueError("Legacy optical sample audit differs from its completed package")
        result[run] = row
    return result, deepcopy(bound)


def _partial_file(document, relative, reader):
    """Use only files named in the retired attempt's unchanged version list."""
    from ..job_management import _identity
    root = Path(document["source_root"])
    path = root / relative
    if (not root.is_absolute() or not path.is_relative_to(root)
            or ".." in Path(relative).parts
            or document["file_versions"].get(Path(relative).as_posix())
                != list(_identity(path, directory=False))):
        raise ValueError("Retired optical sample changed its owned file version: " + str(relative))
    return asset(path)


def _partial_sample(document, relative, reader):
    pin = _partial_file(document, relative, reader)
    value = reader.json(pin)
    arrays = value.get("arrays", {})
    path = reader.resolve_path(arrays.get("path", ""))
    try:
        key = path.relative_to(Path(document["source_root"]))
    except ValueError as error:
        raise ValueError("Retired optical arrays escaped their owned output") from error
    # The original sample sidecar owns this NPZ SHA. The retry records the
    # sidecar's actual current bytes; it does not invent a prior sidecar pin.
    current = _partial_file(document, key, reader)
    if any(current.get(key) != arrays.get(key) for key in ("sha256", "size_bytes")):
        raise ValueError("Retired optical arrays differ from their original sample pin")
    return pin, value


def _partial_sample_contract(value, recipe, run, selected, bridge_pin):
    """Check saved source/audit identity before any missing frame is decoded."""
    provenance = json.loads(value.get("frame_provenance_json", "null"))
    audit = value.get("audit", {})
    expected = optical._source_contract(recipe.document, run, selected)
    active = {str(row["facies_id"]) for row in recipe.document["roles"]
              if row["analysis_role"] == "porous_medium"}
    if (value.get("format") != "darsia-current-optical-samples-v1"
            or value.get("recipe_sha256") != recipe.sha256 or value.get("run") != run
            or value.get("case_id") != selected["case_id"]
            or not isinstance(provenance, dict) or provenance.get("optical_context_verified") is not True
            or provenance.get("bridge") != bridge_pin
            or provenance.get("optical_source_contract") != expected
            or audit.get("source_binding") != expected or audit.get("frame_provenance") != provenance
            or audit.get("run") != run or audit.get("frame_index") != selected["frame_index"]
            or audit.get("elapsed_h") != selected["elapsed_h"]
            or audit.get("sampling_stride_px") != recipe.document["sampling"]["stride_px"]
            or audit.get("facies_interior_radius_px") != recipe.document["sampling"]["interior_radius_px"]
            or set(value.get("identities", {})) != active or set(audit.get("sample_counts", {})) != active):
        raise ValueError("Retired optical sample differs from its original recipe/source audit: " + run)


def _reused_partial_runs(bound, identity, *, reader, _seen=()):
    """Recover genuine corrected samples from a retired failed owned attempt.

    Failure is retained as failure. Only its saved sample bytes are reused;
    no previous path package or completed Setup receipt is manufactured.
    """
    if bound is None:
        return {}, {}, None
    key = str(reader.resolve_path(bound["path"])).casefold()
    if key in _seen or len(_seen) >= 8:
        raise ValueError("Retired optical reuse ancestry repeats or exceeds its bounded retry chain")
    previous = reader.json(bound)
    if (previous.get("format") != PARTIAL_REUSE_FORMAT or previous.get("complete") is not False
            or any(previous.get(key) != identity[key] for key in
                   ("operations", "frame_selection", "optical_sources"))
            or previous.get("image_basis", {}).get("runs") != identity["image_basis"]
            or not previous.get("retirement") or not isinstance(previous.get("file_versions"), dict)):
        raise ValueError("Retired optical samples belong to another image basis, source or selection")
    failure = reader.json(previous["failed_attempt"])
    retired = reader.json(previous["retirement"]["retirement"])
    if (failure.get("run") != "setup-image-preparation" or not failure.get("failed_utc")
            or failure.get("attempt_id") != Path(previous["source_root"]).name
            or retired.get("run") != failure["run"] or retired.get("attempt_id") != failure["attempt_id"]
            or any(retired.get(key) is not True for key in
                   ("complete", "all_absent", "worker_os_references_closed"))
            or retired.get("retirement_basis") != "owned_process_incarnations_absent"):
        raise ValueError("Retired optical samples lack their genuine failed image attempt")
    complete, partial = {}, {}
    if previous.get("prior_reuse") is not None:
        ancestor = reader.json(previous["prior_reuse"])
        if any(ancestor.get(key) != previous.get(key) for key in
               ("choice", "choice_authority", "raw_execution_identity", "correction_order")):
            raise ValueError("Retired optical reuse ancestry changes saved corrections or RAW identity")
        complete, partial, _ = _reused_partial_runs(previous["prior_reuse"], identity,
            reader=reader, _seen=(*_seen, key))
    for run, operation in identity["operations"].items():
        selected = identity["frame_selection"]["runs"][run]
        inherited = complete.get(run, partial.get(run))
        sample_documents, pins, sources = {}, {}, {}
        for index in range(len(selected)):
            relative = f"colour-paths/{run}/{index:04d}-samples/samples.json"
            if relative not in previous["file_versions"]:
                continue
            pin, value = _partial_sample(previous, relative, reader)
            pins[str(index)], sample_documents[index] = pin, value
            sources[index] = deepcopy(value.get("audit", {}).get("source_binding", {}).get("source"))
        recipe_relative = f"colour-paths/{run}/sampling_recipe.json"
        saved_recipe = None
        if recipe_relative in previous["file_versions"]:
            saved_recipe = reader.json(_partial_file(previous, recipe_relative, reader))
            if saved_recipe.get("input") != dict(operation=operation,
                    image_basis=identity["image_basis"][run], selection=selected,
                    optical_sources=identity["optical_sources"][run]):
                raise ValueError("Retired optical sampling recipe changes its actual inputs: " + run)
        if not pins:
            _check_retired_checkpoint(previous, run, inherited, identity, reader)
            continue
        bridge_pin = _partial_file(previous, f"colour-paths/{run}/bridge/bridge.json", reader)
        bridge = reader.json(bridge_pin)
        profile_pin = _partial_file(previous, f"colour-paths/{run}/bridge/raw_source_binding.json", reader)
        first = next(iter(sample_documents.values()))["audit"]["source_binding"]
        if (bridge.get("format") != "darsia-build-native-bridge-v1"
                or bridge.get("operation_result") != operation
                or bridge.get("photometry_source") != identity["image_basis"][run]["source_binding"]
                or bridge.get("raw_source_binding") != profile_pin
                or first.get("clearwater") != identity["optical_sources"][run]["clearwater_source"]["raw"]
                or first.get("build_context") != operation
                or first.get("native_configuration") != bridge.get("native_configuration")):
            raise ValueError("Retired optical bridge differs from the retained completed photometry: " + run)
        if first.get("profile") != profile_pin:
            if (inherited is None or first.get("profile") != inherited["recipe"]["runs"][run]["profile"]
                    or reader.json(first["profile"]) != reader.json(profile_pin)):
                raise ValueError("Retired optical bridge changes the original sampling profile: " + run)
        build = reader.json(operation)
        plan = reader.json(build["setup_plan"])
        request = reader.json(build["raw_build_request"])
        if (first.get("adopted_revision") != plan["adopted_revision"]
                or first.get("native_configuration") != plan["native_parameters"]
                or first.get("baseline") != request["baseline"]):
            raise ValueError("Retired optical sources differ from their genuine completed Build: " + run)
        # No RAW is opened for a fully sampled run. In an interrupted run only
        # still-missing images are pinned; the ordinary producer needs them.
        original_recipe = (saved_recipe["recipe"] if saved_recipe is not None
                           else inherited["recipe"] if inherited is not None else None)
        if original_recipe is not None:
            original = optical.recipe_from_document(original_recipe).document
            frames = original["runs"][run]["frames"]
            if len(frames) != len(selected):
                raise ValueError("Retired optical sampling recipe changes its selected frame count: " + run)
            for index, source in sources.items():
                if source != frames[index]["source"]:
                    raise ValueError("Retired optical sample changes a saved acquisition: " + run)
            # Validate paths/times without opening any stored RAW acquisition.
            _check_selection(dict(run=run, recipe=original_recipe,
                samples=[None] * len(frames)), selected, reader=reader)
        else:
            frames = _selected_source_frames(run, selected, reader, sources)
        roles, _adopted = _role_metadata(first["adopted_revision"], reader)
        recipe = _recipe({run: dict(baseline=first["baseline"], clearwater=first["clearwater"],
            profile=first["profile"], frames=frames)}, roles, first["adopted_revision"], operation,
            first["native_configuration"], identity["optical_sources"][run]["flash_bounds"])
        if original_recipe is not None and recipe.document != original_recipe:
            raise ValueError("Retired optical samples change their saved sampling recipe: " + run)
        for index, value in sample_documents.items():
            _partial_sample_contract(value, recipe, run, frames[index], bridge_pin)
        if inherited is not None:
            if inherited["recipe"] != recipe.document:
                raise ValueError("Retired optical retry changes an already measured run recipe: " + run)
            inherited_pins = (dict(enumerate(inherited["samples"])) if "samples" in inherited
                              else {int(key):pin for key,pin in inherited["samples_by_index"].items()})
            pins = {**{str(index):pin for index,pin in inherited_pins.items()}, **pins}
        row = dict(run=run, recipe=recipe.document,
            operation=deepcopy(operation), image_basis=deepcopy(identity["image_basis"][run]),
            phase_inference_performed=False, model_training_performed=False)
        if len(pins) == len(frames):
            row["samples"] = [pins[str(index)] for index in range(len(frames))]
            _check_selection(row, selected, reader=reader)
            complete[run] = row
            partial.pop(run, None)
        else:
            partial[run] = dict(row, samples_by_index=pins)
        _check_retired_checkpoint(previous, run, complete.get(run, partial.get(run)), identity, reader)
    return complete, partial, deepcopy(bound)


def _check_retired_checkpoint(document, run, row, identity, reader):
    """A retry checkpoint records the actual inherited/current sample pins."""
    relative = f"colour-paths/{run}/prepared_run.json"
    if relative not in document["file_versions"]:
        return
    saved = reader.json(_partial_file(document, relative, reader))
    if (row is None or "samples" not in row or saved.get("format") != PREPARED_RUN_FORMAT
            or saved.get("complete") is not True
            or saved.get("input") != dict(operation=identity["operations"][run],
                image_basis=identity["image_basis"][run], selection=identity["frame_selection"]["runs"][run],
                optical_sources=identity["optical_sources"][run])
            or saved.get("result", {}).get("recipe") != row["recipe"]
            or saved.get("result", {}).get("samples") != row["samples"]):
        raise ValueError("Retired optical checkpoint differs from its genuine sample ancestry: " + run)


def prepare_setup_colour_paths(operations, image_basis, choice_authority, output, *,
        workers=1, reader=None, on_progress=None, run_producer=None, package_writer=None,
        reuse_receipt=None, reuse_partial=None):
    """Produce a separate per-facies colour-path package for every run."""
    from .setup_image_preparation import retained_series_aliases, image_basis_reader
    reader = current_input_reader(reader)
    authority_path = reader.path(choice_authority)
    preflight = preflight_setup_colour_paths(choice_authority, list(operations), reader=reader)
    selected = preflight["frame_selection"]
    if not operations or set(image_basis["runs"]) != set(operations):
        raise ValueError("Setup colour paths need every included run's actual image basis")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / "receipt.json"
    identity = dict(operations=deepcopy(operations), choice_authority=deepcopy(choice_authority),
        image_basis=deepcopy(image_basis["runs"]), frame_selection=selected,
        optical_sources=deepcopy(preflight["optical_sources"]),
        producer=asset(Path(__file__)))
    if receipt.is_file():
        saved = reader.json(asset(receipt))
        if saved.get("input") != identity or saved.get("complete") is not True:
            raise ValueError("Existing colour paths belong to another Setup image basis")
        colour_path_packages(saved, reader, require_per_run=True)
        return asset(receipt)
    aliases = retained_series_aliases(operations, reader)
    reader = image_basis_reader(reader, dict(source_aliases=aliases))
    optical_sources = preflight["optical_sources"]
    arguments = [(run, operation, image_basis["runs"][run], selected["runs"][run],
                  str(root / run), aliases, optical_sources[run]) for run, operation in operations.items()]
    if reuse_receipt is not None and reuse_partial is not None:
        raise ValueError("Select one genuine completed or failed optical sample source")
    prepared, reused_from = _reused_runs(reuse_receipt, identity, reader=reader)
    retained, partial, reused_failed = _reused_partial_runs(reuse_partial, identity, reader=reader)
    prepared.update(retained)
    arguments = [(*row, partial[row[0]]) if row[0] in partial else row for row in arguments]
    checkpoints = {}
    def report(run):
        checkpoints[run] = _prepared_checkpoint(next(row for row in arguments if row[0] == run),
                                                prepared[run], reader=reader)
        if on_progress:
            on_progress(dict(state="colour_paths", run=run, position=len(prepared), total=len(arguments),
                             message="Corrected colour-path images: " + run))
    for run in prepared:
        report(run)
    remaining = [row for row in arguments if row[0] not in prepared]
    if remaining and int(workers) != 1 and run_producer is None:
        from concurrent.futures import wait, FIRST_COMPLETED
        from ..execution import detect_host_resources
        from ..worker_processes import process_pool
        resources = detect_host_resources()
        requested = int(workers) or resources["cpu"]["physical_cores"]
        count = max(1, min(requested, len(remaining), resources["cpu"]["physical_cores"],
            resources["memory"]["available_bytes"] // (20 * 1024**3)))
        with process_pool(count, log_folder=root / "worker_logs") as pool:
            # Do not submit the whole series: an early failure must stop new
            # runs immediately, rather than execute every queued RAW reader.
            todo = iter(remaining)
            pending = {pool.submit(_prepare_run, args): args[0] for args in
                       [next(todo) for _ in range(min(count, len(remaining)))]}
            while pending:
                done, _ = wait(pending, timeout=1., return_when=FIRST_COMPLETED)
                for future in done:
                    run = pending.pop(future)
                    try:
                        prepared[run] = future.result()
                    except Exception as error:
                        for other in pending:
                            other.cancel()
                        if on_progress:
                            on_progress(dict(state="colour_paths_failed", run=run,
                                position=len(prepared), total=len(arguments),
                                message="Colour-path image preparation failed for " + run + ": " + str(error)))
                        # Runs already in progress still finish and keep their own
                        # checkpoints. Name every failure, so one retry can fix all.
                        failures = {run: error}
                        for other, name in pending.items():
                            if other.cancelled():
                                continue
                            try:
                                other.result()
                            except Exception as later:
                                failures[name] = later
                        if len(failures) > 1 and on_progress:
                            on_progress(dict(state="colour_paths_failed", run=run,
                                position=len(prepared), total=len(arguments),
                                message="Colour-path image preparation failed for " + "; ".join(
                                    name + ": " + str(value) for name, value in failures.items())))
                        raise RuntimeError("Colour-path image preparation failed for "
                                           + ", ".join(failures)) from error
                    report(run)
                for _ in range(len(done)):
                    args = next(todo, None)
                    if args is not None:
                        pending[pool.submit(_prepare_run, args)] = args[0]
    else:
        for args in remaining:
            try:
                prepared[args[0]] = (run_producer or _prepare_run)(args)
            except Exception as error:
                raise RuntimeError("Colour-path image preparation failed for " + args[0]) from error
            report(args[0])
    fitted, scales = {}, {}
    for run in operations:
        row = prepared[run]
        recipe = _check_selection(row, selected["runs"][run], reader=reader)
        fit = optical.fit_pooled_color_paths(recipe, (load_samples(pin, reader) for pin in row["samples"]))
        fitted[run] = recipe, fit
        for facies, interpretation in fit[0].items():
            colours = np.asarray(interpretation.color_path.colors, dtype=np.float64)
            length = float(np.sum(np.linalg.norm(np.diff(colours, axis=0), axis=1)))
            if not np.isfinite(length) or length <= 0:
                raise ValueError("A supported run/facies colour path has no positive optical distance")
            scales[str(facies)] = max(scales.get(str(facies), 0.), length)
    axis = dict(contract=SIGNAL_AXIS_CONTRACT,
        units="corrected-clearwater-relative-rgb-euclidean-distance", scales_by_facies=scales,
        source_cohort=list(operations), scale_rule="maximum-supported-path-length-per-facies",
        frozen_for_production=True)
    packages = {run: (package_writer or optical.save_fitted_path_package)(root / "packages" / run,
        recipe, *fit, signal_axis=axis) for run, (recipe, fit) in fitted.items()}
    colour_path_packages(dict(format=FORMAT, complete=True, runs=list(operations),
        scope=COLOUR_PATH_SCOPE, packages_by_run=packages, signal_axis=axis), reader, require_per_run=True)
    for pin in (choice_authority, *operations.values(),
                *(pin for value in optical_sources.values() for pin in value.get("sources", []))):
        reader.path(pin)
    return _write_json(receipt, dict(format=FORMAT, complete=True, input=identity,
        operations=deepcopy(operations), choice_authority=deepcopy(choice_authority),
        runs=list(operations), image_count_per_run=selected["max_images_per_run"],
        sampled_frame_counts={run: len(selected["runs"][run]) for run in operations},
        frame_selection={key: deepcopy(value) for key, value in selected.items() if key != "runs"},
        # Runs finish in any order when prepared in parallel; the receipt keeps the run order.
        scope=COLOUR_PATH_SCOPE, packages_by_run=packages,
        prepared_runs={run: checkpoints[run] for run in operations},
        signal_axis=axis,
        reused_from=reused_from, reused_from_failed_attempt=reused_failed,
        input_representation=REPRESENTATION,
        phase_inference_performed=False, model_training_performed=False, mass_fit_performed=False))
