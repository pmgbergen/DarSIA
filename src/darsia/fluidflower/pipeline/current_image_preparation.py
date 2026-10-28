"""Image preparation from current Build sources and the user's optical choices."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..image_normalisation import selected_corrections
from ..mass.n1_pipeline import asset
from ..optical_reference import frame_facing_sand_boundary, stable_band_mask, reference_samples, propose_reference
from .current_inputs import current_input_reader


def load_image_context(operation, *, reader, include_baseline=True):
    """Read only the three Build images used by optical preparation.

    The completed Build already owns and validates its physical outputs. This
    read-only optical view does not reopen depth, porosity or protocol payloads
    and does not claim to qualify a mass-analysis context. A display-only view
    uses the two categorical grids and certified crop, not scientific RGB.
    """
    import darsia
    from ..mass.native_optical_source import _grid_identity
    result = reader.json(operation)
    if (result.get("operation") != "build" or result.get("complete") is not True
            or result.get("worker_references_closed") is not True):
        raise ValueError("Image preparation requires a completed geometry Build")
    request = reader.json(result["raw_build_request"])
    plan = reader.json(result["setup_plan"])
    certificate = reader.json(result["baseline_renderer"])
    run = result["run"]
    if (request.get("run_id") != run or certificate.get("run_id") != run
            or certificate["inputs"]["raw"] != request["baseline"]
            or certificate["inputs"]["profile"] != request["profile"]
            or certificate["outputs"] != result["produced_files"]):
        raise ValueError("The optical Build images do not share the same source and camera")
    outputs = {Path(row["path"]).name: row for row in certificate["outputs"]}
    selected = {role: outputs[role + ".npz"] for role in ("shape_corrected_baseline", "labels", "facies")}
    images = {role: darsia.imread(reader.path(pin)) for role, pin in selected.items()
              if include_baseline or role != "shape_corrected_baseline"}
    baseline = images.pop("shape_corrected_baseline", None)
    grid = _grid_identity(baseline if baseline is not None else images["labels"])
    if any(_grid_identity(image) != grid for image in images.values()):
        raise ValueError("The optical Build images use different physical grids")
    if baseline is None:
        certified_grid = certificate["renderer"]["source_grid"]
        if any(certified_grid[key] != grid[key] for key in ("num_voxels", "voxel_size")):
            raise ValueError("The display crop differs from the certified Build grid")
    return SimpleNamespace(run=run, baseline_rgb=None if baseline is None else baseline.img,
        display_renderer=deepcopy(certificate.get("renderer")),
        raw_orientation=certificate.get("homography_convention", {}).get("raw_orientation"),
        labels=images["labels"].img,
        facies=images["facies"].img, grid=grid, provenance=dict(
            operation_result=deepcopy(operation), setup_plan=deepcopy(result["setup_plan"]),
            adopted_revision=deepcopy(plan["adopted_revision"]),
            native_configuration=deepcopy(request["profile"]),
            raw_build_request=deepcopy(result["raw_build_request"]),
            baseline_renderer=deepcopy(result["baseline_renderer"]),
            camera_binding=deepcopy(certificate["inputs"]["camera_binding"]),
            raw_acquisition=deepcopy(request["baseline"]), physical_archives=selected))


def facies_support(context, width_mm, *, reader):
    """Mapped sand outside the band along the rig frame, with its facies, for one light field.

    The band (``width_mm`` inward from the sand facing the frame; 0 for none)
    can hold indicator that already changed colour at the start, so it is left
    out. Same adopted material roles as ``stable_support``.
    """
    if (isinstance(width_mm, bool) or not isinstance(width_mm, (int, float))
            or not np.isfinite(width_mm) or width_mm < 0):
        raise ValueError("Choose the band along the frame in millimetres (0 leaves nothing out)")
    sand, band, table = _frame_band(context, width_mm, reader=reader)
    mask = sand & ~band
    if not mask.any():
        raise ValueError("The band along the frame leaves no mapped sand for the light field")
    return dict(mask=mask, facies=np.asarray(context.facies), width_mm=float(width_mm),
        pixel_size_mm=(np.asarray(context.grid["voxel_size"], float) * 1000.).tolist(),
        domain="mapped sand outside the band along the frame; one level per facies",
        adopted_revision=deepcopy(context.provenance["adopted_revision"]),
        material_table=deepcopy(table),
        source_labels=deepcopy(context.provenance["physical_archives"]["labels"]),
        source_facies=deepcopy(context.provenance["physical_archives"]["facies"]))


def _frame_band(context, width_mm, *, reader):
    """Mapped sand and its band ``width_mm`` inward from the sand facing the frame (adopted roles only)."""
    from ..geometry_raster_sources import _full_csv
    adopted = reader.json(context.provenance["adopted_revision"])
    table = adopted["artifacts"]["source_mode_files"]["facies_properties"]
    columns, rows, ids = _full_csv(reader.path(table), strict=True)
    names = {name.strip().casefold(): name for name in columns}
    roles = {key: row[names["analysis_role"]] for key, row in zip(ids, rows)}
    config = adopted["configuration"]
    enabled = [row["label_id"] for row in config["regions"] if row.get("enabled", True)]
    known = np.isin(context.labels, enabled)
    sand = known & np.isin(context.facies, [key for key, role in roles.items() if role == "porous_medium"])
    if width_mm == 0:
        return sand, np.zeros(sand.shape, bool), table
    other = known & np.isin(context.facies, [key for key, role in roles.items()
        if role not in {"porous_medium", "frame", "silicone", "excluded"}])
    boundary = frame_facing_sand_boundary(sand, other_material=other)
    spacing = np.asarray(context.grid["voxel_size"], float) * 1000.
    return sand, stable_band_mask(sand, width_mm, spacing, frame_facing_boundary=boundary), table


def stable_support(context, width_mm, *, reader):
    """Use the adopted material roles, never guessed labels or phase predictions."""
    _sand, mask, table = _frame_band(context, width_mm, reader=reader)
    spacing = np.asarray(context.grid["voxel_size"], float) * 1000.
    if not mask.any():
        raise ValueError("The selected stable band contains no mapped sand next to the frame")
    return dict(mask=mask, width_mm=float(width_mm), pixel_size_mm=spacing.tolist(),
        domain="mapped sand facing the frame; water surfaces and internal boundaries excluded",
        adopted_revision=deepcopy(context.provenance["adopted_revision"]),
        material_table=deepcopy(table),
        source_labels=deepcopy(context.provenance["physical_archives"]["labels"]),
        source_facies=deepcopy(context.provenance["physical_archives"]["facies"]))


def prepare_image_photometry(operations, choice, choice_authority, output, *,
        reader=None, on_progress=None, context_loader=None, collector=None, producer=None,
        workers=1, materialize_baselines=False):
    """Fit one shared preparation once, retaining only reference and current RAW.

    Called from the existing GUI preparation job. Old saved-native choices
    deliberately retain their original route. This never changes the choice.
    """
    from ..native.full_photometry import collect_baseline, prepare_independent_photometry
    reader = current_input_reader(reader)
    corrections = selected_corrections(choice)
    if corrections["color_method"] == "saved_native":
        if choice.get("stable_area") or not corrections["lighting"]:
            raise ValueError("Choose a current colour method to use an independent stable-area light fit")
        return None
    if "color_method" not in choice and not corrections["lighting"]:
        return None  # Existing Off records retain their original producer.
    if corrections.get("light_field"):
        if not choice.get("exclusion_band"):
            raise ValueError("Choose the band along the frame in Image preparation before fitting lighting")
    elif corrections["lighting"] and not choice.get("stable_area"):
        raise ValueError("Choose the stable sand-band width in Image preparation before fitting lighting")
    root = Path(output)
    cfa_policy = None
    if corrections["color_method"] == "uniform24":
        from ..native.cfa_clipping_diagnostics import FIT_VALIDITY_POLICY
        cfa_policy = FIT_VALIDITY_POLICY
    saved_path = root / "independent_photometry.json"
    if saved_path.is_file():
        saved = reader.json(asset(saved_path))
        if set(saved["runs"]) != set(operations):
            raise ValueError("Prepared image corrections belong to a different run selection")
        for run, row in saved["runs"].items():
            bound = reader.json(row["source_binding"])
            model = reader.json(bound["models"]["fit"]["sidecar"])
            if (model["choice"] != choice or model["source"]["operation_result"] != operations[run]):
                raise ValueError("Prepared image corrections belong to another choice or Build")
            if model.get("cfa_validity_policy") != cfa_policy:
                raise ValueError("Prepared colour corrections use another measured-CFA validity policy")
        return saved
    loader = context_loader or load_image_context
    collect = collector or collect_baseline

    class Supports(dict):
        def get(self, key, default=None):
            # The producer asks immediately after loading each baseline. Drop
            # the full native mask after fitting that run instead of retaining
            # a second 39-run image cache.
            return self.pop(key, default)

    supports = Supports()
    count = 0
    def baseline(run):
        nonlocal count
        if on_progress:
            on_progress(dict(state="preparing_image_corrections", run=run,
                             position=count, total=len(operations)))
        context = loader(operations[run], reader=reader)
        if corrections.get("light_field"):
            supports[run] = facies_support(context, choice["exclusion_band"]["width_mm"], reader=reader)
        elif corrections["lighting"]:
            supports[run] = stable_support(context, choice["stable_area"]["width_mm"], reader=reader)
        value = collect(context, reader, require_chart=corrections["color_method"] != "off")
        count += 1
        return value
    lazy = {run: (lambda run=run: baseline(run)) for run in operations}
    options = dict(choice_authority=choice_authority)
    if cfa_policy is not None:
        options["cfa_validity_policy"] = cfa_policy
    if materialize_baselines:
        options["materialize_baselines"] = True
    if int(workers) == 1 or len(operations) == 1:
        return (producer or prepare_independent_photometry)(lazy, choice, supports, root, **options)
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    from ..execution import detect_host_resources
    resources = detect_host_resources()
    limit = max(1, min(int(workers) or resources["cpu"]["physical_cores"], len(operations),
        resources["memory"]["available_bytes"] // (12 * 1024**3)))
    # Bounded read-only prefetch: reference + at most the actual worker budget.
    # The original producer keeps its ordered fits/writes and one target at a time.
    reference = choice.get("reference_run", next(iter(operations)))
    order = iter([reference, *[run for run in operations if run != reference]])
    with ThreadPoolExecutor(max_workers=limit) as pool:
        pending = {}
        def fill():
            while len(pending) < limit:
                run = next(order, None)
                if run is None:
                    break
                pending[run] = pool.submit(copy_context().run, baseline, run)
        def get(run):
            fill()
            value = pending.pop(run).result()
            fill()
            return value
        fill()
        prefetched = {run: (lambda run=run: get(run)) for run in operations}
        return (producer or prepare_independent_photometry)(prefetched, choice, supports, root, **options)


def preview_from_operation(operation, width_mm, *, reader=None, context_loader=None,
                           context_cache=None, display_loader=None, excluded=False):
    """Read just the selected completed baseline; preview writes no job files.

    The band is the same pixels for both lighting methods. With ``excluded``
    it is the band that sand-type equalisation leaves out of its light fit.
    """
    reader = current_input_reader(reader)
    # One selected baseline only. Changing the band must not decode the same
    # Build archive again; a changed immutable operation pin invalidates it.
    cached = context_cache if context_cache is not None else {}
    if cached.get("operation") == operation and context_cache is not None:
        reader.json(operation)  # Recheck the selected small immutable receipt.
        archives = _preview_assets(cached["context"])
        if _preview_archive_stamps(archives, reader) != cached.get("archive_stamps"):
            cached.clear()
    if cached.get("operation") != operation:
        context = (context_loader(operation, reader=reader) if context_loader is not None
                   else load_image_context(operation, reader=reader, include_baseline=False))
        shown = (display_loader or render_preparation_display)(context, reader=reader)
        rgb = shown["rgb"]
        cached.clear()
        cached.update(operation=deepcopy(operation), context=context, rgb=rgb,
                      display=deepcopy(shown["display"]),
                      archive_stamps=None if context_cache is None else _preview_archive_stamps(
                          _preview_assets(context), reader))
    context, rgb = cached["context"], cached["rgb"]
    display_note = "Camera display · before selected lighting/ColorChecker correction. "
    note = display_note + ("Band left out of the light fit, measured inward from the sand-map edge facing the frame."
                           if excluded else "Stable area measured inward from the sand-map edge facing the frame.")
    support, error = None, None
    if width_mm > 0:
        try:
            support = stable_support(context, width_mm, reader=reader)
        except ValueError as failure:
            # This is a display only: never substitute a guessed mask for an
            # unavailable one, but retain the exact selected baseline.
            error = str(failure)
            note = display_note + ("Band unavailable: " if excluded else "Stable area unavailable: ") + error
    else:
        note = display_note + ("No band is left out; all mapped sand enters the light fit." if excluded
                               else "Choose a positive stable-band width to display its sampled area.")
    return dict(run=context.run, rgb=rgb, stable_mask=None if support is None else support["mask"],
        grid=deepcopy(context.grid), baseline=deepcopy(context.provenance),
        display=deepcopy(cached["display"]),
        source_pins={} if support is None else {
            key: deepcopy(value) for key, value in support.items() if key != "mask"},
        stable_area_error=error, note=note)


def render_preparation_display(context, *, reader, display_reader=None):
    """Place the Layers camera-WB display on the exact certified native grid."""
    import cv2
    from ..native.raw_full_display import read_full_raw_display
    provenance, renderer = context.provenance, context.display_renderer
    if not isinstance(renderer, dict) or renderer.get("source_kind") != "raw":
        raise ValueError("This Build has no declared RAW photographic display renderer")
    raw, profile = provenance["raw_acquisition"], provenance["native_configuration"]
    shape = tuple(context.grid["num_voxels"])
    if (renderer.get("raw_acquisition") != raw or tuple(renderer.get("native_shape_hw", ())) != shape
            or context.raw_orientation is None):
        raise ValueError("The display photograph and native crop need the same certified acquisition")
    located_raw = reader.physical_descriptor(raw)
    packet = (display_reader or read_full_raw_display)(located_raw["path"],
        profile_path=reader.physical_descriptor(profile)["path"], profile_sha256=profile["sha256"],
        profile_id=renderer["profile_id"], expected_sha256=raw["sha256"],
        expected_orientation=context.raw_orientation)
    rgb = np.asarray(packet["image_rgb"])
    if (packet.get("display_only") is not True or packet.get("image_encoding") != "rgb_uint8"
            or packet["source_raw"]["sha256"] != raw["sha256"]
            or packet["profile"]["sha256"] != profile["sha256"]
            or packet["profile"]["profile_id"] != renderer["profile_id"]
            or packet["raw_orientation"] != context.raw_orientation
            or tuple(packet["camera_u_wh"]) != tuple(renderer["camera_u_wh"])
            or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3):
        raise ValueError("The photographic display does not match the certified RAW camera grid")
    transform = np.asarray(packet["camera_u_to_photo_h"], float) @ np.asarray(renderer["native_to_camera_u_h"], float)
    if transform.shape != (3, 3) or not np.isfinite(transform).all() or abs(np.linalg.det(transform)) < 1e-12:
        raise ValueError("The photographic display has no valid native-to-photo transform")
    shown = cv2.warpPerspective(rgb, transform, (shape[1], shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return dict(rgb=shown, display=dict(display_only=True, raw_acquisition=deepcopy(raw),
        native_configuration=deepcopy(profile), baseline_renderer=deepcopy(provenance["baseline_renderer"]),
        conversion=deepcopy(packet["display_conversion"]), raw_orientation=packet["raw_orientation"],
        native_to_photo_h=transform.tolist(), native_shape_hw=list(shape),
        corrections_applied=False, display_cache_reused=packet.get("display_cache_reused", False)))


def _preview_assets(context):
    pins = dict(context.provenance.get("physical_archives", {}))
    for name in ("raw_acquisition", "native_configuration"):
        if name in context.provenance:
            pins[name] = context.provenance[name]
    return pins


def _preview_archive_stamps(archives, reader):
    """Observe only selected files; a changed archive must pass its SHA again."""
    stamps = {}
    for role, pin in archives.items():
        path = (Path(reader.physical_descriptor(pin)["path"])
                if hasattr(reader, "physical_descriptor") else Path(pin["path"]))
        stat = path.stat()
        stamps[role] = (stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino)
    return stamps


def proposal_from_operations(operations, width_mm, use_lighting, use_color, *, reader=None,
        check_cancelled=None, on_progress=None, context_loader=None, collector=None, light_field=None):
    """Read baselines once each and retain only small comparable optical samples.

    With ``light_field`` (the saved choice's light-field record), each start
    photo is compared with its own same-facies light field divided out, on the
    sand outside the band ``width_mm`` along the frame.
    """
    from ..native.full_photometry import collect_baseline
    reader = current_input_reader(reader)
    values, pins, grid = [], {}, None
    for index, (run, operation) in enumerate(operations.items()):
        if check_cancelled:
            check_cancelled()
        if on_progress:
            on_progress(dict(run=run, position=index, total=len(operations)))
        context = (context_loader or load_image_context)(operation, reader=reader)
        current_grid = {key: context.grid[key] for key in ("num_voxels", "dimensions", "voxel_size")}
        if grid is not None and grid != current_grid:
            raise ValueError("Reference candidates need the same physical crop and image grid")
        grid = current_grid
        if use_lighting and light_field is not None:
            from ..native.full_photometry import _facies_light_fit, _light_on_grid, _working_rgb
            support = facies_support(context, width_mm, reader=reader)
            baseline = (collector or collect_baseline)(context, reader, require_chart=use_color)
            cell = int(baseline["parameters"]["photometry"]["native_area_log_int_8"])
            shape = tuple(max(1, int(n) // cell) for n in baseline["rgb_native"].shape[:2])
            light, mask = _facies_light_fit(baseline["rgb_native"], support, shape,
                degree=light_field["degree"], native_to_u=baseline["native_to_camera_u_h"],
                chart_roi_rc=baseline.get("chart_roi_rc"))
            flat = _working_rgb(baseline["rgb_native"], shape) * np.exp(
                _light_on_grid(light, shape)).astype(np.float32)[..., None]
            values.append(reference_samples(run, flat, mask, chart_rgb=baseline.get("chart_rgb")))
        else:
            support = stable_support(context, width_mm, reader=reader) if use_lighting else None
            baseline = (collector or collect_baseline)(context, reader, require_chart=use_color)
            mask = support["mask"] if support else np.zeros(context.labels.shape, bool)
            values.append(reference_samples(run, baseline["rgb_native"], mask,
                                            chart_rgb=baseline.get("chart_rgb")))
        pins[run] = deepcopy(baseline["source_pins"])
        del baseline, context, support, mask
    result = propose_reference(values, use_lighting=use_lighting, use_color=use_color)
    for row in result["ranking"]:
        row["score"] = row["mean_log_correction"]
    result.update(source_pins=pins,
        note="Smallest mean optical correction across baseline images; no chemical or phase ranking.")
    return result


def colour_fit_preview_from_operations(operations, run, choice, *, reader=None,
        check_cancelled=None, on_progress=None, context_loader=None, collector=None,
        cfa_validity_policy=None):
    """Fit only the selected baseline/reference in memory with the actual producer."""
    from ..image_correction import NeutralToneFitError, linear_to_srgb
    from ..native.full_photometry import collect_baseline, _colour_full
    reader = current_input_reader(reader)
    method = selected_corrections(choice)["color_method"]
    reference_run = choice.get("reference_run")
    if method not in ("uniform24", "darsia_original"):
        raise ValueError("Save an independent ColorChecker correction before checking its fit")
    if run not in operations or reference_run not in operations:
        raise ValueError("The selected baseline and saved reference need completed current geometry")
    def collect(selected):
        if check_cancelled:
            check_cancelled()
        if on_progress:
            on_progress(dict(run=selected))
        context = ((context_loader or load_image_context)(operations[selected], reader=reader,
                                                         include_baseline=False))
        return (collector or collect_baseline)(context, reader, require_chart=True)
    reference = collect(reference_run)
    ref_chart = linear_to_srgb(reference["chart_rgb"], dtype=np.float32)
    reference_validity = reference_evidence = current_evidence = None
    if cfa_validity_policy is not None:
        from ..native.cfa_clipping_diagnostics import FIT_VALIDITY_POLICY, chart_fit_validity
        if cfa_validity_policy != FIT_VALIDITY_POLICY or method != "uniform24":
            raise ValueError("Unknown or inapplicable preview CFA colour-fit validity policy")
        reference_validity, reference_evidence = chart_fit_validity(reference["sensor"],
            reference["parameters"]["registration"], reference["chart_roi_rc"],
            camera_rgb=reference["rgb_full"])
    current, stage = reference, "reference"
    try:
        for selected in dict.fromkeys((reference_run, run)):
            if selected != reference_run:
                current, stage = collect(selected), "selected baseline"
            validity = {}
            if cfa_validity_policy is not None:
                if selected == reference_run:
                    current_validity, current_evidence = reference_validity, reference_evidence
                else:
                    current_validity, current_evidence = chart_fit_validity(current["sensor"],
                        current["parameters"]["registration"], current["chart_roi_rc"],
                        camera_rgb=current["rgb_full"])
                validity = dict(source_validity=current_validity, reference_validity=reference_validity,
                    validity_evidence=dict(source=current_evidence, reference=reference_evidence))
            corrected, fitted = _colour_full(
                current["rgb_full"] / np.float32(current["white_range_dn"]),
                method, ref_chart, current.get("chart_roi_rc"), **validity)
            del corrected
        return dict(run=run, reference_run=reference_run, method=method, accepted=True,
            complete=False, diagnostics=deepcopy(fitted), choice=deepcopy(choice),
            source=deepcopy(current["source_pins"]),
            note="Read-only ColorChecker fit preview; no model, job result or Setup completion was saved.")
    except NeutralToneFitError as error:
        sensor = current.get("sensor")
        clipping = current_evidence or dict(status="unavailable", reason="This baseline has no original sensor context")
        if sensor is not None and current_evidence is None:
            try:
                from ..native.cfa_clipping_diagnostics import chart_clipping_diagnostics
                clipping = chart_clipping_diagnostics(sensor, current["parameters"]["registration"],
                                                      error.diagnostics["roi_rc"])
            except Exception as diagnostic_error:
                # Optional evidence never substitutes a fit or hides the original rejection.
                clipping = dict(status="unavailable", reason=str(diagnostic_error))
        return dict(run=run, reference_run=reference_run, failed_run=current["run"], stage=stage,
            method=method, accepted=False, complete=False, error=str(error),
            diagnostics=deepcopy(error.diagnostics), choice=deepcopy(choice),
            source=deepcopy(current["source_pins"]),
            sensor={} if sensor is None else {key:deepcopy(sensor.metadata[key]) for key in (
                "white_level", "black_level", "sensor_saturated_count") if key in sensor.metadata},
            sensor_clipping=clipping,
            note="Rejected by the original physical bounds. No model, job result or Setup completion was saved.")


def _registered_build_operations(root, series_id, registrations, *, reader=None, snapshot_reader=None,
                                 operations_cache=None):
    """Find current geometry in explicitly registered jobs, using metadata only."""
    from ..jobs_metadata import JobSnapshotReader
    from ..series_geometry import load_series_geometry_project
    from ..geometry_facies import GeometryRevisionStore
    from ..series_native_context import read_series_native_binding
    reader = current_input_reader(reader)
    native = read_series_native_binding(root)
    if native is None:
        raise ValueError("Save the current camera configuration before previewing image preparation")
    project = load_series_geometry_project(Path(root))
    if project is None:
        raise ValueError("Save the reference geometry before previewing image preparation")
    current_geometry = GeometryRevisionStore(project).current_path
    current = reader.json(asset(current_geometry))
    registrations = tuple(registrations)
    key = None if operations_cache is None else (str(root), series_id, native.sha256, current,
           [(row.get("root"), row.get("job")) for row in registrations])
    if operations_cache is not None and operations_cache.get("authority") == key:
        return deepcopy(operations_cache["operations"])
    read_snapshot = snapshot_reader or JobSnapshotReader()
    for registration in reversed(registrations):
        snapshot = read_snapshot(registration)
        facts = snapshot.get("job_facts", {})
        if facts.get("pipeline") != "series-setup" or facts.get("series_id") != series_id:
            continue
        if snapshot.get("errors"):
            image_errors = [row.get("error") for row in snapshot.get("runs", [])
                            if row.get("operation") == "image-preparation"
                            and row.get("state") in ("Failed", "Metadata error")]
            if (any(error not in image_errors for error in snapshot["errors"])
                    or any(row.get("state") == "Metadata error"
                           and row.get("operation") != "image-preparation"
                           for row in snapshot.get("runs", []))):
                continue
        plan_pin = facts.get("setup_plan") or {}
        if not plan_pin.get("sha256"):
            continue
        path = Path(snapshot["root"]) / "setup-controls" / "plan.json"
        plan = reader.json(dict(path=str(path), sha256=plan_pin["sha256"],
                               size_bytes=plan_pin.get("bytes", plan_pin.get("size_bytes"))))
        if reader.json(plan["adopted_revision"]) != current:
            continue
        builds = plan.get("build_plans", [])
        if not builds or any(row["native_parameters"]["sha256"] != native.sha256 for row in builds):
            continue
        operations = {row["run"]: row["result"]["files"]["operation_result.json"]
                      for row in snapshot["runs"]
                      if row.get("operation") == "build" and row.get("state") == "Complete"}
        # Geometry is an input to image preparation, not its output. The new
        # final image-preparation operation may still be pending or failed.
        if set(operations) == {row["run_id"] for row in builds}:
            if operations_cache is not None:
                operations_cache.clear()
                operations_cache.update(authority=deepcopy(key), operations=deepcopy(operations))
            return operations
    raise ValueError("Complete the geometry Build for the current crop before previewing image preparation")


def get_preparation_preview(root, series_id, run, band_width_mm, *, registrations,
                            check_cancelled=None, on_progress=None, context_cache=None,
                            operations_cache=None, excluded=False):
    if check_cancelled:
        check_cancelled()
    operations = _registered_build_operations(root, series_id, registrations,
                                              operations_cache=operations_cache)
    if run not in operations:
        raise ValueError("The selected run is not in the current completed geometry Build")
    result = preview_from_operation(operations[run], band_width_mm, context_cache=context_cache,
                                    excluded=excluded)
    if check_cancelled:
        check_cancelled()
    return result


def get_reference_proposal(root, series_id, band_width_mm, use_lighting, use_color, *,
                          registrations, check_cancelled=None, on_progress=None, light_field=None):
    if check_cancelled:
        check_cancelled()
    operations = _registered_build_operations(root, series_id, registrations)
    return proposal_from_operations(operations, band_width_mm, use_lighting, use_color,
                                   check_cancelled=check_cancelled, on_progress=on_progress,
                                   light_field=light_field)


def get_colour_fit_preview(root, series_id, run, *, registrations, check_cancelled=None,
                          on_progress=None, operations_cache=None):
    from ..image_normalisation import saved_choice
    from ..normalisation_step import read_series_context
    _raw, document, runs = read_series_context(root, series_id)
    choice = saved_choice(document, series_id, runs)
    if choice is None:
        raise ValueError("Save image settings before checking the ColorChecker fit")
    operations = _registered_build_operations(root, series_id, registrations,
                                              operations_cache=operations_cache)
    policy = None
    if selected_corrections(choice)["color_method"] == "uniform24":
        from ..native.cfa_clipping_diagnostics import FIT_VALIDITY_POLICY
        policy = FIT_VALIDITY_POLICY
    return colour_fit_preview_from_operations(operations, run, choice,
        check_cancelled=check_cancelled, on_progress=on_progress, cfa_validity_policy=policy)
