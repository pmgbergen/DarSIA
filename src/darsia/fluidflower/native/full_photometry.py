"""Correct the complete lens-corrected RAW photograph before the common crop.

The original chart matcher, light fitter, colour/field arithmetic and measured
Build transform remain the producers. The prepared model records this distinct
pixel basis; historical native-crop photometry remains a separate mode.
"""
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from . import photometry, raw_chart, raw_positioning, source

ORDER = "darsia-full-photograph-colour-light-before-crop-v1"
INDEPENDENT = "darsia-independent-full-photometry-v1"


def collect_baseline(context, reader=None, *, require_chart=True, chart_roi_rc=None):
    """Read the current Build's RAW/full-photo basis without any old light fit."""
    from ..pipeline.current_inputs import current_input_reader
    from ..pipeline.current_phase_recipe import _runtime_parameters
    reader = current_input_reader(reader)
    pins = deepcopy(context.provenance)
    parameters = _runtime_parameters(pins["native_configuration"], reader)
    certificate = reader.json(pins["baseline_renderer"])
    renderer = certificate["renderer"]
    camera = pins["camera_binding"]
    acquisition = dict(pins["raw_acquisition"])
    acquisition["path"] = str(reader.path(acquisition))
    sensor = raw_positioning.read_sensor(acquisition["path"],
        expected_sha256=acquisition["sha256"], parameters=parameters["registration"],
        profile_id=camera["profile_id"])
    full = raw_positioning.camera_u(sensor, parameters=parameters["registration"])
    h = np.asarray(renderer["native_to_camera_u_h"], np.float64)
    shape = tuple(context.grid["num_voxels"])
    if (full.dtype != np.float32 or full.ndim != 3 or full.shape[0] != 3
            or h.shape != (3, 3) or not np.isfinite(h).all()
            or len(shape) != 2 or abs(np.linalg.det(h)) < 1e-12
            or renderer["profile_id"] != camera["profile_id"]):
        raise ValueError("The completed Build has no matching full RAW photograph geometry")
    scale = float(sensor.metadata["white_level"] - sensor.metadata["black_level"])
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("The RAW photograph has no positive sensor range")
    native = _native_rgb(full / np.float32(scale), h, shape)
    values = dict(run=context.run, rgb_full=full, rgb_native=native,
        native_to_camera_u_h=h, white_range_dn=scale, sensor=sensor,
        source_pins=pins, parameters=parameters, grid=deepcopy(context.grid),
        shape_assets=deepcopy(renderer["shape_corrections"]),
        image_metadata=deepcopy(certificate["inputs"]["uncorrected_grid"]["metadata"]))
    values["chart_rgb"] = values["chart_roi_rc"] = None
    if require_chart:
        chart, roi = _chart(full / np.float32(scale), chart_roi_rc)
        from ..image_correction import srgb_to_linear
        values.update(chart_rgb=srgb_to_linear(chart, dtype=np.float32), chart_roi_rc=roi)
    return values


def _native_rgb(full_unit, native_to_u, shape):
    """Sample linear full-photo values on the already measured Build grid."""
    height, width = map(int, shape)
    transform = np.linalg.inv(np.asarray(native_to_u, np.float64))
    return np.stack([cv2.warpPerspective(channel, transform, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan")) for channel in full_unit], axis=-1).astype(np.float32)


def _chart(unit_rgb, roi=None):
    """Use the existing detector/extractor, with its explicit sRGB interface."""
    from ..image_correction import _reference_colorchecker, linear_to_srgb
    srgb = linear_to_srgb(np.moveaxis(unit_rgb, 0, -1), dtype=np.float32)
    cv2.setRNGSeed(0)
    if roi is None:
        from ..phase_training.radiometry import detect_chart
        _, roi, _ = detect_chart(np.nan_to_num(srgb, nan=0., posinf=0., neginf=0.))
    return _reference_colorchecker(srgb, corner="upper_right", roi_rc=roi)


def _colour_full(unit_rgb, method, reference_swatches, roi, *, source_validity=None,
                 reference_validity=None, validity_evidence=None):
    """Call the existing two colour producers; neither adds a light field."""
    if method == "off":
        return np.asarray(unit_rgb, np.float32), dict(method="off")
    from .. import image_correction as original
    srgb = original.linear_to_srgb(np.moveaxis(unit_rgb, 0, -1), dtype=np.float32)
    reference_swatches = np.asarray(reference_swatches, np.float32)
    cv2.setRNGSeed(0)
    if method == "darsia_original":
        import darsia
        correction = darsia.ColorCorrection(
            base=darsia.CustomColorChecker(reference_colors=reference_swatches),
            config=dict(roi=np.asarray(roi).tolist(), active=True, whitebalancing=True,
                        colorbalancing="affine", balancing="darsia", clip=False))
        corrected = correction.correct_array(srgb)
        details = dict(method=method, producer="darsia.ColorCorrection",
            whitebalancing=True, colorbalancing="affine", balancing="darsia", clip=False)
    elif method == "uniform24":
        validity = {}
        if source_validity is not None or reference_validity is not None:
            validity = dict(source_validity=source_validity, reference_validity=reference_validity,
                            validity_evidence=validity_evidence)
        fitted = original.fit_uniform24_to_reference_swatches(srgb, reference_swatches,
            roi_rc=np.asarray(roi).tolist(), settings=dict(clip=False), **validity)
        corrected = original.apply_uniform24_array_transform(srgb, fitted["transform"])
        details = dict(method=method, producer="fit_uniform24_to_reference_swatches",
                       transform=fitted["transform"], diagnostics=fitted["diagnostics"])
    else:
        raise ValueError("Unknown independently prepared colour method")
    output = np.moveaxis(original.srgb_to_linear(corrected, dtype=np.float32), -1, 0)
    output[:, ~np.isfinite(unit_rgb).all(axis=0)] = np.nan
    return output, details


def _working_rgb(rgb, shape):
    return cv2.resize(np.asarray(rgb, np.float32), tuple(map(int, shape[::-1])),
                      interpolation=cv2.INTER_AREA)


def _light_fit(target, reference, mask, parameters):
    """Reuse the original robust optical polynomial solver on stable pixels."""
    target, reference = np.asarray(target), np.asarray(reference)
    mask = np.asarray(mask, bool)
    if target.shape != reference.shape or target.shape != mask.shape + (3,):
        raise ValueError("Light fitting requires the same prepared physical grid")
    valid = (mask & np.isfinite(target).all(axis=2) & np.isfinite(reference).all(axis=2)
             & (target > 0).all(axis=2) & (reference > 0).all(axis=2)
             & (target < 1).all(axis=2) & (reference < 1).all(axis=2))
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.log(reference[..., 1]) - np.log(target[..., 1])
    shape = mask.shape
    grid = photometry.ReferenceGrid(shape, (0, 0), shape, parameters=parameters)
    coefficients, intercepts, counts = photometry._fit_observed_field(
        delta, mask.astype(np.int16), valid, grid, parameters=parameters)
    return dict(coefficients=coefficients.tolist(), intercept=intercepts[1],
        sample_count=counts[1], grid_shape_hw=list(shape),
        producer="darsia.fluidflower.native.photometry._fit_observed_field",
        signal="log(reference linear green) - log(current linear green)")


def _facies_light_fit(rgb_native, support, shape, *, degree, native_to_u=None, chart_roi_rc=None):
    """One run's light field from same-facies sand on its start photo, as the field that divides it out.

    ``support`` holds the native sand mask outside the excluded band and the
    native facies ids. The fit is evaluated as zero at the ColorChecker centre
    (at the sand median without a chart), so the chart's colour correction
    keeps its level and only the spatial light variation is removed. Returns
    the field for ``_apply_light`` and the working-grid cells it was fitted on.
    """
    from ..optical_reference import facies_light_field, polynomial_terms
    native_mask = np.asarray(support["mask"], bool)
    facies = np.asarray(support["facies"])
    if native_mask.shape != facies.shape or native_mask.shape != tuple(np.shape(rgb_native)[:2]):
        raise ValueError("The same-facies sand differs from its completed Build grid")
    cells = np.full(shape, -1, np.int64)
    for value in np.unique(facies[native_mask]):
        whole = cv2.resize(((facies == value) & native_mask).astype(np.float32), tuple(shape[::-1]),
                           interpolation=cv2.INTER_AREA) >= .999
        cells[whole] = int(value)
    green = _working_rgb(rgb_native, shape)[..., 1].astype(np.float64)
    usable = (cells >= 0) & np.isfinite(green) & (green > 0) & (green < 1)
    log_green = np.full(shape, np.nan)
    log_green[usable] = np.log(green[usable])
    fit = facies_light_field(log_green, np.where(usable, cells, -1), degree=degree)
    gh, gw = shape
    nh, nw = native_mask.shape
    if chart_roi_rc is not None and native_to_u is not None:
        corners = np.asarray(chart_roi_rc, float)  # (row, column) corners in the full photograph
        centre = np.linalg.inv(np.asarray(native_to_u, float)) @ np.array(
            [corners[:, 1].mean(), corners[:, 0].mean(), 1.0])
        xn, yn = centre[0] / centre[2], centre[1] / centre[2]
        x = (((xn + .5) * gw / nw - .5) - gw / 2) / (gw / 2)
        y = (((yn + .5) * gh / nh - .5) - gh / 2) / (gh / 2)
        anchor, at = float(polynomial_terms(x, y, degree) @ fit["coefficients"]), "colorchecker"
    else:
        rows, cols = np.nonzero(usable)
        anchor = float(np.median(polynomial_terms((cols - gw / 2) / (gw / 2), (rows - gh / 2) / (gh / 2),
                                                  degree) @ fit["coefficients"]))
        at = "sand-median"
    light = dict(coefficients=(-fit["coefficients"]).tolist(), intercept=anchor, grid_shape_hw=list(shape),
        producer="darsia.fluidflower.optical_reference.facies_light_field",
        signal="log(start-photo linear green) = facies level + shared field; the field is divided out",
        degree=fit["degree"], block_cells=fit["block_cells"], sample_count=fit["sample_count"],
        facies_levels={str(key): value for key, value in fit["facies_levels"].items()},
        interquartile_log_spread={str(key): value for key, value in fit["interquartile_log_spread"].items()},
        anchor=at, excluded_band_mm=float(support["width_mm"]))
    return light, usable


def _light_on_grid(light, shape):
    """Evaluate a fitted light field at the centres of ``shape``'s cells (its own working grid)."""
    from ..optical_reference import polynomial_terms
    gh, gw = shape
    rows, cols = np.mgrid[0:gh, 0:gw]
    coefficients = np.asarray(light["coefficients"], float)
    degree = int(round((np.sqrt(8 * (len(coefficients) + 1) + 1) - 3) / 2))
    return light["intercept"] + polynomial_terms((cols - gw / 2) / (gw / 2), (rows - gh / 2) / (gh / 2),
                                                 degree) @ coefficients


def _lighting_application(corrections, source_purpose=None):
    """How the light field reaches a photograph, recorded with every corrected image."""
    if corrections.get("light_field"):
        return "start-photo-field"
    return "fixed-baseline-field" if source_purpose == "clearwater" else "per-frame-stable-area"


def _apply_light(unit_rgb, fitted, native_to_u, native_shape, *, tile_rows=128):
    """Evaluate the fitted scalar field equally on RGB; source pixels stay put."""
    output = np.empty_like(unit_rgb)
    inverse = np.linalg.inv(np.asarray(native_to_u, np.float64))
    gh, gw = fitted["grid_shape_hw"]
    nh, nw = native_shape
    coefficients = fitted["coefficients"]
    degree_max = int(round((np.sqrt(8 * (len(coefficients) + 1) + 1) - 3) / 2))
    for first in range(0, output.shape[1], tile_rows):
        stop = min(first + tile_rows, output.shape[1])
        yy, xx = np.mgrid[first:stop, :output.shape[2]]
        den = inverse[2, 0] * xx + inverse[2, 1] * yy + inverse[2, 2]
        if np.any(np.abs(den) < 1e-12):
            raise ValueError("Light field projects to reference infinity")
        xn = np.clip((inverse[0, 0]*xx + inverse[0, 1]*yy + inverse[0, 2])/den, 0, nw-1)
        yn = np.clip((inverse[1, 0]*xx + inverse[1, 1]*yy + inverse[1, 2])/den, 0, nh-1)
        # OpenCV's resized sample centres, then ReferenceGrid.normalized_coordinates.
        x = (((xn + .5)*gw/nw - .5) - gw/2)/(gw/2)
        y = (((yn + .5)*gh/nh - .5) - gh/2)/(gh/2)
        field = np.full(x.shape, fitted["intercept"], np.float64)
        index = 0
        for degree in range(1, degree_max + 1):
            for ax in range(degree + 1):
                field += coefficients[index] * x**ax * y**(degree-ax)
                index += 1
        if index != len(coefficients):
            raise ValueError("Unknown optical polynomial coefficient count")
        output[:, first:stop] = unit_rgb[:, first:stop] * np.exp(field).astype(np.float32)[None]
    return output


def prepare_independent_photometry(baselines, choice, stable_supports, output, *, choice_authority,
                                   materialize_baselines=False, cfa_validity_policy=None):
    """Prepare one reference plus one target at a time, from actual Build sources.

    ``baselines`` maps run IDs to lazy zero-argument loaders (or already loaded
    dictionaries from ``collect_baseline``). Only the chosen reference is kept.
    ``stable_supports`` maps runs to their boolean native masks and provenance.
    This writes inputs for the same per-frame consumer used by training/P1.
    """
    from ..image_normalisation import selected_corrections
    from ..mass.n1_pipeline import _write_json, _write_npz, asset
    from ..pipeline.job_queue import name
    corrections = selected_corrections(choice)
    if cfa_validity_policy is not None:
        from .cfa_clipping_diagnostics import FIT_VALIDITY_POLICY, chart_fit_validity
        if cfa_validity_policy != FIT_VALIDITY_POLICY or corrections["color_method"] != "uniform24":
            raise ValueError("Unknown or inapplicable explicit CFA colour-fit validity policy")
    if corrections["color_method"] == "saved_native":
        raise ValueError("Saved native photometry keeps its original producer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    runs = tuple(baselines)
    if not runs or len(set(runs)) != len(runs):
        raise ValueError("Distinct completed Build baselines are required")
    reference_run = choice.get("reference_run", runs[0])
    if reference_run not in baselines:
        raise ValueError("The selected reference has no completed Build baseline")
    get = lambda run: baselines[run]() if callable(baselines[run]) else baselines[run]
    reference = get(reference_run)
    ref_chart = None
    reference_validity = reference_evidence = None
    if corrections["color_method"] != "off":
        if reference.get("chart_rgb") is None:
            raise ValueError("The selected colour reference has no observed ColorChecker")
        from ..image_correction import linear_to_srgb
        ref_chart = linear_to_srgb(reference["chart_rgb"], dtype=np.float32)
    if cfa_validity_policy is not None:
        reference_validity, reference_evidence = chart_fit_validity(reference["sensor"],
            reference["parameters"]["registration"], reference["chart_roi_rc"],
            camera_rgb=reference["rgb_full"])
    reference_options = ({} if reference_validity is None else dict(
        source_validity=reference_validity, reference_validity=reference_validity,
        validity_evidence=dict(source=reference_evidence, reference=reference_evidence)))
    ref_full, _ = _colour_full(reference["rgb_full"] / np.float32(reference["white_range_dn"]),
        corrections["color_method"], ref_chart, reference.get("chart_roi_rc"), **reference_options)
    ref_native = _native_rgb(ref_full, reference["native_to_camera_u_h"], reference["rgb_native"].shape[:2])
    del ref_full
    choice_pin = _write_json(output/"image_normalisation_choice.json", dict(
        format="darsia-prepared-image-normalisation-v1", choice=deepcopy(choice),
        source_series=deepcopy(choice_authority)))
    result = dict(format=INDEPENDENT, reference_run=reference_run, runs={})
    proposal_samples = []
    for run in runs:
        baseline = reference if run == reference_run else get(run)
        if baseline["run"] != run:
            raise ValueError("A baseline loader returned another run")
        folder = output/name(run)
        folder.mkdir()
        native_shape = baseline["rgb_native"].shape[:2]
        parameters = baseline["parameters"]["photometry"]
        cell = int(parameters["native_area_log_int_8"])
        shape = tuple(max(1, int(n)//cell) for n in native_shape)
        reference_rgb = _working_rgb(ref_native, shape)
        # A lazy collector may publish one large mask with its returned frame.
        # Consume that entry so the coordinator retains only reference+target.
        support = (stable_supports.pop(run, None) if callable(baselines[run])
                   else stable_supports.get(run))
        if corrections["lighting"]:
            if not isinstance(support, dict):
                raise ValueError("Light fitting requires this run's selected stable area")
            mask = np.asarray(support["mask"])
            if mask.dtype != bool or mask.shape != native_shape:
                raise ValueError("The stable area differs from its completed Build grid")
            mask = cv2.resize(mask.astype(np.float32), shape[::-1], interpolation=cv2.INTER_AREA) >= .999
        else:
            mask = np.zeros(shape, bool)
        equalised = None
        if corrections.get("light_field"):
            # The run's own start photo: one light field from same-facies sand
            # outside the excluded band, applied unchanged to all its photos.
            if not isinstance(support, dict) or "facies" not in support:
                raise ValueError("Equalising sand types needs this run's same-facies sand")
            equalised, mask = _facies_light_fit(baseline["rgb_native"], support, shape,
                degree=choice["light_field"]["degree"], native_to_u=baseline["native_to_camera_u_h"],
                chart_roi_rc=baseline.get("chart_roi_rc"))
        if corrections["lighting"] or corrections["color_method"] != "off":
            from ..optical_reference import reference_samples
            if equalised is not None:
                # Runs are compared with their own light divided out, outside the band.
                flat = _working_rgb(baseline["rgb_native"], shape) * np.exp(
                    _light_on_grid(equalised, shape)).astype(np.float32)[..., None]
                proposal_samples.append(reference_samples(run, flat, mask, chart_rgb=baseline.get("chart_rgb")))
            else:
                proposal_samples.append(reference_samples(run, baseline["rgb_native"],
                    np.asarray(support["mask"], bool) if support is not None else np.zeros(native_shape, bool),
                    chart_rgb=baseline.get("chart_rgb")))
        baseline_validity = baseline_evidence = None
        validity = {}
        if cfa_validity_policy is not None:
            if run == reference_run:
                baseline_validity, baseline_evidence = reference_validity, reference_evidence
            else:
                baseline_validity, baseline_evidence = chart_fit_validity(baseline["sensor"],
                    baseline["parameters"]["registration"], baseline["chart_roi_rc"],
                    camera_rgb=baseline["rgb_full"])
            validity = dict(source_validity=baseline_validity, reference_validity=reference_validity,
                            validity_evidence=dict(source=baseline_evidence, reference=reference_evidence))
        corrected, color = _colour_full(baseline["rgb_full"] / np.float32(baseline["white_range_dn"]),
            corrections["color_method"], ref_chart, baseline.get("chart_roi_rc"), **validity)
        if equalised is not None:
            light = equalised
        elif corrections["lighting"]:
            current = _working_rgb(_native_rgb(corrected, baseline["native_to_camera_u_h"], native_shape), shape)
            light = _light_fit(current, reference_rgb, mask, parameters)
        else:
            light = None
        arrays = dict(reference_rgb=reference_rgb, stable_mask=mask,
            reference_swatches=np.empty((0,), np.float32) if ref_chart is None else ref_chart)
        if cfa_validity_policy is not None:
            arrays.update(reference_validity=reference_validity, baseline_validity=baseline_validity)
        data = _write_npz(folder/"model.npz", arrays)
        record = dict(format=INDEPENDENT, run=run, choice=deepcopy(choice),
            data=data, source=deepcopy(baseline["source_pins"]),
            reference_source=deepcopy(reference["source_pins"]),
            stable_support=None if support is None else {key:deepcopy(value) for key,value in support.items()
                                                         if key not in ("mask", "facies")},
            native_shape_hw=list(native_shape), native_to_camera_u_h=baseline["native_to_camera_u_h"].tolist(),
            baseline_chart_roi_rc=None if baseline.get("chart_roi_rc") is None else np.asarray(baseline["chart_roi_rc"]).tolist(),
            baseline_color_fit=color, baseline_light_fit=light,
            correction_order="selected ColorChecker producer; scalar light from stable area; common crop",
            color_interface="sRGB transfer of camera-linear RGB / black-subtracted white range; inverse transfer after colour",
            old_photometry_used=False)
        if cfa_validity_policy is not None:
            record.update(cfa_validity_policy=cfa_validity_policy,
                reference_validity_evidence=reference_evidence, baseline_validity_evidence=baseline_evidence)
        sidecar = _write_json(folder/"model.json", record)
        metadata = deepcopy(baseline["image_metadata"])
        metadata.pop("type", None)
        metadata.pop("color_space", None)
        models = dict(fit=dict(data=data, sidecar=sidecar),
                      reference=dict(data=data, sidecar=sidecar), old_field=None)
        from .raw_rig_source import execution_identity
        identity = execution_identity()
        runtime_parameters = deepcopy(baseline["parameters"])
        runtime_parameters.setdefault("source", {})["raw_execution_identity"] = identity
        native_configuration = _write_json(folder/"native_configuration.json", runtime_parameters)
        binding = dict(contract="darsia-raw-rig-source-v1", run=run,
            profile_id=baseline["source_pins"]["camera_binding"]["profile_id"],
            native_shape_hw=list(native_shape), baseline_raw=deepcopy(baseline["source_pins"]["raw_acquisition"]),
            baseline_reference_to_native_h=np.eye(3).tolist(), shape_assets=deepcopy(baseline["shape_assets"]),
            image_metadata=metadata, models=models,
            execution_identity=identity,
            correction_order=ORDER, photometric_method=INDEPENDENT,
            normalisation=dict(choice=deepcopy(choice), record=choice_pin))
        source_binding = _write_json(folder/"raw_source_binding.json", binding)
        result["runs"][run] = dict(source_binding=source_binding, fit=sidecar,
                                    native_configuration=native_configuration)
        if materialize_baselines:
            # Preserve the same baseline arithmetic as correct_independent;
            # this frame is still decoded and its colour fit already exists.
            if corrections["lighting"]:
                corrected = _apply_light(corrected, light, baseline["native_to_camera_u_h"], native_shape)
            signal = (baseline["rgb_full"] if not corrections["lighting"] and corrections["color_method"] == "off"
                      else corrected * np.float32(baseline["white_range_dn"]))
            with np.errstate(divide="ignore", invalid="ignore"):
                log = np.where(signal > parameters["minimum_sensor_signal"], np.log(signal), np.nan).astype(np.float32)
            native_log = crop_corrected(log, baseline["native_to_camera_u_h"], native_shape,
                coverage_tolerance=baseline["parameters"]["support"]["coverage_tolerance"])
            pixels = _write_npz(folder/"baseline.npz", dict(log_linear_rgb=native_log))
            result["runs"][run]["baseline"] = _write_json(folder/"baseline.json", dict(
                format="darsia-setup-corrected-baseline-v1", run=run,
                raw_acquisition=deepcopy(binding["baseline_raw"]), photometry_source=source_binding,
                native_configuration=native_configuration, products=pixels,
                native_shape_hw=list(native_shape), native_to_camera_u_h=baseline["native_to_camera_u_h"].tolist(),
                choice=deepcopy(choice), correction_order=ORDER,
                correction=dict(method=INDEPENDENT, color=deepcopy(color), light=deepcopy(light),
                    normalisation_choice=deepcopy(choice), old_photometry_used=False, source_purpose=None,
                    lighting_application=_lighting_application(corrections),
                    output_encoding="chw-float32-log-camera-linear-rgb-with-nan-invalid",
                    full_photo_shape_hw=list(baseline["rgb_full"].shape[1:])),
                encoding="chw-float32-log-camera-linear-rgb-with-nan-invalid"))
        del corrected
        if baseline is not reference:
            del baseline
    if proposal_samples:
        from ..optical_reference import propose_reference
        result["reference_proposal"] = propose_reference(proposal_samples,
            use_lighting=corrections["lighting"], use_color=corrections["color_method"] != "off")
    else:
        result["reference_proposal"] = None
    result["record"] = _write_json(output/"independent_photometry.json", result)
    return result


def load_independent_model(processor):
    """Load only this freshly prepared run's small signal/mask arrays."""
    from ..image_normalisation import selected_corrections
    from .io import read_bound_json
    model = processor.binding["models"]["fit"]
    record = read_bound_json(processor._asset(model["sidecar"]), model["sidecar"]["sha256"])
    if (record.get("format") != INDEPENDENT or record.get("run") != processor.binding["run"]
            or record.get("choice") != processor.binding["normalisation"]["choice"]
            or record.get("data") != model["data"]
            or record["source"]["raw_acquisition"]["sha256"] != processor.binding["baseline_raw"]["sha256"]
            or record.get("old_photometry_used") is not False):
        raise ValueError("Fresh photometric inputs differ from their RAW source binding")
    with np.load(processor._asset(model["data"]), allow_pickle=False) as z:
        arrays = {key:z[key] for key in ("reference_rgb", "stable_mask", "reference_swatches")}
        if record.get("cfa_validity_policy") is not None:
            from .cfa_clipping_diagnostics import FIT_VALIDITY_POLICY
            if record["cfa_validity_policy"] != FIT_VALIDITY_POLICY:
                raise ValueError("Unknown persisted CFA colour-fit validity policy")
            for key in ("reference_validity", "baseline_validity"):
                arrays[key] = z[key]
                if arrays[key].dtype != bool or arrays[key].shape != (4, 6, 3):
                    raise ValueError("Persisted CFA validity must be boolean 4x6xRGB")
            for kind, source_role in (("reference", "reference_source"), ("baseline", "source")):
                evidence = record[kind + "_validity_evidence"]
                if (evidence.get("fit_validity_policy") != FIT_VALIDITY_POLICY
                        or evidence["source"]["sha256"] != record[source_role]["raw_acquisition"]["sha256"]
                        or not np.array_equal(np.asarray(evidence["valid_patch_channels"]), arrays[kind + "_validity"])):
                    raise ValueError("Persisted CFA validity differs from its exact reference or baseline")
    corrections = selected_corrections(record["choice"])
    if arrays["reference_rgb"].shape != arrays["stable_mask"].shape + (3,) or arrays["stable_mask"].dtype != bool:
        raise ValueError("Fresh light input uses another grid or mask")
    return dict(record=record, corrections=corrections, **arrays)


def correct_independent(processor, sensor, h_ref, h_b, h_ref_b, *, source_purpose=None):
    """Apply independently selected current producers on the full RAW photo."""
    from darsia._opencv_runtime import scientific_thread, completed_frame
    before = scientific_thread("independent_photometry:before")
    p = processor.processor.independent
    model, options = p["record"], p["corrections"]
    full = (processor.baseline_camera_u if sensor.source["sha256"] == processor.baseline_sensor.source["sha256"]
            else raw_positioning.camera_u(sensor, parameters=processor.parameters["registration"]))
    _, renderer = processor._render(sensor, camera_u=full)
    h = np.asarray(renderer["native_to_camera_u_h"], np.float64)
    scale = np.float32(sensor.metadata["white_level"] - sensor.metadata["black_level"])
    unit = full/scale
    roi = model["baseline_chart_roi_rc"]
    chart = None
    if options["color_method"] != "off":
        # Locate the current chart independently of the rig crop/registration.
        exact = sensor.source["sha256"] == processor.baseline_sensor.source["sha256"]
        _, roi = _chart(unit, roi if exact else None)
        chart = dict(roi_rc=np.asarray(roi).tolist(), sample_grid="complete-camera-U")
    validity = {}
    if model.get("cfa_validity_policy") is not None:
        from .cfa_clipping_diagnostics import FIT_VALIDITY_POLICY, chart_fit_validity
        if model["cfa_validity_policy"] != FIT_VALIDITY_POLICY or options["color_method"] != "uniform24":
            raise ValueError("Unknown or inapplicable persisted CFA colour-fit validity policy")
        if sensor.source["sha256"] == processor.baseline_sensor.source["sha256"]:
            current_validity = p["baseline_validity"]
            current_evidence = model["baseline_validity_evidence"]
        else:
            current_validity, current_evidence = chart_fit_validity(sensor,
                processor.parameters["registration"], roi, camera_rgb=full)
        validity = dict(source_validity=current_validity, reference_validity=p["reference_validity"],
            validity_evidence=dict(source=current_evidence, reference=model["reference_validity_evidence"]))
    corrected, colour = _colour_full(unit, options["color_method"], p["reference_swatches"], roi, **validity)
    light = None
    if options["lighting"]:
        if source_purpose == "clearwater" or options.get("light_field"):
            # A wash before indicator addition has another chemical colour, and
            # CO2 changes the sand's colour during a run. Re-fitting light on
            # sand would erase that measured difference. Carry the start
            # photo's field, at the start photo's sand positions.
            light = deepcopy(model["baseline_light_fit"])
            light_h = h @ np.asarray(h_b, np.float64)
        else:
            native = _native_rgb(corrected, h, processor.binding["native_shape_hw"])
            current = _working_rgb(native, p["stable_mask"].shape)
            light = _light_fit(current, p["reference_rgb"], p["stable_mask"], processor.parameters["photometry"])
            light_h = h
        corrected = _apply_light(corrected, light, light_h, processor.binding["native_shape_hw"])
    signal = (full if not options["lighting"] and options["color_method"] == "off"
              else corrected*scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        log = np.where(signal > processor.parameters["photometry"]["minimum_sensor_signal"],
                       np.log(signal), np.nan).astype(np.float32)
    result = crop_corrected(log, h, processor.binding["native_shape_hw"],
                            coverage_tolerance=processor.parameters["support"]["coverage_tolerance"])
    return result, dict(contract=processor.binding["contract"], processor=processor.identity,
        sources=dict(camera_arw=dict(sensor.source)), renderer=renderer, chart=chart,
        correction_order=ORDER, exact_B=sensor.source["sha256"] == processor.baseline_sensor.source["sha256"],
        photometry=dict(method=INDEPENDENT, color=colour, light=light,
                        normalisation_choice=deepcopy(model["choice"]), old_photometry_used=False,
                        source_purpose=source_purpose,
                        lighting_application=_lighting_application(options, source_purpose),
                        output_encoding="chw-float32-log-camera-linear-rgb-with-nan-invalid",
                        full_photo_shape_hw=list(full.shape[1:])),
        **{processor.parameters["source"]["symbols"]["reference_to_frame_h_key"]: np.asarray(h_ref).tolist()},
        H_v5_B_to_frame_native=np.asarray(h_b).tolist(),
        source_identity=dict(kind="raw-acquisition",raw_sha256=sensor.source["sha256"]),
        parent_cache=None, cpu_only=True, training=False, jpeg_inputs=False,
        opencv_ipp=completed_frame(before, "independent_photometry:after"))


def crop_corrected(log_rgb, native_to_camera_u_h, shape_hw, *, coverage_tolerance):
    """Resample corrected linear camera DN, then retain the scientific log ABI."""
    log = np.asarray(log_rgb)
    if log.dtype != np.float32 or log.ndim != 3 or log.shape[0] != 3:
        raise ValueError("Full corrected photograph requires CHW float32 log camera RGB")
    valid = np.isfinite(log).all(axis=0)
    h = np.asarray(native_to_camera_u_h, np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
        raise ValueError("The common crop requires its measured camera-U transform")
    height, width = map(int, shape_hw)
    matrix = np.linalg.inv(h)
    support = cv2.warpPerspective(valid.astype(np.float32), matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    result = np.empty((3, height, width), np.float32)
    keep = support >= np.float32(1 - coverage_tolerance)
    for channel in range(3):
        with np.errstate(over="ignore", invalid="ignore"):
            linear = np.exp(np.where(valid, log[channel], -np.inf)).astype(np.float32)
        cropped = cv2.warpPerspective(linear, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        keep &= np.isfinite(cropped) & (cropped > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            result[channel] = np.log(cropped)
    result[:, ~keep] = np.nan
    return result


def reference_area_log(full_linear, reference_to_camera_u_h, *, parameters):
    """Original reference-cell arithmetic means sampled from the complete photo.

    A target crop never determines whether a ColorChecker or light-reference
    cell exists. Unobserved cells remain NaN for the existing masked fitter.
    """
    _, height, width = parameters["native_shape_chw"]
    gh, gw = parameters["area_grid_shape_chw"][1:]
    scale = parameters["native_area_log_int_8"]
    if gh * scale > height or gw * scale != width:
        raise ValueError("The original photometric reference grid is inconsistent")
    matrix = np.linalg.inv(np.asarray(reference_to_camera_u_h, np.float64))
    area = np.empty((3, gh, gw), np.float32)
    for channel in range(3):
        reference = cv2.warpPerspective(full_linear[channel], matrix, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
        mean = reference[:gh * scale].reshape(gh, scale, gw, scale).mean(axis=(1, 3), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            area[channel] = np.where(mean > parameters["minimum_sensor_signal"], np.log(mean), np.nan)
    return area


def correct(processor, sensor, h_ref, h_b, h_ref_b, *, source_purpose=None):
    """Execute full photograph → chart/light → colour/field → common crop."""
    if processor.binding.get("photometric_method") == INDEPENDENT:
        return correct_independent(processor, sensor, h_ref, h_b, h_ref_b, source_purpose=source_purpose)
    from ..image_normalisation import selected_corrections
    selected = selected_corrections(processor.binding["normalisation"]["choice"])
    if selected["color_method"] == "saved_native" and not selected["lighting"]:
        raise ValueError("Saved native correction bundles colour and light; choose an independent colour method")
    if selected["color_method"] not in {"off", "saved_native"} or (selected["lighting"] and selected["color_method"] == "off"):
        raise ValueError("The selected independent corrections need newly prepared optical inputs")
    from darsia._opencv_runtime import scientific_thread, completed_frame
    before = scientific_thread("full_photometry:before")
    parameters, p = processor.parameters, processor.processor
    full = (processor.baseline_camera_u if sensor.source["sha256"] == processor.baseline_sensor.source["sha256"]
            else raw_positioning.camera_u(sensor, parameters=parameters["registration"]))
    # This renderer establishes the measured output coordinates only. Its
    # uncorrected crop is discarded; scientific output is cropped after correction.
    _, renderer = processor._render(sensor, camera_u=full)
    native_to_u = np.asarray(renderer["native_to_camera_u_h"], np.float64)
    ref_to_u = native_to_u @ np.asarray(h_ref, np.float64)
    exact_b = sensor.source["sha256"] == processor.baseline_sensor.source["sha256"]
    choice = processor.binding["normalisation"]["choice"]
    normalise = choice["method"] != "off"
    chart = None
    light = None
    light_info = dict(mode="exact-source-bound-B", zero_field=True)
    if not normalise:
        light_info = dict(mode="off", photometric_correction_applied=False)
    elif not exact_b and source_purpose != "clearwater":
        registration = deepcopy(parameters["registration"])
        registration["common"]["sensor"]["content_wh"] = list(full.shape[1:][::-1])
        xy_u8, usable, chart = raw_chart.register(
            baseline_sensor=processor.baseline_sensor, frame_sensor=sensor,
            baseline_camera_u=processor.baseline_camera_u, frame_camera_u=full,
            baseline_chart_xy8=source.chart_baseline_coordinates(p.fit, p.grid, h_ref_b, parameters=parameters),
            baseline_chart_log_rgb=p.fit.CB,
            baseline_native_to_u_h=np.asarray(processor.baseline_renderer["native_to_camera_u_h"]),
            frame_native_to_u_h=np.eye(3), frame_native_linear_rgb=full, parameters=registration)
        area = reference_area_log(full, ref_to_u, parameters=parameters["photometry"])
        xy_ref8 = photometry.chart_xy_to_reference(xy_u8, ref_to_u, parameters=parameters["photometry"])
        light, light_info = photometry.frame_light_from_native_grid(area, p.fit, p.grid,
            np.eye(3), xy_ref8.astype(np.float32), usable, parameters=parameters["photometry"])
        chart.update(sample_grid="complete-camera-U", samples_before_common_crop=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_full = np.where(full > parameters["photometry"]["minimum_sensor_signal"], np.log(full), np.nan).astype(np.float32)
    valid = np.isfinite(log_full).all(axis=0)
    corrected = (photometry.apply_native(log_full, p.fit, p.reference, p.grid, ref_to_u,
        source_eligible_mask=valid, light_field_ref8=light, parameters=parameters["photometry"])
        if normalise else log_full)
    if normalise and p.old_field is not None:
        corrected = photometry.apply_old_field_native(corrected, p.old_field, p.grid, ref_to_u,
            tile_rows=parameters["photometry"]["apply_native_default_tile_rows"], parameters=parameters["photometry"])
    result = crop_corrected(corrected, native_to_u, processor.binding["native_shape_hw"],
        coverage_tolerance=parameters["support"]["coverage_tolerance"])
    return result, dict(contract=processor.binding["contract"], processor=processor.identity,
        sources=dict(camera_arw=dict(sensor.source)), renderer=renderer, chart=chart,
        photometry=dict(correction_order=ORDER, full_photo_shape_hw=list(full.shape[1:]),
            normalisation_choice=deepcopy(choice), per_frame_light=light_info,
            old_light_residual_applied=normalise and p.old_field is not None,
            output_encoding="chw-float32-log-camera-linear-rgb-with-nan-invalid",
            finite_eligible_pixel_count=int(np.isfinite(result).all(axis=0).sum()),
            crop_interpolation="bilinear corrected linear camera DN, then log; unsupported pixels remain NaN"),
        exact_B=exact_b, correction_order=ORDER,
        **{parameters["source"]["symbols"]["reference_to_frame_h_key"]: np.asarray(h_ref).tolist()},
        H_v5_B_to_frame_native=np.asarray(h_b).tolist(),
        source_identity=dict(kind="raw-acquisition", raw_sha256=sensor.source["sha256"]),
        parent_cache=None, cpu_only=True, training=False, jpeg_inputs=False,
        opencv_ipp=completed_frame(before, "full_photometry:after"))
