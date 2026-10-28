"""Frozen-input integration of the physical pixel-domain mass rule.

Reference geometry supplies the flash support, dilution state and an explicitly
adopted full-grid measured depth field. It never supplies candidate mass maps,
candidate facies, a reporting mask, or candidate transport weights.
"""

from collections.abc import Mapping

import numpy as np
from .phase_expert_knowledge import expert_knowledge_of


def validate_policy(policy):
    """Require explicit domain decisions and scalar-table fields before loading data."""
    if not isinstance(policy, Mapping):
        raise ValueError("An explicit pixel-domain policy mapping is required")
    required = {
        "contract", "adopted_for_execution", "fixture_scope_complete",
        "fixture_names", "injector_hardware_excluded", "authorized_decisions",
        "max_column_distance_m", "raster_scale", "raster_shift",
        "geometry_chunk_pixels", "below_floor_phase_unknown_policy",
        "include_estimated_mass", "table_fields", "baseline_coverage_tolerance",
        "heldout_fit_run",
    }
    if not required.issubset(policy):
        raise ValueError("Incomplete pixel-domain policy: " + str(sorted(required - set(policy))))
    if (policy["contract"] != "darsia-pixel-domain-v1"
            or policy["adopted_for_execution"] is not True
            or policy["fixture_scope_complete"] is not True
            or type(policy["injector_hardware_excluded"]) is not bool
            or policy["below_floor_phase_unknown_policy"] != "unknown-no-capacity"
            or policy["include_estimated_mass"] is not False
            or policy["heldout_fit_run"] is not False):
        raise ValueError("Pixel-domain decisions are not the explicitly adopted policy")
    names = policy["fixture_names"]
    if (not isinstance(names, list) or not names
            or any(not isinstance(x, str) or not x for x in names)
            or len(set(names)) != len(names)):
        raise ValueError("Complete distinct fixture names are required")
    distance = policy["max_column_distance_m"]
    if isinstance(distance, bool) or not np.isfinite(distance) or distance <= 0:
        raise ValueError("An explicit positive column distance is required")
    tolerance = policy["baseline_coverage_tolerance"]
    if isinstance(tolerance, bool) or not np.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError("Explicit baseline coverage tolerance must lie in [0,1)")
    if type(policy["geometry_chunk_pixels"]) is not int or policy["geometry_chunk_pixels"] < 1:
        raise ValueError("Positive measured-depth chunk size required")
    fields = policy["table_fields"]
    needed = {"M_measured_g", "M_edge_g", "U_g", "phase_unknown_no_capacity_area_cm2"}
    if (not isinstance(fields, list) or not needed.issubset(fields)
            or len(set(fields)) != len(fields)
            or any(not isinstance(x, str) or not x for x in fields)):
        raise ValueError("The scalar table must include measured, edge, unknown and unclassified area")
    return policy


def _source_geometry(pipeline, spec, shape):
    policy = validate_policy(pipeline.config["pixel_domain"])
    reader = pipeline.reader
    reader.json(policy["authorized_decisions"])
    declared = spec["pixel_domain_geometry"]
    source = reader.json(declared["source_receipt"])
    pointer = declared["json_pointer"]
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("A literal source-geometry JSON pointer is required")
    for key in pointer[1:].split("/"):
        source = source[key.replace("~1", "/").replace("~0", "~")]
    if (source["run"] != pipeline.config["run"]
            or source["frame_index"] != declared["source_frame_index"]
            or source["source_sha256"] != spec["source_binding"]["source_sha256"]):
        raise ValueError("Pixel-domain polygons belong to another source frame")
    for name, original in (
        ("inner_polygon_native", "frame_inner_polygon_native"),
        ("fixture_polygons_native", "frame_fixture_polygons_native"),
        ("native_origin_xy", "native_origin_xy"),
        ("phase_shape_hw", "phase_shape_hw"),
    ):
        if declared[name] != source[original]:
            raise ValueError("Pixel-domain geometry differs from its source: " + name)
    if (declared["phase_shape_hw"] != list(shape)
            or declared["native_origin_xy"] != spec["current"]["native_origin_xy"]
            or declared["inner_polygon_native"] != spec["inner_polygon_native"]
            or set(declared["fixture_polygons_native"]) != set(policy["fixture_names"])):
        raise ValueError("Complete configured polygons must share the current native grid")
    # Historical source scope flags remain unchanged; the separate adopted
    # decision binds the complete exclusion list for this method.
    return policy, declared


def _reference_basis(pipeline, spec, shape):
    from .concentration_joint import dilution_factor
    from .n1_pipeline import _array_identity

    basis, reader, config = spec["frozen_response_basis"], pipeline.reader, pipeline.config
    original = reader.json(basis["result"])
    expected = dict(
        source_sha256=spec["source_binding"]["source_sha256"],
        phase_sha256=spec["labels"]["sha256"],
        calibration_sha256=config["calibration"]["sha256"],
    )
    if (original.get("complete") is not True or original["run"] != config["run"]
            or original["frame_index"] != spec["frame_index"]
            or original["elapsed_h"] != spec["elapsed_h"]
            or any(original["bindings"][key] != value or basis[key] != value
                   for key, value in expected.items())):
        raise ValueError("Frozen response state belongs to another source, phase or calibration")
    static = original["static"]
    if "storage" in static or static["asset"] is None:
        raise ValueError("A frozen response basis must be a full-profile reference frame")
    for name in ("sha256", "size_bytes"):
        if basis["geometry"][name] != static["asset"][name]:
            raise ValueError("Frozen response geometry differs from its producer")
    if (basis["arrays"] != static["arrays"]
            or basis["per_frame_geometry"] != static["per_frame_geometry"]):
        raise ValueError("Frozen response array declarations differ")
    fields = reader.archive(basis["geometry"], tuple(static["arrays"]))
    for name, value in fields.items():
        if value.shape != shape or _array_identity(value) != static["arrays"][name]:
            raise ValueError("Frozen physical field differs: " + name)
    weights = fields["transport_weight_l"]
    if weights.dtype != np.float64 or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Frozen flash support requires genuine finite transport geometry")
    support = weights > 0
    if not support.any():
        raise ValueError("Frozen flash reference support is empty")
    old = original["scalar_result"]
    if (old["parameters"] != pipeline.parameters or original["anchor"] != config["anchor"]
            or old["precision"] != pipeline.math.precision):
        raise ValueError("Frozen dilution parameters or observed-volume anchor differ")
    correction = pipeline.calibration["sandmap"]
    if (any(correction[key] != config["physical_correction"][key]
            for key in ("sha256", "size_bytes"))
            or not np.array_equal(
                np.asarray(original["sandmap"]["source_binding"]["correction_matrix"], np.float64),
                pipeline.correction)):
        raise ValueError("Frozen calibration and original physical correction differ")
    factor = dilution_factor(
        spec["elapsed_h"], old["affected_pore_volume_m3"], config["anchor"],
        pipeline.parameters["beta"],
    )
    if factor != old["dilution_factor"]:
        raise ValueError("Frozen dilution state does not reproduce its original factor")
    depth = spec["pixel_domain_geometry"]["target_depth"]
    if (depth.get("array_key") != "depth_m"
            or any(depth[key] != basis["geometry"][key] for key in ("sha256", "size_bytes"))):
        raise ValueError("Target depth must be the explicitly bound full-grid measured field")
    reader.path(depth)
    target_depth = fields["depth_m"]
    if target_depth.dtype != np.float64 or not np.isfinite(target_depth).all() or np.any(target_depth <= 0):
        raise ValueError("Target measured depth must be positive float64 on the full grid")
    return support, target_depth, factor, dict(
        result=basis["result"], geometry=basis["geometry"],
        flash_support="positive original physical transport support; not a candidate domain",
        flash_support_pixels=int(support.sum()),
        dilution_affected_volume_m3=old["affected_pore_volume_m3"], dilution_factor=factor,
        target_depth=depth, target_depth_role="uniform full-grid measured physical input",
        target_depth_measurements=config["shared"]["depth_points"],
        target_depth_source_geometry=spec["pixel_domain_geometry"]["source_receipt"],
        physical_correction=config["physical_correction"],
        reference_weights_used_as_candidate_weights=False,
        original_maps_used_as_candidate_maps=False,
    )


def prepare_frame(pipeline, spec, *, current, valid, baseline, coverage, clear_rgb,
                  clear_supported, labels, area, current_matrix):
    """Build raw polygon support and column facies before any historical floor cut.

    All incoming RGB buffers retain the original decoded working precision.
    Source validity defines visibility; baseline/clear-water validity separately
    controls application of the frozen colour model. Depth is the uniformly
    applied adopted measured field, while porosity/facies are newly transported.
    """
    import cv2
    from ..geometry.pixels import polygon_pixel_domain, visible_pixels, extend_column_geometry
    from . import n1_geometry as geometry
    from .n1_models import FrameData
    from .n1_pipeline import _atmosphere, _dose

    config, shared, math = pipeline.config, pipeline.config["shared"], pipeline.math
    shape = labels.shape
    policy, declared = _source_geometry(pipeline, spec, shape)
    basis, depth, factor, reference = _reference_basis(pipeline, spec, shape)
    domain, inside, fixtures = polygon_pixel_domain(
        declared["inner_polygon_native"], declared["fixture_polygons_native"], list(shape),
        native_origin_xy=declared["native_origin_xy"],
        raster_scale=policy["raster_scale"], raster_shift=policy["raster_shift"],
    )
    revision = dict(phase_shape_hw=list(shape), native_origin_xy=declared["native_origin_xy"])
    raw_geometry = spec["current"]["binding"]["geometry"]
    x, y = geometry.frame_coordinates(
        pipeline.reference_matrix, raw_geometry, revision, fields=config["geometry_fields"]
    )
    transported = geometry.transport_physical_reference(
        pipeline.physical, current_matrix @ pipeline.correction, shape, math, cv2
    )
    columns = extend_column_geometry(
        transported["facies"], transported["domain"], domain, y,
        porosity_by_facies=shared["porosity_by_facies"], swi_by_facies=shared["swi_by_facies"],
    )
    known = domain & ~columns.geometry_unknown
    if not np.isfinite(current[valid]).all() or np.any(current[valid] < 0):
        raise ValueError("Source-valid RGB must retain finite nonnegative scientific values")
    visible = visible_pixels(domain, valid)
    optical = (visible & (coverage >= 1 - policy["baseline_coverage_tolerance"]) & clear_supported
               & np.isfinite(baseline).all(-1) & (baseline >= 0).all(-1)
               & np.isfinite(clear_rgb).all(-1) & (clear_rgb > 0).all(-1))
    floor = geometry.transport_curve(
        pipeline.floor, H_source_to_target=np.asarray(
            raw_geometry[config["geometry_fields"]["source_to_native"]], np.float64),
        source_origin_xy=shared["floor_origin_xy"],
        target_origin_xy=revision["native_origin_xy"], target_width=shape[1],
    )
    good = np.flatnonzero(np.isfinite(floor))
    if not len(good):
        raise ValueError("The bound floor has no transported columns")
    floor[:good[0]], floor[good[-1] + 1:] = floor[good[0]], floor[good[-1]]
    if not np.isfinite(floor).all():
        raise ValueError("The bound floor has an interior gap")
    below_floor = domain & (np.arange(shape[0])[:, None] > floor[None, :])
    state = _atmosphere(config, spec)
    frame = FrameData(
        run_id=config["run"], frame_index=spec["frame_index"], elapsed_h=float(spec["elapsed_h"]),
        current_rgb=current, baseline_rgb=baseline, clear_water_rgb=clear_rgb,
        clear_water_supported_mask=clear_supported, phase_labels=labels,
        facies=columns.facies, domain=domain, supported_mask=known & optical,
        mass_eligible_mask=known & optical, depth_m=depth, porosity=columns.porosity,
        pixel_area_m2=area, dimensions_m=tuple(config["dimensions_m"]),
        pressure_bar=state["pressure_bar"], temperature_c=state["temperature_c"],
        pressure_gradient_bar_per_m=state.get("pressure_gradient_bar_per_m", 0.0),
        temperature_gradient_c_per_m=state.get("temperature_gradient_c_per_m", 0.0),
        nominal_injected_mass_g=_dose(config, spec), recipe=dict(config["recipe"]),
        provenance=dict(source_binding=spec["source_binding"], frozen_response_basis=reference),
        working_dtype=math.precision, excluded_masks={}, unknown_physical_mask=domain & ~known,
    )
    return frame, x, y, visible, dict(
        flash_support=basis, dilution_factor=factor, reference=reference,
        geometry=columns, below_floor=below_floor, inside=inside, fixtures=fixtures,
        policy=policy, optical_support=optical, transported_physical=transported,
    )


def evaluate(pipeline, spec, *, output_capture=None):
    """Apply frozen optics once to new physical support and report all mass parts."""
    import darsia
    from .n1_models import prepare_response
    from .pixels import evaluate_pixels

    from ..storage_profile import stores_colour_coordinate

    frame, x, y, visible, context = pipeline.load_frame(spec)
    from .n1_arithmetic import analysis_gap_scope
    approved_gap, analysis_scope = analysis_gap_scope(frame)
    analysis_domain = frame.domain if approved_gap is None else frame.domain & ~approved_gap
    analysis_visible = visible if approved_gap is None else visible & ~approved_gap
    config, math = pipeline.config, pipeline.math
    capture = dict(colour_coordinate=True) if stores_colour_coordinate(config) else {}
    response = prepare_response(
        frame, pipeline.models, math, config["shared"]["swi_by_facies"],
        solubility_domain=context["flash_support"], **capture,
    )
    weights = np.asarray(response["analysis"].geometry.cached_voxel_volume, np.float64) * 1000.0
    if (not np.isfinite(weights).all() or np.any(weights < 0)
            or np.any(weights[~frame.known_physical_domain] != 0)):
        raise ValueError("Native geometry requires finite weights and zero outside known support")
    pixel = evaluate_pixels(
        domain=analysis_domain, visible=analysis_visible, geometry_known=frame.known_physical_domain,
        colour_available=response["quantifiable"], phase_labels=frame.phase_labels,
        relative_aqueous=response["prepared"].aqueous_relative, swi=response["swi"],
        solubility=response["physics"].solubility_co2,
        gas_density=response["physics"].density_gaseous_co2, transport_weight_l=weights,
        y_m=y, max_distance_m=context["policy"]["max_column_distance_m"],
        gain=pipeline.parameters["gain"], frozen_dilution_factor=context["dilution_factor"],
        force_estimated=context["below_floor"] & analysis_domain,
        unknown_without_capacity=context["below_floor"] & (expert_knowledge_of(frame).unknown) & analysis_domain,
        pixel_math=math,
    )
    n16 = None
    if "n16" in context["policy"]:
        n16 = _prepare_n16(pipeline, spec, frame, x, y, analysis_visible, context, response, pixel, weights,
                           **({} if approved_gap is None else dict(approved_gap=approved_gap)))
    if output_capture is not None:
        if n16 is None:
            raise ValueError("Pixel-domain coarse storage requires the qualified measured-mass bridge")
        output_capture.update(frame=frame, x_m=x, y_m=y,
            volume_m3=response["analysis"].geometry.cached_voxel_volume)
    image = response["analysis"].scalar(frame.phase_labels, frame.elapsed_h)
    reporting_domain = frame.domain if n16 is None else n16.get("reporting_domain", n16["masks"]["physical"])
    regions = dict(full_rig=reporting_domain)
    for name in ("box1", "box2"):
        roi = pipeline.boxes[name]
        roi = roi.get("roi", roi) if isinstance(roi, dict) else roi
        regions[name] = reporting_domain & darsia.roi_to_mask(
            darsia.CoordinateArray(roi), image, mode="voxels"
        ).img

    def totals(mask):
        positive = expert_knowledge_of(frame).co2
        measured = mask & pixel["masks"]["measured"] & positive
        capacity = pixel["per_pixel_mass_g"]["unknown_capacity"]
        total = float(math.integrate(pixel["maps"]["mass_total"], weights, measured))
        return dict(
            total_mass_g=total,
            aqueous_mass_g=float(math.integrate(pixel["maps"]["mass_aqueous"], weights, measured)),
            gas_mass_g=float(math.integrate(pixel["maps"]["mass_free_gas"], weights, measured)),
            M_measured_g=total,
            M_edge_g=float(math.integrate(pixel["maps"]["mass_total"], weights,
                                          mask & pixel["masks"]["concentration_estimated"] & positive)),
            U_g=None if np.isnan(capacity[mask]).any() else float(math.integrate(
                capacity, np.ones_like(weights), mask)),
        )

    domains = {name: totals(mask) for name, mask in regions.items()}
    full = domains["full_rig"]
    area = np.asarray(frame.pixel_area_m2, np.float64)
    unclassified = pixel["masks"]["unknown_without_capacity"]
    no_capacity_area = float(np.sum(area[unclassified])) * 1e4
    scalar = dict(
        contract=context["policy"]["contract"], parameters=pipeline.parameters,
        precision=math.precision, pixel_backend=math.facts()["backend"],
        pixel_backend_facts=dict(math.facts()), domains=domains, excluded_mass={},
        **full, affected_pore_volume_m3=float(np.sum(weights[
            frame.domain & expert_knowledge_of(frame).co2])) / 1000.0,
        fields=dict(main_mass_eligible_mask=pixel["masks"]["measured"]),
        dilution_factor=context["dilution_factor"],
        scientific_eligibility=False, production_permission_granted=False,
    )
    main, edge, unknown = full["M_measured_g"], full["M_edge_g"], full["U_g"]
    dose = frame.nominal_injected_mass_g
    from .n1_statistics import interval_residual

    residual = (None if unknown is None or dose is None or dose <= 0 else
                interval_residual(main + edge, dose, unknown))
    prediction = dict(
        run=frame.run_id, frame_index=frame.frame_index, elapsed_h=frame.elapsed_h,
        selected_target_h=spec["selected_target_h"], precision=math.precision,
        parameters=pipeline.parameters, total_mass_g=main, main_mass_g=main,
        M_measured_g=main, M_edge_g=edge, U_g=unknown,
        E_lo_g=edge, E_hi_g=edge, U_rest_g=unknown, W_g=unknown,
        W_over_d=None if unknown is None or dose is None or dose <= 0 else unknown / dose,
        interval_lower_g=main + edge,
        interval_upper_g=None if unknown is None else main + edge + unknown,
        injected_mass_g=dose, excluded_parts={},
        signed_interval_residual=residual, delta=None if residual is None else abs(residual),
        dilution_factor=context["dilution_factor"],
        phase_labels_changed=False, calibration_refitted=False,
        evaluation_is_leave_one_run_out=False, heldout_fit_run=context["policy"]["heldout_fit_run"],
        round4_spec_sha256=pipeline.method_bindings.get("round4_spec_sha256"),
        amendment_sha256=pipeline.method_bindings.get("amendment_sha256"),
        scope="pixel_domain_evaluation", main_geometry_modified=True,
        main_labels_modified=False, map_mass_rescaling=False, overflow_in_objective=False,
        scientific_eligibility=False, production_permission_granted=False,
        estimated_mass_included_in_reported_M=False,
    )
    columns = context["geometry"]
    flags = dict(pixel["masks"], domain=frame.domain, visible=visible,
                 geometry_filled=columns.geometry_filled,
                 geometry_unknown=columns.geometry_unknown)
    extra = {name + "_mask": value for name, value in flags.items()}
    extra.update(
        mass_contribution_mask=frame.domain & expert_knowledge_of(frame).co2 & (
            pixel["masks"]["measured"] | pixel["masks"]["concentration_estimated"]),
        measured_mass_g=pixel["per_pixel_mass_g"]["measured"].astype(math.dtype),
        edge_mass_g=pixel["per_pixel_mass_g"]["edge"].astype(math.dtype),
        unknown_capacity_g=pixel["per_pixel_mass_g"]["unknown_capacity"].astype(math.dtype),
        concentration_donor_row=pixel["donor_rows"],
        concentration_donor_distance_m=pixel["donor_distance_m"].astype(math.dtype),
        geometry_donor_row=columns.donor_rows,
        geometry_donor_distance_m=columns.donor_distance_m.astype(math.dtype),
    )
    audit = dict(
        contract=context["policy"]["contract"], policy=context["policy"],
        source_geometry=spec["pixel_domain_geometry"]["source_receipt"],
        original_response_basis=context["reference"],
        field_role="candidate depth/area are bound measured input; facies/phi/weights newly computed",
        fixture_pixels={name: int(mask.sum()) for name, mask in context["fixtures"].items()},
        domain_pixels=int(frame.domain.sum()), visible_pixels=int(visible.sum()),
        phase_unknown_no_capacity_pixels=int(unclassified.sum()),
        phase_unknown_no_capacity_area_cm2=no_capacity_area,
        phase_labels_changed=False, source_RGB_changed=False,
        below_floor_clipped_from_domain=False,
        original_finite_identity_selection_used_in_computation=False,
        output_map_dtype=math.precision, mass_integration_dtype="float64",
        reduction_dtype="float64", geometry_operand_dtype="float64",
        kernel=pixel["audit"],
    )
    evaluated = dict(
        response=response, result=scalar, prediction=prediction,
        export=dict(products=pixel["maps"], units=pixel["units"]),
        statistics=audit, sandmap=audit, extra_products=extra,
        geometry_fields=dict(pixel_area_m2=area, depth_m=frame.depth_m,
                             porosity=frame.porosity, facies=frame.facies,
                             transport_weight_l=weights),
        pixel_domain=dict(audit, M_measured_g=main, M_edge_g=edge, U_g=unknown),
    )
    if n16 is not None:
        _apply_n16(evaluated, n16, pipeline, spec, regions)
    if approved_gap is not None:
        _approved_scope(evaluated, approved_gap, analysis_scope, regions)
    return evaluated


def _approved_scope(evaluated, mask, scope, regions):
    """Label unchanged finite pixel carriers separately from public unknowns."""
    from copy import deepcopy
    from .n1_arithmetic import scoped_result
    result, extra = evaluated["result"], evaluated["extra_products"]
    frame = evaluated["response"]["frame"]
    exported = evaluated["export"]
    exported["quantified_products"] = {name: exported["products"][name]
        for name in ("mass_total", "mass_aqueous", "mass_free_gas")}
    exported["products"] = dict(exported["products"])
    for name in ("concentration_aqueous", "saturation_free_gas", "mass_total", "mass_aqueous", "mass_free_gas"):
        value = exported["products"][name].copy()
        value[mask] = np.nan
        exported["products"][name] = value
    exported["products"]["approved_analysis_gap_mask"] = mask.copy()
    result["fields"]["approved_analysis_gap_mask"] = mask.copy()
    result["fields"]["unquantified_phase_mask"] = extra["concentration_unknown_mask"] | mask
    result["excluded_mass"]["approved_analysis_gap"] = dict(total_mass_g=None,
        aqueous_mass_g=None, gas_mass_g=None, pixel_count=scope["approved_analysis_gap_pixels"],
        area_cm2=None, unknown_mass_g=None, meaning="outside-analysis-unquantified")
    result["domains"] = {name: scoped_result(row, dict(scope, domain=name,
        approved_analysis_gap_pixels=int((mask & regions[name]).sum()),
        phase255_pixels=int((mask & regions[name] & (expert_knowledge_of(frame).unknown)).sum())))
        for name, row in result["domains"].items()}
    for name in ("concentration_unknown_mask", "unknown_without_capacity_mask", "geometry_unknown_mask"):
        extra[name] = extra[name] | mask
    extra["approved_analysis_gap_mask"] = mask.copy()
    for name in tuple(extra):
        if name.endswith("mass_g") or name == "unknown_capacity_g":
            value = extra[name].copy()
            value[mask] = np.nan
            extra[name] = value
    for name in ("result", "prediction", "statistics", "sandmap", "pixel_domain", "export"):
        evaluated[name] = scoped_result(evaluated[name], scope)
    for name in ("statistics", "sandmap", "pixel_domain"):
        audit = evaluated[name]
        audit["phase_unknown_no_capacity_pixels"] = int(extra["unknown_without_capacity_mask"].sum())
        audit["quantified_phase_unknown_no_capacity_area_cm2"] = audit["phase_unknown_no_capacity_area_cm2"]
        audit["phase_unknown_no_capacity_area_cm2"] = None
        if "U_g" in audit:
            audit["quantified_U_g"] = audit["U_g"]
    if "n16_producer" in evaluated:
        evaluated["n16_producer"]["analysis_scope"] = deepcopy(scope)
        for name in ("statistics", "pixel_domain"):
            evaluated[name]["n16"]["analysis_scope"] = deepcopy(scope)


def _prepare_n16(pipeline, spec, frame, x, y, visible, context, response, pixel, weights, *, approved_gap=None):
    """Source-bound N16 execution; historical configurations never enter here."""
    import darsia
    from ..geometry.pixels import (hidden_physical_geometry,
        hidden_boundary_observation, below_photo_column_support)
    from .n1_geometry import project_points
    from .pixels import (round3_hidden_solubility_capacity,
        hidden_mass_by_adjacency, measured_component_maps, _sum_axis_zero)
    policy = context["policy"]["n16"]
    if (not isinstance(policy, dict) or set(policy) != {"contract", "authority", "capacity_selector"}
            or policy["contract"] != "darsia-n16-q85-a-v1"
            or policy["capacity_selector"] != "literal-R3-hidden255-solubility"):
        raise ValueError("Explicit N16 Q85 A policy required")
    # The configuration pins the decision record's bytes; the reader verifies them.
    from ..series_decisions import recorded_decision
    authority = recorded_decision(pipeline.reader.json(policy["authority"]), "N16 hidden-mass decision")
    if len(context["fixtures"]) != 3 or context["policy"]["max_column_distance_m"] != .020:
        raise ValueError("N16 requires three hidden polygons and the original 20 mm column rule")
    declared = spec["pixel_domain_geometry"]
    source = pipeline.reader.json(declared["source_receipt"])
    for key in declared["json_pointer"][1:].split("/"):
        source = source[key.replace("~1", "/").replace("~0", "~")]
    masks = declared.get("n16_masks")
    if (not isinstance(masks, dict) or set(masks) != {"below_photo_strip", "not_sand"}
            or source.get("n16_masks") != masks):
        raise ValueError("Explicit source-bound N16 below-photo/not-sand masks required")

    def mask(name):
        bound = masks[name]
        if bound is None:
            return np.zeros(frame.domain.shape, bool)
        value = pipeline.reader.array(bound)
        if value.dtype != bool or value.shape != frame.domain.shape:
            raise ValueError("N16 bound " + name + " mask differs from the native grid")
        return value

    strip, not_sand = mask("below_photo_strip"), mask("not_sand")
    # A declared campaign strip is required, never guessed from the floor or RGB.
    run = str(frame.run_id).lower().removeprefix("ac")
    if run.isdigit() and 17 <= int(run) <= 38 and masks["below_photo_strip"] is None:
        raise ValueError("AC17-38 needs its measured, source-bound 8 mm below-photo strip")
    if np.any((strip | not_sand) & ~context["inside"]) or np.any(strip & not_sand):
        raise ValueError("N16 strip/not-sand masks must be disjoint inside the rig")
    if approved_gap is not None:
        strip = strip & ~approved_gap
    raw_geometry = spec["current"]["binding"]["geometry"]
    matrix = (np.asarray(pipeline.reference_matrix, np.float64) @ np.linalg.inv(
        np.asarray(raw_geometry[pipeline.config["geometry_fields"]["source_to_native"]], np.float64)))

    def metric_polygon(points):
        points = np.asarray(points, np.float64)
        px, py = project_points(matrix, points[:, 0], points[:, 1])
        return np.column_stack((px, py)).tolist()

    observation = hidden_boundary_observation(frame.phase_labels, visible & ~strip,
        x, y, inner_polygon_m=metric_polygon(declared["inner_polygon_native"]),
        fixture_polygons_m={name: metric_polygon(points)
            for name, points in declared["fixture_polygons_native"].items()})
    transported = context["transported_physical"]
    physical = hidden_physical_geometry(transported["facies"], transported["domain"],
        context["inside"], context["fixtures"], y,
        porosity_by_facies=pipeline.config["shared"]["porosity_by_facies"],
        swi_by_facies=pipeline.config["shared"]["swi_by_facies"])
    inside = physical["physical_mask"] & ~not_sand
    hidden = physical["hidden_mask"]
    reporting_domain = inside
    if approved_gap is not None:
        inside = inside & ~approved_gap
        hidden = hidden & ~approved_gap
    if np.any(hidden & (not_sand | strip)) or np.any(hidden & (frame.phase_labels != 255)):
        raise ValueError("Hidden sand must retain original phase255 and cannot overlap the bottom strip/not-sand")
    columns = physical["geometry"]
    valid = inside & ~columns.geometry_unknown
    if np.any(inside & ~valid):
        raise ValueError("Full physical geometry has an undefined column; no maximum is substituted")
    area = np.asarray(frame.pixel_area_m2, np.float64)
    depth = np.asarray(frame.depth_m, np.float64)
    phi = np.asarray(columns.porosity, np.float64)
    # Exact NativeAnalysis Geometry construction/order, on the separate Q52 domain.
    geometry = darsia.ExtrudedPorousGeometry(porosity=np.where(inside, phi, 0.),
        depth=np.where(inside, depth, 0.), space_dim=2, num_voxels=inside.shape,
        dimensions=list(frame.dimensions_m))
    nominal_volume = geometry.voxel_volume.copy()
    nominal_area = float(np.prod(frame.dimensions_m)) / np.prod(inside.shape)
    geometry.voxel_volume = nominal_volume * (area / nominal_area)
    geometry.cached_voxel_volume = geometry.voxel_volume.copy()
    pv = np.asarray(geometry.cached_voxel_volume, np.float64)
    capacity = (round3_hidden_solubility_capacity(hidden_mask=hidden,
        physical_properties_valid_mask=valid, pixel_area_m2=area, depth_m=depth,
        porosity=phi, solubility=response["physics"].solubility_co2)
        if observation["co2_adjacent"] else None)
    unknown = hidden_mass_by_adjacency(hidden_mask=hidden, adjacency=observation,
        elapsed_h=frame.elapsed_h, round3_capacity_g=capacity)
    bottom = below_photo_column_support(strip, visible, frame.phase_labels, y)
    bottom_mass = np.zeros(inside.shape, np.float64)
    selected = bottom["estimated_mask"]
    donor_rows = np.maximum(bottom["donor_rows"], 0)
    donor_concentration = pixel["maps"]["concentration_aqueous"][donor_rows, np.arange(inside.shape[1])[None, :]]
    # Original N15 column estimate is donor aqueous concentration * target litres.
    np.multiply(np.asarray(donor_concentration, np.float64), pv * 1000.,
        out=bottom_mass, where=selected)
    measured_support = inside & ~strip & pixel["masks"]["measured"]
    estimate_support = inside & ~hidden & ~measured_support
    explicit_estimate = pixel["masks"]["concentration_estimated"] | strip
    if np.any(estimate_support & ~explicit_estimate):
        raise ValueError("Non-hidden physical pixels lack an actual estimate/zero rule")
    masses = measured_component_maps(pixel, frame.phase_labels, weights)
    if np.any(strip & pixel["masks"]["measured"] & expert_knowledge_of(frame).co2):
        raise ValueError("The source-bound below-photo strip cannot be an observed positive mass pixel")
    if any(np.any(value[~inside] != 0) for value in masses.values()):
        raise ValueError("Measured contribution outside the explicit physical domain")
    edge = pixel["per_pixel_mass_g"]["edge"].copy()
    if np.any(edge[~inside] != 0):
        raise ValueError("Edge contribution outside the explicit physical domain")
    edge[strip] = bottom_mass[strip]
    if not np.isfinite(edge).all() or not np.isfinite(bottom_mass).all():
        raise ValueError("Actual edge/bottom contribution is not finite")
    saturation = pixel["maps"]["saturation_free_gas"]
    saturation_valid = inside & ~hidden & np.isfinite(saturation)
    gas = inside & (expert_knowledge_of(frame).free_gas)
    gas_pv = np.zeros(inside.shape, np.float64)
    np.multiply(np.asarray(saturation, np.float64), pv, out=gas_pv, where=gas & saturation_valid)
    native_extensive_totals = dict(physical_area_m2=_sum_axis_zero(np.where(inside, area, 0.)),
        bulk_volume_m3=_sum_axis_zero(np.where(inside, area * depth, 0.)),
        pore_volume_m3=_sum_axis_zero(np.where(inside, pv, 0.)),
        gas_pore_volume_m3=_sum_axis_zero(gas_pv))
    precision = dict(provider_fields=pipeline.math.precision,
        original_component_contributions="float64", original_geometry="float64",
        original_mass_accumulation="float64", native_component_export=pipeline.math.precision,
        stored_quantities=pipeline.math.precision, stored_reduction="float64",
        original_five_field_identity_path_recast=False,
        export_conversion="new contribution exports only: existing math.dtype policy; original geometry archive remains float64",
        storage_conversion="existing reducers: float64 accumulation then grid.precision")
    return dict(hidden=hidden, unknown=unknown, bottom=bottom, bottom_mass=bottom_mass,
        measured_masses=masses, edge=edge, M_edge_g=_sum_axis_zero(edge),
        masks=dict(physical=inside, valid=valid, measured=measured_support,
                   estimated=estimate_support, saturation_valid=saturation_valid),
        geometry=dict(depth_m=depth, porosity=phi, pixel_pore_volume_m3=pv,
                      facies=columns.facies, physical_area_m2=area),
        totals=native_extensive_totals, policy=policy, authority=authority,
        precision=precision, observation=observation,
        capacity_inputs=dict(pixel_area_m2=area, depth_m=depth, porosity=phi,
            physical_properties_valid_mask=valid, hidden_mask=hidden,
            solubility_g_per_l=response["physics"].solubility_co2),
        **({} if approved_gap is None else dict(reporting_domain=reporting_domain)))


def _apply_n16(evaluated, n16, pipeline, spec, regions):
    """Keep measured identity, edge and hidden channels separate in actual outputs."""
    from .pixels import _sum_axis_zero
    from .n1_statistics import interval_residual
    hidden, unknown, masks = n16["hidden"], n16["unknown"], n16["masks"]
    result, prediction = evaluated["result"], evaluated["prediction"]
    main, edge, upper = result["M_measured_g"], n16["M_edge_g"], unknown["U_g"]
    for name, region in regions.items():
        result["domains"][name].update(M_edge_g=_sum_axis_zero(np.where(region, n16["edge"], 0.)),
            U_g=_sum_axis_zero(np.where(region, unknown["U_per_pixel_g"], 0.)))
    result.update(M_edge_g=edge, U_g=upper)
    dose = prediction["injected_mass_g"]
    residual = None if dose is None or dose <= 0 else interval_residual(main, dose, upper)
    prediction.update(M_edge_g=edge, U_g=upper, E_lo_g=edge, E_hi_g=edge,
        U_rest_g=upper, W_g=upper, W_over_d=None if dose is None or dose <= 0 else upper/dose,
        interval_lower_g=main, interval_upper_g=main+upper,
        signed_interval_residual=residual, delta=None if residual is None else abs(residual),
        interval_basis="M_maalt to M_maalt+U; M_kant separate until blind edge-label GO")
    exports = evaluated["extra_products"]
    exports.update(hidden_mask=hidden, physical_mask=masks["physical"],
        physical_properties_valid_mask=masks["valid"],
        below_photo_estimated_mask=n16["bottom"]["estimated_mask"],
        below_photo_zero_mask=n16["bottom"]["zero_mask"],
        below_photo_unknown_mask=n16["bottom"]["unknown_mask"],
        below_photo_donor_row=n16["bottom"]["donor_rows"],
        below_photo_donor_distance_m=n16["bottom"]["donor_distance_m"].astype(pipeline.math.dtype),
        below_photo_estimated_mass_g=n16["bottom_mass"].astype(pipeline.math.dtype),
        edge_mass_g=n16["edge"].astype(pipeline.math.dtype))
    strip = n16["bottom"]["estimated_mask"] | n16["bottom"]["zero_mask"]
    exports["concentration_unknown_mask"] &= ~strip
    exports["unknown_without_capacity_mask"] &= ~strip
    exports["concentration_estimated_mask"] |= n16["bottom"]["estimated_mask"]
    exports["mass_contribution_mask"] |= n16["bottom"]["estimated_mask"]
    # U0 remains a factual rule receipt with hidden coverage, never a placeholder map.
    exports.pop("unknown_capacity_g")
    if unknown["U_layer"] is not None:
        exports["unknown_capacity_g"] = unknown["U_layer"].astype(pipeline.math.dtype)
    for name, value in n16["measured_masses"].items():
        exports["measured_" + name + "_mass_g"] = value.astype(pipeline.math.dtype)
    evaluated["geometry_fields"]["n16_solubility_g_per_l"] = n16["capacity_inputs"]["solubility_g_per_l"]
    for name, value in n16["geometry"].items():
        evaluated["geometry_fields"]["full_physical_" + name] = value
    remaining_unclassified = exports["unknown_without_capacity_mask"]
    unclassified_area = float(np.sum(n16["geometry"]["physical_area_m2"][remaining_unclassified])) * 1e4
    for audit in (evaluated["pixel_domain"], evaluated["statistics"]):
        audit.update(phase_unknown_no_capacity_pixels=int(remaining_unclassified.sum()),
                     phase_unknown_no_capacity_area_cm2=unclassified_area)
    receipt = dict(unknown["receipt"], contract=n16["policy"]["contract"],
        U_total_g=upper, M_measured_g=main, M_edge_g=edge,
        authority=n16["policy"]["authority"], authority_record=n16["authority"],
        capacity_selector=n16["policy"]["capacity_selector"],
        hidden_capacity_expression="where(valid,area*depth*phi,0.)*solubility*1000",
        gas_contact_report=n16["observation"], U_layer_present=unknown["U_layer"] is not None,
        below_photo_rule="strictly above same-column visible label1/2 donor within20mm, else zero; unavailable donor concentration stops",
        bottom_strip_unknown_pixels=0, measured_mass_unchanged=True,
        M_kant_in_measured_mass=False, U_in_measured_mass=False, precision=n16["precision"])
    evaluated["pixel_domain"].update(M_edge_g=edge, U_g=upper, n16=receipt)
    evaluated["statistics"]["n16"] = receipt
    evaluated["n16_producer"] = n16
