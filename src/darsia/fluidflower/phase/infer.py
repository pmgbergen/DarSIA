"""Ordered frozen optical decisions with explicit acquisition and model roles."""

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from ..native.support import compute_feature_support
from . import domain, features, geometry, heads, placement, regional, rules as decisions
from .history import envelope_ordinal
from .metrics import area_filtered_gas, array_pair_identity


@dataclass(frozen=True)
class NativeCentreResult:
    """Native optical decisions before categorical domain merging."""

    labels: np.ndarray
    feature_support: np.ndarray
    raw_feature_support: np.ndarray
    domain: np.ndarray
    preminimum_gas: np.ndarray
    g3_pixel: np.ndarray
    regional_refit: np.ndarray
    flatfield_valid: np.ndarray
    provenance: dict


def _independent_alignments(arguments, *, feature_parameters):
    """Preserve the two historical reference-mask conventions before sharing scores."""
    named = dict(arguments)
    native_previous = (
        named.get("previous") is not None
        and tuple(named["previous_origin_xy"]) == (0, 0)
        and np.asarray(named["previous"]).shape == np.asarray(named["current"]).shape
    )
    if native_previous:
        named = dict(named, previous_source_valid_mask=None)
    inherited_named = dict(named)
    native_baseline = (
        tuple(named["baseline_origin_xy"]) == (0, 0)
        and np.asarray(named["baseline"]).shape == np.asarray(named["current"]).shape
    )
    if native_baseline:
        inherited_named["baseline_source_valid_mask"] = None
    inherited_base, inherited_previous, inherited_audit = (
        features.align_temporal_inputs(**inherited_named, parameters=feature_parameters)
    )
    aligned_base, aligned_previous, alignment = features.align_temporal_inputs(
        **named, parameters=feature_parameters
    )
    same_baseline = array_pair_identity(inherited_base, aligned_base)
    same_previous = (
        inherited_previous is None
        and aligned_previous is None
        or inherited_previous is not None
        and aligned_previous is not None
        and array_pair_identity(inherited_previous, aligned_previous)["bit_identical"]
    )
    if not same_baseline["bit_identical"] or not same_previous:
        raise ValueError(
            "Independently aligned inherited and probability RGB differ; sharing scores is forbidden"
        )
    return (
        inherited_base,
        aligned_base,
        aligned_previous,
        alignment,
        dict(
            inherited_alignment=inherited_audit,
            g3_alignment=alignment,
            baseline_rgb_bit_identity=same_baseline,
            previous_rgb_bit_identical=bool(same_previous),
        ),
    )


def infer_native_centre(
    current_log,
    *,
    current_valid,
    source_baseline_log,
    baseline_valid,
    current_h,
    photo_sand_mask,
    probability_spacing_m,
    inherited_spacing_m,
    source_origin_xy,
    flat_baseline_log,
    flat_baseline_valid,
    previous_log,
    previous_valid,
    previous_h,
    previous_origin_xy,
    labelled_probability_encoding,
    model_bundle: Mapping,
    parameters: Mapping,
    input_provenance: Mapping,
    matched_encoding_receipt=None,
    output_capture=None,
):
    """Infer the native centre from its own colour and independently bound references.

    Optical inherited probabilities and the frozen gas refinement retain their
    separate metric axes, alignment masks, matrix products and probability
    storage round trips. All model documents, support tolerances and physical
    decision thresholds are supplied explicitly; this function performs no fit.
    """
    from darsia._opencv_runtime import scientific_thread, completed_frame
    ipp_before = scientific_thread("infer_native_centre:before")
    fp, hp, rp = (parameters[name] for name in ("features", "heads", "regional"))
    runtime = parameters["native"]
    threads, block_rows = runtime["threads"], runtime["block_rows"]
    if type(threads) is not int or not 1 <= threads <= runtime["maximum_threads"]:
        raise ValueError("Native-centre threads exceed their configured bound")
    if type(block_rows) is not int or block_rows < 1:
        raise ValueError("Native-centre tile rows must be positive")
    current = features.decode_native_log(current_log, current_valid, parameters=fp)
    fixed_cache = model_bundle.get("fixed_baseline_decode_cache")
    if fixed_cache is None:
        baseline = features.decode_native_log(
            source_baseline_log, baseline_valid, parameters=fp
        )
    else:
        baseline = fixed_cache.decode(
            runtime["temporal_cache_role"],
            source_baseline_log,
            baseline_valid,
            decoder=model_bundle["fixed_baseline_decoder"],
        )
    previous = (
        None
        if previous_log is None
        else features.decode_native_log(previous_log, previous_valid, parameters=fp)
    )
    shape = current.shape[:2]
    photo_sand = np.asarray(photo_sand_mask)
    if photo_sand.dtype != bool or photo_sand.shape != shape:
        raise ValueError("Native-centre requires exact photo-sand HW bool")
    spacing = np.asarray(probability_spacing_m, float)
    inherited_spacing = np.asarray(inherited_spacing_m, float)
    for axes in (spacing, inherited_spacing):
        if axes.shape != (2,) or not np.isfinite(axes).all() or (axes <= 0).any():
            raise ValueError(
                "Native-centre requires separately bound nominal y/x spacing"
            )
    if (previous_log is None) != (previous_valid is None) or (previous_log is None) != (
        previous_h is None
    ):
        raise ValueError(
            "Native-centre predecessor must provide log/support/H together"
        )
    if flat_baseline_valid is None or flat_baseline_log.shape[1:] != shape:
        raise ValueError(
            "Native flat-field baseline must be same-grid source-supported log RAW"
        )
    if fixed_cache is None:
        features.decode_native_log(
            flat_baseline_log, flat_baseline_valid, parameters=fp
        )
    else:
        fixed_cache.validate_fixed(
            runtime["flat_cache_role"],
            flat_baseline_log,
            flat_baseline_valid,
            decoder=model_bundle["fixed_baseline_decoder"],
        )
    cv2.setNumThreads(threads)
    with threadpool_limits(limits=threads):
        support, raw_support = compute_feature_support(
            current_valid=current_valid,
            baseline_valid=baseline_valid,
            h_baseline_to_current=current_h,
            previous_valid=previous_valid,
            h_baseline_to_previous=previous_h,
            baseline_origin_xy=source_origin_xy,
            previous_origin_xy=previous_origin_xy,
            parameters=parameters["feature_support"],
        )
        domain = photo_sand & support
        inherited_base, aligned_base, aligned_previous, alignment, independent = (
            _independent_alignments(
                dict(
                    baseline=baseline,
                    current=current,
                    previous=previous,
                    h_baseline_to_current=current_h,
                    h_baseline_to_previous=previous_h,
                    required_valid_mask=support,
                    baseline_origin_xy=source_origin_xy,
                    previous_origin_xy=previous_origin_xy,
                    baseline_source_valid_mask=baseline_valid,
                    previous_source_valid_mask=previous_valid,
                ),
                feature_parameters=fp,
            )
        )
        predictor = heads.CpuFeaturePredictor(
            model_bundle["original_model"],
            aligned_base,
            block_rows=block_rows,
            input_encoding=runtime["input_encoding"],
            parameters=hp,
            feature_parameters=fp,
        )
        flat_reference = model_bundle.get("flatfield_reference")
        if flat_reference is None:
            flat_reference = features.prepare_flatfield_reference(
                flat_baseline_log,
                source_valid_mask=flat_baseline_valid,
                tile_rows=block_rows,
                parameters=fp,
            )
        registered = features.register_flatfield_reference(
            flat_reference, current_h, target_shape_hw=shape,
            source_origin_xy=tuple(runtime["flatfield_source_origin_xy"]),
            target_origin_xy=tuple(runtime["flatfield_target_origin_xy"]),
            parameters=fp
        )
        current_features = predictor.prepare_rgb(current)
        previous_features = (
            predictor.baseline
            if aligned_previous is None
            else predictor.prepare_rgb(aligned_previous)
        )
        generic = np.empty((*shape, 2), np.float32)
        presence = np.empty(shape, np.float32)
        gas_scores = np.full(shape, np.nan, np.float32)
        flat_valid = np.zeros(shape, bool)
        tiles = iter(
            features.iter_flatfield_feature_tiles(
                current_log,
                registered,
                current_valid_mask=current_valid,
                tile_rows=block_rows,
                parameters=fp,
            )
        )
        for first, last, feature_rows in predictor.feature_blocks(
            current_features,
            previous_features,
            dynamic_available=aligned_previous is not None,
        ):
            flat_first, flat_last, flat_rows, valid = next(tiles)
            if (first, last) != (flat_first, flat_last):
                raise ValueError(
                    "Native-centre optical and flat-field tile rows differ"
                )
            generic[first:last] = heads.score_packed_heads(
                feature_rows, predictor.weights, predictor.bias
            ).reshape(last - first, shape[1], 2)
            presence[first:last] = heads.score_packed_presence(
                feature_rows, predictor.weights, predictor.bias
            ).reshape(last - first, shape[1])
            safe = np.where(valid[..., None], flat_rows, 0).reshape(
                -1, runtime["gas_feature_count"]
            )
            scores = heads.gas_probability(
                model_bundle["gas_head"], feature_rows, safe, parameters=hp
            ).reshape(last - first, shape[1])
            gas_scores[first:last] = np.where(valid, scores, np.nan)
            flat_valid[first:last] = valid
        inherited, co2_audit = regional.original_co2_labels(
            current,
            inherited_base,
            generic[..., 0],
            generic[..., 1],
            domain,
            inherited_spacing,
            model_bundle["regional_original"],
            model_bundle["profile"],
            parameters=rp,
        )
        shared = regional.build_shared_regional_context(
            current,
            aligned_base,
            presence,
            domain,
            spacing,
            regional.Settings.from_mapping(
                model_bundle["regional_original"]["settings"], parameters=rp
            ),
            parameters=rp,
        )
        pure = regional.prepare_conditional_gas(
            shared, np.nan_to_num(gas_scores, nan=0.0), parameters=rp
        )
        region_scores = regional.probability(
            pure["features"],
            model_bundle["regional_refit"]["heads"]["gas_within_co2"],
            parameters=rp,
        )
        decoded_pixel, decoded_region, encoding = heads.encode_decode_probabilities(
            gas_scores,
            region_scores,
            flat_valid,
            labelled=labelled_probability_encoding,
            parameters=hp,
        )
        nominal_area = float(
            np.prod(spacing) * runtime["square_metres_to_square_millimetres"]
        )
        labels, gas_audit = decisions.apply_gas_configuration(
            inherited,
            decoded_pixel,
            decoded_region,
            nominal_area,
            model_bundle["gas_decision_settings"],
            parameters=parameters["decisions"],
            **({} if output_capture is None else dict(output_capture=output_capture)),
        )
        preminimum = decisions.preminimum_gas_selection(
            decoded_pixel, decoded_region, parameters=parameters["decisions"],
            **({} if output_capture is None else dict(output_capture=output_capture.setdefault("seam_pair", {}))),
        )
    if output_capture is not None and "seam_pair" in output_capture:
        output_capture["seam_pair"]["actual_returned_selection"] = preminimum.copy()
    provenance = dict(
        runtime["provenance"],
        inference_performed=True,
        input_provenance=dict(input_provenance),
        models=model_bundle["bindings"],
        temporal_alignment=alignment,
        independent_alignment_audit=independent,
        native_shape_hw=list(shape),
        spacing_m=spacing.tolist(),
        nominal_pixel_area_mm2=nominal_area,
        co2_context=co2_audit,
        gas_configuration=gas_audit,
        probability_encoding=encoding,
        matched_encoding_receipt=matched_encoding_receipt,
        feature_support_pixels=int(support.sum()),
        domain_pixels=int(domain.sum()),
        flat_valid_pixels=int(flat_valid.sum()),
        threads=threads,
        opencv_ipp=completed_frame(ipp_before, "infer_native_centre:after"),
    )
    if fixed_cache is not None:
        provenance["fixed_baseline_decode_cache"] = fixed_cache.receipt()
    return NativeCentreResult(
        labels,
        support,
        raw_support,
        domain,
        preminimum,
        decoded_pixel,
        decoded_region,
        flat_valid,
        provenance,
    )


def infer_rig_edge(
    current_log,
    aligned_baseline_log,
    domain,
    *,
    aligned_previous_log,
    spacing_m,
    pixel_area_mm2,
    source_baseline_log,
    baseline_to_current_h,
    source_baseline_valid,
    feature_shape_hw,
    model_bundle: Mapping,
    parameters: Mapping,
    output_capture=None,
):
    """Infer supported rig pixels with masked filters and measured area guards.

    The inherited CO2 partition uses its original generic gas evidence; the
    downstream frozen gas head only replaces that partition. The rig regional
    model is distinct from the native-centre original regional model. No labels
    or newly filled categorical pixels enter an optical feature calculation.
    """
    from darsia._opencv_runtime import scientific_thread, completed_frame
    ipp_before = scientific_thread("infer_rig_edge:before")
    fp, hp, rp = (parameters[name] for name in ("features", "heads", "regional"))
    runtime = parameters["edge"]
    if pixel_area_mm2 is None:
        raise ValueError("Rig inference needs its recorded per-pixel physical area")
    valid = (
        np.asarray(domain, bool)
        & np.isfinite(current_log).all(0)
        & np.isfinite(aligned_baseline_log).all(0)
    )
    if aligned_previous_log is not None:
        valid &= np.isfinite(aligned_previous_log).all(0)
    current, valid = features.log_to_unit_rgb(current_log, valid, parameters=fp)
    baseline, _ = features.log_to_unit_rgb(aligned_baseline_log, valid, parameters=fp)
    previous = (
        None
        if aligned_previous_log is None
        else features.log_to_unit_rgb(aligned_previous_log, valid, parameters=fp)[0]
    )
    predictor = heads.CpuFeaturePredictor(
        model_bundle["original_model"],
        baseline,
        valid=valid,
        feature_shape_hw=feature_shape_hw,
        block_rows=runtime["block_rows"],
        input_encoding=runtime["input_encoding"],
        parameters=hp,
        feature_parameters=fp,
    )
    prepared_current = predictor.prepare_rgb(current)
    prepared_previous = (
        predictor.baseline if previous is None else predictor.prepare_rgb(previous)
    )
    shape = valid.shape
    presence, legacy_gas = np.empty(shape, np.float32), np.empty(shape, np.float32)
    for first, last, feature_rows in predictor.feature_blocks(
        prepared_current, prepared_previous, dynamic_available=previous is not None
    ):
        scores = heads.score_packed_heads(
            feature_rows, predictor.weights, predictor.bias
        ).reshape(last - first, shape[1], 2)
        presence[first:last], legacy_gas[first:last] = scores[..., 0], scores[..., 1]
    flat, flat_valid = features.masked_flatfield_features(
        current_log,
        aligned_baseline_log,
        valid,
        source_baseline_log=source_baseline_log,
        h_baseline_to_current=baseline_to_current_h,
        source_baseline_valid=source_baseline_valid,
        parameters=fp,
    )
    gas_scores = heads.gas_probability(
        model_bundle["gas_head"],
        np.empty((*shape, 0), np.float32),
        np.where(flat_valid[..., None], flat, 0),
        parameters=hp,
    ).astype(np.float32)
    gas_scores[~flat_valid] = np.nan
    regional_model = model_bundle["regional_edge"]
    settings = regional.Settings.from_mapping(regional_model["settings"], parameters=rp)
    shared = regional.build_shared_regional_context(
        current, baseline, presence, valid, spacing_m, settings, parameters=rp
    )
    co2_prepared = regional.prepare_conditional_gas(shared, legacy_gas, parameters=rp)
    raw = decisions.frozen_native_partition(
        presence, legacy_gas, valid, parameters=parameters["decisions"]
    )
    labels, co2_audit = regional.predict(
        co2_prepared, regional_model, raw, valid, parameters=rp
    )
    labels, island_audit = regional.weighted_co2_output_guard(
        labels, valid, pixel_area_mm2, parameters=rp
    )
    prepared_gas = regional.prepare_conditional_gas(
        shared, np.nan_to_num(gas_scores, nan=0.0), parameters=rp
    )
    gas_region = regional.probability(
        prepared_gas["features"],
        regional_model["heads"]["gas_within_co2"],
        parameters=rp,
    )
    gas_region = cv2.resize(gas_region, shape[::-1], interpolation=cv2.INTER_LINEAR)
    co2 = np.isin(labels, (1, 2)) & valid
    gas_selected = (
        co2
        & flat_valid
        & (gas_scores >= runtime["pixel_threshold"])
        & (gas_region >= runtime["regional_threshold"])
    )
    preminimum_gas = gas_selected.copy()
    if output_capture is not None:
        from .probability_storage import pair
        pair(output_capture, gas_scores, gas_region, valid & flat_valid, co2,
             preminimum_gas, branch="edge", pixel_threshold=runtime["pixel_threshold"],
             regional_threshold=runtime["regional_threshold"])
    gas_selected, minimum_audit = area_filtered_gas(
        gas_selected,
        pixel_area_mm2,
        minimum_area_mm2=runtime["minimum_area_mm2"],
        connectivity=runtime["gas_connectivity"],
    )
    if output_capture is not None:
        from .probability_storage import transition, changed_rule
        transition(output_capture, "edge-minimum", preminimum_gas, gas_selected)
        changed_rule(output_capture, "edge_minimum", preminimum_gas, gas_selected)
    labels[co2] = np.where(gas_selected[co2], 2, 1)
    labels, final_support, support_audit = (
        decisions.require_categorical_feature_support(
            labels, valid, flat_valid, parameters=parameters["decisions"]
        )
    )
    audit = dict(
        runtime["provenance"],
        optical_context_native_shape_hw=list(feature_shape_hw),
        optical_spatial_sigma_px=float(predictor.spatial_sigma),
        co2=co2_audit,
        co2_islands=island_audit,
        gas_minimum=minimum_audit,
        categorical_feature_support=support_audit,
        opencv_ipp=completed_frame(ipp_before, "infer_rig_edge:after"),
    )
    return labels, final_support, audit, dict(co2=co2, pre_minimum_gas=preminimum_gas)


def apply_locked_postprocessing(
    inputs, shift_context, delta, envelopes, *, backend, parameters: Mapping, output_capture=None
):
    """Apply frozen gas placement, top removal and ordered floor decisions.

    The parent envelopes and their acquisition times are supplied data. The
    temporary historical floor fill is absent because its complete removal
    preceded the saved final labels. Categorical fill enters only the declared
    floor-support population, never optical features or a new model fit.
    """
    dp = parameters["decisions"]
    runtime = parameters["chain"]
    for name in ("native_inference_provenance", "edge_inference_provenance"):
        proof = inputs[name]
        if (
            not isinstance(proof, Mapping)
            or proof.get("inference_performed") is not True
        ):
            raise ValueError("Phase production needs numerical " + name)
    if len(inputs["source_sha256"]) != runtime["source_digest_length"]:
        raise ValueError("Phase production needs an exact camera acquisition identity")
    p1, p1_audit, _ = decisions.evaluate_placement(
        shift_context,
        runtime["placement_mode"],
        float(delta),
        inputs["base_labels"],
        inputs["modelled"],
        inputs["pixel_area_mm2"],
        filled_support=inputs["filled_support"],
        top_keep=inputs["top_keep"],
        open_components=inputs["open_components"],
        barrier_mask=inputs["barrier_mask"],
        frozen_base_rule_b=inputs["frozen_base_rule_b"],
        source_component_ids=list(inputs["source_component_ids"]),
        parameters=dp,
        heads_parameters=parameters["heads"],
        regional_parameters=parameters["regional"],
        **({} if output_capture is None else dict(output_capture=output_capture["placement"])),
    )
    if output_capture is not None:
        from .probability_storage import state
        state(output_capture["postprocessing"], "placement", p1)
    top = decisions.apply_top_rule(
        p1,
        p1,
        np.zeros(p1.shape, bool),
        inputs["pixel_area_mm2"],
        backend=backend,
        parameters=dp,
    )
    if output_capture is not None:
        state(output_capture["postprocessing"], "top", top.labels, eligible_rule="post_top")
    ordinal = envelope_ordinal(
        inputs["elapsed_h"], envelopes["u1_elapsed_h"], envelopes["u2_elapsed_h"]
    )
    floor_support = (inputs["modelled"] | inputs["filled_support"]) & inputs[
        "corrected_domain"
    ]
    floor = decisions.apply_floor_round(
        top.labels,
        inputs["below_floor"],
        floor_support,
        None,
        inputs["l1_zone"],
        envelopes["env1"],
        envelopes["env2"],
        float(inputs["elapsed_h"]),
        ordinal,
        runtime["first_envelope_ordinal"],
        runtime["second_envelope_ordinal"],
        inputs["pixel_area_mm2"],
        backend=backend,
        corrected_domain_mask=inputs["corrected_domain"],
        parameters=dp,
    )
    if output_capture is not None:
        state(output_capture["postprocessing"], "ordered-floor", floor.labels, eligible_rule="post_floor")
    if not floor.qa["all_pass"]:
        raise ValueError("Phase frame failed the unchanged ordered floor QA")
    labels = decisions.remove_contact_fill_b(
        floor.labels,
        floor.fill_reason_a,
        floor.fill_reason_b,
        floor.step_masks[runtime["contact_step"]],
        parameters=dp,
    )
    if output_capture is not None:
        state(output_capture["postprocessing"], "contact-fill-removal", labels, eligible_rule="post_contact")
    if floor.fill_reason_b.any():
        raise AssertionError("An absent carryover mask cannot create contact reason b")
    audit = dict(
        runtime["postprocessing_provenance"],
        run=inputs["run"],
        elapsed_h=float(inputs["elapsed_h"]),
        source_sha256=inputs["source_sha256"],
        p1=p1_audit,
        top=top.qa,
        floor=floor.qa,
        gas_envelopes={
            key: value
            for key, value in envelopes.items()
            if key not in ("env1", "env2")
        },
        envelope_ordinal=ordinal,
        parent_u1_u2_camera_times_preserved=True,
        native_inference=inputs["native_inference_provenance"],
        edge_inference=inputs["edge_inference_provenance"],
    )
    return (
        labels,
        audit,
        dict(
            removed_top=top.removed_top,
            **floor.step_masks,
            contact_reason_a=floor.fill_reason_a,
            contact_reason_b=floor.fill_reason_b,
        ),
    )


def infer_labels(frame, previous, model_bundle: Mapping, rules: Mapping, *, output_capture=None):
    """Infer one acquisition on its unchanged native and complete rig grids.

    ``frame`` contains admitted source arrays, photograph masks, transforms,
    physical areas and static component geometry. ``previous`` is its exact
    preceding selected acquisition. The baseline bundle retains separate
    native flat-field, temporal and full-rig reference roles. Retained labels
    can resolve only explicitly admitted categorical-fill populations; they
    cannot supply classifier features. No filesystem or campaign lookup,
    model fitting, temporal interpolation or canvas resampling occurs here.
    """
    if model_bundle.get("family") == "fresh-native-logistic":
        from .selected_model import infer_labels as infer_selected
        return infer_selected(frame, previous, model_bundle, rules, output_capture=output_capture)
    from darsia._opencv_runtime import scientific_thread, completed_frame
    ipp_before = scientific_thread("infer_labels:before")
    probability_capture = None if output_capture is None else output_capture.get("gas_probability_storage")
    if probability_capture is not None:
        if type(probability_capture) is not dict or probability_capture:
            raise ValueError("A fresh optional probability capture is required for every frame")
        probability_capture.update(edge={}, centre={}, composition={}, placement={}, postprocessing={})
    runtime, gp, dp = rules["chain"], rules["geometry"], rules["decisions"]
    baseline = model_bundle["baseline"]
    source = frame["source"]
    previous_digest = source["previous_source_sha256"]
    if (previous_digest is None) != (previous is None):
        raise ValueError("Acquisition history needs its exact declared predecessor")
    if previous is not None and previous["source"]["source_sha256"] != previous_digest:
        raise ValueError("Acquisition history identifies another predecessor")
    full, full_valid, context = frame["full_log"], frame["full_valid"], frame["context"]
    native_input = frame["native_input"]
    current_h = np.asarray(frame["current_h"], np.float64)
    source_box = tuple(gp["source_box_xywh"])
    canvas_h = domain.canvas_homography(current_h, source_box_xywh=source_box)
    aligned_base, baseline_support = geometry.align_scientific(
        baseline["full_log"],
        baseline["full_valid"],
        canvas_h,
        target_shape_hw=full.shape[1:],
        parameters=gp,
        feature_parameters=rules["features"],
    )
    previous_full = previous_support = previous_h = None
    previous_temporal = None
    if previous is not None:
        previous_h = np.asarray(previous["current_h"], np.float64)
        previous_full, previous_support = geometry.align_scientific(
            previous["full_log"],
            previous["full_valid"],
            domain.canvas_homography(
                current_h @ np.linalg.inv(previous_h), source_box_xywh=source_box
            ),
            target_shape_hw=full.shape[1:],
            parameters=gp,
            feature_parameters=rules["features"],
        )
        previous_temporal = previous["temporal_input"]
    fresh, fresh_support, edge_audit, edge_decisions = infer_rig_edge(
        full,
        aligned_base,
        context.rig_domain,
        aligned_previous_log=previous_full,
        spacing_m=runtime["rig_spacing_m"],
        pixel_area_mm2=frame["original_area"],
        source_baseline_log=baseline["full_log"],
        baseline_to_current_h=canvas_h,
        source_baseline_valid=baseline["full_valid"],
        feature_shape_hw=gp["native_shape_hw"],
        model_bundle=model_bundle["edge"],
        parameters=rules,
        **({} if probability_capture is None else dict(output_capture=probability_capture["edge"])),
    )
    fresh_support &= context.rig_domain & full_valid & baseline_support
    if previous_support is not None:
        fresh_support &= previous_support
    fresh[~fresh_support] = runtime["unresolved_label"]
    components, source_ids = frame["components"], frame["source_component_ids"]
    special_anchor = (
        source.get("studio_anchor") and source["parent_source"]["source_changed"]
    )
    if special_anchor:
        base, base_b = decisions.apply_barrier_rule_b(
            fresh,
            components["open_components"],
            source_ids,
            pixel_area_mm2=rules["domain"]["nominal_rule_b_pixel_area_mm2"],
            barrier_mask=components["barrier_mask"],
            parameters=dp,
        )
        if probability_capture is not None:
            from .probability_storage import state
            state(probability_capture["composition"], "edge-before-Rule-B", fresh)
            state(probability_capture["composition"], "edge-Rule-B", base, eligible_rule="edge_B")
        base, classes = domain.fill_rig_labels(
            base,
            context.rig_domain,
            frame["observed"] | fresh_support,
            hidden=context.hardware_excluded,
            parameters=rules["domain"],
        )
        if probability_capture is not None:
            state(probability_capture["composition"], "categorical-fill", base)
        centre_audit = dict(runtime["changed_source_provenance"])
        seam_audit = dict(rule_b=base_b, **runtime["changed_source_seam_provenance"])
        masks = {}
    else:
        temporal = baseline["temporal_input"]
        centre = infer_native_centre(
            native_input.log,
            current_valid=native_input.valid,
            source_baseline_log=temporal.log,
            baseline_valid=temporal.valid,
            current_h=current_h,
            photo_sand_mask=frame["photo_sand"],
            probability_spacing_m=frame["probability_spacing_m"],
            inherited_spacing_m=frame["inherited_spacing_m"],
            source_origin_xy=temporal.origin_xy,
            flat_baseline_log=baseline["native_input"].log,
            flat_baseline_valid=baseline["native_input"].valid,
            previous_log=None if previous_temporal is None else previous_temporal.log,
            previous_valid=(
                None if previous_temporal is None else previous_temporal.valid
            ),
            previous_h=previous_h,
            previous_origin_xy=(
                runtime["empty_previous_origin_xy"]
                if previous_temporal is None
                else previous_temporal.origin_xy
            ),
            labelled_probability_encoding=frame["labelled_probability_encoding"],
            model_bundle=model_bundle["native"],
            parameters=rules,
            input_provenance=frame["input_provenance"],
            matched_encoding_receipt=frame["encoding_contract"],
            **({} if probability_capture is None else dict(output_capture=probability_capture["centre"])),
        )
        if output_capture is not None:
            output_capture.update(g3_pixel=centre.g3_pixel, flatfield_valid=centre.flatfield_valid,
                                  source_box_xywh=source_box,
                                  probability_encoding=centre.provenance["probability_encoding"])
        if probability_capture is not None:
            probability_capture["centre"]["encoding"] = centre.provenance["probability_encoding"]
        working_shape = centre.provenance["probability_encoding"][
            "regional_working_shape_hw"
        ]
        release_native, release_audit = domain.old_domain_interpolation_footprint(
            centre.domain, working_shape, parameters=rules["domain"]
        )
        release_native |= domain.outer_domain_ring(centre.domain)
        release = (
            domain.embed_native_canvas(
                release_native,
                source_box_xywh=source_box,
                phase_shape_hw=full.shape[1:],
                fill=False,
            )
            & context.rig_domain
        )
        old_domain = domain.embed_native_canvas(
            centre.domain,
            source_box_xywh=source_box,
            phase_shape_hw=full.shape[1:],
            fill=False,
        )
        base, classes, composition, masks = domain.merge_native_and_rig(
            centre,
            fresh,
            fresh_support,
            context.rig_domain,
            frame["observed"],
            release=release,
            old_domain=old_domain,
            original_area=frame["original_area"],
            open_components=components["open_components"],
            barrier_mask=components["barrier_mask"],
            source_ids=source_ids,
            fresh_pre_minimum=edge_decisions["pre_minimum_gas"],
            source_box_xywh=source_box,
            parameters=rules["domain"],
            rule_parameters=dp,
            prior_cache=frame["prior_cache"],
            **({} if probability_capture is None else dict(output_capture=probability_capture["composition"])),
        )
        centre_audit = centre.provenance
        seam_audit = dict(
            composition["seam"],
            composition=composition,
            released_footprint=release_audit,
        )
    context_domain = context.rig_domain & full_valid & baseline_support
    if previous_support is not None:
        context_domain &= previous_support
    pm = model_bundle["placement"]
    shift = placement.prepare_shift_context(
        pm["original_model"],
        pm["pixel_model"],
        pm["regional_model"],
        full,
        aligned_base,
        context_domain,
        light_epoch_id=frame["light_epoch_id"],
        spacing_m=runtime["rig_spacing_m"],
        feature_shape_hw=gp["native_shape_hw"],
        metadata=frame["placement_metadata"],
        parameters=rules["placement"],
        feature_parameters=rules["features"],
        heads_parameters=rules["heads"],
        regional_parameters=rules["regional"],
        rules_parameters=dp,
        aligned_previous_log=previous_full,
        source_baseline_log=baseline["full_log"],
        h_baseline_to_current=canvas_h,
        source_baseline_valid=baseline["full_valid"],
        base_labels=base,
        joint_pixel_model=pm.get("joint_pixel_model"),
        projected_regional_model=pm.get("projected_regional_model"),
    )
    floor = geometry.transported_floor(
        model_bundle["floor_curve"],
        H_source_to_target=frame["reference_to_current_h"],
        source_origin_xy=gp["floor_source_origin_xy"],
        target_origin_xy=gp["rig_origin_xy"],
        target_shape_hw=base.shape,
        parameters=gp,
    )
    below = context.rig_domain & (np.arange(base.shape[0])[:, None] > floor[None, :])
    corrected = context.rig_domain & ~below
    inputs = dict(
        base_labels=base,
        modelled=classes["modelled"],
        filled_support=classes["filled_observed"] | classes["filled_unobserved"],
        top_keep=frame["top_keep"],
        open_components=components["open_components"],
        barrier_mask=components["barrier_mask"],
        source_component_ids=source_ids,
        frozen_base_rule_b=seam_audit["rule_b"],
        pixel_area_mm2=frame["pixel_area_mm2"],
        corrected_domain=corrected,
        below_floor=below,
        l1_zone=geometry.native_l1(
            model_bundle["zones"],
            H_source_to_target=frame["reference_to_current_h"],
            native_shape_hw=gp["native_shape_hw"],
            source_box_xywh=source_box,
            phase_shape_hw=base.shape,
            zone_id=gp["lower_zone_id"],
            parameters=gp,
        ),
        elapsed_h=float(source["elapsed_h"]),
        run=source["run"],
        source_sha256=source["source_sha256"],
        native_inference_provenance=centre_audit,
        edge_inference_provenance=dict(edge_audit, inference_performed=True),
    )
    labels, qa, post_masks = apply_locked_postprocessing(
        inputs,
        shift,
        runtime["delta"],
        model_bundle["envelopes"],
        backend=runtime["rule_backend"],
        parameters=rules,
        **({} if probability_capture is None else dict(output_capture=probability_capture)),
    )
    classes.update(
        rig_domain=corrected,
        hardware_excluded=context.hardware_excluded,
        below_sand_floor=below,
    )
    for name in ("modelled", "filled_observed", "filled_unobserved"):
        classes[name] &= corrected
    area = frame["pixel_area_mm2"]
    qa.update(
        source_history=dict(
            previous_source_sha256=previous_digest,
            selected=source["selected"],
            studio_anchor=source["studio_anchor"],
            policy=frame["history_policy"],
        ),
        base_seam=seam_audit,
        barrier_geometry=frame["barrier_proof"],
        physical_scale_revision=frame["physical_scale_revision"],
    )
    qa["phase_area"] = dict(
        co2_pixels=int(np.isin(labels, (1, 2)).sum()),
        gas_pixels=int((labels == 2).sum()),
        aqueous_pixels=int((labels == 1).sum()),
        co2_cm2=float(
            area[np.isin(labels, (1, 2))].sum(dtype=np.float64)
            / runtime["square_millimetres_per_square_centimetre"]
        ),
        gas_cm2=float(
            area[labels == 2].sum(dtype=np.float64)
            / runtime["square_millimetres_per_square_centimetre"]
        ),
        aqueous_cm2=float(
            area[labels == 1].sum(dtype=np.float64)
            / runtime["square_millimetres_per_square_centimetre"]
        ),
        physical_scale_approved=True,
        mass_computed=False,
    )
    qa["opencv_ipp"] = completed_frame(ipp_before, "infer_labels:after")
    if probability_capture is not None:
        from .probability_storage import finish
        finish(probability_capture, labels, classes, source_box)
    return dict(
        labels=labels,
        classes=classes,
        pixel_area_mm2=area,
        native_inference=centre_audit,
        edge_inference=edge_audit,
        qa=qa,
        source=source,
        scientific=frame["scientific"],
        seam_masks=masks,
        post_masks=post_masks,
        context_parameters=context.parameters,
    )
