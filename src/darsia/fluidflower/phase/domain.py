"""Source-bound categorical composition on native and complete rig grids.

Consume verified arrays and explicit frozen parameters. Do not load models,
infer geometry, or choose release populations from output differences.
"""

from __future__ import annotations


from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from .metrics import area_filtered_gas, sum_pixel_area, validate_pixel_area
from .metrics import array_identity as _array_identity


@dataclass(frozen=True)
class RetainedSourceInferenceCache:
    """Caller-certified retained categories and predeclared read-only masks."""

    labels: np.ndarray
    cached_parent_modelled: np.ndarray
    tie_reference_eligible: np.ndarray
    prior_fill_eligible: np.ndarray
    allowed_domain: np.ndarray
    parent_fixed_area_mm2: float | None
    audit: dict
    recomputed_old_domain_ring: np.ndarray

    def __post_init__(self):
        for value in (
            self.labels,
            self.cached_parent_modelled,
            self.tie_reference_eligible,
            self.prior_fill_eligible,
            self.allowed_domain,
            self.recomputed_old_domain_ring,
        ):
            value.flags.writeable = False


def _mask_digest(mask):
    return sha256(np.packbits(mask.ravel(), bitorder="little").tobytes()).hexdigest()


def _mask_binding(mask):
    return dict(pixels=int(mask.sum()), sha256=_mask_digest(mask))


def _rule_b(
    labels, components, source_ids, *, pixel_area_mm2, barrier_mask, parameters
):
    from .rules import apply_barrier_rule_b

    return apply_barrier_rule_b(
        labels,
        components,
        source_ids,
        pixel_area_mm2=pixel_area_mm2,
        barrier_mask=barrier_mask,
        parameters=parameters,
    )


def outer_domain_ring(domain):
    """One native four-neighbour ring adjoining the exterior old domain.

    Pad the grid so the image edge belongs to the exterior. Closed internal
    feature-support holes are not an outer edge and keep their existing policy.
    This geometry uses only the bound old mask, never a predicted category.
    """
    exterior = np.pad((~domain).astype(np.uint8), 1, constant_values=1)
    cv2.floodFill(exterior, None, (0, 0), 2, flags=4)
    exterior = exterior == 2
    touches = (
        exterior[:-2, 1:-1]
        | exterior[2:, 1:-1]
        | exterior[1:-1, :-2]
        | exterior[1:-1, 2:]
    )
    return domain & touches


def old_domain_interpolation_footprint(
    domain, working_shape_hw, *, parameters: Mapping
):
    """Release the exact bilinear influence of old exterior-rejected cells.

    Reproduce the original regional domain reduction, including its >.999
    coverage rule. Only rejected cells connected to the padded coarse exterior
    supply influence: a closed coarse support hole is deliberately excluded.
    No categories, classifier scores, or output differences select this mask.
    """
    working_shape = tuple(working_shape_hw)
    coverage = cv2.resize(
        domain.astype(np.float32), working_shape[::-1], interpolation=cv2.INTER_AREA
    )
    coarse_domain = coverage > parameters["coarse_coverage_threshold"]
    exterior = np.pad((~coarse_domain).astype(np.uint8), 1, constant_values=1)
    cv2.floodFill(exterior, None, (0, 0), 2, flags=4)
    rejected_exterior = exterior[1:-1, 1:-1] == 2
    influence = cv2.resize(
        rejected_exterior.astype(np.float32),
        domain.shape[::-1],
        interpolation=cv2.INTER_LINEAR,
    )
    footprint = domain & (influence > 0)
    # This is a source-geometry bound, not a fresh probability prediction.
    # Even a score of one in every allowed coarse cell cannot exceed it.
    upper = cv2.resize(
        coarse_domain.astype(np.float32),
        domain.shape[::-1],
        interpolation=cv2.INTER_LINEAR,
    )
    forced_rejection = domain & (upper < parameters["upper_bound_rejection_threshold"])
    return footprint, dict(
        contract=parameters["interpolation_contract"],
        native_shape_hw=list(domain.shape),
        original_working_shape_hw=list(working_shape),
        working_cell_native_yx=[
            domain.shape[0] / working_shape[0],
            domain.shape[1] / working_shape[1],
        ],
        original_domain_mask_sha256=_mask_digest(domain),
        coarse_domain_mask_sha256=_mask_digest(coarse_domain),
        exterior_rejected_coarse_cells=int(rejected_exterior.sum()),
        exterior_rejected_coarse_mask_sha256=_mask_digest(rejected_exterior),
        interpolation_footprint_pixels=int(footprint.sum()),
        interpolation_footprint_mask_sha256=_mask_digest(footprint),
        domain_reduction=parameters["domain_reduction_description"],
        exterior_criterion="four-connected rejected coarse cells connected to padded exterior; closed internal coarse holes excluded",
        interpolation="cv2.INTER_LINEAR to original native grid; positive exterior-zero interpolation weight",
        upper_bound_definition=parameters["upper_bound_definition"],
        upper_bound_forced_rejection_pixels=int(forced_rejection.sum()),
        upper_bound_forced_rejection_in_exterior_footprint_pixels=int(
            (forced_rejection & footprint).sum()
        ),
        upper_bound_forced_rejection_mask_sha256=_mask_digest(forced_rejection),
        original_coarse_scores_or_categories_used=False,
        eligibility_selected_from_output_difference=False,
    )


def fill_rig_labels(
    model_labels,
    rig_domain,
    observed,
    *,
    hidden=None,
    tie_break_reference=None,
    tie_break_mask=None,
    audit=None,
    parameters: Mapping,
):
    """Fill nearest categories without promoting extrapolated pixels to modelled.

    The explicit domain and observation masks distinguish photographed fill
    from unobserved fill. A caller-certified parent category can win only an
    exact minimum-distance tie, with its contract supplied in ``parameters``.
    """
    labels, domain, visible = (
        np.asarray(model_labels),
        np.asarray(rig_domain),
        np.asarray(observed),
    )
    if (
        labels.dtype != np.uint8
        or domain.dtype != bool
        or visible.dtype != bool
        or labels.shape != domain.shape
        or visible.shape != domain.shape
    ):
        raise TypeError("Rig classes and labels must share one explicit grid")
    hidden = (
        np.zeros(domain.shape, bool) if hidden is None else np.asarray(hidden, bool)
    )
    if hidden.shape != domain.shape:
        raise ValueError("Hidden ColorChecker mask differs from the rig grid")
    if np.any(hidden & domain):
        raise ValueError(
            "Hidden ColorChecker/hardware must be excluded from the rig domain"
        )
    supported = domain & visible & np.isin(labels, (0, 1, 2))
    if not supported.any():
        raise ValueError("No modelled source pixel exists for nearest-label filling")
    filled = domain & ~supported
    result = np.full(domain.shape, 255, np.uint8)
    result[supported] = labels[supported]
    if filled.any():
        distances, indices = distance_transform_edt(
            ~supported, return_distances=True, return_indices=True
        )
        result[filled] = labels[indices[0, filled], indices[1, filled]]
        tie_audit = dict(
            contract=parameters["tie_fill_contract"],
            applicable=False,
            preference_candidates=0,
            equal_distance_preferences=0,
            unique_nearest_unchanged=True,
            modelled_support_unchanged=True,
        )
        if tie_break_reference is not None or tie_break_mask is not None:
            preferred, eligible = np.asarray(tie_break_reference), np.asarray(
                tie_break_mask
            )
            if (
                preferred.dtype != np.uint8
                or preferred.shape != domain.shape
                or eligible.dtype != bool
                or eligible.shape != domain.shape
                or np.any(eligible & ~np.isin(preferred, (0, 1, 2)))
            ):
                raise ValueError(
                    "Nearest-fill tie preference requires a source-bound categorical reference/mask"
                )
            # The tie policy is fixed before filling. An inherited category
            # may win ONLY if at least one modelled pixel with that category
            # is at the exact same minimum Euclidean raster distance. It
            # never substitutes a farther neighbour or calls a fill modelled.
            candidates = np.argwhere(filled & eligible & (result != preferred))
            count = 0
            for yy, xx in candidates:
                minimum_squared = int(
                    (int(yy) - int(indices[0, yy, xx])) ** 2
                    + (int(xx) - int(indices[1, yy, xx])) ** 2
                )
                radius = int(np.ceil(distances[yy, xx]))
                y0, y1 = max(0, int(yy) - radius), min(
                    domain.shape[0], int(yy) + radius + 1
                )
                x0, x1 = max(0, int(xx) - radius), min(
                    domain.shape[1], int(xx) + radius + 1
                )
                iy, ix = np.ogrid[y0:y1, x0:x1]
                exact_distance = (iy - int(yy)) ** 2 + (
                    ix - int(xx)
                ) ** 2 == minimum_squared
                matches = (
                    supported[y0:y1, x0:x1]
                    & exact_distance
                    & (labels[y0:y1, x0:x1] == preferred[yy, xx])
                )
                if matches.any():
                    result[yy, xx] = preferred[yy, xx]
                    count += 1
            tie_audit.update(
                applicable=True,
                preference_candidates=len(candidates),
                equal_distance_preferences=count,
            )
        if audit is not None:
            if not isinstance(audit, dict):
                raise TypeError("Fill receipt audit must be a dictionary")
            audit.update(
                tie_audit,
                modelled_pixels=int(supported.sum()),
                filled_observed_pixels=int((filled & visible).sum()),
                filled_unobserved_pixels=int((filled & ~visible).sum()),
            )
    return result, dict(
        modelled=supported,
        filled_observed=filled & visible,
        filled_unobserved=filled & ~visible,
    )


def preserve_prior_categorical_fill(
    cache,
    final_labels,
    *,
    modelled,
    filled_observed,
    filled_unobserved,
    parameters: Mapping,
):
    """Preserve old resolved categories on the predeclared final fill population.

    Call after nearest fill and all numerical inference operators. This pure
    helper changes no modelled pixel and promotes no filled pixel to modelled.
    ``binding=None`` at preparation leaves changed/new acquisitions opted out.
    All masks use the unchanged native grid; the caller binds any packed mask
    files into the new receipt and keeps the strict interior comparison.
    """
    if not isinstance(cache, RetainedSourceInferenceCache):
        raise TypeError("Prepare the source-bound categorical policy before inference")
    values = np.asarray(final_labels)
    masks = tuple(
        np.asarray(mask) for mask in (modelled, filled_observed, filled_unobserved)
    )
    if (
        values.dtype != np.uint8
        or values.shape != cache.labels.shape
        or not np.isin(values, (0, 1, 2, 255)).all()
        or any(mask.dtype != bool or mask.shape != values.shape for mask in masks)
    ):
        raise ValueError("Final categorical fill has invalid native dtype/grid/masks")
    supported, observed, unobserved = masks
    if (
        np.any(supported & (observed | unobserved))
        or np.any(observed & unobserved)
        or np.any((supported | observed | unobserved) & (values == 255))
    ):
        raise ValueError(
            "Final modelled and filled populations must be resolved and disjoint"
        )
    eligible = cache.prior_fill_eligible
    inherited_observed = eligible & ~supported & observed
    inherited_unobserved = eligible & ~supported & unobserved
    inherited = inherited_observed | inherited_unobserved
    now_modelled = eligible & supported
    now_modelled_changed = now_modelled & (values != cache.labels)
    result = values.copy()
    result[inherited] = cache.labels[inherited]
    changed = inherited & (values != result)
    populations = dict(
        prior_fill_eligible=eligible.copy(),
        inherited_fill_observed=inherited_observed,
        inherited_fill_unobserved=inherited_unobserved,
        inherited_categorical_fill=inherited,
        prior_fill_eligible_now_modelled=now_modelled,
        prior_fill_eligible_now_modelled_category_changed=now_modelled_changed,
    )
    audit = dict(
        contract=parameters["prior_fill_contract"],
        run=cache.audit["run"],
        frame_index=cache.audit["frame_index"],
        current_source_sha256=cache.audit["current_source_sha256"],
        source_binding=cache.audit.get("binding"),
        parent_labels_map=cache.audit.get("parent_labels_map"),
        parent_scientific_assets=cache.audit.get("parent_scientific_assets"),
        prepared_before_fresh_inference=cache.audit["prepared_before_fresh_inference"],
        cache_enabled=cache.audit["cache_enabled"],
        eligibility_selected_from_output_difference=False,
        modelled_cache_expanded=False,
        filled_pixels_promoted_to_modelled=False,
        newly_modelled_eligible_policy="Fresh categories retained and differences reported; no silent demotion",
        unique_nearest_fill_policy="Old resolved category preserved on the entire predeclared eligible final-fill population, including unique-nearest mismatches",
        mask_hash_encoding="little-endian packbits, flattened native row-major grid",
        native_shape_hw=list(values.shape),
        changed_after_fill_pixels=int(changed.sum()),
        changed_after_fill_mask_sha256=_mask_digest(changed),
        final_prior_fill_categories_exact=bool(
            np.array_equal(result[inherited], cache.labels[inherited])
        ),
        modelled_categories_unchanged=bool(
            np.array_equal(result[supported], values[supported])
        ),
        independent_final_QA_required=True,
        fresh_parity_claimed=False,
        scientific_mass_approval=False,
    )
    for name, mask in populations.items():
        audit[name + "_pixels"] = int(mask.sum())
        audit[name + "_mask_sha256"] = _mask_digest(mask)
    return result, populations, audit


def crop_native_canvas(full, source_box_xywh):
    """Return the recorded native rectangle as a view without interpolation."""
    x, y, width, height = source_box_xywh
    return full[y : y + height, x : x + width]


def embed_native_canvas(native, *, source_box_xywh, phase_shape_hw, fill):
    """Embed native values at their recorded box, retaining dtype and pixels."""
    x, y, width, height = source_box_xywh
    result = np.full(phase_shape_hw, fill, np.asarray(native).dtype)
    result[y : y + height, x : x + width] = native
    return result


def canvas_homography(native_h, *, source_box_xywh):
    """Conjugate a native homography by the recorded canvas offset."""
    x, y = source_box_xywh[:2]
    translate = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], np.float64)
    return translate @ np.asarray(native_h, np.float64) @ np.linalg.inv(translate)


def revise_seam_labels(
    old_labels,
    fresh_labels,
    ring,
    old_domain,
    old_pre_min,
    fresh_pre_min,
    modelled,
    original_area_mm2,
    *,
    open_components,
    source_component_ids,
    barrier_mask,
    parameters: Mapping,
    rule_parameters: Mapping,
    output_capture=None,
):
    """Recompute incident gas components while preserving all other categories.

    ``old_pre_min`` is the decoded original G3/refit threshold selection before
    its CO2 intersection or minimum. ``fresh_pre_min`` is the fresh threshold
    selection. Both are boolean masks on the complete unchanged canvas.
    ``ring`` is the caller-sealed native outer ring or source-derived exterior
    coarse-domain interpolation footprint. ``fresh_labels`` already includes
    explicit nearest fill there. Filled pixels can change CO2 in the release
    but cannot supply modelled gas.

    The old/new pre-minimum union graph identifies incident components even
    when a ring change splits a formerly connected gas candidate. Unaffected
    components retain their saved post-minimum, post-Rule-B categories. A new
    Rule B reachability change that would affect that retained population is
    an explicit error, requiring a larger declared revision scope.
    """
    old, fresh = np.asarray(old_labels), np.asarray(fresh_labels)
    if (
        old.dtype != np.uint8
        or old.ndim != 2
        or not old.size
        or fresh.dtype != np.uint8
        or fresh.shape != old.shape
        or not np.isin(old, (0, 1, 2, 255)).all()
        or not np.isin(fresh, (0, 1, 2, 255)).all()
    ):
        raise ValueError("Seam labels require matching native uint8 categorical grids")
    ring, domain, inherited, evaluated, supported = tuple(
        np.asarray(value)
        for value in (ring, old_domain, old_pre_min, fresh_pre_min, modelled)
    )
    if any(
        mask.dtype != bool or mask.shape != old.shape
        for mask in (ring, domain, inherited, evaluated, supported)
    ):
        raise ValueError("Seam masks require matching explicit bool grids")
    if np.any(ring & ~domain) or np.any(ring & (fresh == 255)):
        raise ValueError(
            "The released old-domain ring must have resolved fresh or filled labels"
        )
    area = validate_pixel_area(original_area_mm2, old.shape)
    sources = tuple(source_component_ids)
    old_co2 = (old == 1) | (old == 2)
    result = old.copy()
    result[ring] = fresh[ring]
    new_co2 = (result == 1) | (result == 2)
    old_selection = np.where(domain, inherited, evaluated)
    new_selection = np.where(domain & ~ring, inherited, evaluated)
    old_gas_candidates = old_selection & old_co2 & supported
    new_gas_candidates = new_selection & new_co2 & supported
    union = old_gas_candidates | new_gas_candidates
    count, components = cv2.connectedComponents(union.astype(np.uint8), connectivity=4)
    incident = np.zeros(count, bool)
    incident[np.unique(components[ring & union])] = True
    incident[0] = False
    affected = incident[components]
    scoped_candidates = new_gas_candidates & affected
    kept, minimum_audit = area_filtered_gas(
        scoped_candidates,
        area,
        minimum_area_mm2=parameters["gas_minimum_area_mm2"],
        connectivity=parameters["gas_connectivity"],
    )
    if output_capture is not None:
        from .probability_storage import changed_rule
        changed_rule(output_capture, "seam_minimum", scoped_candidates, kept)
    commit = supported & new_co2 & (affected | ring)
    if output_capture is not None:
        from .probability_storage import capture_seam_population
        capture_seam_population(output_capture, locals())
    result[commit] = 1
    result[kept] = 2

    # Reachability depends on the corrected CO2 union, so evaluate Rule B on
    # that entire union after the scoped gas minimum. Its nominal area serves
    # only the inherited diagnostic; report physical removal with original area.
    _, old_rule_b = _rule_b(
        old,
        open_components,
        sources,
        pixel_area_mm2=parameters["nominal_rule_b_pixel_area_mm2"],
        barrier_mask=barrier_mask,
        parameters=rule_parameters,
    )
    gas_before_b = result == 2
    revised, rule_b = _rule_b(
        result,
        open_components,
        sources,
        pixel_area_mm2=parameters["nominal_rule_b_pixel_area_mm2"],
        barrier_mask=barrier_mask,
        parameters=rule_parameters,
    )
    if output_capture is not None:
        changed_rule(output_capture, "seam_B", result, revised)
    changed_reachability = sorted(
        set(old_rule_b["allowed_component_ids"]) ^ set(rule_b["allowed_component_ids"])
    )
    retained_scope = ~ring & ~affected
    retained_gas_or_candidates = retained_scope & ((old == 2) | new_gas_candidates)
    reachability_outside_scope = retained_gas_or_candidates & np.isin(
        open_components, changed_reachability
    )
    changed_by_b_outside_scope = retained_scope & (revised != result)
    if reachability_outside_scope.any() or changed_by_b_outside_scope.any():
        raise ValueError(
            "Rule B requires a wider declared gas scope: "
            f"reachable component IDs changed={changed_reachability}; "
            f"outside-scope candidates={int(reachability_outside_scope.sum())}; "
            f"outside-scope gas removals={int(changed_by_b_outside_scope.sum())}"
        )
    result = revised
    changed = result != old
    changed_co2 = old_co2 != ((result == 1) | (result == 2))
    changed_gas = (old == 2) != (result == 2)
    if (
        np.any(changed_co2 & ~ring)
        or np.any(changed & retained_scope)
        or np.any((old == 255) & ~ring & (result != 255))
    ):
        raise AssertionError("Seam revision exceeded its predeclared CO2/gas scope")
    removed_by_b = gas_before_b & (result != 2)
    rule_b["removed_gas_area_original_variable_cm2"] = sum_pixel_area(
        removed_by_b, area, unit="cm2"
    )
    masks = dict(
        old_pre_minimum_gas=old_gas_candidates,
        corrected_pre_minimum_gas=new_gas_candidates,
        affected_gas_graph=affected,
        affected_gas_after_minimum=kept,
        changed_labels=changed,
        changed_co2=changed_co2,
        changed_gas=changed_gas,
    )
    audit = dict(
        contract=parameters["seam_contract"],
        cpu_only=True,
        training=False,
        ring=_mask_binding(ring),
        old_domain=_mask_binding(domain),
        native_shape_hw=list(old.shape),
        mask_hash_encoding="little-endian packbits, row-major native grid",
        co2_outside_ring_exact=True,
        categories_outside_ring_and_affected_gas_exact=True,
        released_mask=_mask_binding(ring),
        co2_outside_released_mask_exact=True,
        ring_argument_definition="caller-sealed native boundary ring or source-derived exterior coarse-domain interpolation footprint; no output-derived dilation",
        unresolved_outside_ring_exact=True,
        affected_union_component_count=int(incident.sum()),
        union_component_count=count - 1,
        affected_graph_policy="four-connected union of old and corrected pre-minimum modelled gas candidates; components incident to fixed ring, including splits",
        minimum=minimum_audit,
        minimum_scope="corrected pre-minimum gas inside the affected union graph only",
        filtering_area_policy="original rig variable area before measured-area scaling; no global refilter",
        original_area=(
            dict(representation="constant", value_mm2=float(area))
            if np.isscalar(area)
            else dict(representation="native-array", sha256=_array_identity(area))
        ),
        operator_order=[
            "ring CO2 correction",
            "combined literal gas threshold selection",
            "affected-component original-area four-connected minimum",
            "Rule B",
        ],
        old_rule_b_allowed_component_ids=old_rule_b["allowed_component_ids"],
        changed_rule_b_reachable_component_ids=changed_reachability,
        rule_b=rule_b,
        masks={name: _mask_binding(mask) for name, mask in masks.items()},
        mass_eligible=False,
        bottom_mass_allowed=False,
    )
    return result, audit, masks


def merge_native_and_rig(
    centre,
    fresh,
    fresh_support,
    rig_domain,
    observed,
    *,
    release,
    old_domain,
    original_area,
    open_components,
    barrier_mask,
    source_ids,
    fresh_pre_minimum,
    source_box_xywh,
    parameters: Mapping,
    rule_parameters: Mapping,
    prior_cache=None,
    output_capture=None,
):
    """Compose native and rig decisions, preserving the frozen operator order.

    ``source_box_xywh`` embeds the native centre without resampling. Retained
    categories apply only to predeclared final-fill populations; the released
    seam consumes a separate fresh fill and the original physical area grid.
    ``parameters`` supplies frozen cutoffs/contracts, while ``rule_parameters``
    configures each distinct barrier-rule invocation.
    """

    shape = rig_domain.shape
    support = fresh_support.copy()
    edge_labels, edge_b = _rule_b(
        fresh,
        open_components,
        source_ids,
        pixel_area_mm2=parameters["nominal_rule_b_pixel_area_mm2"],
        barrier_mask=barrier_mask,
        parameters=rule_parameters,
    )
    if output_capture is not None:
        from .probability_storage import transition, changed_rule
        transition(output_capture, "edge-Rule-B", fresh, edge_labels)
        changed_rule(output_capture, "edge_B", fresh, edge_labels)
    x, y, width, height = source_box_xywh
    central = np.asarray(centre.labels).copy()
    central_support = (
        np.asarray(centre.domain)
        & np.asarray(centre.flatfield_valid)
        & np.isin(central, (0, 1, 2))
        & crop_native_canvas(rig_domain, source_box_xywh)
    )
    central[~central_support] = 255
    central_b, native_b = _rule_b(
        central,
        crop_native_canvas(open_components, source_box_xywh),
        source_ids,
        pixel_area_mm2=parameters["nominal_rule_b_pixel_area_mm2"],
        barrier_mask=crop_native_canvas(barrier_mask, source_box_xywh),
        parameters=rule_parameters,
    )
    if output_capture is not None:
        transition(output_capture, "centre-Rule-B", central, central_b)
        changed_rule(output_capture, "centre_B", central, central_b)
    merged = edge_labels.copy()
    crop = merged[y : y + height, x : x + width]
    crop[central_support] = central_b[central_support]
    crop[~central_support] = edge_labels[y : y + height, x : x + width][
        ~central_support
    ]
    support[y : y + height, x : x + width] = (
        central_support | fresh_support[y : y + height, x : x + width]
    )
    if output_capture is not None:
        from .probability_storage import state
        state(output_capture, "centre-edge-composition", merged)
    observed = observed | support
    fill_audit = {}
    kwargs = {}
    if prior_cache is not None:
        kwargs = dict(
            tie_break_reference=embed_native_canvas(
                prior_cache.labels,
                source_box_xywh=source_box_xywh,
                phase_shape_hw=shape,
                fill=255,
            ),
            tie_break_mask=embed_native_canvas(
                prior_cache.tie_reference_eligible,
                source_box_xywh=source_box_xywh,
                phase_shape_hw=shape,
                fill=False,
            ),
        )
    filled, classes = fill_rig_labels(
        merged, rig_domain, observed, audit=fill_audit, parameters=parameters, **kwargs
    )
    if output_capture is not None:
        state(output_capture, "categorical-fill", filled)
    prior_audit = None
    if prior_cache is not None:
        native, _, prior_audit = preserve_prior_categorical_fill(
            prior_cache,
            crop_native_canvas(filled, source_box_xywh),
            modelled=crop_native_canvas(classes["modelled"], source_box_xywh),
            filled_observed=crop_native_canvas(
                classes["filled_observed"], source_box_xywh
            ),
            filled_unobserved=crop_native_canvas(
                classes["filled_unobserved"], source_box_xywh
            ),
            parameters=parameters,
        )
        filled[y : y + height, x : x + width] = native
    if output_capture is not None:
        state(output_capture, "preserved-categorical-fill", filled)
    # Fresh ring fill is a separate evaluation, just as in the saved seam
    # worker. It must not inherit source-bound old fill on the released ring.
    # The saved seam worker filled the fresh numerical decision before B;
    # revise_seam_labels subsequently applies B to the assembled graph.
    fresh_filled, fresh_classes = fill_rig_labels(
        fresh, rig_domain, observed, parameters=parameters
    )
    for name in ("modelled", "filled_observed", "filled_unobserved"):
        classes[name][release] = fresh_classes[name][release]
    old_pre = embed_native_canvas(
        np.asarray(centre.preminimum_gas, bool),
        source_box_xywh=source_box_xywh,
        phase_shape_hw=shape,
        fill=False,
    )
    revised, seam_audit, seam_masks = revise_seam_labels(
        filled,
        fresh_filled,
        release,
        old_domain,
        old_pre,
        fresh_pre_minimum,
        classes["modelled"],
        original_area,
        open_components=open_components,
        source_component_ids=source_ids,
        barrier_mask=barrier_mask,
        parameters=parameters,
        rule_parameters=rule_parameters,
        **({} if output_capture is None else dict(output_capture=output_capture)),
    )
    if output_capture is not None:
        output_capture["central_support"] = central_support.copy()
        output_capture["release"] = release.copy()
        state(output_capture, "seam-minimum-and-Rule-B", revised)
    return (
        revised,
        classes,
        dict(
            edge_rule_b=edge_b,
            native_rule_b=native_b,
            fill=fill_audit,
            inherited_fill=prior_audit,
            seam=seam_audit,
        ),
        seam_masks,
    )
