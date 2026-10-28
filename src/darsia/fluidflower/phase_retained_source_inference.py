"""Explicit inference-cache reuse for unchanged FF_AC source acquisitions.

Prepare this policy before fresh inference. The inherited categorical result
is an inference cache, never training evidence or proof of fresh-pipeline
parity. New pixels use the frozen heads; missing features are filled later.
The cache is not rescaled, rethresholded, or used on a reselected acquisition.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

from .phase_saved_review_metadata import configured_parameters
from .phase.metrics import sum_pixel_area, validate_pixel_area
from .project import sha256_file

CONTRACT = "ffac-retained-source-inference-cache-v1"
PRIOR_FILL_CONTRACT = "ffac-retained-source-categorical-fill-v1"
OLD_DOMAIN_RING_CONTRACT = "ffac-old-domain-one-pixel-fresh-ring-v1"
OLD_DOMAIN_INTERPOLATION_CONTRACT = "ffac-old-domain-exterior-interpolation-footprint-v1"
ENCODING_CONTRACT = "ffac-frozen-probability-storage-contract-v1"
PROBABILITY_CONTRACT = "ffac-g3-downstream-probability-layers-v1"
FRAME_CONTRACT = "ffac-gas-barrier-rule-b-candidate-v1"
MASK_NAMES = ("photo_sand_domain", "feature_support", "regional_native_domain",
              "flatfield_valid", "feature_eligible", "optical_presence_ge_half",
              "frozen_gas", "frozen_unresolved")
GAS_RULES = dict(pixel_threshold=.7, regional_threshold=.6,
                 minimum_area_mm2=2.5, connectivity=4,
                 top_surface_applied=True, rule_b_applied=True)


def _asset(item, kind):
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise ValueError("Missing SHA-bound " + kind)
    digest = item.get("sha256")
    path = Path(item["path"])
    if (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not path.is_file() or sha256_file(path) != digest):
        raise ValueError("Changed SHA-bound " + kind)
    return path


def _json(item, kind):
    return json.loads(_asset(item, kind).read_text(encoding="utf-8-sig"))


def _identity(run, frame_index, source_sha256):
    if (not isinstance(run, str) or re.fullmatch(r"ac[0-9]{2}", run) is None
            or type(frame_index) is not int or frame_index < 0
            or not isinstance(source_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None):
        raise ValueError("Retained inference needs an explicit run/index/source SHA")


def _mask_digest(mask):
    return sha256(np.packbits(mask.ravel(), bitorder="little").tobytes()).hexdigest()


def _outer_domain_ring(domain):
    """One native four-neighbour ring adjoining the exterior old domain.

    Pad the grid so the image edge belongs to the exterior. Closed internal
    feature-support holes are not an outer edge and keep their existing policy.
    This geometry uses only the bound old mask, never a predicted category.
    """
    from .phase.domain import outer_domain_ring
    return outer_domain_ring(domain)


def _old_domain_interpolation_footprint(domain, working_shape_hw):
    """Release the exact bilinear influence of old exterior-rejected cells.

    Reproduce the original regional domain reduction, including its >.999
    coverage rule. Only rejected cells connected to the padded coarse exterior
    supply influence: a closed coarse support hole is deliberately excluded.
    No categories, classifier scores, or output differences select this mask.
    """
    working_shape = tuple(working_shape_hw)
    coverage = cv2.resize(domain.astype(np.float32), working_shape[::-1],
                          interpolation=cv2.INTER_AREA)
    coarse_domain = coverage > .999
    exterior = np.pad((~coarse_domain).astype(np.uint8), 1, constant_values=1)
    cv2.floodFill(exterior, None, (0, 0), 2, flags=4)
    rejected_exterior = exterior[1:-1, 1:-1] == 2
    influence = cv2.resize(rejected_exterior.astype(np.float32), domain.shape[::-1],
                           interpolation=cv2.INTER_LINEAR)
    footprint = domain & (influence > 0)
    # This is a source-geometry bound, not a fresh probability prediction.
    # Even a score of one in every allowed coarse cell cannot exceed it.
    upper = cv2.resize(coarse_domain.astype(np.float32), domain.shape[::-1],
                       interpolation=cv2.INTER_LINEAR)
    forced_rejection = domain & (upper < .5)
    return footprint, dict(contract=OLD_DOMAIN_INTERPOLATION_CONTRACT,
        native_shape_hw=list(domain.shape), original_working_shape_hw=list(working_shape),
        working_cell_native_yx=[domain.shape[0]/working_shape[0], domain.shape[1]/working_shape[1]],
        original_domain_mask_sha256=_mask_digest(domain),
        coarse_domain_mask_sha256=_mask_digest(coarse_domain),
        exterior_rejected_coarse_cells=int(rejected_exterior.sum()),
        exterior_rejected_coarse_mask_sha256=_mask_digest(rejected_exterior),
        interpolation_footprint_pixels=int(footprint.sum()),
        interpolation_footprint_mask_sha256=_mask_digest(footprint),
        domain_reduction="cv2.INTER_AREA native bool->float32; coarse domain coverage >0.999",
        exterior_criterion="four-connected rejected coarse cells connected to padded exterior; closed internal coarse holes excluded",
        interpolation="cv2.INTER_LINEAR to original native grid; positive exterior-zero interpolation weight",
        upper_bound_definition="U=INTER_LINEAR(coarse_domain.astype(float32)); any reconstructed regional field is <=U",
        upper_bound_forced_rejection_pixels=int(forced_rejection.sum()),
        upper_bound_forced_rejection_in_exterior_footprint_pixels=int((forced_rejection & footprint).sum()),
        upper_bound_forced_rejection_mask_sha256=_mask_digest(forced_rejection),
        original_coarse_scores_or_categories_used=False,
        eligibility_selected_from_output_difference=False)


def probability_encoding_contract(receipt, native_shape_hw):
    """Delegate original storage admission to its one canonical implementation."""
    from .pipeline.phase_inputs import probability_encoding_contract as admit
    return admit(receipt, native_shape_hw,
                 parameters=configured_parameters("retained"))


@dataclass(frozen=True)
class RetainedSourceInferenceCache:
    labels: np.ndarray
    cached_parent_modelled: np.ndarray
    tie_reference_eligible: np.ndarray
    prior_fill_eligible: np.ndarray
    allowed_domain: np.ndarray
    parent_fixed_area_mm2: float | None
    audit: dict
    recomputed_old_domain_ring: np.ndarray

    def __post_init__(self):
        from .phase.domain import RetainedSourceInferenceCache as CanonicalCache
        CanonicalCache.__post_init__(self)


def prepare_retained_source_cache(binding, *, run, frame_index,
        current_source_sha256, current_log, allowed_domain, producer_audit=None,
        recompute_old_domain_ring=False,
        recompute_old_domain_interpolation_footprint=False):
    """Seal fixed eligibility before inference; no guessed frame correspondence.

    ``binding=None`` explicitly opts out of reuse for a changed/new acquisition.
    A supplied, mismatched binding fails instead of falling back to a guessed
    neighbour. ``current_log`` is the retained native CHW float32 input, before
    any spatial alignment or feature computation.

    ``recompute_old_domain_ring=True`` explicitly releases the old domain's
    outermost native ring from all parent categorical reuse. Its classes must
    come from the same fresh frozen-head evaluation as the added domain,
    including selection before the four-connected gas minimum. The historical
    default retains the previous immutable cache policy.

    ``recompute_old_domain_interpolation_footprint=True`` instead releases the
    source-derived bilinear influence of exterior coarse-domain rejections,
    together with the native ring. Its working shape and masks are bound by
    the original probability receipt. No output discrepancy sets its extent.
    """
    _identity(run, frame_index, current_source_sha256)
    if type(recompute_old_domain_ring) is not bool or type(recompute_old_domain_interpolation_footprint) is not bool:
        raise TypeError("Old domain ring recomputation requires an explicit bool policy")
    release_enabled = recompute_old_domain_ring or recompute_old_domain_interpolation_footprint
    current, domain = np.asarray(current_log), np.asarray(allowed_domain)
    if (current.dtype != np.float32 or current.ndim != 3 or current.shape[0] != 3
            or domain.dtype != bool or domain.shape != current.shape[1:]):
        raise ValueError("Retained inference requires native float32 and explicit bool domain")
    empty = np.full(domain.shape, 255, np.uint8)
    if binding is None:
        return RetainedSourceInferenceCache(empty, np.zeros(domain.shape, bool),
            np.zeros(domain.shape, bool), np.zeros(domain.shape, bool), domain.copy(), None,
            dict(contract=CONTRACT, run=run, frame_index=frame_index,
                 current_source_sha256=current_source_sha256,
                 prepared_before_fresh_inference=True, cache_enabled=False,
                 cache_reason="explicit no reuse for changed/new source acquisition",
                 cached_parent_modelled_pixels=0, tie_reference_eligible_pixels=0,
                 tie_reference_eligible_mask_sha256=_mask_digest(np.zeros(domain.shape, bool)),
                 prior_categorical_fill_contract=PRIOR_FILL_CONTRACT,
                 prior_fill_eligible_pixels=0,
                 prior_fill_eligible_mask_sha256=_mask_digest(np.zeros(domain.shape, bool)),
                 old_domain_ring=dict(contract=OLD_DOMAIN_RING_CONTRACT,
                     enabled=release_enabled, applicable=False,
                     interpolation_footprint_requested=recompute_old_domain_interpolation_footprint,
                     recomputed_pixels=0, recomputed_mask_sha256=_mask_digest(np.zeros(domain.shape, bool)),
                     prepared_before_fresh_inference=True,
                     eligibility_selected_from_output_difference=False),
                 training_input=False,
                 fresh_parity_claimed=False, mass_eligible=False), np.zeros(domain.shape, bool))
    if (binding.get("contract") != CONTRACT or binding.get("run") != run
            or binding.get("frame_index") != frame_index):
        raise ValueError("Retained inference binding has a different explicit run/index")
    native = _json(binding.get("parent_native_manifest"), "parent native manifest")
    frame_receipt = _json(binding.get("parent_frame_receipt"), "parent categorical receipt")
    exported = _json(binding.get("parent_probability_receipt"), "parent probability receipt")
    if (native.get("run") != run or native.get("analysis_image_shape_hw") != list(domain.shape)
            or frame_index >= len(native.get("frame_sources") or [])):
        raise ValueError("Parent native source does not contain this exact frame/grid")
    frame = native["frame_sources"][frame_index]
    if frame.get("source_sha256") != current_source_sha256:
        raise ValueError("A changed acquisition cannot reuse parent inference")
    receipt_binding = frame_receipt.get("binding") or {}
    if (frame_receipt.get("contract") != FRAME_CONTRACT or frame_receipt.get("complete") is not True
            or receipt_binding.get("run") != run or receipt_binding.get("frame_index") != frame_index
            or receipt_binding.get("native_manifest_sha256") != binding["parent_native_manifest"]["sha256"]):
        raise ValueError("Parent categorical proof is not bound to this native frame")
    encoding = probability_encoding_contract(exported, domain.shape)
    export_binding, metadata = exported["binding"], exported["metadata"]
    if export_binding.get("run") != run or export_binding.get("frame_index") != frame_index:
        raise ValueError("Parent probability receipt belongs to a different frame")
    original_native_asset = dict(path=metadata.get("native_manifest_path"),
                                 sha256=export_binding.get("native_manifest_sha256"))
    original_native = _json(original_native_asset, "original probability native manifest")
    if (original_native.get("run") != run
            or original_native.get("analysis_image_shape_hw") != list(domain.shape)
            or frame_index >= len(original_native.get("frame_sources") or [])):
        raise ValueError("Probability source uses a different native frame/grid")
    original_frame = original_native["frame_sources"][frame_index]
    scientific_keys = ("source_sha256", "v5_raw_sha256", "v5_log_linear_rgb_path",
        "v5_log_linear_rgb_sha256", "v5_log_linear_valid_mask_path", "v5_log_linear_valid_mask_sha256")
    if any(not frame.get(key) or frame[key] != original_frame.get(key) for key in scientific_keys):
        raise ValueError("Parent source alias changed scientific acquisition/cache bindings")
    if (not isinstance(producer_audit, dict) or producer_audit.get("cpu_only") is not True
            or producer_audit.get("training") is not False
            or (producer_audit.get("parent_cache") or {}).get("parent_float_bytes_preserved") is not True
            or (producer_audit.get("processor") or {}).get("parent_native_manifest_sha256") !=
                binding["parent_native_manifest"]["sha256"]
            or ((producer_audit.get("sources") or {}).get("camera_jpeg") or {}).get("sha256") != current_source_sha256
            or ((producer_audit.get("sources") or {}).get("camera_arw") or {}).get("sha256") != frame["v5_raw_sha256"]):
        raise ValueError("Producer did not audit preserved parent scientific bits")
    assets = {}
    for name in ("v5_log_linear_rgb", "v5_log_linear_valid_mask"):
        item = dict(path=frame[name + "_path"], sha256=frame[name + "_sha256"])
        assets[name] = item
        _asset(item, "parent scientific " + name)
    parent = np.load(assets["v5_log_linear_rgb"]["path"], mmap_mode="r", allow_pickle=False)
    valid = np.load(assets["v5_log_linear_valid_mask"]["path"], mmap_mode="r", allow_pickle=False)
    if (parent.dtype != np.float32 or parent.shape != current.shape
            or valid.dtype != bool or valid.shape != domain.shape
            or not np.isfinite(parent[:, valid]).all()):
        raise ValueError("Parent scientific cache/validity has invalid native dtype/grid")
    # Verify all parent-supported pixels, not merely the pixels we later reuse.
    if (not np.isfinite(current[:, valid]).all()
            or not np.array_equal(current[:, valid].view(np.uint32), parent[:, valid].view(np.uint32))):
        raise ValueError("Current scientific float bits differ on parent-valid support")
    packed_asset = exported["files"].get("extra_masks.packbits.npz")
    with np.load(_asset(packed_asset, "original inference masks"), allow_pickle=False) as archive:
        packed = archive["masks"]
        if packed.dtype != np.uint8 or packed.shape != (len(MASK_NAMES), (domain.size + 7) // 8):
            raise ValueError("Original inference mask packing/grid changed")
        masks = {name: np.unpackbits(row, bitorder="little", count=domain.size).reshape(domain.shape).astype(bool)
                 for name, row in zip(MASK_NAMES, packed)}
    if not np.array_equal(masks["regional_native_domain"], masks["photo_sand_domain"] & masks["feature_support"]):
        raise ValueError("Original regional domain differs from its explicit photo/feature support")
    label_asset = frame_receipt.get("labels_map")
    with Image.open(_asset(label_asset, "parent categorical L proof")) as image:
        labels = np.array(image)
    if labels.dtype != np.uint8 or labels.shape != domain.shape or not np.isin(labels, (0, 1, 2, 255)).all():
        raise ValueError("Parent categorical proof does not share the exact native grid")
    cached = (domain & valid & masks["regional_native_domain"] & masks["flatfield_valid"]
              & ~masks["frozen_unresolved"] & np.isin(labels, (0, 1, 2)))
    original_cached_pixels = int(cached.sum())
    native_ring = (_outer_domain_ring(masks["regional_native_domain"])
                   if release_enabled else np.zeros(domain.shape, bool))
    interpolation = None
    release = native_ring
    if recompute_old_domain_interpolation_footprint:
        footprint, interpolation = _old_domain_interpolation_footprint(
            masks["regional_native_domain"], encoding["regional_working_shape_hw"])
        release = native_ring | footprint
        interpolation.update(source_assets=dict(original_masks=packed_asset,
            original_probability_receipt=binding["parent_probability_receipt"]),
            native_ring_pixels=int(native_ring.sum()),
            native_ring_mask_sha256=_mask_digest(native_ring),
            released_eligible_pixels=int((release & domain).sum()),
            released_eligible_mask_sha256=_mask_digest(release & domain),
            allowed_domain_clipping_applied=True)
    ring = domain & release
    cached &= ~ring
    # An existing category can resolve an exact nearest-distance tie without
    # making an invalid-feature pixel modelled. Unique nearest assignments are
    # unaffected; caller must record these pixels as categorical extrapolation.
    tie_eligible = domain & valid & masks["photo_sand_domain"] & np.isin(labels, (0, 1, 2)) & ~ring
    # This population is declared from old support and source proof, before
    # inference or observing any resulting discrepancy. It is categorical
    # extrapolation, distinct from both modelled reuse and nearest-distance ties.
    prior_fill = tie_eligible & ~cached
    fixed_area = validate_pixel_area(float(metadata["pixel_area_mm2"]), domain.shape)
    audit = dict(contract=CONTRACT, run=run, frame_index=frame_index,
        current_source_sha256=current_source_sha256, cache_enabled=True,
        prepared_before_fresh_inference=True, inference_cache_not_training_evidence=True,
        binding=binding, original_probability_native_manifest=original_native_asset,
        parent_labels_map=label_asset, parent_scientific_assets=assets,
        parent_valid_pixels=int(valid.sum()), parent_scientific_bits_verified=True,
        producer_parent_cache=producer_audit["parent_cache"], probability_encoding=encoding,
        cache_eligibility=("allowed domain AND original photo/feature support AND original flatfield-valid AND parent scientific-valid AND original frozen label!=255 AND parent category0/1/2"
                           + (" AND NOT predeclared recomputed old boundary release" if release_enabled else "")),
        cached_parent_modelled_pixels=int(cached.sum()), cached_parent_modelled_mask_sha256=_mask_digest(cached),
        old_domain_ring=dict(contract=OLD_DOMAIN_RING_CONTRACT,
            enabled=release_enabled, applicable=True,
            native_ring_requested=recompute_old_domain_ring,
            interpolation_footprint_requested=recompute_old_domain_interpolation_footprint,
            released_mask_definition=("native outer ring UNION source-derived exterior-rejected coarse-cell bilinear interpolation footprint"
                                     if recompute_old_domain_interpolation_footprint else "native one-pixel outer ring"),
            native_ring_pixels=int((native_ring & domain).sum()),
            interpolation_footprint=interpolation,
            original_domain_mask_sha256=_mask_digest(masks["regional_native_domain"]),
            original_domain="SHA-bound original photo_sand_domain AND feature_support",
            geometry=("one native exterior pixel ring plus exact bilinear influence of original exterior-rejected coarse cells; clipped to allowed domain"
                      if recompute_old_domain_interpolation_footprint else
                      "one native pixel with a four-connected neighbour in the exterior of the old domain; closed internal holes excluded; clipped to allowed domain"),
            recomputed_pixels=int(ring.sum()), recomputed_mask_sha256=_mask_digest(ring),
            released_cached_pixels=original_cached_pixels-int(cached.sum()),
            prepared_before_fresh_inference=True,
            eligibility_selected_from_output_difference=False,
            cache_and_prior_fill_excluded=True,
            classification_policy="fresh frozen-head evaluation with the added domain before four-connected gas minimum; unsupported pixels retain explicit fill flags"),
        tie_reference_eligible_pixels=int(tie_eligible.sum()),
        tie_reference_eligible_mask_sha256=_mask_digest(tie_eligible),
        tie_reference_policy="SHA-bound parent category may select only an equally nearest modelled neighbour; not modelled reuse, not a unique-nearest override",
        prior_categorical_fill_contract=PRIOR_FILL_CONTRACT,
        prior_fill_eligibility=("allowed domain AND parent scientific-valid AND original photo-sand AND resolved parent category0/1/2 AND NOT cached parent modelled"
                                + (" AND NOT predeclared recomputed old boundary release" if release_enabled else "")),
        prior_fill_eligible_pixels=int(prior_fill.sum()),
        prior_fill_eligible_mask_sha256=_mask_digest(prior_fill),
        prior_fill_policy="Preserve the prior category only after final nearest fill on non-modelled filled pixels; never demote fresh-modelled pixels or select eligibility from changed pixels",
        parent_fixed_area_mm2=fixed_area, inherited_parent_rules_retained=True,
        fresh_parity_claimed=False, mass_eligible=False)
    return RetainedSourceInferenceCache(labels, cached, tie_eligible, prior_fill,
                                       domain.copy(), fixed_area, audit, ring)


def _gas_area_crossings(result, cached, old_fixed_area, current_area, minimum):
    count, components, stats, _ = cv2.connectedComponentsWithStats((result == 2).astype(np.uint8), connectivity=4)
    if count == 1:
        return dict(component_count=0, retained_component_count=0, crossings=[],
                    component_filter_reapplied=False)
    area = validate_pixel_area(current_area, result.shape)
    physical = (stats[:, cv2.CC_STAT_AREA].astype(np.float64) * area if np.isscalar(area)
                else np.bincount(components.ravel(), weights=area.ravel(), minlength=count))
    retained = np.bincount(components[cached & (result == 2)], minlength=count)
    events = []
    for identity in range(1, count):
        if not retained[identity]:
            continue
        old_area = float(stats[identity, cv2.CC_STAT_AREA]) * old_fixed_area
        new_area = float(physical[identity])
        if (old_area >= minimum) != (new_area >= minimum):
            events.append(dict(component_id=identity, pixels=int(stats[identity, cv2.CC_STAT_AREA]),
                cached_parent_gas_pixels=int(retained[identity]), fixed_area_mm2=old_area,
                revised_area_mm2=new_area, fixed_passes_minimum=old_area >= minimum,
                revised_passes_minimum=new_area >= minimum))
    return dict(component_count=count - 1, retained_component_count=int(np.count_nonzero(retained[1:])),
                crossings=events, component_filter_reapplied=False,
                audit_extent="final inherited-plus-fresh component before nearest fill; Top/Rule B may fragment an inherited component",
                crossing_policy="report only; cached categorical result is not rethresholded")


def load_retained_pre_minimum_gas_selection(cache):
    """Decode SHA-bound G3/refit selection before any component minimum.

    This returns only the literal .7/.6 threshold mask. The caller intersects
    it with the corrected CO2 union, combines the fresh boundary selection,
    and applies the unchanged four-connected minimum and Rule B. Reusing a
    post-minimum gas category would lose rejected candidates that can acquire
    a connection through the recomputed ring. No new inference is claimed.
    """
    if not isinstance(cache, RetainedSourceInferenceCache):
        raise TypeError("Pre-minimum gas selection requires a sealed source cache")
    shape = cache.labels.shape
    if not cache.audit["cache_enabled"]:
        return np.zeros(shape, bool), dict(
            contract="ffac-source-bound-pre-minimum-g3-selection-v1",
            applicable=False, selected_pixels=0, training_input=False)
    receipt_asset = cache.audit["binding"]["parent_probability_receipt"]
    receipt = _json(receipt_asset, "parent probability receipt")
    encoding = probability_encoding_contract(receipt, shape)
    if encoding != cache.audit["probability_encoding"]:
        raise ValueError("Prepared source probability encoding changed")
    files = receipt["files"]
    g3_asset, regional_asset = files["g3_pixel.npy"], files["regional_refit.npy"]
    raw_g3 = np.load(_asset(g3_asset, "pre-minimum G3 probabilities"), allow_pickle=False)
    raw_regional = np.load(_asset(regional_asset, "pre-minimum refit probabilities"), allow_pickle=False)
    expected_g3 = np.uint8 if encoding["layout"] != "native-float16" else np.float16
    expected_regional_shape = (shape if encoding["layout"] == "native-float16"
                               else tuple(encoding["regional_working_shape_hw"]))
    if (raw_g3.dtype != expected_g3 or raw_g3.shape != shape
            or raw_regional.dtype != np.float16 or raw_regional.shape != expected_regional_shape):
        raise ValueError("Pre-minimum probability array dtype/grid changed")
    g3 = raw_g3.astype(np.float32)
    if raw_g3.dtype == np.uint8:
        g3 /= np.float32(254.)
        g3[raw_g3 == 255] = np.nan
    regional = raw_regional.astype(np.float32)
    if (np.isinf(g3).any() or np.any((g3[np.isfinite(g3)] < 0) | (g3[np.isfinite(g3)] > 1))
            or not np.isfinite(regional).all() or np.any((regional < 0) | (regional > 1))):
        raise ValueError("Pre-minimum probability arrays contain invalid scores")
    regional = cv2.resize(regional, shape[::-1], interpolation=cv2.INTER_LINEAR)
    selected = (g3 >= GAS_RULES["pixel_threshold"]) & (regional >= GAS_RULES["regional_threshold"])
    return selected, dict(contract="ffac-source-bound-pre-minimum-g3-selection-v1",
        applicable=True, probability_receipt=receipt_asset,
        probability_assets={"g3_pixel.npy": g3_asset, "regional_refit.npy": regional_asset},
        probability_encoding=encoding,
        pixel_threshold=GAS_RULES["pixel_threshold"], regional_threshold=GAS_RULES["regional_threshold"],
        regional_resize="decoded float32 cv2.INTER_LINEAR to native grid before threshold",
        selected_pixels=int(selected.sum()), selected_mask_sha256=_mask_digest(selected),
        co2_union_intersection_applied=False, minimum_applied=False, rule_b_applied=False,
        source_unchanged_only=True, new_inference_claimed=False, training_input=False,
        mass_eligible=False)


def merge_retained_source_inference(cache, fresh_labels, fresh_support, *, pixel_area_mm2,
                                   gas_rules):
    """Commit fresh classes outside sealed cache; then caller fills missing data.

    ``fresh_labels`` must already have its frozen thresholds, area-weighted
    minimum, photographic top exclusion and Rule B applied. This operation
    never repeats these operators over cached pixels.
    """
    if not isinstance(cache, RetainedSourceInferenceCache):
        raise TypeError("Prepare the retained-source inference cache before fresh inference")
    values, supported = np.asarray(fresh_labels), np.asarray(fresh_support)
    if (values.dtype != np.uint8 or values.shape != cache.labels.shape
            or supported.dtype != bool or supported.shape != values.shape
            or not np.isin(values, (0, 1, 2, 255)).all()
            or np.any(supported & ~np.isin(values, (0, 1, 2)))):
        raise ValueError("Fresh categorical support has invalid dtype/grid/labels")
    if not isinstance(gas_rules, dict) or any(gas_rules.get(key) != value for key, value in GAS_RULES.items()):
        raise ValueError("Fresh inference did not declare the unchanged G3/minimum/Top/Rule B operators")
    area = validate_pixel_area(pixel_area_mm2, values.shape)
    cached = cache.cached_parent_modelled
    fresh = cache.allowed_domain & supported & ~cached
    result = np.full(values.shape, 255, np.uint8)
    result[fresh] = values[fresh]
    result[cached] = cache.labels[cached]
    modelled = cached | fresh
    reused_exact = np.array_equal(result[cached], cache.labels[cached])
    if np.any(cached & fresh) or not reused_exact:
        raise AssertionError("Retained and fresh inference populations must be disjoint")
    crossing = (_gas_area_crossings(result, cached, cache.parent_fixed_area_mm2, area, 2.5)
                if cache.parent_fixed_area_mm2 is not None else
                dict(component_count=None, retained_component_count=0, crossings=[], component_filter_reapplied=False))
    audit = dict(cache.audit)
    audit.update(contract=CONTRACT,
        cached_parent_modelled_pixels=int(cached.sum()), freshly_modelled_pixels=int(fresh.sum()),
        modelled_pixels=int(modelled.sum()), unmodelled_inside_allowed_pixels=int((cache.allowed_domain & ~modelled).sum()),
        cached_parent_modelled_mask_sha256=_mask_digest(cached), freshly_modelled_mask_sha256=_mask_digest(fresh),
        mask_hash_encoding="little-endian packbits, flattened native row-major grid",
        cached_parent_categories_exact=reused_exact,
        fresh_evaluation_commit_policy="commit only outside predeclared cache; full feature context evaluation may still be needed",
        gas_rules=dict(GAS_RULES), retained_component_area_crossings=crossing,
        inherited_component_rule_limit="The cached result already includes parent Top/Rule B/minimum operators. Merge can alter component connectivity; final combined-rule consistency needs a separate audit, never a silent global refilter of the cache.",
        cached_parent_area_cm2=sum_pixel_area(cached, area, unit="cm2"),
        fresh_area_cm2=sum_pixel_area(fresh, area, unit="cm2"),
        nearest_fill_applied=False, independent_final_QA_required=True,
        fresh_parity_claimed=False, scientific_mass_approval=False)
    return result, dict(cached_parent_modelled=cached.copy(), freshly_modelled=fresh,
                        modelled=modelled, recomputed_old_domain_ring=cache.recomputed_old_domain_ring.copy()), audit


def preserve_prior_categorical_fill(cache, final_labels, *, modelled,
                                    filled_observed, filled_unobserved):
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
    masks = tuple(np.asarray(mask) for mask in (modelled, filled_observed, filled_unobserved))
    if (values.dtype != np.uint8 or values.shape != cache.labels.shape
            or not np.isin(values, (0, 1, 2, 255)).all()
            or any(mask.dtype != bool or mask.shape != values.shape for mask in masks)):
        raise ValueError("Final categorical fill has invalid native dtype/grid/masks")
    supported, observed, unobserved = masks
    if (np.any(supported & (observed | unobserved)) or np.any(observed & unobserved)
            or np.any((supported | observed | unobserved) & (values == 255))):
        raise ValueError("Final modelled and filled populations must be resolved and disjoint")
    eligible = cache.prior_fill_eligible
    inherited_observed = eligible & ~supported & observed
    inherited_unobserved = eligible & ~supported & unobserved
    inherited = inherited_observed | inherited_unobserved
    now_modelled = eligible & supported
    now_modelled_changed = now_modelled & (values != cache.labels)
    result = values.copy()
    result[inherited] = cache.labels[inherited]
    changed = inherited & (values != result)
    populations = dict(prior_fill_eligible=eligible.copy(),
        inherited_fill_observed=inherited_observed,
        inherited_fill_unobserved=inherited_unobserved,
        inherited_categorical_fill=inherited,
        prior_fill_eligible_now_modelled=now_modelled,
        prior_fill_eligible_now_modelled_category_changed=now_modelled_changed)
    audit = dict(contract=PRIOR_FILL_CONTRACT, run=cache.audit["run"],
        frame_index=cache.audit["frame_index"],
        current_source_sha256=cache.audit["current_source_sha256"],
        source_binding=cache.audit.get("binding"),
        parent_labels_map=cache.audit.get("parent_labels_map"),
        parent_scientific_assets=cache.audit.get("parent_scientific_assets"),
        prepared_before_fresh_inference=cache.audit["prepared_before_fresh_inference"],
        cache_enabled=cache.audit["cache_enabled"],
        eligibility_selected_from_output_difference=False,
        modelled_cache_expanded=False, filled_pixels_promoted_to_modelled=False,
        newly_modelled_eligible_policy="Fresh categories retained and differences reported; no silent demotion",
        unique_nearest_fill_policy="Old resolved category preserved on the entire predeclared eligible final-fill population, including unique-nearest mismatches",
        mask_hash_encoding="little-endian packbits, flattened native row-major grid",
        native_shape_hw=list(values.shape),
        changed_after_fill_pixels=int(changed.sum()),
        changed_after_fill_mask_sha256=_mask_digest(changed),
        final_prior_fill_categories_exact=bool(np.array_equal(result[inherited], cache.labels[inherited])),
        modelled_categories_unchanged=bool(np.array_equal(result[supported], values[supported])),
        independent_final_QA_required=True, fresh_parity_claimed=False,
        scientific_mass_approval=False)
    for name, mask in populations.items():
        audit[name + "_pixels"] = int(mask.sum())
        audit[name + "_mask_sha256"] = _mask_digest(mask)
    return result, populations, audit


__all__ = ["CONTRACT", "PRIOR_FILL_CONTRACT", "OLD_DOMAIN_RING_CONTRACT", "OLD_DOMAIN_INTERPOLATION_CONTRACT", "ENCODING_CONTRACT", "GAS_RULES", "RetainedSourceInferenceCache",
           "probability_encoding_contract", "prepare_retained_source_cache",
           "load_retained_pre_minimum_gas_selection", "merge_retained_source_inference",
           "preserve_prior_categorical_fill"]


def __getattr__(name):
    keys = {"CO2_SHA256": "co2", "G3_SHA256": "g3", "REFIT_SHA256": "regional_refit"}
    if name not in keys:
        raise AttributeError(name)
    return configured_parameters("retained")["expected_models"][keys[name]]
