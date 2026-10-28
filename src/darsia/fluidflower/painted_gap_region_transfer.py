"""Suggested class pixels through existing placements; never an approval."""
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np


def _readonly(value):
    """Own immutable pixels instead of borrowing a mutable GUI draft."""
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)


def component_transfer_coverage(source_component, source_native_to_painting_h,
                                target_component, target_native_to_painting_h):
    """Measure descriptive overlap with the original carrier, without classes.

    Zero-valued scratch pixels have no scientific label meaning here. Only the
    carrier's explicit support is consumed; no class, draft or proof is made.
    The literal placements and nearest/support semantics match region transfer.
    """
    from .geometry_raster_sources import carry_label_reference
    from .map_placement import _matrix

    for component in (source_component, target_component):
        if (not isinstance(component, np.ndarray) or component.dtype != bool
                or component.ndim != 2 or min(component.shape) < 1 or not component.any()):
            raise ValueError("Class transfer needs the exact nonempty native gap components")
    source = _matrix(source_native_to_painting_h, "Source native-to-painting placement")
    target = _matrix(target_native_to_painting_h, "Target native-to-painting placement")
    carried = carry_label_reference(np.zeros(source_component.shape, np.uint8),
        reference_matrix=np.linalg.inv(source).tolist(),
        target_matrix=np.linalg.inv(target).tolist(),
        target_shape_hw=list(target_component.shape),
        reference_filled=np.zeros(source_component.shape, bool),
        source_fill_status="KNOWN", reference_support=source_component)
    matched = target_component & carried["support"]
    return dict(matched_pixels=int(np.count_nonzero(matched)),
                remaining_pixels=int(np.count_nonzero(target_component & ~matched)))


def transfer_region_masks(class_masks, source_component, source_native_to_painting_h,
                          target_component, target_native_to_painting_h):
    """Preview only the marked source pixels which meet this actual target gap.

    Both matrices already map a run's own native grid to the same painting.
    The existing categorical carrier supplies its original nearest sampling;
    its explicit support removes all edge replication from these suggestions.
    Unmarked target pixels stay unmarked. Sand remains a descriptive draft key.
    """
    from .geometry_raster_sources import carry_label_reference
    from .map_placement import _matrix
    from .painted_gap_regions import REGION_CLASSIFICATIONS

    classes = tuple(REGION_CLASSIFICATIONS)
    for component in (source_component, target_component):
        if (not isinstance(component, np.ndarray) or component.dtype != bool
                or component.ndim != 2 or min(component.shape) < 1 or not component.any()):
            raise ValueError("Class transfer needs the exact nonempty native gap components")
    if not isinstance(class_masks, Mapping) or set(class_masks) != set(classes):
        raise ValueError("Class transfer needs the complete explicit region palette")
    codes = np.zeros(source_component.shape, np.uint8)
    marked = np.zeros(source_component.shape, bool)
    for index, name in enumerate(classes, 1):
        mask = class_masks[name]
        if (not isinstance(mask, np.ndarray) or mask.dtype != bool
                or mask.shape != source_component.shape
                or np.any(mask & ~source_component) or np.any(mask & marked)):
            raise ValueError("Source class masks must be disjoint inside their original native component")
        codes[mask] = index
        marked |= mask
    if not marked.any():
        raise ValueError("Mark at least one area before suggesting classifications in another run")
    source = _matrix(source_native_to_painting_h, "Source native-to-painting placement")
    target = _matrix(target_native_to_painting_h, "Target native-to-painting placement")
    carried = carry_label_reference(codes,
        reference_matrix=np.linalg.inv(source).tolist(),
        target_matrix=np.linalg.inv(target).tolist(),
        target_shape_hw=list(target_component.shape),
        reference_filled=np.zeros(source_component.shape, bool),
        source_fill_status="KNOWN", reference_support=marked)
    matched = target_component & carried["support"]
    masks = {name: _readonly(matched & (carried["labels"] == index))
             for index, name in enumerate(classes, 1)}
    unmarked = target_component & ~matched
    return dict(class_masks=MappingProxyType(masks), unmarked_mask=_readonly(unmarked),
        matched_pixels=int(np.count_nonzero(matched)),
        remaining_pixels=int(np.count_nonzero(unmarked)),
        source_to_target_h=carried["metrics"]["composition"])


def transfer_region_draft(review, verified_packet, source_snapshot):
    """Seed one genuine destination draft; its own human must still confirm it."""
    from .painted_gap_regions import region_draft_factory, validate_gap_region_snapshot

    source_snapshot = validate_gap_region_snapshot(source_snapshot, review=review)
    draft = region_draft_factory(review, verified_packet)
    sources = [row for row in review["runs"] if row["run_id"] == source_snapshot.run_id]
    targets = [row for row in review["runs"] if row["run_id"] == draft.run_id]
    if len(sources) != 1 or len(targets) != 1:
        raise ValueError("Both suggestions must belong to this exact reviewed cohort")
    target = targets[0]
    if verified_packet["native_to_painting_h"] != target["native_to_painting_h"]:
        raise ValueError("The destination preview has another native map placement")
    if (draft.run_id, draft.component_id) == (source_snapshot.run_id, source_snapshot.component_id):
        raise ValueError("Keep the original local draft when returning to its own question")
    result = transfer_region_masks(source_snapshot.class_masks, source_snapshot.editable_mask,
        sources[0]["native_to_painting_h"], draft.editable_mask, target["native_to_painting_h"])
    draft.set_preview_class_masks(dict(result["class_masks"]), source_context=dict(
        review=review["review_binding"], run_id=source_snapshot.run_id,
        component_id=source_snapshot.component_id, review_row_sha256=source_snapshot.review_row_sha256))
    return draft, result
