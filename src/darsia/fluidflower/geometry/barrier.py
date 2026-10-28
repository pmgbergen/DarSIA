"""Geometry-only admission and transport of already frozen static barriers."""

from __future__ import annotations

import hashlib
from typing import Mapping

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from ..phase.domain import canvas_homography, embed_native_canvas


def array_bytes_sha256(array):
    """Hash contiguous array bytes using the original frozen-component identity.

    This byte-only identity intentionally excludes dtype/shape prefixes; callers
    admit those separately through the component-grid checks."""
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()


def validate_component_arrays(values, shape):
    """Admit static integer component IDs and boolean barrier/domain arrays.

    All three arrays must occupy the exact native shape, with nonnegative IDs
    and no open component inside a barrier or outside the frozen reference domain."""
    components, barrier, domain = (
        values[key] for key in ("open_components", "barrier_mask", "reference_domain")
    )
    if (
        components.shape != tuple(shape)
        or not np.issubdtype(components.dtype, np.integer)
        or np.any(components < 0)
        or (barrier.dtype != bool)
        or (domain.dtype != bool)
        or (barrier.shape != components.shape)
        or (domain.shape != components.shape)
        or np.any((components > 0) & (barrier | ~domain))
    ):
        raise ValueError(
            "Barrier Rule B component bundle differs from its native domain"
        )
    return values


def extend_static_components(
    open_components, barrier_mask, rig_domain, *, parameters: Mapping
):
    """Extend known static barrier/open-component classes into accepted rig pixels.

    Euclidean nearest-index transport retains the original SciPy tie convention
    and every inherited class. No phase categories, relabelling, or boundary
    fitting are used. The descriptive audit contract is explicit parameter DATA."""
    components = np.asarray(open_components)
    barrier = np.asarray(barrier_mask)
    domain = np.asarray(rig_domain)
    if (
        components.ndim != 2
        or components.dtype.kind not in "iu"
        or np.any(components < 0)
    ):
        raise TypeError("Frozen open components require a nonnegative integer grid")
    if (
        barrier.dtype != bool
        or domain.dtype != bool
        or barrier.shape != components.shape
        or (domain.shape != components.shape)
    ):
        raise TypeError(
            "Frozen barrier and accepted rig domain require the exact same boolean phase grid"
        )
    if np.any(barrier & (components > 0)):
        raise ValueError("Frozen pixels cannot be both a barrier and an open component")
    known = barrier | (components > 0)
    extension = domain & ~known
    if extension.any() and (not known.any()):
        raise ValueError(
            "An accepted rig cannot inherit classes from an empty frozen reference"
        )
    extended_components = components.copy()
    extended_barrier = barrier.copy()
    if extension.any():
        nearest = distance_transform_edt(
            ~known, return_distances=False, return_indices=True
        )
        rows, cols = nearest[:, extension]
        extended_components[extension] = components[rows, cols]
        extended_barrier[extension] = barrier[rows, cols]
    if not np.array_equal(
        extended_components[known], components[known]
    ) or not np.array_equal(extended_barrier[known], barrier[known]):
        raise ValueError("Geometric extension changed an inherited frozen class")
    if np.any(domain & ~(extended_barrier | (extended_components > 0))):
        raise ValueError("Accepted rig contains unclassified Rule B geometry")
    audit = dict(
        contract=parameters["contract"],
        assumption="nearest frozen static barrier or same open-component ID",
        method="scipy Euclidean nearest index; deterministic library tie convention",
        source_component_ids_unchanged=True,
        reference_classes_preserved=True,
        phase_shape_hw=list(components.shape),
        known_pixels=int(known.sum()),
        extended_pixels=int(extension.sum()),
        extended_barrier_pixels=int((extension & extended_barrier).sum()),
        extended_open_pixels=int((extension & (extended_components > 0)).sum()),
        inherited_open_ids=[int(value) for value in np.unique(components) if value > 0],
        frozen_open_components_sha256=array_bytes_sha256(components),
        frozen_barrier_sha256=array_bytes_sha256(barrier),
        known_mask_sha256=array_bytes_sha256(known),
        extension_mask_sha256=array_bytes_sha256(extension),
        phase_categories_used=False,
        components_relabelled=False,
    )
    return (
        dict(
            open_components=extended_components,
            barrier_mask=extended_barrier,
            known=known,
            geometric_extension=extension,
            reference_domain=domain.copy(),
        ),
        audit,
    )


def prepare_static_components(
    values,
    rig_domain,
    *,
    native_shape_hw,
    source_box_xywh,
    source_to_frame_h,
    source_ids,
    unchanged_source,
    source_binding,
    parameters: Mapping,
):
    """Place admitted static components on a frame's complete physical canvas.

    An unchanged acquisition embeds its own original component bundle without
    warping. A fresh acquisition transports its separately admitted frozen
    reference by the recorded native homography using nearest-neighbour IDs
    and boolean barriers, then extends only accepted previously unknown rig
    pixels. The source binding and IDs remain explicit; no phase output is
    used to prepare this geometry.
    """
    arrays = validate_component_arrays(values, native_shape_hw)
    if unchanged_source:
        ids = embed_native_canvas(
            arrays["open_components"],
            source_box_xywh=source_box_xywh,
            phase_shape_hw=rig_domain.shape,
            fill=0,
        )
        barrier = embed_native_canvas(
            arrays["barrier_mask"],
            source_box_xywh=source_box_xywh,
            phase_shape_hw=rig_domain.shape,
            fill=False,
        )
        proof = dict(exact_same_acquisition=True, source_components=source_binding)
    else:
        transform = canvas_homography(
            source_to_frame_h, source_box_xywh=source_box_xywh
        )
        ids = cv2.warpPerspective(
            embed_native_canvas(
                arrays["open_components"],
                source_box_xywh=source_box_xywh,
                phase_shape_hw=rig_domain.shape,
                fill=0,
            ),
            transform,
            rig_domain.shape[::-1],
            flags=cv2.INTER_NEAREST,
        )
        barrier = cv2.warpPerspective(
            embed_native_canvas(
                arrays["barrier_mask"],
                source_box_xywh=source_box_xywh,
                phase_shape_hw=rig_domain.shape,
                fill=False,
            ).astype(np.uint8),
            transform,
            rig_domain.shape[::-1],
            flags=cv2.INTER_NEAREST,
        ).astype(bool)
        proof = dict(
            exact_same_acquisition=False,
            frozen_reference=source_binding,
            own_source_H=source_to_frame_h,
        )
    arrays, extension = extend_static_components(
        ids, barrier, rig_domain, parameters=parameters
    )
    return arrays, tuple(source_ids), dict(proof, extension=extension)
