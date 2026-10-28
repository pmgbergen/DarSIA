"""Original numeric transport of explicit Build fields and RAW references.

The caller owns accepted source domains and all source/phase provenance joins.
These array adapters do not infer roles, fill geometry or certify eligibility.
"""
from __future__ import annotations

from copy import deepcopy

import cv2
import numpy as np

from ..geometry.context import homography
from .n1_geometry import transport_physical_reference


def _shape(value):
    shape = tuple(value)
    if (len(shape) != 2 or any(isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, np.integer)) or item < 1 for item in shape)):
        raise ValueError("A positive integer height/width grid is required")
    return tuple(map(int, shape))


def _mask(value, shape):
    mask = np.asarray(value)
    if mask.dtype != bool or mask.shape != shape:
        raise ValueError("An exact boolean domain on the declared grid is required")
    return mask


def _canvas(matrix, origin, parameters):
    origin = np.asarray(origin, np.float64)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("Two finite native origin coordinates are required")
    measured = homography(matrix, parameters=parameters)
    translation = np.array([[1., 0., -origin[0]], [0., 1., -origin[1]], [0., 0., 1.]])
    return homography(translation @ measured, parameters=parameters)


def registered_build_fields(context, build_to_frame_native_h, native_origin_xy,
                            shape_hw, domain, math, geometry_parameters, *, source_domain):
    """Transport original saved Build fields without extending physical support.

    ``source_domain`` comes from the caller's accepted Build-label/role join.
    Returned ``domain`` is its original nearest-neighbour transport; known and
    unknown physical masks are restricted to the separately supplied target
    domain. No facies or continuous values are filled or imputed.
    """
    source_shape, shape = _shape(context.grid["num_voxels"]), _shape(shape_hw)
    source_domain, target_domain = _mask(source_domain, source_shape), _mask(domain, shape)
    reference = dict(facies=np.asarray(context.facies), depth_m=np.asarray(context.depth_m),
                     porosity=np.asarray(context.porosity), domain=source_domain)
    if any(value.shape != source_shape for value in reference.values()):
        raise ValueError("Saved Build fields differ from their original grid")
    facies = reference["facies"]
    if (not np.issubdtype(facies.dtype, np.integer)
            or np.any(facies < 0) or np.any(facies > 255)):
        raise ValueError("Build facies must retain integer categories within uint8 range")
    matrix = _canvas(build_to_frame_native_h, native_origin_xy, geometry_parameters)
    result = transport_physical_reference(reference, matrix, shape, math, cv2)
    known = target_domain & result["physical_supported_mask"]
    result.update(target_domain=target_domain.copy(), physical_supported_mask=known,
                  unknown_physical_mask=target_domain & ~known,
                  source_to_frame_h=matrix.copy(), provenance=dict(
                      build_context=deepcopy(context.provenance), source_grid=deepcopy(context.grid),
                      build_to_frame_native_h=np.asarray(build_to_frame_native_h, np.float64).tolist(),
                      native_origin_xy=list(native_origin_xy), shape_hw=list(shape),
                      transport="darsia.fluidflower.mass.n1_geometry.transport_physical_reference"))
    return result


def registered_scientific_reference(rgb, valid, source_build_to_native_h, source_origin,
                                    current_build_to_native_h, current_origin, shape, math, *,
                                    geometry_parameters, require_positive=False):
    """Apply the original continuous reference warp and full-coverage support.

    Baseline RGB may contain exact zero. A clear-water caller explicitly sets
    ``require_positive=True``. Unsupported source pixels are zeroed before the
    original warp; partially covered output values remain, with valid=False.
    """
    rgb = np.asarray(rgb)
    if (rgb.dtype != np.float32 or rgb.ndim != 3 or rgb.shape[-1] != 3
            or np.dtype(math.dtype) != np.dtype(np.float32)):
        raise TypeError("Scientific reference transport requires float32 RGB and pixel math")
    if type(require_positive) is not bool:
        raise TypeError("Positive reference support must be an explicit boolean")
    source_shape, shape = _shape(rgb.shape[:2]), _shape(shape)
    valid = _mask(valid, source_shape)
    source = _canvas(source_build_to_native_h, source_origin, geometry_parameters)
    current = _canvas(current_build_to_native_h, current_origin, geometry_parameters)
    matrix = homography(current @ np.linalg.inv(source), parameters=geometry_parameters)
    supported = valid & np.isfinite(rgb).all(-1)
    supported &= (rgb > 0).all(-1) if require_positive else (rgb >= 0).all(-1)
    registered = math.warp_continuous(np.where(supported[..., None], rgb, np.float32(0)),
                                      matrix, shape, cv2_module=cv2)
    coverage = math.warp_continuous(supported, matrix, shape, cv2_module=cv2)
    target_valid = (coverage >= 1 - 1e-6) & np.isfinite(registered).all(-1)
    target_valid &= ((registered > 0).all(-1) if require_positive
                     else (registered >= 0).all(-1))
    return dict(rgb=registered, valid=target_valid, coverage=coverage,
                source_to_current_h=matrix.copy(), provenance=dict(
                    source_build_to_native_h=np.asarray(source_build_to_native_h, np.float64).tolist(),
                    source_origin_xy=list(source_origin),
                    current_build_to_native_h=np.asarray(current_build_to_native_h, np.float64).tolist(),
                    current_origin_xy=list(current_origin), shape_hw=list(shape),
                    require_positive=require_positive,
                    transport="darsia.fluidflower.mass.production_precision.PixelMath.warp_continuous"))
