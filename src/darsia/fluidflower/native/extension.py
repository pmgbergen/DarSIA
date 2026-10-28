"""Frozen sensor correction for disjoint peripheral camera windows.

The registered native centre is reused exactly. Added windows use recorded
light coefficients, without fitting new radiometry or promoting hardware.
"""
from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
import cv2
import numpy as np
from .io import sha256_file
from . import registration, photometry

def frozen_light_from_receipt(info, grid, *, parameters: Mapping):
    """Evaluate the recorded polynomial; do not fit on the added ring."""
    if info.get('mode') == 'exact-source-bound-B' and info.get('zero_field') is True:
        return None
    if info.get('central_frozen_v5_parity') is not True:
        raise ValueError('Ring requires the existing certified frame-light coefficients')
    if tuple(info.get('grid_origin_cell_xy', ())) != tuple(grid.origin_cell_xy):
        raise ValueError('Frame-light receipt belongs to another reference grid')
    degree = info['degree']
    coef = np.asarray(info['coefficients'], np.float64)
    if (
        degree not in parameters['extension']['allowed_light_degrees']
        or coef.shape != ((degree + 1) * (degree + 2) // 2,)
        or not np.isfinite(coef).all()
    ):
        raise ValueError('Invalid frozen frame-light polynomial')
    yy, xx = np.mgrid[:grid.shape_hw[0], :grid.shape_hw[1]]
    return (photometry.light_basis((xx + grid.origin_cell_xy[0] - parameters['extension']['light_cell_center']) / parameters['extension']['light_cell_scale'], (yy + grid.origin_cell_xy[1] - parameters['extension']['light_cell_center']) / parameters['extension']['light_cell_scale'], degree, parameters=parameters['photometry']) @ coef).astype(np.float32)

def peripheral_rectangles(mask, native_hw, origin_xy, *, parameters: Mapping):
    """Disjoint minimal camera-ring windows, excluding the reused native centre."""
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.ndim != 2:
        raise TypeError('Camera-ring eligibility requires a boolean canvas mask')
    height, width = mask.shape
    x0, y0 = (-int(x) for x in origin_xy)
    nh, nw = native_hw
    if x0 < 0 or y0 < 0 or x0 + nw > width or (y0 + nh > height):
        raise ValueError('Native centre lies outside the requested canvas')
    rectangles = []
    for ax, ay, bx, by in ((0, 0, width, y0), (0, y0 + nh, width, height), (0, y0, x0, y0 + nh), (x0 + nw, y0, width, y0 + nh)):
        ys, xs = np.nonzero(mask[ay:by, ax:bx])
        if len(xs):
            rectangles.append((ax + int(xs.min()), ay + int(ys.min()), ax + int(xs.max()) + 1, ay + int(ys.max()) + 1))
    return rectangles

def sample_sensor_window(demosaiced, renderer, rectangle, *, origin_xy, float_planes=None, parameters: Mapping):
    """Same frozen Sony inverse map and bilinear RAW interpolation as the centre."""
    image = np.asarray(demosaiced)
    if (
        image.dtype != np.uint16
        or image.shape != (*tuple(parameters['registration']['common']['sensor']['raw_wh'])[::-1], 3)
    ):
        raise TypeError('Ring must use the full-resolution uint16 RAW demosaic')
    h = np.asarray(renderer['review_to_camera_h'], np.float64)
    if (
        h.shape != (3, 3)
        or not np.isfinite(h).all()
        or abs(np.linalg.det(h)) < parameters['extension']['homography_min_determinant']
    ):
        raise ValueError('Invalid source-certified camera transform')
    profile = renderer['profile']
    distortion = tuple(renderer['sony_distortion_table'])
    if distortion != {key: tuple(value['distortion']) for key, value in parameters['registration']['common']['profiles'].items()}[profile]:
        raise ValueError('Ring differs from its frozen Sony distortion profile')
    a, b, c, d = rectangle
    result = np.empty((3, d - b, c - a), np.float32)
    ca = (renderer['sony_ca_red_table'], None, renderer['sony_ca_blue_table'])
    planes = [image[..., channel].astype(np.float32) for channel in range(3)] if float_planes is None else float_planes
    if len(planes) != 3 or any((plane.dtype != np.float32 or plane.shape != image.shape[:2] for plane in planes)):
        raise TypeError('RAW sampler planes must preserve full-sensor float32 geometry')
    for first in range(b, d, parameters['extension']['tile_rows']):
        last = min(d, first + parameters['extension']['tile_rows'])
        yy, xx = np.mgrid[first:last, a:c].astype(np.float32)
        xx += origin_xy[0]
        yy += origin_xy[1]
        for channel in range(3):
            (rx, ry) = registration.raw_coordinates(
                xx,
                yy,
                h,
                {key: value['coefficients'] for key, value in parameters['registration']['common']['profiles'].items()}[profile],
                distortion,
                ca[channel],
                parameters=parameters['registration']['common']['lens'],
            )
            result[channel, first - b:last - b] = cv2.remap(
                planes[channel],
                rx.astype(np.float32),
                ry.astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=np.nan,
            )
    return result

def extend_prepared_source(native_log, producer_audit, processor, ring_eligible, *, parameters: Mapping):
    """Return full-canvas scientific float data and an explicit extension audit.

    ``ring_eligible`` is the photo-certified rig domain minus actual hardware,
    intersected with the camera footprint. Sensor/photometry-invalid pixels
    remain NaN and are filled categorically by the downstream caller.
    """
    native = np.asarray(native_log)
    eligible = np.asarray(ring_eligible)
    if native.dtype != np.float32 or native.shape != (3, *tuple(parameters['extension']['native_shape_hw'])):
        raise TypeError('Ring centre must be the prepared CHW float32 scientific source')
    if (
        eligible.dtype != bool
        or eligible.ndim != 2
        or eligible.shape[0] < tuple(parameters['extension']['minimum_canvas_hw'])[0]
        or eligible.shape[1] < tuple(parameters['extension']['minimum_canvas_hw'])[1]
    ):
        raise ValueError('Camera-ring domain does not share the phase canvas')
    output = np.full((3, *eligible.shape), np.nan, np.float32)
    ox, oy = (-x for x in tuple(parameters['extension']['native_origin_xy']))
    output[:, oy:oy + tuple(parameters['extension']['native_shape_hw'])[0], ox:ox + tuple(parameters['extension']['native_shape_hw'])[1]] = native
    rectangles = peripheral_rectangles(
        eligible,
        native_hw=tuple(parameters['extension']['native_shape_hw']),
        origin_xy=tuple(parameters['extension']['native_origin_xy']),
        parameters=parameters,
    )
    audit = dict(
        contract=parameters['extension']['contract'],
        cpu_only=True,
        training=False,
        new_photometry_fit=False,
        origin_xy=list(tuple(parameters['extension']['native_origin_xy'])),
        canvas_hw=list(eligible.shape),
        native_centre_bytes_preserved=True,
        requested_ring_pixels=sum((int(eligible[b:d, a:c].sum()) for a, b, c, d in rectangles)),
        model_input_encoding='chw-float32-log-camera-linear-rgb',
        windows=[],
    )
    if not rectangles:
        audit['finite_ring_pixels'] = 0
        return (output, np.isfinite(output).all(0), audit)
    if processor.identity != producer_audit.get('processor'):
        raise ValueError("Peripheral correction differs from the prepared source's frozen calibration")
    raw_source = producer_audit['sources']['camera_arw']
    raw_path = Path(raw_source['path'])
    if sha256_file(raw_path) != raw_source['sha256']:
        raise ValueError('Peripheral RAW source changed after scientific preparation')
    renderer = producer_audit['renderer']
    if renderer['sources']['arw_sha256'] != raw_source['sha256']:
        raise ValueError('Peripheral camera transform belongs to another RAW source')
    p = processor.processor
    if renderer['run'].casefold() != p.run.casefold():
        raise ValueError('Peripheral RAW sensor belongs to another run')
    h = np.asarray(producer_audit[parameters['extension']['reference_to_frame_h_key']], np.float64)
    light = frozen_light_from_receipt(producer_audit['photometry']['per_frame_light'], p.grid, parameters=parameters)
    registration.check_demosaic_order(parameters=parameters['registration'])
    import rawpy
    with rawpy.imread(str(raw_path)) as raw:
        if (
            raw.raw_image_visible.shape != tuple(parameters['registration']['common']['sensor']['raw_wh'])[::-1]
            or tuple(raw.black_level_per_channel) != tuple(parameters['registration']['common']['sensor']['black_levels'])
            or raw.white_level != renderer['white_level']
            or not np.array_equal(raw.raw_pattern, parameters['registration']['common']['sensor']['cfa_pattern'])
            or raw.color_desc != parameters['registration']['common']['sensor']['color_description'].encode('ascii')
        ):
            raise ValueError('Peripheral RAW sensor metadata differs from the frozen source')
        black = np.maximum(raw.raw_image_visible.astype(np.int32) - parameters['registration']['common']['sensor']['black_level'], 0).astype(np.uint16)
    demosaic = cv2.demosaicing(black, getattr(cv2, parameters['registration']['edge_aware']['demosaic_code']))
    planes = [demosaic[..., channel].astype(np.float32) for channel in range(3)]
    for rectangle in rectangles:
        a, b, c, d = rectangle
        linear = sample_sensor_window(
            demosaic,
            renderer,
            rectangle,
            origin_xy=tuple(parameters['extension']['native_origin_xy']),
            float_planes=planes,
            parameters=parameters,
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            log = np.where(linear > parameters['extension']['minimum_sensor_signal'], np.log(linear), np.nan).astype(np.float32)
        corrected = photometry.apply_native(
            log,
            p.fit,
            p.reference,
            p.grid,
            h,
            source_eligible_mask=eligible[b:d, a:c],
            light_field_ref8=light,
            destination_origin_xy=(a + tuple(parameters['extension']['native_origin_xy'])[0], b + tuple(parameters['extension']['native_origin_xy'])[1]),
            parameters=parameters['photometry'],
        )
        if p.old_field is not None:
            corrected = photometry.apply_old_field_native(
                corrected,
                p.old_field,
                p.grid,
                h,
                destination_origin_xy=(a + tuple(parameters['extension']['native_origin_xy'])[0], b + tuple(parameters['extension']['native_origin_xy'])[1]),
                parameters=parameters['photometry'],
            )
        valid = np.isfinite(corrected).all(0)
        corrected[:, ~valid] = np.nan
        output[:, b:d, a:c] = corrected
        audit['windows'].append(dict(canvas_xyxy=list(rectangle), eligible_pixels=int(eligible[b:d, a:c].sum()), finite_scientific_pixels=int(valid.sum())))
    audit['finite_ring_pixels'] = sum((row['finite_scientific_pixels'] for row in audit['windows']))
    audit['raw_source'] = raw_source
    audit['frozen_frame_light'] = producer_audit['photometry']['per_frame_light']
    audit['unsupported_ring_pixels'] = audit['requested_ring_pixels'] - audit['finite_ring_pixels']
    return (output, np.isfinite(output).all(0), audit)
