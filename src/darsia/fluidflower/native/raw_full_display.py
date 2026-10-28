"""Complete RAW field of view on the existing calibrated camera-U plane.

Only display RGB is rectified. Sensor, U and saved-native coordinates remain
separate; no scientific image, frame, registration or map is changed.
"""
from dataclasses import dataclass, replace
from copy import deepcopy
from pathlib import Path
from threading import RLock

import numpy as np

from . import registration
from .io import sha256_file


# Shared by Layers and Image preparation. One full photo is enough to avoid
# decoding it again on navigation; this cache never contains scientific DN.
_DISPLAY_CACHE = {}
_DISPLAY_CACHE_LOCK = RLock()
_DISPLAY_CACHE_BYTES = 192 * 1024**2


def _display_copy(packet, *, reused):
    result = {key: value if isinstance(value, np.ndarray) else deepcopy(value)
              for key, value in packet.items()}
    result['display_cache_reused'] = reused
    return result


def _points(value):
    value = np.asarray(value, np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError('Display coordinates must be finite xy points')
    return value


@dataclass(frozen=True)
class RawDisplayMapping:
    sensor_wh: tuple
    camera_u_wh: tuple
    profile_id: str
    coefficients: dict
    distortion: tuple
    lens: dict
    origin_u_xy: tuple = (0., 0.)

    @property
    def camera_u_to_photo_h(self):
        x, y = self.origin_u_xy
        return np.array([[1., 0., -x], [0., 1., -y], [0., 0., 1.]])

    def camera_u_to_display_xy(self, points):
        return _points(points) - np.asarray(self.origin_u_xy)

    def display_to_camera_u_xy(self, points):
        return _points(points) + np.asarray(self.origin_u_xy)

    def camera_u_to_sensor_xy(self, points):
        points = _points(points)
        x, y = registration.raw_coordinates(points[:, 0], points[:, 1], np.eye(3),
            self.coefficients, self.distortion, parameters=self.lens)
        return np.column_stack((x, y))

    def sensor_to_camera_u_xy(self, points):
        """Invert the original monotone piecewise-linear radial scale exactly."""
        points = _points(points)
        p = self.coefficients
        centre = np.array([p['cu'], p['cv']])
        delta = points - centre - np.asarray(self.lens['raw_offset_xy']) - [p['tx'], p['ty']]
        sensor_radius = np.linalg.norm(delta, axis=1)
        radius = np.asarray(self.lens['knots'], np.float64) * p['R']
        scale = 1. + p['amp'] * np.asarray(self.distortion, np.float64) / self.lens['distortion_denominator']
        slope = np.diff(scale) / np.diff(radius)
        intercept = scale[:-1] - slope * radius[:-1]
        if (not np.all(scale > 0) or not np.all(np.diff(radius) > 0)
                or np.any(intercept + 2 * slope * radius[:-1] <= 0)
                or np.any(intercept + 2 * slope * radius[1:] <= 0)):
            raise ValueError('The bound G lens map is not invertible on the full sensor')
        mapped = radius * scale
        interval = np.clip(np.searchsorted(mapped, sensor_radius, side='right') - 1, 0, len(slope)-1)
        a, b = intercept[interval], slope[interval]
        outside = sensor_radius >= mapped[-1]
        a[outside], b[outside] = scale[-1], 0.
        discriminant = a*a + 4*b*sensor_radius
        if np.any(discriminant < 0):
            raise ValueError('The bound G lens inverse is undefined')
        inverse_radius = 2*sensor_radius / (a + np.sqrt(discriminant))
        ratio = np.divide(inverse_radius, sensor_radius, out=np.ones_like(inverse_radius), where=sensor_radius != 0)
        return centre + delta * ratio[:, None]

    def sensor_boundary_xy(self):
        width, height = self.sensor_wh
        # Include the full pixel footprint, not only the centres of four corners.
        x = np.arange(width + 1, dtype=np.float64) - .5
        y = np.arange(height + 1, dtype=np.float64) - .5
        return np.concatenate((np.column_stack((x, np.full_like(x, -.5))),
            np.column_stack((x, np.full_like(x, height-.5))),
            np.column_stack((np.full_like(y, -.5), y)),
            np.column_stack((np.full_like(y, width-.5), y))))


def mapping_from_parameters(parameters, profile_id, *, origin_u_xy=(0., 0.)):
    common = parameters['common']
    profile = common['profiles'][profile_id]
    mapping = RawDisplayMapping(tuple(common['sensor']['raw_wh']), tuple(common['sensor']['camera_wh']),
        profile_id, deepcopy(profile['coefficients']), tuple(profile['distortion']),
        deepcopy(common['lens']), tuple(origin_u_xy))
    if (len(mapping.distortion) != len(mapping.lens['knots']) or len(mapping.distortion) < 2
            or mapping.lens['knots'][0] != 0
            or not np.isfinite([*mapping.coefficients.values(), *mapping.distortion,
                               *mapping.lens['knots'], *origin_u_xy]).all()
            or any(type(value) is not int or value <= 0 for value in (*mapping.sensor_wh, *mapping.camera_u_wh))):
        raise ValueError('The full-sensor display needs the exact declared calibrated grids')
    return mapping


def full_raw_display_extent(mappings):
    """One U extent enclosing every actual RAW boundary, including curved edges."""
    points = [value.sensor_to_camera_u_xy(value.sensor_boundary_xy()) for value in mappings]
    if not points:
        raise ValueError('The complete RAW display needs its calibrated profile')
    points = np.concatenate(points)
    low = np.floor(points.min(axis=0)).astype(np.int64) - 2
    high = np.ceil(points.max(axis=0)).astype(np.int64) + 2
    size = high-low+1
    if np.any(size <= 0) or int(size[0])*int(size[1]) > 4*max(np.prod(value.sensor_wh) for value in mappings):
        raise ValueError('The calibrated full-sensor field exceeds its display bound')
    return int(low[0]), int(low[1]), int(size[0]), int(size[1])


def read_full_raw_display(path, *, profile_path, profile_sha256, profile_id,
                          expected_sha256=None, expected_orientation=None, extent=None):
    """Reuse one exactly bound display photograph; never cache scientific data."""
    from .raw_positioning import _version
    path, profile_path = Path(path), Path(profile_path)
    key = (str(path.resolve()), _version(path), str(profile_path.resolve()), _version(profile_path),
           profile_sha256, profile_id, None if extent is None else tuple(extent))
    with _DISPLAY_CACHE_LOCK:
        cached = _DISPLAY_CACHE.get('packet') if _DISPLAY_CACHE.get('key') == key else None
        if cached is not None:
            if expected_sha256 is not None and cached['source_raw']['sha256'] != expected_sha256:
                raise ValueError('The original RAW differs from its saved display binding')
            if expected_orientation is not None and cached['raw_orientation'] != expected_orientation:
                raise ValueError('The original RAW orientation differs from its saved coordinate convention')
            return _display_copy(cached, reused=True)
        packet = _read_full_raw_display(path, profile_path=profile_path,
            profile_sha256=profile_sha256, profile_id=profile_id, expected_sha256=expected_sha256,
            expected_orientation=expected_orientation, extent=extent)
        arrays = [value for value in packet.values() if isinstance(value, np.ndarray)]
        _DISPLAY_CACHE.clear()
        if sum(value.nbytes for value in arrays) <= _DISPLAY_CACHE_BYTES:
            for value in arrays:
                value.setflags(write=False)
            _DISPLAY_CACHE.update(key=key, packet=packet)
        return _display_copy(packet, reused=False)


def _read_full_raw_display(path, *, profile_path, profile_sha256, profile_id,
                           expected_sha256=None, expected_orientation=None, extent=None):
    """Read only one original RAW and rectify its complete field onto expanded U."""
    import cv2
    import rawpy
    from .raw_camera import read_camera_identity
    from .raw_photograph import _parameters
    from .raw_positioning import _version

    path = Path(path)
    parameters = _parameters(profile_path, profile_sha256, profile_id)
    mapping = mapping_from_parameters(parameters, profile_id)
    policy = parameters['common']['sensor']
    before = _version(path)
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('The original RAW differs from its saved display binding')
    identity = read_camera_identity(path)
    if expected_orientation is not None and identity.orientation != expected_orientation:
        raise ValueError('The original RAW orientation differs from its saved coordinate convention')
    distortion, _red, _blue = registration.read_lens_tables(path, parameters=parameters['common']['lens'])
    if tuple(distortion) != mapping.distortion:
        raise ValueError('The original RAW lens table differs from its bound display profile')
    with rawpy.imread(str(path)) as source:
        if (source.raw_image_visible.shape != mapping.sensor_wh[::-1]
                or not np.array_equal(source.raw_pattern, policy['cfa_pattern'])
                or source.color_desc != policy['color_description'].encode('ascii')
                or list(source.black_level_per_channel) != policy['black_levels']):
            raise ValueError('The original RAW sensor differs from its saved calibration')
        # LibRaw's full visible sensor; no camera JPEG crop or EXIF transpose.
        sensor_rgb = source.postprocess(use_camera_wb=True, output_bps=8, half_size=False, user_flip=0)
    if sensor_rgb.dtype != np.uint8 or sensor_rgb.shape != (*mapping.sensor_wh[::-1], 3):
        raise ValueError('The RAW display decoder returned a different full-sensor grid')
    extent = full_raw_display_extent([mapping]) if extent is None else tuple(extent)
    left, top, width, height = extent
    mapping = replace(mapping, origin_u_xy=(left, top))
    boundary = mapping.camera_u_to_display_xy(mapping.sensor_to_camera_u_xy(mapping.sensor_boundary_xy()))
    if (np.any(boundary < 0) or np.any(boundary[:, 0] >= width) or np.any(boundary[:, 1] >= height)):
        raise ValueError('The supplied display extent would crop the original RAW field')
    image = np.empty((height, width, 3), np.uint8)
    support = np.empty((height, width), bool)
    for y0 in range(0, height, int(policy['tile_rows'])):
        y1 = min(height, y0 + int(policy['tile_rows']))
        yy, xx = np.mgrid[y0:y1, 0:width].astype(np.float32)
        rx, ry = registration.raw_coordinates(xx+left, yy+top, np.eye(3), mapping.coefficients,
            mapping.distortion, parameters=mapping.lens)
        image[y0:y1] = cv2.remap(sensor_rgb, rx.astype(np.float32), ry.astype(np.float32),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        support[y0:y1] = (rx >= 0) & (ry >= 0) & (rx <= mapping.sensor_wh[0]-1) & (ry <= mapping.sensor_wh[1]-1)
    if _version(path) != before or sha256_file(path) != digest:
        raise ValueError('The original RAW changed during its display read')
    return dict(image_rgb=image, image_encoding='rgb_uint8', valid_support=support,
        mapping=mapping, sensor_wh=list(mapping.sensor_wh), camera_u_wh=list(mapping.camera_u_wh),
        camera_u_to_photo_h=mapping.camera_u_to_photo_h.tolist(), extent_u_xywh=list(extent),
        source_raw=dict(path=str(path), sha256=digest, size_bytes=path.stat().st_size),
        profile=dict(path=str(profile_path), sha256=profile_sha256, profile_id=profile_id),
        raw_orientation=identity.orientation, full_raw_sensor_field=True, display_only=True,
        display_conversion=dict(producer='rawpy.RawPy.postprocess', use_camera_wb=True,
            output_bps=8, half_size=False, user_flip=0))
