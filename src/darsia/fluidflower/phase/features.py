"""Optical evidence on unchanged registered grids, with explicit frozen data."""
from __future__ import annotations
from collections.abc import Mapping, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any
import cv2
import numpy as np

@dataclass(frozen=True)
class OpticalFeatureContract:
    """Ordered optical evidence columns selected by a source-bound model."""
    chemistry_adapter: str
    reference_adapter: str
    include_dynamic: bool
    generic_version: str
    input_color_space: str
    change_detector: str
    schema: int
    parameters: Mapping = field(compare=False, repr=False)

    def __post_init__(self):
        p = self.parameters['contract']
        if self.schema != p['schema'] or self.generic_version not in p['versions']:
            raise ValueError('Unsupported optical feature schema')
        if self.input_color_space != p['input_color_space']:
            raise ValueError('Unsupported optical colour space')
        if self.change_detector not in p['change_detectors'].values():
            raise ValueError('Unsupported change detector')
        if self.chemistry_adapter not in p['chemistry_adapters'] or self.reference_adapter not in p['reference_adapters']:
            raise ValueError('Unsupported optical adapter')

    @property
    def full_feature_names(self):
        """Columns before the declared change-evidence ablation."""
        p = self.parameters['contract']
        static, dynamic_names = p['versions'][self.generic_version]
        dynamic = dynamic_names if self.include_dynamic else ()
        return (*static, *dynamic, *p['chemistry_adapters'][self.chemistry_adapter], *p['reference_adapters'][self.reference_adapter])

    @property
    def feature_names(self):
        """Exact model column order after its declared ablation."""
        p = self.parameters['contract']
        names = self.full_feature_names
        if self.change_detector == p['change_detectors']['none']:
            return tuple((name for name in names if name not in p['change_feature_names']))
        return names

    def to_dict(self):
        """Serialize the original feature-contract fields without fitting."""
        return dict(schema=int(self.schema), generic_version=self.generic_version, input_color_space=self.input_color_space, change_detector=self.change_detector, chemistry_adapter=self.chemistry_adapter, reference_adapter=self.reference_adapter, include_dynamic=bool(self.include_dynamic), feature_names=list(self.feature_names))

    @classmethod
    def from_dict(cls, payload, *, parameters):
        """Parse a model contract against explicit schema and default data."""
        d = parameters['contract']['defaults']
        c = cls(schema=int(payload.get('schema', -1)), generic_version=str(payload.get('generic_version') or ''), input_color_space=str(payload.get('input_color_space') or ''), change_detector=str(payload.get('change_detector') or d['change_detector']), chemistry_adapter=str(payload.get('chemistry_adapter') or d['chemistry_adapter']), reference_adapter=str(payload.get('reference_adapter') or d['reference_adapter']), include_dynamic=bool(payload.get('include_dynamic', d['include_dynamic'])), parameters=parameters)
        declared = tuple(payload.get('feature_names') or ())
        if declared and declared != c.feature_names:
            raise ValueError('Optical feature-name contract differs')
        return c

def relative_bright_grain_columns(baseline_gray, baseline_smooth, current_gray, current_smooth, *, xp=np, parameters: Mapping):
    """Learnable registered grain evidence, shared by CPU and CUDA features.

    Local contrast divided by local brightness is invariant to an unclipped
    multiplicative intensity change above the exposure floor. This keeps a
    persistent shiny grain recognizable when dye attenuation changes. The
    bounded floor and range limit dark-pixel noise. New contrast remains a
    separate feature even at an originally bright grain; no gas veto, temporal
    label, or wash-image assumption is imposed.
    """

    def relative(gray, smooth):
        'Validate and apply explicit frozen optical data without fitting or changing pixels.'
        return xp.clip((gray - smooth) / xp.maximum(smooth, xp.float32(parameters['optical']['relative_grain_floor'])), xp.float32(parameters['optical']['relative_grain_min']), xp.float32(parameters['optical']['relative_grain_max']))
    baseline = relative(baseline_gray, baseline_smooth)
    current = relative(current_gray, current_smooth)
    baseline_bright = xp.maximum(baseline, xp.float32(0))
    current_bright = xp.maximum(current, xp.float32(0))
    persistent = xp.minimum(baseline_bright, current_bright)
    new = xp.maximum(current_bright - baseline_bright, xp.float32(0))
    agreement = persistent * persistent / (xp.maximum(baseline_bright, current_bright) + xp.float32(parameters['optical']['relative_grain_floor']))
    return (baseline, current, current - baseline, persistent, new, agreement)

def decode_native_log(log, valid, *, parameters: Mapping):
    """Match RunopticalLinearInput's channel-wise float32 decoder exactly."""
    source, mask = (np.asarray(log), np.asarray(valid))
    if source.dtype != np.float32 or source.ndim != 3 or source.shape[0] != 3 or (mask.dtype != bool) or (mask.shape != source.shape[1:]) or (not np.array_equal(mask, np.isfinite(source).all(axis=0))):
        raise ValueError('Native-centre RAW requires CHW float32 and exact finite HW support')
    result = np.empty((*mask.shape, 3), np.float32)
    with np.errstate(over='ignore', invalid='ignore'):
        for channel in range(3):
            np.exp(source[channel], out=result[..., channel])
            result[..., channel] /= np.float32(parameters['native_decoder']['denominator'])
    if not np.isfinite(result[mask]).all() or np.any(result[mask] < 0) or np.any(result[mask] > 1):
        raise ValueError('Native-centre RAW exceeds the frozen unit representation')
    result[~mask] = 0
    return result

def log_to_unit_rgb(log_rgb, valid, *, parameters: Mapping):
    'Decode supported rig colour with its original float64 exponential before float32 assignment.'
    values, mask = (np.asarray(log_rgb), np.asarray(valid))
    if values.dtype != np.float32 or values.ndim != 3 or values.shape[0] != 3:
        raise TypeError('Rig model input requires CHW float32 scientific log RGB')
    if mask.dtype != bool or mask.shape != values.shape[1:]:
        raise ValueError('Rig validity must match the native scientific grid')
    usable = mask & np.isfinite(values).all(axis=0)
    rgb = np.zeros((*usable.shape, 3), np.float32)
    rgb[usable] = np.exp(values[:, usable].T.astype(np.float64)) / parameters['rig_decoder']['denominator']
    if not np.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 1).any():
        raise ValueError('Rig float input exceeds its fixed RAW denominator; no clipping allowed')
    return (rgb, usable)

def _h(value: Any, *, parameters: Mapping) -> np.ndarray:
    matrix = np.asarray(value, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < parameters['alignment']['homography_floor'] or (abs(float(matrix[2, 2])) < parameters['alignment']['homography_floor']):
        raise ValueError('optical temporal alignment requires an invertible 3x3 H')
    return matrix / matrix[2, 2]

def _rgb(value: Any, name: str, *, parameters: Mapping) -> np.ndarray:
    rgb = np.asarray(value)
    if rgb.dtype != np.float32 or rgb.ndim != 3 or rgb.shape[2] != 3 or (not np.isfinite(rgb).all()) or np.any(rgb < 0) or np.any(rgb > 1):
        raise ValueError(f'{name} must be finite HWC float32 unit RGB')
    return rgb

def warp_reference_to_current(reference: np.ndarray, h_reference_to_current: Any, *, required_valid_mask: np.ndarray | None=None, target_shape_hw: tuple[int, int] | None=None, source_origin_xy: tuple[int, int], source_valid_mask: np.ndarray | None=None, parameters: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Return a reference on the current native grid and its exact support.

    ``source_origin_xy`` locates the reference array's (0, 0) in native review
    coordinates.  An extended source-backed RAW reference can therefore carry
    real sensor pixels beyond the old camera-JPEG crop.  It fails closed if
    any required pixel would still need extrapolation.
    """
    reference = _rgb(reference, 'Reference', parameters=parameters)
    if source_valid_mask is None:
        source_valid = np.ones(reference.shape[:2], np.float32)
    else:
        source_valid = np.asarray(source_valid_mask)
        if source_valid.dtype != np.bool_ or source_valid.shape != reference.shape[:2]:
            raise ValueError('optical reference valid mask must be source HW bool')
        source_valid = source_valid.astype(np.float32)
    if target_shape_hw is None:
        target_shape_hw = reference.shape[:2]
    height, width = (int(target_shape_hw[0]), int(target_shape_hw[1]))
    if height <= 0 or width <= 0 or len(source_origin_xy) != 2:
        raise ValueError('optical reference warp needs a valid target shape/origin')
    origin = (int(source_origin_xy[0]), int(source_origin_xy[1]))
    if tuple(source_origin_xy) != origin:
        raise ValueError('optical source origin must be integer native coordinates')
    h = _h(h_reference_to_current, parameters=parameters)
    source_to_native = np.array([[1.0, 0.0, origin[0]], [0.0, 1.0, origin[1]], [0.0, 0.0, 1.0]], np.float64)
    h_source_to_current = _h(h @ source_to_native, parameters=parameters)
    aligned = cv2.warpPerspective(reference, h_source_to_current, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0, 0.0))
    footprint = cv2.warpPerspective(source_valid, h_source_to_current, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    valid = footprint >= np.float32(1 - parameters['alignment']['coverage_floor'])
    if required_valid_mask is not None:
        required = np.asarray(required_valid_mask)
        if required.dtype != np.bool_ or required.shape != (height, width):
            raise ValueError('Required optical support mask must be native HW bool')
        if np.any(required & ~valid):
            raise ValueError('optical reference warp lacks source pixels in required region')
    return (np.asarray(aligned, np.float32), valid)

def align_temporal_inputs(*, baseline: np.ndarray, current: np.ndarray, previous: np.ndarray | None, h_baseline_to_current: Any, h_baseline_to_previous: Any | None=None, required_valid_mask: np.ndarray | None=None, baseline_origin_xy: tuple[int, int], previous_origin_xy: tuple[int, int], baseline_source_valid_mask: np.ndarray | None=None, previous_source_valid_mask: np.ndarray | None=None, return_support_mask: bool=False, parameters: Mapping) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Align baseline and previous into current native coordinates.

    The two transforms must be certified by source-bound per-frame receipts.
    ``previous`` cannot be used with an omitted previous-frame transform.
    """
    baseline = _rgb(baseline, 'Baseline', parameters=parameters)
    current = _rgb(current, 'Current', parameters=parameters)
    if baseline.shape != current.shape and baseline_origin_xy == (0, 0):
        raise ValueError('Extended baseline needs an explicit native source origin')
    h_current = _h(h_baseline_to_current, parameters=parameters)
    aligned_baseline, baseline_valid = warp_reference_to_current(baseline, h_current, required_valid_mask=required_valid_mask, target_shape_hw=current.shape[:2], source_origin_xy=baseline_origin_xy, source_valid_mask=baseline_source_valid_mask, parameters=parameters)
    aligned_previous = None
    previous_valid = None
    if previous is not None:
        previous = _rgb(previous, 'Previous', parameters=parameters)
        if h_baseline_to_previous is None or (previous.shape != current.shape and previous_origin_xy == (0, 0)):
            raise ValueError('Previous input needs its exact same-grid H')
        h_previous_to_current = _h(h_current @ np.linalg.inv(_h(h_baseline_to_previous, parameters=parameters)), parameters=parameters)
        aligned_previous, previous_valid = warp_reference_to_current(previous, h_previous_to_current, required_valid_mask=required_valid_mask, target_shape_hw=current.shape[:2], source_origin_xy=previous_origin_xy, source_valid_mask=previous_source_valid_mask, parameters=parameters)
    report = dict(baseline_valid_fraction=float(np.mean(baseline_valid)), previous_valid_fraction=None if previous_valid is None else float(np.mean(previous_valid)), reference_motion_policy='warp-baseline-and-previous-to-current-native-v1', baseline_source_origin_xy=list(baseline_origin_xy), previous_source_origin_xy=None if previous is None else list(previous_origin_xy), current_pixels_unchanged=True)
    if return_support_mask:
        report['joint_valid_mask'] = baseline_valid if previous_valid is None else baseline_valid & previous_valid
    return (aligned_baseline, aligned_previous, report)

def normalized_gaussian(values, valid, sigma, *, denominator=None, parameters: Mapping):
    'Normalize source-supported optical convolution without introducing hardware pixels.'
    data, mask = (np.asarray(values, np.float32), np.asarray(valid, bool))
    if data.shape[:2] != mask.shape or not np.isfinite(data).all():
        raise ValueError('Masked optical filter needs finite native values and a matching mask')
    weights = mask.astype(np.float32)
    if denominator is None:
        denominator = cv2.GaussianBlur(weights, (0, 0), float(np.float32(sigma)), borderType=cv2.BORDER_CONSTANT)
    elif denominator.dtype != np.float32 or denominator.shape != mask.shape:
        raise ValueError('Cached optical normalization must share the native float32 grid')
    weighted = data * (weights if data.ndim == 2 else weights[..., None])
    numerator = cv2.GaussianBlur(weighted, (0, 0), float(np.float32(sigma)), borderType=cv2.BORDER_CONSTANT)
    denominator = denominator if data.ndim == 2 else denominator[..., None]
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > parameters['alignment']['coverage_floor'])

class FeatureBuilder:
    """Original float32 optical features, with separate masked convolution."""

    def __init__(self, contract, baseline, *, parameters, input_encoding, block_rows, texture_sigma, valid=None, feature_shape_hw=None):
        """Bind corrected-camera baseline features; texture lengths are in feature-grid pixels."""
        self.parameters = parameters
        self.feature_contract = contract
        p = parameters['optical']
        d = parameters['contract']['defaults']
        if input_encoding not in (p['review_encoding'], p['unit_encoding']):
            raise ValueError('Unknown optical input encoding')
        if contract.chemistry_adapter != d['chemistry_adapter'] or contract.reference_adapter != d['reference_adapter']:
            raise ValueError('Packed features require corrected generic RGB')
        self.input_encoding = input_encoding
        self.cp = np
        self.block_rows, self.texture_sigma = (block_rows, texture_sigma)
        self.spatial_sigma = max(texture_sigma, p['spatial_context_fraction'] * min(baseline.shape[:2]))
        self.valid = None if valid is None else np.array(valid, dtype=bool, copy=True)
        self._denominators = {}
        if self.valid is not None:
            self.valid.setflags(write=False)
        self.baseline = self.prepare_rgb(baseline)
        if feature_shape_hw is not None:
            shape = np.asarray(feature_shape_hw)
            if shape.shape != (2,) or np.any(shape <= 0):
                raise ValueError('Optical context needs its original native shape')
            self.spatial_sigma = max(self.texture_sigma, p['spatial_context_fraction'] * float(shape.min()))
        self.previous = None
        self.previous_key = None

    def blur(self, values, sigma):
        """Keep reflected unmasked or source-normalized masked Gaussian order."""
        if self.valid is None:
            return cv2.GaussianBlur(values, (0, 0), sigmaX=float(np.float32(sigma)), sigmaY=float(np.float32(sigma)))
        key = float(np.float32(sigma))
        if key not in self._denominators:
            self._denominators[key] = cv2.GaussianBlur(self.valid.astype(np.float32), (0, 0), key, borderType=cv2.BORDER_CONSTANT)
        return normalized_gaussian(values, self.valid, sigma, denominator=self._denominators[key], parameters=self.parameters)

    def prepare_rgb(self, image):
        """Prepare colour fractions, texture and ratios in original float32 order."""
        cp = self.cp
        source = np.asarray(image)
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError('Optical input must be HWC RGB')
        if self.input_encoding == self.parameters['optical']['review_encoding']:
            if source.dtype != np.uint8:
                raise ValueError('Review optical input requires uint8 corrected pixels')
            rgb = cp.asarray(source, dtype=cp.float32) / cp.float32(self.parameters['optical']['review_denominator'])
        else:
            if source.dtype != np.float32 or not np.isfinite(source).all() or np.any(source < 0) or np.any(source > 1):
                raise ValueError('optical optical input requires finite float32 unit RGB')
            rgb = cp.asarray(source, dtype=cp.float32)
        gray = cp.mean(rgb, axis=2, dtype=cp.float32)
        smooth = self.blur(gray, self.texture_sigma)
        contrast = gray - smooth
        texture = cp.abs(contrast)
        fraction = rgb / (cp.sum(rgb, axis=2, keepdims=True) + cp.float32(self.parameters['optical']['fraction_epsilon']))
        safe = cp.maximum(rgb, cp.float32(self.parameters['optical']['ratio_floor']))
        ratios = cp.stack([cp.log(safe[..., 0] / safe[..., 1]), cp.log(safe[..., 1] / safe[..., 2]), cp.log(safe[..., 0] / safe[..., 2])], axis=-1)
        return {'rgb': rgb, 'gray': gray, 'smooth': smooth, 'texture': texture, 'signed_contrast': contrast, 'fraction': fraction, 'ratios': ratios}

    def set_baseline(self, baseline):
        """Rebind a frame-aligned optical baseline without rebuilding model weights.

            The caller must certify its geometry and pass the corrected baseline
            warped into the current native frame.  Cached previous pixels are
            cleared because their coordinate system is no longer valid.
            """
        if self.input_encoding != self.parameters['optical']['unit_encoding']:
            raise ValueError('Per-frame baseline rebinding is optical-only')
        if tuple(np.asarray(baseline).shape[:2]) != tuple(self.baseline['gray'].shape):
            raise ValueError('Aligned baseline shape changed')
        self.baseline = self.prepare_rgb(baseline)
        self.previous = None
        self.previous_key = None

    def feature_blocks(self, current, previous, *, dynamic_available):
        """Yield the original ordered static and dynamic optical columns."""
        cp, base = (self.cp, self.baseline)
        delta = current['rgb'] - base['rgb']
        dynamic = current['rgb'] - previous['rgb']
        norm = cp.sqrt(cp.sum(delta * delta, axis=2))
        dnorm = cp.sqrt(cp.sum(dynamic * dynamic, axis=2))
        mean = self.blur(norm, self.spatial_sigma)
        std = cp.sqrt(cp.maximum(self.blur(norm * norm, self.spatial_sigma) - mean * mean, cp.float32(0)))
        dmean = self.blur(dnorm, self.spatial_sigma)
        height, width = norm.shape
        for start in range(0, height, self.block_rows):
            stop = min(start + self.block_rows, height)
            sl = slice(start, stop)
            texture = current['texture'][sl] - base['texture'][sl]
            blocks = [delta[sl], (current['gray'][sl] - base['gray'][sl])[..., None], current['fraction'][sl] - base['fraction'][sl], current['ratios'][sl] - base['ratios'][sl], texture[..., None], cp.abs(texture)[..., None], norm[sl, :, None], mean[sl, :, None], std[sl, :, None], (mean[sl] / (std[sl] + cp.float32(self.parameters['optical']['ratio_floor'])))[..., None]]
            if self.feature_contract.generic_version in {self.parameters['contract']['bright_version'], self.parameters['contract']['relative_bright_version']}:
                baseline_contrast = base['signed_contrast'][sl]
                current_contrast = current['signed_contrast'][sl]
                baseline_bright = cp.maximum(baseline_contrast, cp.float32(0))
                current_bright = cp.maximum(current_contrast, cp.float32(0))
                blocks.extend([base['gray'][sl, :, None], baseline_contrast[..., None], current_contrast[..., None], (current_contrast - baseline_contrast)[..., None], cp.minimum(baseline_bright, current_bright)[..., None], cp.maximum(current_bright - baseline_bright, cp.float32(0))[..., None]])
            if self.feature_contract.generic_version == self.parameters['contract']['relative_bright_version']:
                blocks.extend((value[..., None] for value in relative_bright_grain_columns(base['gray'][sl], base['smooth'][sl], current['gray'][sl], current['smooth'][sl], xp=cp, parameters=self.parameters)))
            if self.feature_contract.include_dynamic:
                blocks.append(cp.full((stop - start, width, 1), int(dynamic_available), dtype=cp.float32))
                if dynamic_available:
                    blocks.extend([dynamic[sl], (current['gray'][sl] - previous['gray'][sl])[..., None], (current['texture'][sl] - previous['texture'][sl])[..., None], dnorm[sl, :, None], dmean[sl, :, None]])
                else:
                    blocks.append(cp.zeros((stop - start, width, self.parameters['optical']['dynamic_unavailable_columns']), dtype=cp.float32))
            yield (start, stop, cp.concatenate(blocks, axis=2).reshape(-1, len(self.feature_contract.feature_names)))

@dataclass(frozen=True)
class FlatfieldReference:
    """Unsmoothed calibrated RGB on the pre-injection source grid."""
    linear_rgb: np.ndarray
    valid: np.ndarray

@dataclass(frozen=True)
class RegisteredFlatfieldReference:
    """Baseline linear RGB and luminance warped before box smoothing."""
    rgb_box5: np.ndarray
    luminance_box5: np.ndarray
    valid: np.ndarray
    box_valid: np.ndarray
    alignment: dict

def _require_signal(value: np.ndarray, name: str, *, parameters: Mapping) -> np.ndarray:
    signal = np.asarray(value)
    if signal.dtype != np.float32 or signal.ndim != 3 or signal.shape[0] != 3:
        raise TypeError(f'{name} requires CHW float32 log-camera RGB')
    if min(signal.shape[1:]) < 1:
        raise ValueError(f'{name} requires a nonempty image')
    return signal

def _require_mask(value: np.ndarray | None, shape: tuple[int, int], name: str, *, parameters: Mapping):
    if value is None:
        return None
    mask = np.asarray(value)
    if mask.dtype != np.bool_ or mask.shape != shape:
        raise TypeError(f'{name} must be source HW bool with the exact image shape')
    return mask

def _require_tile_rows(tile_rows: int, *, parameters: Mapping) -> int:
    if not isinstance(tile_rows, int) or isinstance(tile_rows, bool) or tile_rows < 1:
        raise ValueError('tile_rows must be a positive integer')
    return tile_rows

def _linear_tile(signal: np.ndarray, valid_mask: np.ndarray | None, *, parameters: Mapping):
    """Match the display's calibrated float32 linear values without repair."""
    tile = np.moveaxis(signal, 0, -1)
    valid = np.isfinite(tile).all(axis=-1)
    if valid_mask is not None:
        valid &= valid_mask
    with np.errstate(over='ignore', invalid='ignore'):
        linear = np.exp(np.where(valid[..., None], tile, 0).astype(np.float64)) @ np.asarray(parameters['flatfield']['natural_matrix'], np.float64)
        linear = linear.astype(np.float32)
    valid &= np.isfinite(linear).all(axis=-1)
    linear[~valid] = 0
    np.maximum(linear, 0, out=linear)
    return (linear, valid)

def _box_valid(valid: np.ndarray, *, parameters: Mapping) -> np.ndarray:
    return cv2.erode(valid.astype(np.uint8), np.ones((parameters['flatfield']['box_size'], parameters['flatfield']['box_size']), np.uint8), borderType=cv2.BORDER_REFLECT_101).astype(bool)

def prepare_flatfield_reference(reference_log_camera_rgb: np.ndarray, *, source_valid_mask: np.ndarray | None=None, tile_rows: int=None, parameters: Mapping) -> FlatfieldReference:
    """Calibrate the exact run baseline, retaining its scientific support."""
    if tile_rows is None:
        tile_rows = parameters['optical']['block_rows']
    signal = _require_signal(reference_log_camera_rgb, 'Flatfield reference', parameters=parameters)
    tile_rows = _require_tile_rows(tile_rows, parameters=parameters)
    height, width = signal.shape[1:]
    source_valid = _require_mask(source_valid_mask, (height, width), 'Reference validity', parameters=parameters)
    linear = np.empty((height, width, 3), np.float32)
    valid = np.empty((height, width), bool)
    for start in range(0, height, tile_rows):
        stop = min(start + tile_rows, height)
        linear[start:stop], valid[start:stop] = _linear_tile(signal[:, start:stop], None if source_valid is None else source_valid[start:stop], parameters=parameters)
    linear.setflags(write=False)
    valid.setflags(write=False)
    return FlatfieldReference(linear, valid)

def _shape_hw(value: Sequence[int], *, parameters: Mapping) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError('Flatfield target shape must contain height and width')
    result = tuple(map(int, value))
    if tuple(value) != result or min(result) < 1:
        raise ValueError('Flatfield target shape must be positive integers')
    return result

def _origin_xy(value: Sequence[int], *, parameters: Mapping) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError('Flatfield source and target origins must contain x and y')
    result = tuple(map(int, value))
    if tuple(value) != result:
        raise ValueError('Flatfield origins must be integer native coordinates')
    return result

def register_flatfield_reference(prepared: FlatfieldReference, h_baseline_to_current: np.ndarray, *, target_shape_hw: tuple[int, int], source_origin_xy: tuple[int, int], target_origin_xy: tuple[int, int], parameters: Mapping) -> RegisteredFlatfieldReference:
    """Transport calibrated baseline colour with the change view's exact H."""
    if not isinstance(prepared, FlatfieldReference):
        raise TypeError('Expected an unregistered FlatfieldReference')
    source, source_valid = (prepared.linear_rgb, prepared.valid)
    if source.dtype != np.float32 or source.ndim != 3 or source.shape[-1] != 3 or (source_valid.dtype != np.bool_) or (source_valid.shape != source.shape[:2]) or (not np.isfinite(source).all()) or np.any(source < 0):
        raise TypeError('Unregistered flatfield reference is malformed')
    height, width = _shape_hw(target_shape_hw, parameters=parameters)
    origin, target_origin = (_origin_xy(source_origin_xy, parameters=parameters), _origin_xy(target_origin_xy, parameters=parameters))
    transform = _h(h_baseline_to_current, parameters=parameters)
    source_to_native = np.array([[1.0, 0.0, origin[0]], [0.0, 1.0, origin[1]], [0.0, 0.0, 1.0]])
    native_to_target = np.array([[1.0, 0.0, -target_origin[0]], [0.0, 1.0, -target_origin[1]], [0.0, 0.0, 1.0]])
    source_to_current = _h(native_to_target @ transform @ source_to_native, parameters=parameters)
    footprint = cv2.warpPerspective(source_valid.astype(np.float32), source_to_current, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    valid = footprint >= np.float32(1 - parameters['alignment']['coverage_floor'])
    del footprint
    rgb_box5 = np.empty((height, width, 3), np.float32)
    luminance_box5 = None
    for channel in range(4):
        field = source[..., channel] if channel < 3 else source @ np.asarray(parameters['flatfield']['luminance'], np.float32)
        aligned = cv2.warpPerspective(field, source_to_current, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        aligned[~valid] = 0
        smoothed = cv2.blur(aligned, (parameters['flatfield']['box_size'], parameters['flatfield']['box_size']), borderType=cv2.BORDER_REFLECT_101)
        if channel < 3:
            rgb_box5[..., channel] = smoothed
        else:
            luminance_box5 = smoothed
        del field, aligned, smoothed
    box_valid = _box_valid(valid, parameters=parameters)
    for array in (rgb_box5, luminance_box5, valid, box_valid):
        array.setflags(write=False)
    return RegisteredFlatfieldReference(rgb_box5, luminance_box5, valid, box_valid, {'policy': 'warp-calibrated-linear-rgb-and-luminance-then-box5', 'source_origin_xy': list(origin), 'target_origin_xy': list(target_origin), 'target_shape_hw': [height, width], parameters['flatfield']['alignment_h_key']: transform.tolist()})

def _require_registered(reference: RegisteredFlatfieldReference, shape: tuple[int, int], *, parameters: Mapping):
    if not isinstance(reference, RegisteredFlatfieldReference):
        raise TypeError('reference must be a per-frame RegisteredFlatfieldReference')
    if reference.rgb_box5.shape != (*shape, 3) or reference.luminance_box5.shape != shape or reference.valid.shape != shape or (reference.box_valid.shape != shape):
        raise ValueError('Registered pre-injection reference does not match current geometry')
    if reference.rgb_box5.dtype != np.float32 or reference.luminance_box5.dtype != np.float32 or reference.valid.dtype != np.bool_ or (reference.box_valid.dtype != np.bool_):
        raise TypeError('Registered pre-injection reference is malformed')

def _feature_tile(signal, reference, current_valid, start, stop, *, parameters: Mapping):
    halo_start, halo_stop = (max(0, start - parameters['flatfield']['halo_rows']), min(signal.shape[1], stop + parameters['flatfield']['halo_rows']))
    offset, length = (start - halo_start, stop - start)
    linear, valid = _linear_tile(signal[:, halo_start:halo_stop], None if current_valid is None else current_valid[halo_start:halo_stop], parameters=parameters)
    blurred = cv2.blur(linear, (parameters['flatfield']['box_size'], parameters['flatfield']['box_size']), borderType=cv2.BORDER_REFLECT_101)[offset:offset + length]
    usable = _box_valid(valid, parameters=parameters)[offset:offset + length] & reference.box_valid[start:stop]
    numerator_y = blurred @ np.asarray(parameters['flatfield']['luminance'], np.float32)
    usable &= (blurred > 0).all(axis=-1) & (numerator_y > 0)
    usable &= np.isfinite(blurred).all(axis=-1) & np.isfinite(numerator_y)
    features = np.full((*blurred.shape[:2], parameters['flatfield']['feature_count']), np.nan, np.float32)
    for channel in range(parameters['flatfield']['feature_count']):
        numerator = blurred[..., channel] if channel < 3 else numerator_y
        denominator = reference.rgb_box5[start:stop, :, channel] if channel < 3 else reference.luminance_box5[start:stop]
        logged = np.full(numerator.shape, np.nan, np.float32)
        np.log(numerator, out=logged, where=usable)
        logged -= np.log(np.maximum(denominator, np.float32(parameters['flatfield']['denominator_floor'])))
        features[..., channel] = logged
    usable &= np.isfinite(features).all(axis=-1)
    features[~usable] = np.nan
    return (features, usable)

def iter_flatfield_feature_tiles(current_log_camera_rgb: np.ndarray, reference: RegisteredFlatfieldReference, *, current_valid_mask: np.ndarray | None=None, tile_rows: int=None, parameters: Mapping) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield ``(start, stop, Htile W 4 features, Htile W valid)`` on CPU."""
    if tile_rows is None:
        tile_rows = parameters['optical']['block_rows']
    signal = _require_signal(current_log_camera_rgb, 'Flatfield current frame', parameters=parameters)
    tile_rows = _require_tile_rows(tile_rows, parameters=parameters)
    shape = signal.shape[1:]
    current_valid = _require_mask(current_valid_mask, shape, 'Current validity', parameters=parameters)
    _require_registered(reference, shape, parameters=parameters)
    for start in range(0, shape[0], tile_rows):
        stop = min(start + tile_rows, shape[0])
        features, valid = _feature_tile(signal, reference, current_valid, start, stop, parameters=parameters)
        yield (start, stop, features, valid)

def extract_flatfield_features(current_log_camera_rgb: np.ndarray, reference: RegisteredFlatfieldReference, *, current_valid_mask: np.ndarray | None=None, selection: tuple[np.ndarray, np.ndarray] | None=None, tile_rows: int=None, parameters: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Return features and support on the native grid or at selected pixels.

    ``selection=(rows, columns)`` returns N by 4 features and N validity flags
    in request order, preserving duplicates.  Only tiles touched by selected
    rows are evaluated.  Without selection the result is H by W by 4 plus H
    by W validity.  Every unusable row contains NaN in all four features.
    """
    if tile_rows is None:
        tile_rows = parameters['optical']['block_rows']
    signal = _require_signal(current_log_camera_rgb, 'Flatfield current frame', parameters=parameters)
    tile_rows = _require_tile_rows(tile_rows, parameters=parameters)
    height, width = signal.shape[1:]
    current_valid = _require_mask(current_valid_mask, (height, width), 'Current validity', parameters=parameters)
    _require_registered(reference, (height, width), parameters=parameters)
    if selection is None:
        features = np.empty((height, width, 4), np.float32)
        valid = np.empty((height, width), bool)
        for start, stop, tile_features, tile_valid in iter_flatfield_feature_tiles(signal, reference, current_valid_mask=current_valid, tile_rows=tile_rows, parameters=parameters):
            features[start:stop], valid[start:stop] = (tile_features, tile_valid)
        return (features, valid)
    if len(selection) != 2:
        raise ValueError('selection must contain rows and columns')
    rows, columns = (np.asarray(index) for index in selection)
    if rows.ndim != 1 or columns.shape != rows.shape or rows.dtype.kind not in 'iu' or (columns.dtype.kind not in 'iu'):
        raise TypeError('selection must be equally sized one-dimensional integer rows and columns')
    if np.any(rows < 0) or np.any(rows >= height) or np.any(columns < 0) or np.any(columns >= width):
        raise IndexError('Flatfield selection is outside the native image')
    features = np.empty((rows.size, 4), np.float32)
    valid = np.empty(rows.size, bool)
    tile_ids = rows // tile_rows
    for tile_id in np.unique(tile_ids):
        start = int(tile_id) * tile_rows
        stop = min(start + tile_rows, height)
        selected = np.flatnonzero(tile_ids == tile_id)
        tile_features, tile_valid = _feature_tile(signal, reference, current_valid, start, stop, parameters=parameters)
        local_rows, local_columns = (rows[selected] - start, columns[selected])
        features[selected] = tile_features[local_rows, local_columns]
        valid[selected] = tile_valid[local_rows, local_columns]
    return (features, valid)

def masked_flatfield_features(current_log, aligned_baseline_log, valid, *, source_baseline_log=None, h_baseline_to_current=None, source_baseline_valid=None, parameters: Mapping):
    """Frozen G3's four log colour ratios, with a normalized 5Ãƒâ€”5 mean.

    The linear photographic matrix and channel/luminance definitions are
    unchanged. Only missing neighbours are excluded instead of introducing
    frame pixels or an artificial eight-pixel erosion.
    """
    shape = valid.shape
    transformed = []
    denominator = cv2.blur(valid.astype(np.float32), (parameters['flatfield']['box_size'], parameters['flatfield']['box_size']), borderType=cv2.BORDER_CONSTANT)
    for index, signal in enumerate((current_log, aligned_baseline_log)):
        if index == 1 and source_baseline_log is not None:
            prepared = prepare_flatfield_reference(source_baseline_log, source_valid_mask=source_baseline_valid, parameters=parameters)
            if h_baseline_to_current is None:
                raise ValueError('Flat-field baseline needs its exact frame homography')
            h = np.asarray(h_baseline_to_current, np.float64)
            values = cv2.warpPerspective(prepared.linear_rgb, h, shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            luminance = cv2.warpPerspective(prepared.linear_rgb @ np.asarray(parameters['flatfield']['luminance'], np.float32), h, shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            footprint = cv2.warpPerspective(prepared.valid.astype(np.float32), h, shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if np.any(valid & (footprint < np.float32(1 - parameters['alignment']['coverage_floor']))):
                raise ValueError('G3 uses an unsupported registered baseline pixel')
            values[~valid] = 0
            luminance[~valid] = 0
        else:
            if signal.dtype != np.float32 or signal.shape != (3, *shape):
                raise TypeError('G3 requires unchanged native float32 log input')
            values = np.zeros((*shape, 3), np.float32)
            values[valid] = np.exp(signal[:, valid].T.astype(np.float64)) @ np.asarray(parameters['flatfield']['natural_matrix'], np.float64)
            np.maximum(values, 0, out=values)
            luminance = values @ np.asarray(parameters['flatfield']['luminance'], np.float32)
        if values.dtype != np.float32:
            raise TypeError('G3 requires unchanged native float32 log input')
        values = np.concatenate((values, luminance[..., None]), axis=-1)
        numerator = cv2.blur(values, (parameters['flatfield']['box_size'], parameters['flatfield']['box_size']), borderType=cv2.BORDER_CONSTANT)
        transformed.append(np.divide(numerator, denominator[..., None], out=np.zeros_like(numerator), where=denominator[..., None] > parameters['alignment']['coverage_floor']))
    current, baseline = transformed
    c, b = (current, baseline)
    usable = valid & (c > 0).all(-1)
    features = np.full((*shape, 4), np.nan, np.float32)
    features[usable] = np.log(c[usable] / np.maximum(b[usable], np.float32(parameters['flatfield']['denominator_floor'])))
    return (features, usable)
