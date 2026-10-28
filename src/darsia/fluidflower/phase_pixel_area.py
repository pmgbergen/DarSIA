"""Source-bound pixel areas for the FF_AC rig-plane revision.

Old candidates retain their constant grid area. A declared area revision is
never silently replaced by that constant when its asset or binding is invalid.
Coordinates are native integer pixel centres, including the declared origin
of a larger review canvas. Areas are extensive weights, not RGB resampling.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from .phase.metrics import (
    _shape as _canonical_shape,
    validate_pixel_area as _canonical_validate_pixel_area,
    sum_pixel_area as _canonical_sum_pixel_area,
    filter_components_by_area as _canonical_filter_components_by_area,
)

CONTRACT = "ffac-native-pixel-area-v1"
RIG_MARKER = "ffac_rig_geometry"
RIG_CONTRACT = "ffac-rig-domain-revision-v1"
FIXED_AC28_MM2 = .027053


def _shape(shape):
    return _canonical_shape(shape)


def validate_pixel_area(value, shape_hw):
    """Return a positive finite scalar or matching floating area grid."""
    return _canonical_validate_pixel_area(value, shape_hw)


def inverse_homography_area_map(H_ac28_to_frame_native, shape_hw, *,
                                origin_xy=(0, 0), fixed_area_mm2=FIXED_AC28_MM2,
                                H_ac28_to_metric=None):
    """Evaluate |det J| of frame-to-AC28 at each native pixel centre.

    The optional metric transform is dimensionless relative to the AC28 grid,
    so the recorded AC28 mm² factor is applied exactly once.
    """
    shape = _shape(shape_hw)
    fixed = validate_pixel_area(fixed_area_mm2, shape)
    origin = np.asarray(origin_xy, np.float64)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("Pixel-area native origin must contain two finite coordinates")
    H = np.asarray(H_ac28_to_frame_native, np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-15:
        raise ValueError("Pixel-area homography is singular or invalid")
    inverse = np.linalg.inv(H)
    if H_ac28_to_metric is not None:
        metric = np.asarray(H_ac28_to_metric, np.float64)
        if (metric.shape != (3, 3) or not np.isfinite(metric).all()
                or abs(np.linalg.det(metric)) < 1e-15):
            raise ValueError("AC28 metric homography is singular or invalid")
        inverse = metric @ inverse
    x = np.arange(shape[1], dtype=np.float64) + origin[0]
    y = np.arange(shape[0], dtype=np.float64) + origin[1]
    denominator = inverse[2, 0] * x[None, :] + inverse[2, 1] * y[:, None] + inverse[2, 2]
    if np.any(np.abs(denominator) < 1e-12) or (denominator.min() < 0 < denominator.max()):
        raise ValueError("Pixel-area homography crosses its projective horizon on this grid")
    area = fixed * abs(float(np.linalg.det(inverse))) / np.abs(denominator) ** 3
    return validate_pixel_area(area, shape)


def sum_pixel_area(mask, pixel_area, *, unit="mm2"):
    return _canonical_sum_pixel_area(mask, pixel_area, unit=unit)


@dataclass(frozen=True)
class PixelArea:
    values_mm2: Any
    shape_hw: tuple[int, int]
    provenance: Mapping[str, Any]

    def __post_init__(self):
        value = validate_pixel_area(self.values_mm2, self.shape_hw)
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
        object.__setattr__(self, "values_mm2", value)

    @property
    def values_m2(self):
        return self.values_mm2 * 1e-6

    def sum(self, mask, *, unit="mm2"):
        if np.shape(mask) != self.shape_hw:
            raise ValueError("Area mask belongs to another grid")
        return sum_pixel_area(mask, self.values_mm2, unit=unit)

    def audit(self, domain=None):
        values = np.asarray(self.values_mm2)
        result = dict(self.provenance, shape_hw=list(self.shape_hw),
                      pixel_area_mm2_min=float(values.min()),
                      pixel_area_mm2_max=float(values.max()),
                      variable_pixel_area=values.ndim == 2)
        if domain is not None:
            result["domain_area_cm2"] = self.sum(domain, unit="cm2")
        return result


def _asset(asset, base_dir):
    from darsia.fluidflower.phase_saved_review_metadata import _rig_class_asset as _canonical_rig_class_asset
    return _canonical_rig_class_asset(asset, base_dir)


def _array_asset(asset, base_dir, key):
    path, data, digest = _asset(asset, base_dir)
    loaded = np.load(io.BytesIO(data), allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded as archive:
            value = archive[asset.get("array_key", key)].copy()
    else:
        value = loaded
    return value, dict(path=str(path), sha256=digest)


def crop_native_grid(array, source_origin_xy, shape_hw, origin_xy=(0, 0)):
    """Select an exact native rectangle; no interpolation or silent padding."""
    shape = _shape(shape_hw)
    offset = np.asarray(origin_xy, np.float64) - np.asarray(source_origin_xy, np.float64)
    if offset.shape != (2,) or not np.isfinite(offset).all() or not np.array_equal(offset, np.rint(offset)):
        raise ValueError("Physical grid crop needs exact integer native offsets")
    x, y = map(int, offset)
    if x < 0 or y < 0 or x + shape[1] > array.shape[1] or y + shape[0] > array.shape[0]:
        raise ValueError("Physical grid crop is outside the recorded review canvas")
    return array[y:y + shape[0], x:x + shape[1]]


def rig_phase_shape(revision):
    """Return the explicit analysis grid, independently of photograph height.

    A new rig revision may add rows below the photographic canvas. Historical
    revisions keep their original common photograph/analysis shape.
    """
    photograph = _shape(tuple(revision["canvas_wh"])[::-1])
    shape = _shape(revision.get("phase_shape_hw", photograph))
    if shape[1] != photograph[1] or shape[0] < photograph[0]:
        raise ValueError("Rig phase grid may extend only below its photographic canvas")
    declared = revision.get("photograph_canvas_wh")
    if declared is not None and tuple(declared) != photograph[::-1]:
        raise ValueError("Rig photograph canvas bindings disagree")
    return shape


def load_rig_pixel_area(revision, shape_hw, *, origin_xy=(0, 0), base_dir="."):
    """Load the explicit rig revision and crop its canvas weights if needed."""
    if not isinstance(revision, Mapping) or revision.get("contract") != RIG_CONTRACT:
        raise ValueError("Unsupported rig-domain area revision")
    asset = revision.get("area_map")
    if not asset:
        raise ValueError("Declared rig-domain revision has no bound physical area map")
    full, binding = _array_asset(asset, base_dir, "pixel_area_mm2")
    full_shape = _shape(asset.get("shape_hw", full.shape))
    full = validate_pixel_area(full, full_shape)
    if np.isscalar(full):
        raise ValueError("Rig-domain area revision requires the full spatial map")
    if rig_phase_shape(revision) != full_shape:
        raise ValueError("Rig-domain area map differs from its declared phase grid")
    origin = tuple(revision.get("native_origin_xy", (0, 0)))
    value = crop_native_grid(full, origin, shape_hw, origin_xy)
    return PixelArea(value, _shape(shape_hw), dict(
        contract=CONTRACT, representation="sha-bound-rig-canvas-array",
        rig_domain_contract=RIG_CONTRACT, array=binding,
        pixel_area_mm2_fixed_ac28=float(revision["fixed_area_mm2"]),
        mass_eligible=False, physical_scale_approved=bool(revision.get("physical_scale_approved", False)),
        source_shape_hw=list(full_shape), photograph_canvas_wh=list(revision["canvas_wh"]),
        source_origin_xy=list(origin),
        selected_origin_xy=list(origin_xy)))


def load_rig_classes(revision, *, base_dir=".", shape_hw=None, origin_xy=(0, 0)):
    """Read SHA-bound domain/fill flags, checking their disjoint partition."""
    if not isinstance(revision, Mapping) or revision.get("contract") != RIG_CONTRACT:
        raise ValueError("Unsupported rig-domain class revision")
    path, data, digest = _asset(revision["classes"], base_dir)
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        fields = {name: archive[name].copy() for name in (
            "modelled", "filled_observed", "filled_unobserved")}
        domain = archive["rig_domain"].copy() if "rig_domain" in archive.files else None
    if domain is None:
        domain, _ = _array_asset(revision["rig_domain"], base_dir, "rig_domain")
    full_shape = rig_phase_shape(revision)
    if any(v.dtype != bool or v.shape != full_shape for v in (*fields.values(), domain)):
        raise ValueError("Rig-domain classes must be exact bool masks on the declared canvas")
    total = sum(v.astype(np.uint8) for v in fields.values())
    if np.any(total > 1) or not np.array_equal(total == 1, domain):
        raise ValueError("Rig-domain modelled/filled classes do not form a disjoint complete partition")
    fields["rig_domain"] = domain
    if shape_hw is not None:
        fields = {k: crop_native_grid(v, revision["native_origin_xy"], shape_hw, origin_xy)
                  for k, v in fields.items()}
    for v in fields.values():
        v.flags.writeable = False
    return fields, dict(path=str(path), sha256=digest, contract=RIG_CONTRACT)


def rig_mass_coverage(revision, native_shape_hw, *, base_dir="."):
    """Expose omitted rig strips instead of calling a native crop full mass."""
    shape = rig_phase_shape(revision)
    origin = tuple(revision["native_origin_xy"])
    area = load_rig_pixel_area(revision, shape, origin_xy=origin, base_dir=base_dir)
    fields, binding = load_rig_classes(revision, base_dir=base_dir)
    native_box = np.zeros(shape, bool)
    x, y = -int(origin[0]), -int(origin[1])
    h, w = _shape(native_shape_hw)
    if x < 0 or y < 0 or x + w > shape[1] or y + h > shape[0]:
        raise ValueError("Mass native grid is outside the declared rig canvas")
    native_box[y:y + h, x:x + w] = True
    omitted = fields["rig_domain"] & ~native_box
    return dict(contract=RIG_CONTRACT, classes=binding,
                rig_domain_area_cm2=area.sum(fields["rig_domain"], unit="cm2"),
                native_grid_domain_area_cm2=area.sum(fields["rig_domain"] & native_box, unit="cm2"),
                outside_native_grid_area_cm2=area.sum(omitted, unit="cm2"),
                filled_observed_area_cm2=area.sum(fields["filled_observed"], unit="cm2"),
                filled_unobserved_area_cm2=area.sum(fields["filled_unobserved"], unit="cm2"),
                full_rig_quantification_possible_on_native_grid=not bool(omitted.any()),
                physical_coverage_verified=False)


def load_pixel_area(record, shape_hw, *, base_dir=".", fallback_mm2=None,
                    origin_xy=(0, 0)):
    """Resolve an explicit immutable area descriptor, or a legacy constant.

    Supported recorded representation is a SHA-bound NPY/NPZ mm² map or exact
    inverse-homography parameters. The map takes precedence and is checked
    against the parameters when both are recorded.
    """
    shape = _shape(shape_hw)
    if record is None:
        if fallback_mm2 is None:
            raise ValueError("No recorded pixel area or legacy physical grid area")
        value = validate_pixel_area(fallback_mm2, shape)
        return PixelArea(value, shape, dict(contract="legacy-constant-pixel-area",
                                           pixel_area_mm2=float(value)))
    if not isinstance(record, Mapping) or record.get("contract") != CONTRACT or record.get("unit", "mm2") != "mm2":
        raise ValueError("Unsupported recorded pixel-area contract")
    if tuple(record.get("shape_hw", ())) != shape or tuple(record.get("origin_xy", (0, 0))) != tuple(origin_xy):
        raise ValueError("Recorded pixel area belongs to another grid or native origin")
    parameter_value = None
    if record.get("H_ac28_to_frame_native") is not None:
        parameter_value = inverse_homography_area_map(
            record["H_ac28_to_frame_native"], shape, origin_xy=origin_xy,
            fixed_area_mm2=record["pixel_area_mm2_fixed_ac28"],
            H_ac28_to_metric=record.get("H_ac28_to_metric"))
    provenance = dict(contract=CONTRACT, representation="inverse-homography",
                      pixel_area_mm2_fixed_ac28=float(record["pixel_area_mm2_fixed_ac28"]))
    value = parameter_value
    if record.get("array"):
        path, data, digest = _asset(record["array"], base_dir)
        loaded = np.load(io.BytesIO(data), allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded as archive:
                value = archive[record["array"].get("array_key", "pixel_area_mm2")].copy()
        else:
            value = loaded
        value = validate_pixel_area(value, shape)
        if parameter_value is not None and not np.allclose(value, parameter_value, rtol=2e-6, atol=1e-10):
            raise ValueError("Recorded pixel-area map disagrees with its homography")
        provenance.update(representation="sha-bound-array", array=dict(path=str(path), sha256=digest))
    if value is None:
        raise ValueError("Recorded pixel area has neither map nor homography parameters")
    provenance["parameters"] = {k: record[k] for k in (
        "H_ac28_to_frame_native", "H_ac28_to_metric", "origin_xy") if k in record}
    return PixelArea(value, shape, provenance)


def load_frame_pixel_area(native_manifest_path, frame_index, shape_hw, *,
                          fallback_mm2=None, origin_xy=(0, 0), expected_native_sha256=None):
    """Load one FF_AC frame's declared area revision, retaining old candidates."""
    path = Path(native_manifest_path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_native_sha256 is not None and digest != expected_native_sha256:
        raise ValueError("Pixel-area native timeline SHA256 changed")
    native = json.loads(data.decode("utf-8-sig"))
    rows = native.get("frame_sources") or []
    if type(frame_index) is not int or not 0 <= frame_index < len(rows) or rows[frame_index].get("frame_index") != frame_index:
        raise ValueError("Pixel area requires an exact native frame index")
    row = rows[frame_index]
    if row.get("rig_domain_revision") is not None:
        result = load_rig_pixel_area(row["rig_domain_revision"], shape_hw,
                                     origin_xy=origin_xy, base_dir=path.parent)
        return PixelArea(result.values_mm2, result.shape_hw,
                         dict(result.provenance, native_manifest_sha256=digest, frame_index=frame_index))
    record = row.get("pixel_area")
    base = path.parent
    marker = native.get(RIG_MARKER)
    if marker is not None:
        if not isinstance(marker, Mapping) or not marker.get("manifest"):
            raise ValueError("Rig geometry pixel-area binding is incomplete")
        geometry_path, geometry_bytes, geometry_sha = _asset(marker["manifest"], path.parent)
        geometry = json.loads(geometry_bytes.decode("utf-8-sig"))
        matches = [r for r in geometry.get("frames", ()) if r.get("run", "").casefold() == str(native.get("run", "")).casefold()
                   and r.get("frame_index") == frame_index]
        if len(matches) != 1:
            raise ValueError("Rig geometry has no unique native pixel-area frame")
        bound = matches[0]
        # A geometry manifest may bind the source native SHA to avoid a cycle;
        # the native marker must explicitly record that immutable source SHA.
        expected = marker.get("source_native_manifest_sha256", digest)
        actual = bound.get("native_manifest_sha256") or (bound.get("native_manifest") or {}).get("sha256")
        if actual != expected or bound.get("source_sha256", row.get("source_sha256")) != row.get("source_sha256"):
            raise ValueError("Rig geometry pixel area belongs to another native source")
        record = bound.get("pixel_area")
        if record is None:
            raise ValueError("Declared rig geometry has no frame pixel area")
        base = geometry_path.parent
    result = load_pixel_area(record, shape_hw, base_dir=base, fallback_mm2=fallback_mm2, origin_xy=origin_xy)
    provenance = dict(result.provenance, native_manifest_sha256=digest, frame_index=frame_index)
    if marker is not None:
        provenance["geometry_manifest"] = dict(path=str(geometry_path), sha256=geometry_sha)
    return PixelArea(result.values_mm2, result.shape_hw, provenance)


def filter_components_by_area(mask, pixel_area_mm2, minimum_area_mm2=2.5, *, connectivity=4):
    """Filter components by the sum of physical pixel areas, never pixel count."""
    return _canonical_filter_components_by_area(mask, pixel_area_mm2, minimum_area_mm2=minimum_area_mm2, connectivity=connectivity)


def area_filtered_gas(mask, area_mm2_array, minimum_area_mm2=2.5, connectivity=4):
    """Production inference API for a native physically weighted gas cutoff."""
    return filter_components_by_area(mask, area_mm2_array, minimum_area_mm2,
                                      connectivity=connectivity)


def mass_frame_pixel_area(geometry, frame_index, source_shape_hw, analysis_shape_hw, *, base_dir="."):
    """Return planar m² weights for the distributed CPU mass input contract.

    An optional ``pixel_area_frames`` list contains one exact frame descriptor
    in chronological input order. Reduced grids sum native areas conservatively.
    """
    source_shape, analysis_shape = _shape(source_shape_hw), _shape(analysis_shape_hw)
    frames = geometry.get("pixel_area_frames")
    if frames is None:
        value = float(geometry["width_m"]) * float(geometry["height_m"]) / np.prod(analysis_shape)
        validate_pixel_area(value, analysis_shape)
        return float(value), None
    if not isinstance(frames, list) or not 0 <= frame_index < len(frames):
        raise ValueError("Distributed mass area frames are incomplete")
    row = frames[frame_index]
    if not isinstance(row, Mapping) or row.get("frame_index") != frame_index:
        raise ValueError("Distributed mass area frame identity changed")
    origin = tuple(geometry.get("native_origin_xy", (0, 0)))
    if row.get("rig_domain_revision") is not None:
        area = load_rig_pixel_area(row["rig_domain_revision"], source_shape,
                                   origin_xy=origin, base_dir=base_dir)
    else:
        area = load_pixel_area(row.get("pixel_area"), source_shape, base_dir=base_dir,
                                origin_xy=origin)
    values = np.asarray(np.broadcast_to(area.values_m2, source_shape), np.float64)
    if source_shape != analysis_shape:
        values = cv2.resize(values, analysis_shape[::-1], interpolation=cv2.INTER_AREA) * (np.prod(source_shape) / np.prod(analysis_shape))
        if not np.isclose(values.sum(dtype=np.float64),
                          np.sum(np.broadcast_to(area.values_m2, source_shape), dtype=np.float64),
                          rtol=2e-6, atol=1e-14):
            raise ValueError("Distributed mass physical-area pooling does not conserve total area")
    return validate_pixel_area(values, analysis_shape), area.audit()


def carry_mass_area_bindings(run_input):
    """Carry numerical source area descriptors into the mass-consumer grid."""
    from copy import deepcopy
    result = deepcopy(run_input)
    sources = result.get("source_frames") or []
    declared = [bool(row.get("rig_domain_revision") or row.get("pixel_area")) for row in sources]
    if not any(declared):
        return result
    if not all(declared) or len(sources) != len(result.get("target_times_hours") or []):
        raise ValueError("Numerical mass source has incomplete per-frame physical areas")
    bindings = []
    for index, source in enumerate(sources):
        name = "rig_domain_revision" if source.get("rig_domain_revision") is not None else "pixel_area"
        bindings.append(dict(frame_index=index, **{name: source[name]}))
    result["geometry"]["pixel_area_frames"] = bindings
    return result
