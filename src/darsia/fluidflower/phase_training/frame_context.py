"""Reconstruct source-bound FF_AC V5 physical context for one native frame.

The run baseline can drift relative to later photos.  A per-run mask is thus
insufficient for label sampling or phase-map physics.  This module consumes a
small immutable per-frame registration receipt and one SHA-bound AC28 context;
it never resamples the photograph or moves a saved annotation.

The physical facies boundaries remain provisional.  The output is strictly a
manual-validation context and cannot establish scientific mass eligibility.
"""

from __future__ import annotations
from ..phase_saved_review_metadata import configured_parameters

def _parameters():
    return configured_parameters("training_frame_context")

def _export(name):
    item=_parameters()["exports"][name]
    return tuple(item["value"]) if item["python_type"]=="tuple" else item["value"]

_METADATA_EXPORT_NAMES = ('CONTRACT', 'COHORT_CONTRACT', 'SHAPE', 'REFERENCE_SAFE_KEY', 'RASTER_FIELDS', 'PHOTO_FIELDS', 'PHOTO_SAFE_CORE_CONTRACT', 'PHOTO_SAFE_CORE_GUARD_PX', 'PHOTO_SAFE_CORE_FILLED_CONTRACT', 'PHOTO_SAFE_CORE_HOLE_MAX_AREA_PX', 'PHOTO_SAFE_CORE_VERTICAL_GAP_MAX_PX')

def __getattr__(name):
    if name not in _METADATA_EXPORT_NAMES: raise AttributeError(name)
    return _export(name)

def _verified_original_warp_code_digest(expected):
    """Verify original producer bytes and the actual current method separately."""
    p=_parameters()
    current=p["current_code_source"]
    if Path(current["path"]).resolve()!=Path(__file__).resolve():
        raise ValueError("Current training context is outside its declared code source")
    _check(Path(__file__),current["sha256"],"current training context code")
    if Path(__file__).stat().st_size!=current["size_bytes"]:
        raise ValueError("Current training context code size changed")
    original=p["original_warp_code_sources"].get(expected)
    if not isinstance(original,dict) or original.get("sha256")!=expected:
        raise ValueError("Frame context has no exact original warp-code source binding")
    path=Path(original["path"])
    _check(path,expected,"original warp-code producer")
    if path.stat().st_size!=original["size_bytes"]:
        raise ValueError("Original warp-code producer size changed")
    return expected

from darsia.fluidflower.native import support as _support
from darsia.fluidflower.native.config import require_parameters

from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from darsia.fluidflower.project import sha256_file




def photo_safe_core_policy_metadata(enclosed_hole_policy: str = "keep") -> dict[str, Any]:
    """Retained entry point; policy values come from explicit configuration."""
    return _support.photo_safe_core_policy_metadata(enclosed_hole_policy, parameters=require_parameters('support'))


def apply_photo_safe_core_enclosed_holes(safe, *, enclosed_hole_policy="keep", return_rule_masks=False):
    """Delegate the unchanged support operation to the native core."""
    return _support.apply_photo_safe_core_enclosed_holes(safe, enclosed_hole_policy=enclosed_hole_policy,
        return_rule_masks=return_rule_masks, parameters=require_parameters('support'))


def project_source_photo_safe_core(sand_grid, h_ac28_to_frame_native, *, shape_hw=_export("SHAPE"),
                                    guard_px=_export("PHOTO_SAFE_CORE_GUARD_PX"),
                                    enclosed_hole_policy="keep", return_metadata=False):
    """Retain historical call coordinates while delegating numerical work."""
    return _support.project_source_photo_safe_core(sand_grid, h_ac28_to_frame_native,
        shape_hw=shape_hw, guard_px=guard_px, enclosed_hole_policy=enclosed_hole_policy,
        return_metadata=return_metadata, parameters=require_parameters('support'))


def _source_photo_grid(path: str, digest: str) -> np.ndarray:
    return _support.source_photo_grid(path, digest, parameters=require_parameters('support'))


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _check(path: Path, digest: str, label: str) -> None:
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"V5 per-frame {label} lacks SHA")
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"V5 per-frame {label} changed: {path}")


def _check_static_once(path: Path, digest: str, label: str,
                       cache: dict[tuple[str, str], tuple[int, int]] | None) -> None:
    """Hash a shared immutable asset once per cohort, rechecking file metadata.

    A frame receipt and both acquisition images are still hashed on every
    access.  The 144 MB AC28 reference context and the common cross-run H
    receipt need not be reread 864 times when unchanged.
    """
    if cache is None:
        _check(path, digest, label)
        return
    key = (str(path.resolve()), digest)
    stat = path.stat()
    signature = (int(stat.st_size), int(stat.st_mtime_ns))
    if cache.get(key) != signature:
        _check(path, digest, label)
        cache[key] = signature


def _normal_h(value: Any) -> np.ndarray:
    return _support._normal_h(value, parameters=require_parameters('support'))


def compute_feature_support(*, current_valid, baseline_valid, h_baseline_to_current,
                            previous_valid=None, h_baseline_to_previous=None,
                            baseline_origin_xy=(0, 0), previous_origin_xy=(0, 0)):
    """Delegate source footprint intersections without changing their coordinate origins."""
    return _support.compute_feature_support(current_valid=current_valid,baseline_valid=baseline_valid,
        h_baseline_to_current=h_baseline_to_current,previous_valid=previous_valid,
        h_baseline_to_previous=h_baseline_to_previous,baseline_origin_xy=baseline_origin_xy,
        previous_origin_xy=previous_origin_xy,parameters=require_parameters('support'))


def verify_frame_geometry(receipt_path: Path, source_manifest_path: Path, *,
                          expected_shape_hw: tuple[int, int] = _export("SHAPE"),
                          static_hash_cache: dict[tuple[str, str], tuple[int, int]] | None = None
                          ) -> dict[str, Any]:
    """Verify exact frame/revision/cross-run lineage before deriving any mask."""
    receipt_path = Path(receipt_path)
    source_path = Path(source_manifest_path)
    receipt = _read(receipt_path)
    source_sha = sha256_file(source_path)
    source = _read(source_path)
    if (receipt.get("contract") != _export("CONTRACT")
            or receipt.get("original_source_manifest_sha256") != source_sha
            or receipt.get("run") != str(source.get("run") or "").casefold()
            or receipt.get("native_shape_hw") != list(expected_shape_hw)
            or receipt.get("validation_only") is not True
            or receipt.get("mass_eligible") is not False):
        raise ValueError("V5 frame geometry has another source or acceptance scope")
    index = receipt.get("frame_index")
    frames = source.get("frame_sources") or []
    if (not isinstance(index, int) or not 0 <= index < len(frames)
            or index != frames[index].get("frame_index")):
        raise ValueError("V5 frame geometry has no exact timeline frame")
    frame = frames[index]
    native_sha = frame.get("native_jpeg_sha256") or frame.get("frame_sha256")
    if (receipt.get("native_review_jpeg_sha256") != native_sha
            or receipt.get("baseline_frame0_native_sha256") !=
            (frames[0].get("native_jpeg_sha256") or frames[0].get("frame_sha256"))
            or receipt.get("camera_jpeg_sha256") != frame.get("source_sha256")):
        raise ValueError("V5 frame H is not tied to the original camera/native photos")
    for prefix, digest_key in (
        ("camera_jpeg", "camera_jpeg_sha256"),
        ("native_review_jpeg", "native_review_jpeg_sha256"),
        ("reference_context", "reference_context_sha256"),
        ("crossrun_geometry_receipt", "crossrun_geometry_receipt_sha256"),
    ):
        asset = Path(receipt.get(prefix + "_path") or "")
        if prefix in {"reference_context", "crossrun_geometry_receipt"}:
            _check_static_once(asset, receipt.get(digest_key), prefix,
                               static_hash_cache)
        else:
            _check(asset, receipt.get(digest_key), prefix)
    b_certificate_path = Path(receipt.get("v5_B_registration_certificate_path") or "")
    _check_static_once(b_certificate_path,
                       receipt.get("v5_B_registration_certificate_sha256"),
                       "V5 B registration certificate", static_hash_cache)
    b_certificate = _read(b_certificate_path)
    b_sources = b_certificate.get("sources") or {}
    if (b_certificate.get("run") != receipt["run"]
            or b_sources.get("review_native_jpeg_sha256") !=
            receipt.get("v5_B_native_jpeg_sha256")):
        raise ValueError("V5 B registration belongs to another run/native source")
    for source_key, path_key in (("arw_sha256", "source_arw_path"),
                                 ("camera_jpeg_sha256", "source_camera_jpeg_path"),
                                 ("review_native_jpeg_sha256", "source_review_path")):
        _check(Path(b_certificate.get(path_key) or ""),
               b_sources.get(source_key), f"V5 B {path_key}")
    if (receipt.get("warp_code_sha256") != _verified_original_warp_code_digest(receipt.get("warp_code_sha256"))
            or receipt.get("reference_context_key") != _export("REFERENCE_SAFE_KEY")):
        raise ValueError("V5 frame context was certified with another warp code/mask")
    if (receipt.get("photo_sand_qa_pass") is not True
            or receipt.get("optical_geometry_qa_pass") is not True):
        raise ValueError("V5 source-photo sand or optical geometry lacks independent QA")
    if (receipt.get("mass_eligible") is True
            and receipt.get("physical_scale_qa_pass") is not True):
        raise ValueError("V5 mass use lacks independently checked physical scale")
    photo_scope = receipt.get("photo_sand_qa_scope")
    if photo_scope == _export("PHOTO_SAFE_CORE_CONTRACT"):
        if (receipt.get("photo_sand_transport_qa_pass") is not True
                or receipt.get("training_label_eligibility_certified") is not False
                or receipt.get("phase_map_eligibility_certified") is not False
                or receipt.get("rig_metal_certified") is not False
                or receipt.get("dynamic_overflow_certified") is not False
                or receipt.get("mass_eligible") is not False):
            raise ValueError("Transported B sand core claims unverified labels or rig")
        mask_path = Path(receipt.get("photo_sand_reference_mask_path") or "")
        mask_sha = receipt.get("photo_sand_reference_mask_sha256")
        acceptance_path = Path(receipt.get("photo_sand_reference_receipt_path") or "")
        acceptance_sha = receipt.get("photo_sand_reference_receipt_sha256")
        _check_static_once(mask_path, mask_sha, "source-photo B sand core",
                           static_hash_cache)
        _check_static_once(acceptance_path, acceptance_sha,
                           "source-photo B acceptance", static_hash_cache)
        acceptance = _read(acceptance_path)
        if (acceptance.get("source_manifest_sha256") != source_sha
                or acceptance.get("mask_npz_sha256") != mask_sha
                or acceptance.get("photo_gate_pass") is not True
                or acceptance.get("training_label_eligibility_certified") is not False
                or acceptance.get("phase_map_eligibility_certified") is not False):
            raise ValueError("Transported B sand core has another source/QA scope")
        _source_photo_grid(str(mask_path), mask_sha)
        required_source_gates = ("temporal_registration_pass",
                                 "safe_interior_projection_pass")
    else:
        _check(Path(receipt.get("photo_certified_sand_mask_path") or ""),
               receipt.get("photo_certified_sand_mask_sha256"),
               "source-certified frame sand mask")
        required_source_gates = ("temporal_registration_pass",
                                 "safe_interior_projection_pass",
                                 "domain_frame_exclusion_pass",
                                 "eligible_annotation_support_pass")
    if not all(receipt.get(key) is True for key in required_source_gates):
        raise ValueError("V5 frame geometry lacks source/annotation support QA")
    if (not isinstance(receipt.get("temporal_heldout_sand_p95_px"), (int, float))
            or not math.isfinite(float(receipt["temporal_heldout_sand_p95_px"]))
            or int(receipt.get("temporal_heldout_sand_n") or 0) <= 0):
        raise ValueError("V5 temporal H lacks independent heldout sand evidence")
    crossrun = _read(Path(receipt["crossrun_geometry_receipt_path"]))
    # The photometric B image is a distinct acquisition from timeline frame 0
    # for eleven runs.  Both source coordinate systems must be explicit: an
    # identity BÃ¢â€ â€™frame0 H may only be certified from the exact B source, not
    # inferred from run or frame number.
    h_timeline_to_frame = _normal_h(receipt.get("H_timeline0_to_frame_native"))
    h_b_to_timeline = _normal_h(receipt.get("H_v5_B_to_timeline0_native"))
    h_b_to_frame = _normal_h(receipt.get("H_v5_B_to_frame_native"))
    h_ac28_to_timeline = _normal_h(receipt.get("H_ac28_to_timeline0_native"))
    h_ac28_to_frame = _normal_h(receipt.get("H_ac28_to_frame_native"))
    h_crossrun = _normal_h(crossrun.get("H_ac28_to_native"))
    if (np.max(np.abs(h_crossrun - h_ac28_to_timeline)) > _parameters()["registration_matrix_atol"]
            or np.max(np.abs(_normal_h(h_timeline_to_frame @ h_b_to_timeline)
                              - h_b_to_frame)) > _parameters()["registration_matrix_atol"]
            or np.max(np.abs(_normal_h(h_timeline_to_frame @ h_ac28_to_timeline)
                              - h_ac28_to_frame)) > _parameters()["registration_matrix_atol"]):
        raise ValueError("V5 frame H does not compose its bound B/timeline/AC28 transforms")
    return dict(contract=_export("CONTRACT"), run=receipt["run"], frame_index=index,
                source_manifest_sha256=source_sha,
                receipt_path=str(receipt_path), receipt_sha256=sha256_file(receipt_path),
                native_review_jpeg_sha256=native_sha,
                reference_context_path=receipt["reference_context_path"],
                reference_context_sha256=receipt["reference_context_sha256"],
                 photo_certified_sand_mask_path=receipt.get(
                     "photo_certified_sand_mask_path"),
                 photo_certified_sand_mask_sha256=receipt.get(
                     "photo_certified_sand_mask_sha256"),
                 photo_sand_qa_pass=receipt.get("photo_sand_qa_pass"),
                 photo_sand_qa_scope=photo_scope,
                 photo_sand_reference_mask_path=receipt.get(
                     "photo_sand_reference_mask_path"),
                 photo_sand_reference_mask_sha256=receipt.get(
                     "photo_sand_reference_mask_sha256"),
                 photo_sand_reference_receipt_sha256=receipt.get(
                     "photo_sand_reference_receipt_sha256"),
                 optical_geometry_qa_pass=receipt.get("optical_geometry_qa_pass"),
                 physical_scale_qa_pass=receipt.get("physical_scale_qa_pass"),
                H_ac28_to_frame_native=h_ac28_to_frame,
                H_v5_B_to_frame_native=h_b_to_frame,
                validation_only=True, mass_eligible=False)


@lru_cache(maxsize=2)
def _reference_context(path: str, digest: str,
                       shape_hw: tuple[int, int]) -> dict[str, np.ndarray]:
    source = Path(path)
    _check(source, digest, "reference context")
    with np.load(source, allow_pickle=False) as archive:
        if not set(_export("RASTER_FIELDS")).issubset(archive.files):
            raise ValueError("V5 reference context has missing physical/safe fields")
        values = {key: np.asarray(archive[key]) for key in _export("RASTER_FIELDS")}
        values["x_m"] = np.asarray(archive["x_m"])
        values["y_m"] = np.asarray(archive["y_m"])
    if any(values[key].shape != shape_hw for key in _export("RASTER_FIELDS")):
        raise ValueError("V5 reference context is not on native grid")
    physical = np.asarray(values["physical_sand_domain"], dtype=bool)
    supported = np.asarray(values["physics_supported_domain"], dtype=bool)
    observable = np.asarray(values["observable_phase_domain"], dtype=bool)
    safe = np.asarray(values[_export("REFERENCE_SAFE_KEY")], dtype=bool)
    if (not np.array_equal(np.asarray(values["domain"], dtype=bool), supported)
            or not np.all(observable)
            or not np.any(supported)
            or np.any(physical & ~observable)
            or np.any(supported & ~physical)
            or np.any(safe & ~physical)):
        raise ValueError("V5 supported, physical, observable and safe masks conflict")
    if (values["x_m"].shape != (shape_hw[1],)
            or values["y_m"].shape != (shape_hw[0],)):
        raise ValueError("V5 reference nominal axes are not on native grid")
    for value in values.values():
        value.setflags(write=False)
    return values


def _warp_continuous(source: np.ndarray, h: np.ndarray,
                     shape_hw: tuple[int, int]) -> np.ndarray:
    return cv2.warpPerspective(np.asarray(source, np.float32), h,
                               (shape_hw[1], shape_hw[0]),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0.0)


def _warp_nearest_float(source: np.ndarray, h: np.ndarray,
                        shape_hw: tuple[int, int]) -> np.ndarray:
    # Permeability belongs to a discrete sand facies.  Interpolating across
    # the rig mask combines unrelated facies and propagates its NaN exterior
    # into otherwise valid native porous pixels.
    return cv2.warpPerspective(np.asarray(source, np.float32), h,
                               (shape_hw[1], shape_hw[0]),
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=float("nan"))


def _warp_categorical(source: np.ndarray, h: np.ndarray,
                      shape_hw: tuple[int, int]) -> np.ndarray:
    return cv2.warpPerspective(np.asarray(source, np.uint8), h,
                               (shape_hw[1], shape_hw[0]),
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)


def derive_frame_context(binding: dict[str, Any], *,
                         fields: Iterable[str] = _export("RASTER_FIELDS"),
                         expected_shape_hw: tuple[int, int] = _export("SHAPE")) -> dict[str, np.ndarray]:
    """Warp only source-bound context fields into this frame's native grid.

    Coordinates use the inverse projective map, and physical area uses its
    analytic Jacobian.  The photograph and its annotations are never warped.
    """
    requested = set(fields)
    if not requested.issubset(set(_export("RASTER_FIELDS")) | set(_export("PHOTO_FIELDS"))):
        raise ValueError("Unknown V5 frame context field")
    h = _normal_h(binding["H_ac28_to_frame_native"])
    shape = tuple(expected_shape_hw)
    result: dict[str, np.ndarray] = {}
    if requested & set(_export("PHOTO_FIELDS")):
        if binding.get("photo_sand_qa_pass") is not True:
            raise ValueError("V5 frame sand/photo boundary lacks independent QA")
        if binding.get("photo_sand_qa_scope") == _export("PHOTO_SAFE_CORE_CONTRACT"):
            sand_grid = _source_photo_grid(
                binding["photo_sand_reference_mask_path"],
                binding["photo_sand_reference_mask_sha256"])
            safe = project_source_photo_safe_core(
                sand_grid, h, shape_hw=shape,
                enclosed_hole_policy=binding.get("enclosed_hole_policy", "keep"))
            photo = {"photo_certified_sand_domain": safe,
                     "photo_boundary_uncertain_domain": ~safe}
        else:
            photo_path = Path(binding.get("photo_certified_sand_mask_path") or "")
            _check(photo_path, binding.get("photo_certified_sand_mask_sha256"),
                   "source-certified frame sand mask")
            with np.load(photo_path, allow_pickle=False) as archive:
                if not set(_export("PHOTO_FIELDS")).issubset(archive.files):
                    raise ValueError("V5 source-photo sand asset lacks certified/uncertain masks")
                photo = {key: np.asarray(archive[key]) for key in _export("PHOTO_FIELDS")}
            if (any(value.shape != shape or value.dtype != np.bool_
                    for value in photo.values())
                    or np.any(photo["photo_certified_sand_domain"] &
                              photo["photo_boundary_uncertain_domain"])):
                raise ValueError("V5 source-photo sand/uncertain masks conflict")
        result.update({key: photo[key] for key in requested & set(_export("PHOTO_FIELDS"))})
    if not (requested - set(_export("PHOTO_FIELDS"))):
        return result
    values = _reference_context(binding["reference_context_path"],
                                binding["reference_context_sha256"], shape)
    for key in requested - {"pixel_area_m2"} - set(_export("PHOTO_FIELDS")):
        if key == "observable_phase_domain":
            # The source-backed observable area is the entire unchanged
            # native review content, including documented lower overflow.
            # A projective sand warp must never trim it at the edge.
            result[key] = np.ones(shape, dtype=bool)
        elif key in {"domain", "physics_supported_domain", "physical_sand_domain",
                     "facies", _export("REFERENCE_SAFE_KEY")}:
            raw = _warp_categorical(values[key], h, shape)
            result[key] = raw.astype(bool) if key != "facies" else raw
        elif key == "permeability":
            result[key] = _warp_nearest_float(values[key], h, shape)
        else:
            result[key] = _warp_continuous(values[key], h, shape)
    if "pixel_area_m2" in requested:
        ref_area = _warp_continuous(values["pixel_area_m2"], h, shape)
        inverse = np.linalg.inv(h)
        determinant = abs(float(np.linalg.det(inverse)))
        area = np.empty(shape, np.float32)
        columns = np.arange(shape[1], dtype=np.float64)[None, :]
        for y0 in range(0, shape[0], _parameters()["area_row_block_size"]):
            y1 = min(y0 + _parameters()["area_row_block_size"], shape[0])
            rows = np.arange(y0, y1, dtype=np.float64)[:, None]
            den = inverse[2, 0] * columns + inverse[2, 1] * rows + inverse[2, 2]
            if np.any(np.abs(den) < _parameters()["area_horizon_floor"]):
                raise ValueError("V5 physical grid projects through infinity")
            area[y0:y1] = (ref_area[y0:y1] * determinant / np.abs(den)**3).astype(np.float32)
        result["pixel_area_m2"] = area
    domain = (result["physics_supported_domain"] if "physics_supported_domain" in result
              else _warp_categorical(values["physics_supported_domain"], h, shape).astype(bool))
    if "domain" in result and not np.array_equal(result["domain"], domain):
        raise ValueError("V5 legacy domain alias differs from supported physics")
    if "physical_sand_domain" in result and np.any(domain & ~result["physical_sand_domain"]):
        raise ValueError("V5 supported physics escaped physical sand")
    if "observable_phase_domain" in result:
        if np.any(domain & ~result["observable_phase_domain"]):
            raise ValueError("V5 supported physics escapes observable image support")
        if ("physical_sand_domain" in result and np.any(
                result["physical_sand_domain"] & ~result["observable_phase_domain"])):
            raise ValueError("V5 physical sand escapes observable image support")
    if (_export("REFERENCE_SAFE_KEY") in result and "physical_sand_domain" in result
            and np.any(result[_export("REFERENCE_SAFE_KEY")] & ~result["physical_sand_domain"])):
        raise ValueError("V5 optical safe interior escaped physical sand")
    footprint = cv2.warpPerspective(np.ones(shape, np.float32), h,
                                    (shape[1], shape[0]),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0.0)
    if np.any(domain & (footprint < np.float32(1 - _parameters()["reference_footprint_epsilon"]))):
        raise ValueError("V5 frame porous domain escapes exact reference support")
    for key in requested & {"x_m_native", "y_m_native", "pixel_area_m2", "permeability"}:
        if not np.isfinite(result[key][domain]).all():
            raise ValueError(f"V5 frame {key} nonfinite inside porous domain")
    if "pixel_area_m2" in result and np.any(result["pixel_area_m2"][domain] <= 0):
        raise ValueError("V5 frame pixel area is nonpositive")
    return result


class V5FrameGeometryCohort:
    """One immutable native revision's complete per-frame geometry index.

    This small object may be reconstructed by each process worker from file
    paths.  It does not pickle a 16-million-pixel context or trust a run-name
    lookup.  Individual H receipts are checked on access.
    """

    def __init__(self, native_manifest_path: str | Path):
        path = Path(native_manifest_path)
        native = _read(path)
        marker = native.get("v5_frame_geometry") or {}
        source_path = Path(marker.get("original_source_manifest_path") or "")
        digest = marker.get("original_source_manifest_sha256")
        if marker.get("contract") != _export("COHORT_CONTRACT"):
            raise ValueError("V5 native source lacks a per-frame geometry cohort")
        _check(source_path, digest, "original source manifest")
        source = _read(source_path)
        if (native.get("run") != source.get("run")
                or native.get("frame_count") != source.get("frame_count")
                or (native.get("v5_photometry") or {}).get("source_manifest_sha256") != digest):
            raise ValueError("V5 frame geometry belongs to another source revision")
        rows = marker.get("frames") or []
        if (len(rows) != int(source.get("frame_count") or -1)
                or [row.get("frame_index") for row in rows] != list(range(len(rows)))):
            raise ValueError("V5 per-frame geometry index is incomplete or unordered")
        self.native_manifest_path = path
        self.native_manifest_sha256 = sha256_file(path)
        self.source_manifest_path = source_path
        self.source_manifest_sha256 = digest
        self.rows = rows
        self.shape_hw = tuple(source["analysis_image_shape_hw"])
        self._static_hash_cache: dict[tuple[str, str], tuple[int, int]] = {}

    def frame(self, index: int) -> dict[str, Any]:
        if not isinstance(index, int) or not 0 <= index < len(self.rows):
            raise ValueError("V5 frame index outside its bound source timeline")
        row = self.rows[index]
        path = Path(row.get("receipt_path") or "")
        _check(path, row.get("receipt_sha256"), "frame geometry receipt")
        binding = verify_frame_geometry(path, self.source_manifest_path,
                                        expected_shape_hw=self.shape_hw,
                                        static_hash_cache=self._static_hash_cache)
        if binding["frame_index"] != index:
            raise ValueError("V5 frame receipt index changed")
        return binding

    def context(self, index: int, *, fields: Iterable[str] = _export("RASTER_FIELDS")
                ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        binding = self.frame(index)
        return derive_frame_context(binding, fields=fields,
                                    expected_shape_hw=self.shape_hw), binding

    def H_v5_B_to_frame(self, index: int) -> np.ndarray:
        """Exact corrected V5 B review Ã¢â€ â€™ current native frame transform."""
        return _normal_h(self.frame(index)["H_v5_B_to_frame_native"])

    def H_previous_to_frame(self, previous_index: int,
                            index: int) -> np.ndarray:
        """Previous native frame Ã¢â€ â€™ current native frame via common B anchor."""
        current = self.H_v5_B_to_frame(index)
        previous = self.H_v5_B_to_frame(previous_index)
        return _normal_h(current @ np.linalg.inv(previous))

    def feature_support(self, index: int, previous_index: int | None,
                        linear_input: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Source-supported optical pixels for one unchanged native frame.

        The recorded V5 correction may be finite over less than the complete
        camera content. Baseline/previous references are valid only where all
        four bilinear source neighbours exist. Eight pixels of additional
        support protect Gaussian texture features from zero-transport values.
        An annotation remains in the source inventory even if this mask
        excludes some of its pixels; the cohort-level label gate decides
        whether that loss is acceptable before training.
        """
        from darsia.fluidflower.phase_training.log_linear_input import (
            RunV5LinearInput,
        )

        if (not isinstance(linear_input, RunV5LinearInput)
                or linear_input.manifest_path.resolve() != self.native_manifest_path.resolve()
                or linear_input.manifest_sha256 != self.native_manifest_sha256):
            raise ValueError("V5 feature support needs this exact float source revision")
        native = _read(self.native_manifest_path)
        mode = (native.get("v5_temporal_reference") or {}).get("contract")
        if mode not in ("ffac-v5-native-crop-source-supported-v1",
                        "ffac-v5-source-backed-extended-ring-v1"):
            raise ValueError("V5 feature support has no explicit temporal source mode")
        if previous_index is not None and (not isinstance(previous_index, int)
                                           or previous_index >= index
                                           or previous_index < 0):
            raise ValueError("V5 previous frame index is not earlier than current")

        current_binding = self.frame(index)
        previous_binding = (None if previous_index is None
                            else self.frame(previous_index))

        def exact_mask(frame_id: str, *, extended: bool
                       ) -> tuple[np.ndarray, tuple[int, int], str]:
            if frame_id == "pre_injection":
                photo = (linear_input.review_root / "native_references" /
                         linear_input.run_id / "pre_injection.jpg")
                record = linear_input.reference
            else:
                photo = (linear_input.review_root / "native_frames" /
                         linear_input.run_id / f"{frame_id}.jpg")
                record = linear_input.frames[frame_id]
            valid, origin = linear_input.load_validity(
                photo, frame_id, extended=extended)
            return valid, tuple(origin), str(record["v5_log_linear_rgb_sha256"])

        current_valid, _, current_sha = exact_mask(f"frame_{index:03d}",
                                                    extended=False)
        use_ring = mode == "ffac-v5-source-backed-extended-ring-v1"
        baseline_valid, baseline_origin, baseline_sha = exact_mask(
            "pre_injection", extended=use_ring)

        b_to_current = _normal_h(current_binding["H_v5_B_to_frame_native"])
        previous_sha = None
        previous_origin = None
        previous_valid = None
        if previous_index is not None:
            previous_valid, previous_origin, previous_sha = exact_mask(
                f"frame_{previous_index:03d}", extended=use_ring)
            b_to_previous = _normal_h(
                previous_binding["H_v5_B_to_frame_native"])
        else:
            b_to_previous = None
        safe, raw_support = compute_feature_support(
            current_valid=current_valid, baseline_valid=baseline_valid,
            previous_valid=previous_valid,
            h_baseline_to_current=b_to_current,
            h_baseline_to_previous=b_to_previous,
            baseline_origin_xy=baseline_origin,
            previous_origin_xy=(previous_origin or (0, 0)))
        mask_sha = sha256(np.packbits(safe, bitorder="little").tobytes()).hexdigest()
        receipt = dict(contract="ffac-v5-frame-feature-source-support-v1",
            native_manifest_sha256=self.native_manifest_sha256,
            original_source_manifest_sha256=self.source_manifest_sha256,
            temporal_reference_contract=mode, frame_index=index,
            previous_frame_index=previous_index,
            frame_geometry_receipt_sha256=current_binding["receipt_sha256"],
            previous_geometry_receipt_sha256=(None if previous_binding is None else
                                              previous_binding["receipt_sha256"]),
            current_float_sha256=current_sha, baseline_float_sha256=baseline_sha,
            previous_float_sha256=previous_sha,
            baseline_origin_xy=list(baseline_origin),
            previous_origin_xy=(None if previous_origin is None else
                                list(previous_origin)),
            raw_supported_pixels=int(np.count_nonzero(raw_support)),
            feature_supported_pixels=int(np.count_nonzero(safe)),
            feature_support_erosion_radius_px=_parameters()["feature_support_erosion_radius_px"],
            feature_support_mask_sha256=mask_sha)
        receipt["receipt_sha256"] = sha256(json.dumps(receipt, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return safe, receipt


__all__ = ["CONTRACT", "COHORT_CONTRACT", "REFERENCE_SAFE_KEY",
           "PHOTO_SAFE_CORE_CONTRACT", "PHOTO_SAFE_CORE_FILLED_CONTRACT",
           "PHOTO_SAFE_CORE_GUARD_PX", "PHOTO_SAFE_CORE_HOLE_MAX_AREA_PX",
           "photo_safe_core_policy_metadata", "apply_photo_safe_core_enclosed_holes",
           "project_source_photo_safe_core",
           "compute_feature_support",
           "verify_frame_geometry", "derive_frame_context", "V5FrameGeometryCohort"]
