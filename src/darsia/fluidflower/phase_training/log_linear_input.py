"""Exact, source-bound v5 camera-linear input for optical features.

The display JPEG is intentionally not an input here. The sidecar stores
full-native, black-subtracted, corrected camera-linear RGB in log units.
Its unit encoding is fixed by the trained model and shared by fitting and
inference. Invalid sensor pixels carry no optical evidence.
"""

from __future__ import annotations
from darsia.fluidflower.native.config import require_parameters

import hashlib
import json
import math
from pathlib import Path
import re
from typing import MutableMapping

import numpy as np


MAX_INVALID_PIXEL_FRACTION = 0.001
# Provisional until the source-exact full-native B/H range gate passes. The
# trained model and native manifest must both declare this exact value.
V5_UNIT_DENOMINATOR = 65535.0
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
MODEL_FORMAT = "ffac-v5-raw-full-native-training-v1"
SOURCE_FORMAT = "ffac-v5-native-source-materialization-v1"
EXACT_VALID_MASK_POLICY = "exact_per_frame_sha_mask_v1"
_CONFIGURED_VALID_MASK = object()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_float_support(
    valid: np.ndarray, *, invalid_policy: str | None,
    valid_mask_path: str | Path | None, valid_mask_sha256: str | None,
    required_valid_mask: np.ndarray | None = None,
) -> tuple[int, float]:
    """Check one exact finite-pixel support without inferring missing signal."""
    height, width = valid.shape
    bad_count = int(valid.size - np.count_nonzero(valid))
    bad_fraction = bad_count / valid.size
    if invalid_policy is None:
        if valid_mask_path is not None or valid_mask_sha256 is not None:
            raise ValueError("V5 valid mask requires an explicit invalid policy")
        if bad_fraction > MAX_INVALID_PIXEL_FRACTION:
            raise ValueError(f"Too many invalid v5 pixels: {bad_count}/{valid.size}")
    elif invalid_policy == EXACT_VALID_MASK_POLICY:
        if valid_mask_path is None or valid_mask_sha256 is None:
            raise ValueError("V5 exact valid mask path and SHA-256 are required")
        if not _SHA256.fullmatch(valid_mask_sha256):
            raise ValueError("Expected a 64-hex v5 valid mask SHA-256 digest")
        mask_path = Path(valid_mask_path)
        if _sha256(mask_path) != valid_mask_sha256.lower():
            raise ValueError(f"V5 valid mask SHA-256 mismatch: {mask_path}")
        bound_valid = np.load(mask_path, allow_pickle=False, mmap_mode="r")
        if bound_valid.shape != (height, width) or bound_valid.dtype != np.dtype("bool"):
            raise ValueError("V5 valid mask must be native HW bool")
        if not np.array_equal(valid, bound_valid):
            raise ValueError("V5 valid mask differs from finite scientific input")
    else:
        raise ValueError("Unknown v5 invalid-pixel policy")
    if not np.any(valid):
        raise ValueError("V5 scientific input has no valid pixels")
    if required_valid_mask is not None:
        required = np.asarray(required_valid_mask)
        if required.shape != (height, width) or required.dtype != np.bool_:
            raise ValueError("Required-valid mask must be native HW bool")
        if np.any(required & ~valid):
            raise ValueError("Invalid v5 pixels intersect porous domain or eligible labels")
    return bad_count, bad_fraction


def decode_log_linear(
    path: str | Path,
    *,
    expected_sha256: str | None,
    content_shape_hw: tuple[int, int],
    unit_denominator: float = V5_UNIT_DENOMINATOR,
    diagnostics: MutableMapping | None = None,
    required_valid_mask: np.ndarray | None = None,
    invalid_policy: str | None = None,
    valid_mask_path: str | Path | None = None,
    valid_mask_sha256: str | None = None,
) -> np.ndarray:
    """Decode CHW float32 log RAW to HWC float32 unit RGB without clipping.

    ``expected_sha256=None`` is allowed only for a separately SHA-verified
    local staging asset. The NAS inference caller must pass its manifest SHA.
    Legacy revisions allow at most 0.1% invalid content pixels. A newer
    revision may declare ``exact_per_frame_sha_mask_v1`` and bind a native-HW
    bool mask whose values exactly equal finite RAW support. This permits
    source-backed, non-sand invalid margins without treating transport zeros
    as optical evidence. Callers should also provide the porous-domain and
    eligible-label union in ``required_valid_mask`` during preflight.
    Values above one are rejected instead of silently reinterpreted by the
    generic optical feature normalizer.
    """
    source = Path(path)
    if expected_sha256 is not None:
        if not _SHA256.fullmatch(expected_sha256):
            raise ValueError("Expected a 64-hex SHA-256 digest")
        if _sha256(source) != expected_sha256.lower():
            raise ValueError(f"V5 linear input SHA-256 mismatch: {source}")
    if len(content_shape_hw) != 2 or any(int(v) <= 0 for v in content_shape_hw):
        raise ValueError("Invalid native content shape")
    height, width = map(int, content_shape_hw)
    if not math.isfinite(unit_denominator) or unit_denominator <= 0:
        raise ValueError("Invalid v5 unit encoding denominator")
    log = np.load(source, allow_pickle=False, mmap_mode="r")
    if log.shape != (3, height, width) or log.dtype != np.dtype("float32"):
        raise ValueError(f"Wrong v5 linear shape/dtype: {log.shape}, {log.dtype}")
    valid = np.isfinite(log).all(axis=0)
    bad_count, bad_fraction = _validate_float_support(
        valid, invalid_policy=invalid_policy,
        valid_mask_path=valid_mask_path,
        valid_mask_sha256=valid_mask_sha256,
        required_valid_mask=required_valid_mask)
    rgb = np.empty((height, width, 3), np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        for channel in range(3):
            np.exp(log[channel], out=rgb[..., channel])
            rgb[..., channel] /= np.float32(unit_denominator)
    if not np.isfinite(rgb[valid]).all():
        raise ValueError("V5 linear input overflows after exponentiation")
    max_value = float(np.max(rgb[valid])) if np.any(valid) else 0.0
    if max_value > 1.0:
        raise ValueError(
            f"V5 linear encoding exceeds unit interval ({max_value:.6g}); "
            "choose one fixed denominator before training"
        )
    rgb[~valid] = 0.0
    if diagnostics is not None:
        diagnostics.update(
            path=str(source), sidecar_sha256=expected_sha256,
            native_shape_hw=[height, width], invalid_pixels=bad_count,
            invalid_pixel_fraction=bad_fraction,
            unit_denominator=float(unit_denominator), max_unit_value=max_value,
            invalid_fill="zero-all-channels",
            invalid_policy=invalid_policy or "legacy_fraction_v1",
            valid_mask_sha256=valid_mask_sha256,
        )
    return rgb


def decode_extended_log_linear(
    core_path: str | Path, ring_path: str | Path, *,
    core_sha256: str, ring_sha256: str,
    content_shape_hw: tuple[int, int], pad_px: int,
    unit_denominator: float = V5_UNIT_DENOMINATOR,
    invalid_policy: str | None = None,
    valid_mask_path: str | Path | None = None,
    valid_mask_sha256: str | None = None,
) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    """Decode an exact native float core plus its source-backed RAW border.

    Invalid border locations retain a false support mask. Their RGB zero is
    only a transport value; registered optical sampling must reject every
    destination whose bilinear footprint touches one.
    """
    from darsia.fluidflower.native.sidecar import (
        assemble_extended_log,
    )

    if not isinstance(pad_px, int) or not 1 <= pad_px <= 256:
        raise ValueError("V5 extended reference has invalid native padding")
    native_rgb = decode_log_linear(
        core_path, expected_sha256=core_sha256,
        content_shape_hw=content_shape_hw,
        unit_denominator=unit_denominator,
        invalid_policy=invalid_policy,
        valid_mask_path=valid_mask_path,
        valid_mask_sha256=valid_mask_sha256)
    core_log = np.load(Path(core_path), allow_pickle=False, mmap_mode="r")
    padded_log = assemble_extended_log(core_log, Path(ring_path),
        ring_sha256=ring_sha256, expected_pad_px=pad_px, parameters=require_parameters("extended_ring"))
    valid = np.isfinite(padded_log).all(axis=0)
    rgb = np.zeros((*valid.shape, 3), np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        for channel in range(3):
            rgb[..., channel] = np.exp(np.where(
                valid, padded_log[channel], np.float32(0))) / np.float32(
                    unit_denominator)
    if (not np.isfinite(rgb[valid]).all() or np.any(rgb[valid] < 0)
            or np.any(rgb[valid] > 1)):
        raise ValueError("V5 extended RAW signal exceeds its fixed float encoding")
    rgb[~valid] = 0.0
    if not np.array_equal(rgb[pad_px:-pad_px, pad_px:-pad_px], native_rgb):
        raise ValueError("V5 ring reconstruction changed its native centre RGB")
    return rgb, (-pad_px, -pad_px), valid


def model_v5_settings(model_root: str | Path) -> dict | None:
    """Return the explicit v5 model contract; never infer it from an image."""
    request = Path(model_root) / "training_request.json"
    if not request.is_file():
        return None
    document = json.loads(request.read_text(encoding="utf-8"))
    value = (document.get("settings") or {}).get("v5_raw_photometry")
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("format") != MODEL_FORMAT:
        raise ValueError("Unknown v5 raw photometry model format")
    denominator = value.get("unit_denominator")
    if denominator is None or float(denominator) != V5_UNIT_DENOMINATOR:
        raise ValueError("V5 model unit denominator differs from decoder")
    return dict(value)


def runtime_identity() -> dict:
    return dict(kind="source-bound-v5-log-linear-input-v1",
                decoder_sha256=_sha256(Path(__file__)),
                unit_denominator=V5_UNIT_DENOMINATOR,
                source_format=SOURCE_FORMAT,
                model_format=MODEL_FORMAT)


class RunV5LinearInput:
    """One exact native-review revision's verified float optical input.

    The object has the ``load(photo_path, frame_id)`` interface used by the
    serial, shared-worker and CUDA-lookahead map runners. It checks both the
    materialized display JPEG and scientific float sidecar against the same
    native manifest on each read; there is no JPEG inference fallback.
    """

    def __init__(self, *, native_manifest_path: str | Path, run_id: str,
                 content_shape_hw: tuple[int, int],
                 unit_denominator: float = V5_UNIT_DENOMINATOR,
                 required_valid_mask: np.ndarray | None = None):
        manifest_path = Path(native_manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("v5_photometry") or {}).get("format") != SOURCE_FORMAT:
            raise ValueError("Native source is not a complete v5 float revision")
        if (manifest.get("v5_photometry") or {}).get("unit_denominator") != V5_UNIT_DENOMINATOR:
            raise ValueError("Native v5 source declares another unit denominator")
        if (manifest.get("v5_photometry") or {}).get("display_jpeg_is_training_input") is not False:
            raise ValueError("Native v5 source does not declare display-only JPEG")
        self.review_root = manifest_path.parent.parent.parent
        self.run_id = str(run_id).casefold()
        if manifest_path.parent.name.casefold() != self.run_id:
            raise ValueError("V5 manifest belongs to another run")
        self.manifest_path = manifest_path
        self.manifest_sha256 = _sha256(manifest_path)
        self.content_shape_hw = tuple(map(int, content_shape_hw))
        box = manifest.get("content_box_xywh") or []
        if len(box) != 4 or tuple(map(int, box[2:][::-1])) != self.content_shape_hw:
            raise ValueError("V5 manifest content grid mismatch")
        self.unit_denominator = float(unit_denominator)
        if self.unit_denominator != V5_UNIT_DENOMINATOR:
            raise ValueError("V5 manifest/model unit denominator mismatch")
        self.required_valid_mask = required_valid_mask
        self.reference = manifest.get("reference") or {}
        self.invalid_policy = manifest.get("v5_float_support_policy")
        if self.invalid_policy not in (None, EXACT_VALID_MASK_POLICY):
            raise ValueError("Unknown native v5 float support policy")
        rows = manifest.get("frame_sources") or []
        if len(rows) != int(manifest.get("frame_count") or -1):
            raise ValueError("V5 frame source count mismatch")
        self.frames = {}
        for index, row in enumerate(rows):
            if int(row.get("frame_index", -1)) != index:
                raise ValueError("V5 frame source order changed")
            self.frames[f"frame_{index:03d}"] = row
        self.receipts = []

    def load(self, photo_path: str | Path, frame_id: str, *,
             required_valid_mask: np.ndarray | None | object = _CONFIGURED_VALID_MASK
             ) -> np.ndarray:
        """Decode this exact source, optionally with its own frame-local domain.

        V5 geometry can move inside one run.  A caller that has verified a
        frame receipt passes its current porous/eligible mask here; the
        original configured mask remains the default for older callers.
        """
        record, photo_sha, sidecar, sidecar_sha, mask_path, mask_sha = (
            self._bound_source(photo_path, frame_id))
        report = {}
        required = (self.required_valid_mask if required_valid_mask is _CONFIGURED_VALID_MASK
                    else required_valid_mask)
        rgb = decode_log_linear(sidecar, expected_sha256=sidecar_sha,
                                content_shape_hw=self.content_shape_hw,
                                unit_denominator=self.unit_denominator,
                                required_valid_mask=required,
                                invalid_policy=self.invalid_policy,
                                valid_mask_path=mask_path,
                                valid_mask_sha256=mask_sha,
                                diagnostics=report)
        self.receipts.append(dict(frame_id=frame_id, photo_sha256=photo_sha,
                                  native_manifest_sha256=self.manifest_sha256, **report))
        return rgb

    def _bound_source(self, photo_path: str | Path, frame_id: str
                      ) -> tuple[dict, str, Path, str, Path | None, str | None]:
        """Verify revision paths and display hash without decoding a second time."""
        photo = Path(photo_path)
        if frame_id == "pre_injection":
            expected_photo = (self.review_root / "native_references" /
                              self.run_id / "pre_injection.jpg")
            record = self.reference
            photo_sha = record.get("materialized_native_sha256")
        else:
            if frame_id not in self.frames:
                raise ValueError(f"Unknown v5 source frame: {frame_id}")
            expected_photo = (self.review_root / "native_frames" /
                              self.run_id / f"{frame_id}.jpg")
            record = self.frames[frame_id]
            photo_sha = record.get("native_jpeg_sha256")
        if photo.resolve() != expected_photo.resolve():
            raise ValueError("V5 photo path is not the bound native revision")
        if not photo_sha or _sha256(photo) != photo_sha:
            raise ValueError("V5 display JPEG differs from the native manifest")
        sidecar_path = record.get("v5_log_linear_rgb_path")
        sidecar_sha = record.get("v5_log_linear_rgb_sha256")
        if not sidecar_path or not sidecar_sha:
            raise ValueError("V5 native scientific sidecar is missing")
        sidecar = Path(sidecar_path)
        if not sidecar.resolve().is_relative_to(self.review_root.resolve()):
            raise ValueError("V5 scientific sidecar is outside the native revision")
        expected_sidecar = expected_photo.with_suffix(".log_linear_rgb.npy")
        if sidecar.resolve() != expected_sidecar.resolve():
            raise ValueError("V5 scientific sidecar is not next to its exact frame")
        mask_path, mask_sha = self._bound_valid_mask(record, expected_photo)
        return record, photo_sha, sidecar, sidecar_sha, mask_path, mask_sha

    def _bound_valid_mask(self, record: dict, photo: Path) -> tuple[Path | None, str | None]:
        mask_value = record.get("v5_log_linear_valid_mask_path")
        mask_sha = record.get("v5_log_linear_valid_mask_sha256")
        if self.invalid_policy is None:
            if mask_value is not None or mask_sha is not None:
                raise ValueError("V5 valid mask requires a native support policy")
            return None, None
        if not mask_value or not isinstance(mask_sha, str):
            raise ValueError("V5 native valid mask is missing")
        mask_path = Path(mask_value)
        if not mask_path.resolve().is_relative_to(self.review_root.resolve()):
            raise ValueError("V5 native valid mask is outside its source revision")
        expected = photo.with_suffix(".log_linear_valid_mask.npy")
        if mask_path.resolve() != expected.resolve():
            raise ValueError("V5 native valid mask is not next to its exact frame")
        return mask_path, mask_sha

    def _bound_extended_ring(self, record: dict, photo: Path
                             ) -> tuple[Path, str, int, list[int]]:
        ring_path = Path(record.get("v5_extended_ring_path") or "")
        ring_sha = record.get("v5_extended_ring_sha256")
        pad = record.get("v5_extended_pad_px")
        origin = record.get("v5_extended_origin_xy")
        if (not isinstance(pad, int) or not 1 <= pad <= 256
                or origin != [-pad, -pad]
                or not isinstance(ring_sha, str) or not _SHA256.fullmatch(ring_sha)
                or not ring_path.resolve().is_relative_to(self.review_root.resolve())
                or ring_path.resolve() != photo.with_suffix(".extended_ring.npz").resolve()):
            raise ValueError("V5 reference lacks its exact source-backed border ring")
        return ring_path, ring_sha, pad, origin

    def load_validity(self, photo_path: str | Path, frame_id: str, *,
                      extended: bool = False
                      ) -> tuple[np.ndarray, tuple[int, int]]:
        """Verify source SHA and return optical support without decoding RGB.

        Frame-context preparation needs finite support to constrain temporal
        warps. It must not exponentiate and discard three full-native channels
        for every current/baseline/previous frame. Model inference still
        decodes the same bound file before using any pixel value.
        """
        from darsia.fluidflower.native.sidecar import (
            assemble_extended_log,
        )

        record, _photo_sha, core_path, core_sha, mask_path, mask_sha = (
            self._bound_source(photo_path, frame_id))
        if _sha256(core_path) != core_sha:
            raise ValueError(f"V5 linear input SHA-256 mismatch: {core_path}")
        core_log = np.load(core_path, allow_pickle=False, mmap_mode="r")
        height, width = self.content_shape_hw
        if (core_log.shape != (3, height, width)
                or core_log.dtype != np.dtype("float32")):
            raise ValueError("Wrong v5 linear shape/dtype")
        core_valid = np.isfinite(core_log).all(axis=0)
        _validate_float_support(core_valid, invalid_policy=self.invalid_policy,
                                valid_mask_path=mask_path,
                                valid_mask_sha256=mask_sha)
        if not extended:
            return core_valid, (0, 0)
        ring_path, ring_sha, pad, _origin = self._bound_extended_ring(
            record, Path(photo_path))
        padded_log = assemble_extended_log(
            core_log, ring_path, ring_sha256=ring_sha,
            expected_pad_px=pad, parameters=require_parameters("extended_ring"))
        padded_valid = np.isfinite(padded_log).all(axis=0)
        if not np.array_equal(padded_valid[pad:-pad, pad:-pad], core_valid):
            raise ValueError("V5 extended source validity changed native centre")
        return padded_valid, (-pad, -pad)

    def load_extended(self, photo_path: str | Path, frame_id: str,
                      ) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
        """Read source-backed B/previous border, retaining its validity mask.

        Invalid RAW sensor pixels remain invalid in the returned mask.  The
        RGB array uses zero only as a transport value for those locations;
        registered alignment must require four valid source neighbours for
        every requested destination pixel.  A display JPEG is never used.
        """
        record, photo_sha, core_path, core_sha, mask_path, mask_sha = (
            self._bound_source(photo_path, frame_id))
        photo = Path(photo_path)
        ring_path, ring_sha, pad, origin = self._bound_extended_ring(record, photo)
        rgb, native_origin, valid = decode_extended_log_linear(
            core_path, ring_path, core_sha256=core_sha,
            ring_sha256=ring_sha, content_shape_hw=self.content_shape_hw,
            pad_px=pad, unit_denominator=self.unit_denominator,
            invalid_policy=self.invalid_policy,
            valid_mask_path=mask_path,
            valid_mask_sha256=mask_sha)
        self.receipts.append(dict(frame_id=frame_id, photo_sha256=photo_sha,
            native_manifest_sha256=self.manifest_sha256,
            path=str(core_path), sidecar_sha256=core_sha,
            invalid_policy=self.invalid_policy or "legacy_fraction_v1",
            valid_mask_sha256=mask_sha,
            extended_ring_path=str(ring_path),
            extended_ring_sha256=ring_sha, extended_pad_px=pad,
            extended_origin_xy=origin, extended_invalid_pixels=int(np.count_nonzero(~valid)),
            extended_valid_fraction=float(np.mean(valid))))
        return rgb, native_origin, valid

    def report(self) -> dict:
        return dict(runtime=runtime_identity(), run=self.run_id,
                    native_manifest_path=str(self.manifest_path),
                    native_manifest_sha256=self.manifest_sha256,
                    frames=self.receipts)
