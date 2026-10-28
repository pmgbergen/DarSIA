"""Immutable scientific sidecars and separate display assets on the native grid."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from .io import sha256_file
from ..pipeline.io_ops import publish_file, retry_io
from ..pipeline.job_queue import retry_transient


@dataclass(frozen=True)
class CorrectedFrame:
    """One corrected native frame. Display RGB is HWC float32 in [0, 1]; scientific log-
    camera RGB is CHW float32. Optional masks and border values carry separate explicit
    source certificates.
    """

    rgb: np.ndarray
    log_linear_rgb: np.ndarray
    certificate: dict[str, Any]
    extended_log_linear_rgb: np.ndarray | None = None
    required_valid_mask: np.ndarray | None = None
    filled_invalid_mask: np.ndarray | None = None
    display_modes: dict[str, np.ndarray] | None = None
    photo_sand_mask: np.ndarray | None = None
    display_invalid_mask: np.ndarray | None = None


def fixed_float_input_qa(
    log_linear: np.ndarray,
    *,
    required_valid_mask: np.ndarray | None = None,
    parameters: Mapping,
) -> dict[str, Any]:
    """Reject clipping or missing required support before writing scientific features. The
    sensor denominator and allowed invalid fraction are supplied explicitly.
    """
    valid = np.all(np.isfinite(log_linear), axis=0)
    invalid_fraction = float(1 - np.mean(valid))
    if not np.any(valid) or (
        required_valid_mask is None
        and invalid_fraction > parameters["maximum_invalid_fraction"]
    ):
        raise ValueError("Native corrected float source has excessive invalid pixels")
    if required_valid_mask is not None:
        if (
            required_valid_mask.dtype != np.dtype("bool")
            or required_valid_mask.shape != valid.shape
            or (not required_valid_mask.any())
            or np.any(required_valid_mask & ~valid)
        ):
            raise ValueError(
                "Source required photo/sensor support contains invalid float pixels"
            )
    finite_log = log_linear[:, valid]
    maxima_log = np.max(finite_log, axis=1)
    if np.any(maxima_log > np.log(float(parameters["unit_denominator"]))):
        raise ValueError(
            "Native corrected camera-linear signal exceeds 16-bit unit denominator"
        )
    return dict(
        contract=parameters["nonclipping_contract"],
        unit_denominator=parameters["unit_denominator"],
        valid_pixel_fraction=float(np.mean(valid)),
        invalid_pixel_fraction=invalid_fraction,
        max_corrected_linear_by_channel=[float(x) for x in np.exp(maxima_log)],
        max_corrected_linear=float(np.exp(np.max(maxima_log))),
        fixed_input_nonclipping_pass=True,
        invalid_policy=(
            parameters["exact_valid_mask_policy"]
            if required_valid_mask is not None
            else parameters["legacy_fraction_policy"]
        ),
        required_valid_pixels=(
            int(np.count_nonzero(required_valid_mask))
            if required_valid_mask is not None
            else None
        ),
    )


def raw_detail_display(log_linear: np.ndarray, *, parameters: Mapping) -> np.ndarray:
    """Render a fixed log-range detail view for inspection only; invalid scientific pixels
    appear white without changing the input.
    """
    low, high = (
        np.log(parameters["raw_detail_floor"]),
        np.log(parameters["raw_detail_ceiling"]),
    )
    normalized = np.clip(
        (np.nan_to_num(log_linear, nan=low) - low) / (high - low), 0, 1
    )
    rgb = np.moveaxis(
        np.power(normalized, parameters["raw_detail_exponent"]), 0, -1
    ).astype(np.float32)
    rgb[~np.isfinite(log_linear).all(axis=0)] = parameters["raw_detail_invalid"]
    return rgb


def write_log_linear(path: Path, values: np.ndarray) -> str:
    """Retain the exact full-resolution scientific signal separately from JPEG."""
    def write(stream):
        np.save(stream, np.ascontiguousarray(values), allow_pickle=False)
    return publish_file(path, write, reuse_identical=True)


def copy_verified(source: Path, destination: Path, expected_sha256: str) -> str:
    """Copy a frozen timeline output with atomic receipt-safe resume."""
    if sha256_file(source) != expected_sha256:
        raise ValueError(
            f"Frozen timeline output changed before baseline copy: {source}"
        )
    if retry_transient(destination.exists):
        if sha256_file(destination) != expected_sha256:
            raise ValueError(
                f"Unreceipted baseline copy differs from source: {destination}"
            )
        return expected_sha256
    def copy(outgoing):
        with source.open("rb") as incoming:
            shutil.copyfileobj(incoming, outgoing, length=1 << 20)
    return publish_file(destination, copy, reuse_identical=True,
                        expected_sha256=expected_sha256)


def write_display_jpeg(
    rgb: np.ndarray,
    *,
    run: str,
    index: int,
    count: int,
    elapsed_h: float,
    canvas_wh: tuple[int, int],
    destination: Path,
    parameters: Mapping,
    renderer: Callable,
) -> str:
    """Render and write the original review canvas through an injected renderer. Preserve
    native dimensions and keep the photograph separate from scientific sidecars.
    """
    canvas = renderer(
        rgb,
        run=run,
        index=index,
        count=count,
        elapsed_h=elapsed_h,
        width=canvas_wh[0],
        height=canvas_wh[1],
        native=True,
    )
    if canvas.shape[:2] != tuple(reversed(canvas_wh)):
        raise ValueError("Renderer changed the native review canvas geometry")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.pending")
    try:
        with temporary.open("xb") as stream:
            Image.fromarray(np.asarray(canvas, np.uint8), "RGB").save(
                stream,
                format=parameters["jpeg_format"],
                quality=parameters["jpeg_quality"],
                subsampling=parameters["jpeg_subsampling"],
            )
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(temporary)
        if destination.exists():
            if sha256_file(destination) != digest:
                raise ValueError(
                    f"Unreceipted Native JPEG differs from recomputation: {destination}"
                )
        else:
            os.replace(temporary, destination)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def write_display_invalid_mask(
    path: Path, mask: np.ndarray, *, parameters: Mapping
) -> str:
    """Retain the pre-cleanup full-frame invalid support in a compact archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.pending")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(
                stream, **{parameters["mask_archive_key"]: np.ascontiguousarray(mask)}
            )
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(temporary)
        if path.exists():
            if sha256_file(path) != digest:
                raise ValueError(f"Unreceipted display-invalid mask differs: {path}")
        else:
            os.replace(temporary, path)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def auxiliary_output_fields(record: Mapping, *, parameters: Mapping) -> dict:
    """Select only explicitly configured optional sidecar fields from a receipt."""
    return {
        key: record[key] for key in parameters["field_keys"].values() if key in record
    }


def write_auxiliary_outputs(
    frame: CorrectedFrame,
    target: Path,
    native_sha256: str,
    *,
    run: str,
    index: int,
    count: int,
    elapsed_h: float,
    canvas_wh: tuple[int, int],
    renderer: Callable,
    parameters: Mapping,
) -> dict:
    """Write only optional masks and views actually returned by the processor.

    Receipt field names and view names are supplied as data. The base photograph
    is already published, so its digest is referenced rather than rewritten.
    """
    output = {}
    keys = parameters["field_keys"]
    for field, values in (
        ("photo_sand", frame.photo_sand_mask),
        ("filled_invalid", frame.filled_invalid_mask),
    ):
        if values is not None:
            path = target.with_name(f"{target.stem}.{field}_mask.npy")
            output[keys[field + "_mask_path"]] = str(path)
            output[keys[field + "_mask_sha256"]] = write_log_linear(path, values)
    if frame.display_modes is not None:
        invalid_path = target.with_name(f"{target.stem}.display_invalid_mask.npz")
        output[keys["display_invalid_mask_path"]] = str(invalid_path)
        output[keys["display_invalid_mask_sha256"]] = write_display_invalid_mask(
            invalid_path, frame.display_invalid_mask, parameters=parameters
        )
        modes = {
            parameters["base_mode"]: dict(
                jpeg_path=str(target), jpeg_sha256=native_sha256
            )
        }
        for mode in parameters["extra_modes"]:
            path = target.with_name(f"{target.stem}.{mode}.jpg")
            digest = write_display_jpeg(
                frame.display_modes[mode],
                run=run,
                index=index,
                count=count,
                elapsed_h=elapsed_h,
                canvas_wh=canvas_wh,
                destination=path,
                renderer=renderer,
                parameters=parameters,
            )
            modes[mode] = dict(jpeg_path=str(path), jpeg_sha256=digest)
        output[keys["display_modes"]] = modes
    return output


def split_extended_ring(
    padded_log: np.ndarray,
    native_log: np.ndarray,
    *,
    pad_px: int,
    maximum_center_delta_log: float,
    parameters: Mapping,
) -> tuple[dict[str, np.ndarray], float]:
    """Extract four exact RAW strips after checking native/padded parity."""
    padded = np.asarray(padded_log)
    native = np.asarray(native_log)
    pad = int(pad_px)
    if (
        pad < 1
        or padded.dtype != np.float32
        or native.dtype != np.float32
        or (native.ndim != 3)
        or (native.shape[0] != 3)
        or (padded.shape != (3, native.shape[1] + 2 * pad, native.shape[2] + 2 * pad))
    ):
        raise ValueError("Extended RAW reference has an invalid shape or dtype")
    if np.isinf(padded).any() or np.isinf(native).any():
        raise ValueError("Extended RAW reference contains infinite signal")
    center = padded[:, pad:-pad, pad:-pad]
    if not np.array_equal(np.isfinite(center), np.isfinite(native)):
        raise ValueError("Padded Source correction changed native source validity")
    finite = np.isfinite(native)
    difference = (
        float(np.max(np.abs(center[finite] - native[finite])))
        if np.any(finite)
        else 0.0
    )
    if difference > maximum_center_delta_log:
        raise ValueError("Padded Source correction changed the native source centre")
    strips = {
        "top": np.ascontiguousarray(padded[:, :pad, :]),
        "bottom": np.ascontiguousarray(padded[:, -pad:, :]),
        "left": np.ascontiguousarray(padded[:, pad:-pad, :pad]),
        "right": np.ascontiguousarray(padded[:, pad:-pad, -pad:]),
    }
    return (strips, difference)


def write_extended_ring(
    path: Path,
    padded_log: np.ndarray,
    native_log: np.ndarray,
    *,
    pad_px: int,
    parameters: Mapping,
) -> tuple[str, float]:
    """Write one uncompressed, same-volume atomic ring and return its SHA."""
    path = Path(path)
    strips, difference = split_extended_ring(
        padded_log,
        native_log,
        pad_px=pad_px,
        maximum_center_delta_log=parameters["maximum_center_delta_log"],
        parameters=parameters,
    )
    def write(stream):
        np.savez(stream, **strips, pad_px=np.asarray(pad_px, np.int16),
                 contract=np.asarray(parameters["contract"]))
    digest = publish_file(path, write, reuse_identical=True)
    return digest, difference


def assemble_extended_log(
    native_log: np.ndarray,
    ring_path: Path,
    *,
    ring_sha256: str,
    expected_pad_px: int,
    parameters: Mapping,
) -> np.ndarray:
    """Rebuild the padded log signal from the exact native centre and strips."""
    core = np.asarray(native_log)
    if (
        core.dtype != np.float32
        or core.ndim != 3
        or core.shape[0] != 3
        or np.isinf(core).any()
    ):
        raise ValueError("Native native reference core is not float32 log RGB")
    path = Path(ring_path)
    if sha256_file(path) != ring_sha256:
        raise ValueError("Native extended RAW ring changed")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "top",
            "bottom",
            "left",
            "right",
            "pad_px",
            "contract",
        }:
            raise ValueError("Native extended RAW ring schema changed")
        pad = int(archive["pad_px"])
        if (
            pad != int(expected_pad_px)
            or str(archive["contract"]) != parameters["contract"]
        ):
            raise ValueError("Native extended RAW ring has another padding contract")
        height, width = core.shape[1:]
        expected = dict(
            top=(3, pad, width + 2 * pad),
            bottom=(3, pad, width + 2 * pad),
            left=(3, height, pad),
            right=(3, height, pad),
        )
        strips = {}
        for key, shape in expected.items():
            value = np.asarray(archive[key])
            if (
                value.shape != shape
                or value.dtype != np.float32
                or np.isinf(value).any()
            ):
                raise ValueError(f"Native RAW {key} strip is invalid")
            strips[key] = value
    result = np.empty((3, height + 2 * pad, width + 2 * pad), np.float32)
    result[:, :pad, :] = strips["top"]
    result[:, -pad:, :] = strips["bottom"]
    result[:, pad:-pad, :pad] = strips["left"]
    result[:, pad:-pad, -pad:] = strips["right"]
    result[:, pad:-pad, pad:-pad] = core
    return result


def write_frame_ring(
    path: Path,
    frame: CorrectedFrame,
    native_log: np.ndarray,
    *,
    pad_px: int,
    parameters: Mapping,
) -> dict[str, Any]:
    """Seal only RAW-observed border strips; retain their invalidity mask."""
    padded = frame.extended_log_linear_rgb
    if padded is None:
        raise ValueError("Source processor did not return a source-backed RAW border")
    padded = np.asarray(padded)
    valid = np.isfinite(padded).all(axis=0)
    if not np.any(valid) or np.isinf(padded).any():
        raise ValueError("Source extended RAW source has no valid sensor signal")
    if np.max(padded[:, valid]) > np.log(float(parameters["unit_denominator"])):
        raise ValueError("Source extended RAW source exceeds fixed input denominator")
    digest, difference = write_extended_ring(
        path, padded, native_log, pad_px=pad_px, parameters=parameters
    )
    return _ring_metadata(
        parameters,
        contract=parameters["contract"],
        extended_ring_path=str(path),
        extended_ring_sha256=digest,
        extended_pad_px=pad_px,
        extended_origin_xy=[-pad_px, -pad_px],
        extended_native_center_max_abs_delta_log=difference,
        extended_valid_pixels=int(np.count_nonzero(valid)),
        extended_invalid_pixels=int(np.count_nonzero(~valid)),
        extended_valid_fraction=float(np.mean(valid)),
        extended_max_corrected_linear=float(np.exp(np.max(padded[:, valid]))),
        invalid_sensor_pixels_preserved=True,
    )


def _ring_metadata(parameters: Mapping, **fields) -> dict:
    keys = parameters["field_keys"]
    return {keys.get(key, key): value for key, value in fields.items()}
