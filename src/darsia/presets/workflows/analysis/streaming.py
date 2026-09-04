"""Utilities for streaming low-resolution analysis previews."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

STREAM_LINE_PREFIX = "__DARSIA_STREAM__:"


def _to_uint8_gray(array: np.ndarray) -> np.ndarray:
    """Map scalar data to uint8 grayscale."""
    scalar = np.asarray(array)
    if scalar.ndim != 2:
        raise ValueError(f"Expected 2D scalar array, got shape {scalar.shape}.")
    scalar = scalar.astype(np.float32, copy=False)
    finite_mask = np.isfinite(scalar)
    if not np.any(finite_mask):
        return np.zeros_like(scalar, dtype=np.uint8)
    finite_values = scalar[finite_mask]
    lower = float(np.min(finite_values))
    upper = float(np.max(finite_values))
    if upper <= lower:
        gray = np.zeros_like(scalar, dtype=np.uint8)
        gray[finite_mask] = 255
        return gray
    normalized = np.zeros_like(scalar, dtype=np.float32)
    normalized[finite_mask] = (scalar[finite_mask] - lower) / (upper - lower)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def _to_bgr_array(image_like: Any) -> np.ndarray:
    """Convert darsia-like image objects or arrays to BGR uint8 arrays."""
    if hasattr(image_like, "to_trichromatic"):
        try:
            bgr_image = image_like.to_trichromatic("BGR", return_image=True)
            bgr_array = np.asarray(bgr_image.img)
            if bgr_array.ndim == 3 and bgr_array.shape[2] >= 3:
                if bgr_array.dtype != np.uint8:
                    bgr_array = cv2.normalize(
                        bgr_array[..., :3],
                        None,
                        alpha=0,
                        beta=255,
                        norm_type=cv2.NORM_MINMAX,
                    ).astype(np.uint8)
                return bgr_array[..., :3]
        except Exception:
            pass

    array = np.asarray(image_like.img if hasattr(image_like, "img") else image_like)
    if array.ndim == 2:
        gray = _to_uint8_gray(array)
        return cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
    if array.ndim == 3 and array.shape[2] == 1:
        gray = _to_uint8_gray(array[..., 0])
        return cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
    if array.ndim == 3 and array.shape[2] >= 3:
        rgb_or_bgr = array[..., :3]
        if rgb_or_bgr.dtype == np.uint8:
            pass
        elif np.issubdtype(rgb_or_bgr.dtype, np.floating):
            if np.nanmin(rgb_or_bgr) >= 0.0 and np.nanmax(rgb_or_bgr) <= 1.0:
                rgb_or_bgr = np.clip(rgb_or_bgr * 255.0, 0, 255).astype(np.uint8)
            else:
                rgb_or_bgr = cv2.normalize(
                    rgb_or_bgr,
                    None,
                    alpha=0,
                    beta=255,
                    norm_type=cv2.NORM_MINMAX,
                ).astype(np.uint8)
        else:
            rgb_or_bgr = cv2.normalize(
                rgb_or_bgr,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
            ).astype(np.uint8)
        return cv2.cvtColor(rgb_or_bgr, cv2.COLOR_RGB2BGR)
    raise ValueError(f"Unsupported image shape for streaming: {array.shape}.")


def encode_low_resolution_png(
    image_like: Any,
    max_width: int = 640,
    max_height: int = 480,
) -> bytes:
    """Encode an image-like object as low-resolution PNG bytes."""
    bgr_array = _to_bgr_array(image_like)
    height, width = bgr_array.shape[:2]
    if width == 0 or height == 0:
        raise ValueError(
            f"Cannot encode an image with zero dimensions: width={width}, "
            f"height={height}."
        )
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1.0:
        bgr_array = cv2.resize(
            bgr_array,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    success, encoded = cv2.imencode(".png", bgr_array)
    if not success:
        raise ValueError("Failed to encode stream image.")
    return encoded.tobytes()


def publish_stream_payload(
    stream_callback: Callable[[dict[str, bytes] | None], None] | None,
    payload: dict[str, bytes],
    logger: logging.Logger,
    error_message: str,
) -> None:
    """Publish a payload and guard against callback errors."""
    if stream_callback is None:
        return
    try:
        stream_callback(payload)
    except Exception:
        logger.exception(error_message)
        try:
            stream_callback(None)
        except Exception:
            pass


def publish_stream_images(
    stream_callback: Callable[[dict[str, bytes] | None], None] | None,
    image_payload: dict[str, Any],
    logger: logging.Logger,
    error_message: str,
) -> None:
    """Encode image payload and publish it via stream callback."""
    if stream_callback is None:
        return
    try:
        encoded_payload = {
            key: encode_low_resolution_png(image)
            for key, image in image_payload.items()
        }
        publish_stream_payload(
            stream_callback=stream_callback,
            payload=encoded_payload,
            logger=logger,
            error_message=error_message,
        )
    except Exception:
        logger.exception(error_message)
        try:
            stream_callback(None)
        except Exception:
            pass


def write_stream_frame(cache_dir: Path, key: str, seq: int, png_bytes: bytes) -> None:
    """Atomically write one stream frame file to cache_dir/key/seq.png.

    One file per (key, seq) is retained (not overwritten) so the Streaming
    Preview panel can scrub back through every frame of a run, not just the
    latest.
    """
    key_dir = cache_dir / key
    key_dir.mkdir(parents=True, exist_ok=True)
    target = key_dir / f"{seq}.png"
    tmp = key_dir / f"{seq}.png.tmp"
    tmp.write_bytes(png_bytes)
    os.replace(tmp, target)


def encode_stream_notify_line(payload: dict[str, bytes] | None, seq: int) -> str:
    """Encode a tiny stdout line noting which keys were just (re)written.

    Carries no image bytes, only key names, so it can never be split across
    a pipe-buffer boundary in a way that matters. payload=None encodes as a
    "clear the stream" notification (keys=None).
    """
    keys = None if payload is None else sorted(payload.keys())
    return f"{STREAM_LINE_PREFIX}{json.dumps({'keys': keys, 'seq': seq})}"


def try_decode_stream_notify_line(line: str) -> tuple[bool, dict[str, Any] | None]:
    """Return (False, None) if line isn't a stream line, else (True, decoded)."""
    if not line.startswith(STREAM_LINE_PREFIX):
        return False, None
    return True, json.loads(line[len(STREAM_LINE_PREFIX) :])


def build_file_cache_stream_callback(
    cache_dir: Path,
    seq_cell: dict[str, int],
) -> Callable[[dict[str, bytes] | None], None]:
    """Return a stream_callback that writes frames to cache_dir and prints a
    notify line to stdout for each publish (flushed immediately, since stdout
    is block-buffered when not attached to a TTY).

    seq_cell (e.g. {"n": 0}) is shared with the caller's progress_callback
    wrapper so the two can be correlated: this callback increments and uses
    it as the on-disk frame id, and the progress wrapper tags its own notify
    line with the same value, since it always fires immediately afterwards
    for the same image.
    """

    def _callback(payload: dict[str, bytes] | None) -> None:
        seq_cell["n"] += 1
        seq = seq_cell["n"]
        if payload is not None:
            for key, png_bytes in payload.items():
                write_stream_frame(cache_dir, key, seq, png_bytes)
        print(encode_stream_notify_line(payload, seq), flush=True)

    return _callback
