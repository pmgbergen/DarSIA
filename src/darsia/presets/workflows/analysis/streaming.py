"""Utilities for streaming low-resolution analysis previews."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np


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
            key: encode_low_resolution_png(image) for key, image in image_payload.items()
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
