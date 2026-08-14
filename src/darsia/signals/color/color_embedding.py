"""Color embedding abstractions and shared canonical transform utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

import darsia

if TYPE_CHECKING:
    from darsia.presets.workflows.rig import Rig


class ColorEmbeddingBasis(StrEnum):
    """Label space used by color embedding, calibration, and analysis workflows."""

    LABELS = "labels"
    FACIES = "facies"
    GLOBAL = "global"


def parse_color_embedding_basis(
    value: str | ColorEmbeddingBasis | None,
    default: ColorEmbeddingBasis = ColorEmbeddingBasis.FACIES,
) -> ColorEmbeddingBasis:
    """Parse user/config input into a :class:`ColorEmbeddingBasis`."""

    if value is None:
        return default
    if isinstance(value, ColorEmbeddingBasis):
        return value
    if isinstance(value, str):
        token = value.lower().strip()
        return ColorEmbeddingBasis(token)
    raise TypeError(f"Unsupported color embedding basis value type: {type(value)}")


def calibration_basis_folder(basis: str | ColorEmbeddingBasis) -> str:
    """Return standard folder suffix for basis-aware calibration artifacts."""

    parsed = parse_color_embedding_basis(basis)
    return f"from_{parsed.value}"


@dataclass
class ColorEmbeddingRuntime:
    """Runtime context for color embedding execution."""

    rig: "Rig"
    cache: dict[str, "ColorEmbeddingTransform"] = field(default_factory=dict)


class ColorEmbeddingTransform(ABC):
    """Canonical transform converting 3D color to 1D scalar signal."""

    @abstractmethod
    def __call__(self, image: darsia.Image) -> darsia.ScalarImage:
        raise NotImplementedError


class ColorEmbedding(ABC):
    """Abstract color embedding descriptor."""

    embedding_id: str
    mode: darsia.ColorMode
    basis: ColorEmbeddingBasis

    @abstractmethod
    def canonical_transform(
        self, runtime: ColorEmbeddingRuntime
    ) -> ColorEmbeddingTransform:
        raise NotImplementedError

    def to_scalar_image(
        self, image: darsia.Image, runtime: ColorEmbeddingRuntime
    ) -> darsia.ScalarImage:
        if self.embedding_id not in runtime.cache:
            runtime.cache[self.embedding_id] = self.canonical_transform(runtime)
        return runtime.cache[self.embedding_id](image)

    def get_labels(self, runtime: ColorEmbeddingRuntime | "Rig") -> darsia.Image:
        if self.basis == ColorEmbeddingBasis.GLOBAL:
            if isinstance(runtime, ColorEmbeddingRuntime):
                return darsia.zeroes_like(runtime.rig.baseline, dtype=int)
            else:
                return darsia.zeroes_like(runtime.baseline, dtype=int)
        elif self.basis == ColorEmbeddingBasis.FACIES:
            if isinstance(runtime, ColorEmbeddingRuntime):
                return runtime.rig.facies
            else:
                return runtime.facies
        elif self.basis == ColorEmbeddingBasis.LABELS:
            if isinstance(runtime, ColorEmbeddingRuntime):
                return runtime.rig.labels
            else:
                return runtime.labels
        else:
            raise ValueError(f"Unsupported color embedding basis '{self.basis}'.")


def to_scalar_image(template: darsia.Image, values: np.ndarray) -> darsia.ScalarImage:
    metadata = template.metadata()
    metadata.pop("color_space", None)
    return darsia.ScalarImage(img=values, **metadata)


def channel_index(color_space: str, channel: str) -> int:
    channel_map = {
        "RGB": {"r": 0, "g": 1, "b": 2},
        "BGR": {"b": 0, "g": 1, "r": 2},
        "HSV": {"h": 0, "s": 1, "v": 2},
        "HLS": {"h": 0, "l": 1, "s": 2},
        "LAB": {"l": 0, "a": 1, "b": 2},
    }
    mapping = channel_map.get(color_space.upper(), {})
    if channel not in mapping:
        raise ValueError(
            f"Unsupported channel '{channel}' for color space '{color_space}'."
        )
    return mapping[channel]


def normalized_trichromatic(
    image: darsia.Image,
    color_space: str,
    mode: darsia.ColorMode,
    baseline: darsia.Image | None = None,
) -> tuple[np.ndarray, str]:
    if not isinstance(image, darsia.OpticalImage):
        raise ValueError("Color-based modes require an optical image input.")

    cs = color_space.upper().strip()
    if mode == darsia.ColorMode.RELATIVE:
        if baseline is None:
            raise ValueError("Relative color embedding requires baseline image.")
        if cs not in {"RGB", "BGR"}:
            raise NotImplementedError(
                f"Relative mode is currently only supported for RGB/BGR, not '{cs}'."
            )
        arr = image.img.astype(np.float32) - baseline.img.astype(np.float32)
        if np.max(np.abs(arr)) > 1.0:
            arr = arr / 255.0
        return arr, cs

    converted = image.to_trichromatic(cs, return_image=True)
    arr = converted.img.astype(np.float32, copy=False)
    if cs in {"RGB", "BGR"}:
        # NOTE: Scaling in RGB mode not needed, as the to_trichromatic conversion
        # already normalizes to [0, 1].
        if arr.dtype in [int, np.uint8, np.uint16]:
            arr = arr / 255.0
    elif cs in {"HSV", "HLS"}:
        hue = arr[..., 0]
        hue_scale = 360.0 if np.max(hue) > 180.0 and np.min(hue) >= 0.0 else 180.0
        arr[..., 0] = hue / hue_scale
        # NOTE: Only scaling of hue channel needed, as saturation and value/lightness
        # channels are already in [0, 1] range from to_trichromatic conversion.
        if arr[..., 1:].dtype in [int, np.uint8, np.uint16]:
            arr[..., 1:] = arr[..., 1:] / 255.0
    elif cs == "LAB":
        if np.nanmin(arr[..., 1:]) < 0.0:
            arr[..., 0] = arr[..., 0] / 100.0
            arr[..., 1] = (arr[..., 1] + 127.0) / 254.0
            arr[..., 2] = (arr[..., 2] + 127.0) / 254.0
        else:
            arr = arr / 255.0
    else:
        raise ValueError(f"Unsupported color space '{color_space}'.")
    return np.clip(arr, 0.0, 1.0), cs
