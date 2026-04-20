"""Centralized color embedding abstractions and canonical transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import darsia
from darsia.presets.workflows.basis import CalibrationBasis, calibration_basis_folder

if TYPE_CHECKING:
    from darsia.presets.workflows.rig import Rig


class ColorEmbeddingBasis(StrEnum):
    """Basis for color embedding calibration/application."""

    LABELS = "labels"
    FACIES = "facies"
    SINGLE = "single"


def parse_color_embedding_basis(value: str | ColorEmbeddingBasis) -> ColorEmbeddingBasis:
    if isinstance(value, ColorEmbeddingBasis):
        return value
    return ColorEmbeddingBasis(value.lower().strip())


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
    calibration_root: Path

    @abstractmethod
    def canonical_transform(self, runtime: ColorEmbeddingRuntime) -> ColorEmbeddingTransform:
        raise NotImplementedError

    def to_scalar_image(
        self, image: darsia.Image, runtime: ColorEmbeddingRuntime
    ) -> darsia.ScalarImage:
        if self.embedding_id not in runtime.cache:
            runtime.cache[self.embedding_id] = self.canonical_transform(runtime)
        return runtime.cache[self.embedding_id](image)

    def calibration_basis(self) -> CalibrationBasis:
        if self.basis == ColorEmbeddingBasis.SINGLE:
            raise NotImplementedError(
                "Basis 'single' is not implemented for this calibration workflow."
            )
        return (
            CalibrationBasis.FACIES
            if self.basis == ColorEmbeddingBasis.FACIES
            else CalibrationBasis.LABELS
        )

    def labels_for_runtime(self, runtime: ColorEmbeddingRuntime) -> darsia.Image:
        if self.basis == ColorEmbeddingBasis.SINGLE:
            raise NotImplementedError(
                "Basis 'single' is not implemented for runtime color embeddings."
            )
        if self.basis == ColorEmbeddingBasis.FACIES:
            if not hasattr(runtime.rig, "facies"):
                raise ValueError(
                    "Embedding basis is 'facies' but rig has no facies image loaded."
                )
            return runtime.rig.facies
        return runtime.rig.labels


def _to_scalar_image(template: darsia.Image, values: np.ndarray) -> darsia.ScalarImage:
    metadata = template.metadata()
    metadata.pop("color_space", None)
    return darsia.ScalarImage(img=values, **metadata)


def _channel_index(color_space: str, channel: str) -> int:
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


def _normalized_trichromatic(
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
        if np.max(arr) > 1.0:
            arr = arr / 255.0
    elif cs in {"HSV", "HLS"}:
        hue = arr[..., 0]
        hue_scale = 360.0 if np.max(hue) > 180.0 and np.min(hue) >= 0.0 else 180.0
        arr[..., 0] = hue / hue_scale
        if np.max(arr[..., 1:]) > 1.0:
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


@dataclass
class ColorPathEmbeddingTransform(ColorEmbeddingTransform):
    """Canonical transform for color path embedding."""

    analysis: darsia.ConcentrationAnalysis

    def __call__(self, image: darsia.Image) -> darsia.ScalarImage:
        return self.analysis(image)


@dataclass
class ColorChannelEmbeddingTransform(ColorEmbeddingTransform):
    """Canonical transform for color channel embedding."""

    color_space: str
    channel: str
    mode: darsia.ColorMode
    baseline: darsia.Image | None

    def __call__(self, image: darsia.Image) -> darsia.ScalarImage:
        trichromatic, color_space = _normalized_trichromatic(
            image=image,
            color_space=self.color_space,
            mode=self.mode,
            baseline=self.baseline,
        )
        idx = _channel_index(color_space, self.channel)
        return _to_scalar_image(image, trichromatic[..., idx].astype(np.float32))


@dataclass
class ColorRangeEmbeddingTransform(ColorEmbeddingTransform):
    """Canonical transform for color range embedding."""

    color_space: str
    ranges: list[tuple[float | None, float | None]]
    mode: darsia.ColorMode
    baseline: darsia.Image | None

    def __call__(self, image: darsia.Image) -> darsia.ScalarImage:
        trichromatic, color_space = _normalized_trichromatic(
            image=image,
            color_space=self.color_space,
            mode=self.mode,
            baseline=self.baseline,
        )
        mask = np.ones(trichromatic.shape[:2], dtype=bool)
        for channel, (lower, upper) in enumerate(self.ranges):
            values = trichromatic[..., channel]
            if lower is None and upper is None:
                component_mask = np.ones_like(values, dtype=bool)
            elif lower is None:
                component_mask = values <= float(upper)
            elif upper is None:
                component_mask = values >= float(lower)
            elif (
                channel == 0
                and color_space in {"HSV", "HLS"}
                and float(lower) > float(upper)
            ):
                component_mask = np.logical_or(
                    values >= float(lower), values <= float(upper)
                )
            else:
                component_mask = np.logical_and(
                    values >= float(lower), values <= float(upper)
                )
            mask = np.logical_and(mask, component_mask)
        return _to_scalar_image(image, mask.astype(np.float32))


@dataclass
class ColorPathEmbedding(ColorEmbedding):
    """Color path embedding configuration."""

    embedding_id: str
    mode: darsia.ColorMode
    basis: ColorEmbeddingBasis
    calibration_root: Path
    num_segments: int = 1
    ignore_labels: list[int] = field(default_factory=list)
    resolution: int = 51
    threshold_baseline: float = 0.0
    threshold_calibration: float = 0.0
    baseline_image_paths: list[Path] = field(default_factory=list)
    data: object | None = None
    reference_label: int = 0
    rois: list[str] = field(default_factory=list)
    ignore_baseline_spectrum: str = "expanded"
    histogram_weighting: str = "threshold"
    calibration_mode: str = "auto"

    @property
    def color_paths_folder(self) -> Path:
        return (
            self.calibration_root
            / "color_paths"
            / calibration_basis_folder(self.calibration_basis())
        )

    @property
    def baseline_color_spectrum_folder(self) -> Path:
        return self.calibration_root / "baseline_color_spectrum"

    @property
    def color_range_file(self) -> Path:
        return self.calibration_root / "color_range"

    @property
    def color_to_mass_folder(self) -> Path:
        return (
            self.calibration_root
            / "color_to_mass"
            / calibration_basis_folder(self.calibration_basis())
        )

    def canonical_transform(self, runtime: ColorEmbeddingRuntime) -> ColorEmbeddingTransform:
        labels = self.labels_for_runtime(runtime)
        color_paths = darsia.LabelColorPathMap.load(self.color_paths_folder)
        interpolation = {
            label: darsia.ColorPathInterpolation(
                color_path=path,
                color_mode=self.mode,
                values=path.equidistant_distances,
            )
            for label, path in color_paths.items()
        }
        model = darsia.CombinedModel(
            [
                darsia.HeterogeneousModel(
                    interpolation,
                    labels,
                    ignore_labels=self.ignore_labels,
                )
            ]
        )
        analysis = darsia.ConcentrationAnalysis(
            base=runtime.rig.baseline if self.mode == darsia.ColorMode.RELATIVE else None,
            labels=labels,
            restoration=None,
            model=model,
            **{"diff option": "plain", "restoration -> model": False},
        )
        return ColorPathEmbeddingTransform(analysis=analysis)


@dataclass
class ColorRangeEmbedding(ColorEmbedding):
    """Color range embedding configuration."""

    embedding_id: str
    mode: darsia.ColorMode
    basis: ColorEmbeddingBasis
    calibration_root: Path
    color_space: str
    ranges: list[tuple[float | None, float | None]]

    @property
    def config_file(self) -> Path:
        return self.calibration_root / "embedding.json"

    def canonical_transform(self, runtime: ColorEmbeddingRuntime) -> ColorEmbeddingTransform:
        if self.basis != ColorEmbeddingBasis.SINGLE:
            raise NotImplementedError(
                "Color range embedding currently only supports basis='single'."
            )
        return ColorRangeEmbeddingTransform(
            color_space=self.color_space,
            ranges=self.ranges,
            mode=self.mode,
            baseline=runtime.rig.baseline,
        )


@dataclass
class ColorChannelEmbedding(ColorEmbedding):
    """Color channel embedding configuration."""

    embedding_id: str
    mode: darsia.ColorMode
    basis: ColorEmbeddingBasis
    calibration_root: Path
    color_space: str
    channel: str

    @property
    def config_file(self) -> Path:
        return self.calibration_root / "embedding.json"

    def canonical_transform(self, runtime: ColorEmbeddingRuntime) -> ColorEmbeddingTransform:
        if self.basis != ColorEmbeddingBasis.SINGLE:
            raise NotImplementedError(
                "Color channel embedding currently only supports basis='single'."
            )
        return ColorChannelEmbeddingTransform(
            color_space=self.color_space,
            channel=self.channel,
            mode=self.mode,
            baseline=runtime.rig.baseline,
        )

