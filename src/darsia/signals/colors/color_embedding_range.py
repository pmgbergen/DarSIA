"""Color-range embedding configuration and transform."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import darsia
from darsia.presets.workflows.basis import CalibrationBasis
from darsia.signals.colors.color_embedding import (
    ColorEmbedding,
    ColorEmbeddingRuntime,
    ColorEmbeddingTransform,
    normalized_trichromatic,
    to_scalar_image,
)


@dataclass
class ColorRangeEmbeddingTransform(ColorEmbeddingTransform):
    """Canonical transform for color range embedding."""

    color_space: str
    ranges: list[tuple[float | None, float | None]]
    mode: darsia.ColorMode
    baseline: darsia.Image | None

    def __call__(self, image: darsia.Image) -> darsia.ScalarImage:
        trichromatic, color_space = normalized_trichromatic(
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
        return to_scalar_image(image, mask.astype(np.float32))


@dataclass
class ColorRangeEmbedding(ColorEmbedding):
    """Color range embedding configuration."""

    embedding_id: str
    mode: darsia.ColorMode
    basis: CalibrationBasis
    calibration_root: Path
    color_space: str
    ranges: list[tuple[float | None, float | None]]

    @property
    def config_file(self) -> Path:
        return self.calibration_root / "embedding.json"

    def canonical_transform(
        self, runtime: ColorEmbeddingRuntime
    ) -> ColorEmbeddingTransform:
        if self.basis != CalibrationBasis.GLOBAL:
            raise NotImplementedError(
                "Color range embedding currently only supports basis='global'."
            )
        return ColorRangeEmbeddingTransform(
            color_space=self.color_space,
            ranges=self.ranges,
            mode=self.mode,
            baseline=runtime.rig.baseline,
        )
