"""Color-channel embedding configuration and transform."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import darsia
from darsia.presets.workflows.basis import CalibrationBasis
from darsia.signals.color.color_embedding import (
    ColorEmbedding,
    ColorEmbeddingRuntime,
    ColorEmbeddingTransform,
    channel_index,
    normalized_trichromatic,
    to_scalar_image,
)


@dataclass
class ColorChannelEmbeddingTransform(ColorEmbeddingTransform):
    """Canonical transform for color channel embedding."""

    color_space: str
    channel: str
    mode: darsia.ColorMode
    baseline: darsia.Image | None

    def __call__(self, image: darsia.Image) -> darsia.ScalarImage:
        trichromatic, color_space = normalized_trichromatic(
            image=image,
            color_space=self.color_space,
            mode=self.mode,
            baseline=self.baseline,
        )
        idx = channel_index(color_space, self.channel)
        return to_scalar_image(image, trichromatic[..., idx].astype(np.float32))


@dataclass
class ColorChannelEmbedding(ColorEmbedding):
    """Color channel embedding configuration."""

    embedding_id: str
    mode: darsia.ColorMode
    basis: CalibrationBasis
    calibration_root: Path
    color_space: str
    channel: str

    @property
    def config_file(self) -> Path:
        return self.calibration_root / "embedding.json"

    def canonical_transform(
        self, runtime: ColorEmbeddingRuntime
    ) -> ColorEmbeddingTransform:
        if self.basis != CalibrationBasis.GLOBAL:
            raise NotImplementedError(
                "Color channel embedding currently only supports basis='global'."
            )
        return ColorChannelEmbeddingTransform(
            color_space=self.color_space,
            channel=self.channel,
            mode=self.mode,
            baseline=runtime.rig.baseline,
        )
