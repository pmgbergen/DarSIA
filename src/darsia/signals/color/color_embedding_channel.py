"""Color-channel embedding configuration and transform."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import darsia
from darsia.signals.color.color_embedding import (
    ColorEmbedding,
    ColorEmbeddingBasis,
    ColorEmbeddingRuntime,
    ColorEmbeddingTransform,
    channel_index,
    normalized_trichromatic,
    to_scalar_image,
)
from darsia.signals.color.color_embedding_range import (
    ColorRangeEmbedding,
    ColorRangeEmbeddingTransform,
)


@dataclass
class ColorChannelEmbeddingTransform(ColorEmbeddingTransform):
    """Canonical transform for color channel embedding."""

    color_space: str
    channel: str
    mode: darsia.ColorMode
    baseline: darsia.Image | None
    mask_embedding_transform: ColorRangeEmbeddingTransform | None = None

    def __call__(self, image: darsia.Image) -> darsia.ScalarImage:
        trichromatic, color_space = normalized_trichromatic(
            image=image,
            color_space=self.color_space,
            mode=self.mode,
            baseline=self.baseline,
        )
        idx = channel_index(color_space, self.channel)
        scalar_image = to_scalar_image(image, trichromatic[..., idx].astype(np.float32))

        # Retsrict to mask if provided.
        if self.mask_embedding_transform is not None:
            mask_image = self.mask_embedding_transform(image)
            scalar_image.img = scalar_image.img * mask_image.img

        return scalar_image


@dataclass
class ColorChannelEmbedding(ColorEmbedding):
    """Color channel embedding configuration."""

    embedding_id: str
    mode: darsia.ColorMode
    basis: ColorEmbeddingBasis
    root: Path
    color_space: str
    channel: str
    mask_embedding: ColorRangeEmbedding | None = None

    @property
    def config_file(self) -> Path:
        return self.root / "embedding.json"

    def canonical_transform(
        self, runtime: ColorEmbeddingRuntime
    ) -> ColorEmbeddingTransform:
        if self.basis != ColorEmbeddingBasis.GLOBAL:
            raise NotImplementedError(
                "Color channel embedding currently only supports basis='global'."
            )

        return ColorChannelEmbeddingTransform(
            color_space=self.color_space,
            channel=self.channel,
            mode=self.mode,
            baseline=runtime.rig.baseline,
            mask_embedding_transform=(
                self.mask_embedding.canonical_transform(runtime)
                if self.mask_embedding is not None
                else None
            ),
        )
