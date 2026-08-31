"""Color-path embedding configuration and transform."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import darsia
from darsia.signals.color.color_embedding import (
    ColorEmbedding,
    ColorEmbeddingBasis,
    ColorEmbeddingRuntime,
    ColorEmbeddingTransform,
)


@dataclass
class ColorPathEmbeddingTransform(ColorEmbeddingTransform):
    """Canonical transform for color path embedding."""

    analysis: darsia.ConcentrationAnalysis

    def __call__(self, image: darsia.Image) -> darsia.ScalarImage:
        return self.analysis(image)


@dataclass
class ColorPathEmbedding(ColorEmbedding):
    """Color path embedding configuration."""

    embedding_id: str
    mode: darsia.ColorMode
    basis: ColorEmbeddingBasis
    root: Path
    reference_label: int = 0

    @property
    def color_paths_csv_file(self) -> Path:
        """Color-paths CSV file path."""
        return self.root / "color_paths.csv"

    @property
    def metadata_file(self) -> Path:
        """Calibration metadata file path."""
        return self.root / "metadata.json"

    @property
    def figures_folder(self) -> Path:
        """Diagnostic figures folder."""
        return self.root / "figures"

    @property
    def baseline_color_spectrum_folder(self) -> Path:
        return self.root / "baseline_color_spectrum"

    @property
    def color_to_mass_folder(self) -> Path:
        return self.root / "interpolation" / "mass"

    def canonical_transform(
        self, runtime: ColorEmbeddingRuntime
    ) -> ColorEmbeddingTransform:
        labels = self.get_labels(runtime)
        color_paths = darsia.LabelColorPathMap.load_csv(
            self.color_paths_csv_file
        )
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
                )
            ]
        )
        analysis = darsia.ConcentrationAnalysis(
            base=(
                runtime.rig.baseline if self.mode == darsia.ColorMode.RELATIVE else None
            ),
            labels=labels,
            restoration=None,
            model=model,
            **{"diff option": "plain", "restoration -> model": False},
        )
        return ColorPathEmbeddingTransform(analysis=analysis)
