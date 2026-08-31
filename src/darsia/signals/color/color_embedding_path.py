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
    calibration_basis_folder,
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
    data: Path | None = None

    def __post_init__(self) -> None:
        """Set default data path if unset."""
        if self.data is None:
            self.data = self.root / "color_paths.csv"

    @property
    def color_paths_folder(self) -> Path:
        return self.root / "color_paths" / calibration_basis_folder(self.basis)

    @property
    def baseline_color_spectrum_folder(self) -> Path:
        return self.root / "baseline_color_spectrum"

    @property
    def color_range_file(self) -> Path:
        return self.root / "color_range"

    @property
    def color_to_mass_folder(self) -> Path:
        return self.root / "color_to_mass" / calibration_basis_folder(self.basis)

    # TODO: Flatten.
    @property
    def color_paths_csv_file(self) -> Path:
        """Resolved color-paths CSV path.

        Always returns the path from `data` (set by __post_init__ to canonical
        location if not explicitly provided).
        """
        return Path(self.data)

    def canonical_transform(
        self, runtime: ColorEmbeddingRuntime
    ) -> ColorEmbeddingTransform:
        labels = self.get_labels(runtime)
        csv_path = self.color_paths_csv_file
        if csv_path.exists():
            color_paths = darsia.LabelColorPathMap.load_csv(csv_path)
        else:
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
