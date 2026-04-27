"""Color-path embedding configuration and transform."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import darsia
from darsia.presets.workflows.basis import (
    CalibrationBasis,
    calibration_basis_folder
)
from darsia.signals.color.color_embedding import (
    ColorEmbedding,
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
    basis: CalibrationBasis
    calibration_root: Path
    num_segments: int = 1
    ignore_labels: list[int] = field(default_factory=list)
    resolution: int = 51
    threshold_baseline: float = 0.0
    threshold_calibration: float = 0.0
    baseline_data: object | None = None
    data: object | None = None
    reference_label: int = 0
    rois: list[str] = field(default_factory=list)
    ignore_baseline_spectrum: str = "expanded"
    histogram_weighting: str = "threshold"
    calibration_mode: str = "auto"

    @property
    def color_paths_folder(self) -> Path:
        return (
            self.calibration_root / "color_paths" / calibration_basis_folder(self.basis)
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
            / calibration_basis_folder(self.basis)
        )

    def canonical_transform(
        self, runtime: ColorEmbeddingRuntime
    ) -> ColorEmbeddingTransform:
        labels = self.get_labels(runtime)
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
            base=(
                runtime.rig.baseline if self.mode == darsia.ColorMode.RELATIVE else None
            ),
            labels=labels,
            restoration=None,
            model=model,
            **{"diff option": "plain", "restoration -> model": False},
        )
        return ColorPathEmbeddingTransform(analysis=analysis)
