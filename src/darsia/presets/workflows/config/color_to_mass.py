"""Configuration for color to mass calibration"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

from darsia.signals.color.color_embedding import (
    ColorEmbeddingBasis,
    parse_color_embedding_basis,
)

from .data_registry import DataRegistry
from .roi_registry import RoiRegistry, _load_roi_key_list
from .time_data import TimeData
from .utils import _get_key, _get_section_from_toml

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ColorToMassConfig:
    """Configuration for color to mass calibration"""

    mode: str = "manual"
    """Calibration mode (e.g., 'manual', 'automatic')."""
    fluid: str | None = "co2"
    """Fluid type for mass analysis (e.g., 'tracer', 'co2')."""
    data: TimeData | None = field(default=None, metadata={"hidden": True})
    """Calibration data configuration."""
    calibration_folder: Path = field(default_factory=Path)
    """Path to the calibration folder."""
    basis: ColorEmbeddingBasis = ColorEmbeddingBasis.LABELS
    """Label-space basis used for calibration (`facies` or `labels`)."""
    threshold: float = 0.2
    """Sensitivity threshold used when deactivating insensitive color paths."""
    rois: list[str] = field(default_factory=list)
    """Registry key names of ROIs used for manual calibration integration plots."""

    def load(
        self,
        path: Path,
        data: Path | None,
        results: Path | None = None,
        data_registry: DataRegistry | None = None,
        roi_registry: "RoiRegistry | None" = None,
    ) -> "ColorToMassConfig":
        """Load color to mass config from a toml file from [section].

        Args:
            path: Path to the TOML file.
            data: Path to the data folder.
            results: Path to the results folder.
            data_registry: Optional global :class:`DataRegistry` for resolving
                ``data = "key"`` or ``data = ["key1", "key2"]`` references.

        """
        # Get section
        sec = _get_section_from_toml(path, "color_to_mass")

        # Mode and fluid
        self.mode = _get_key(sec, "mode", default="manual", required=False, type_=str)
        self.fluid = _get_key(sec, "fluid", default="co2", required=False, type_=str)
        self.basis = parse_color_embedding_basis(
            _get_key(
                sec, "basis", default=ColorEmbeddingBasis.LABELS.value, required=False
            )
        )
        self.threshold = _get_key(
            sec, "threshold", default=0.2, required=False, type_=float
        )
        self.rois = _load_roi_key_list(
            sec,
            "rois",
            context="color_to_mass.rois",
            roi_registry=roi_registry,
        )

        # Calibration data – centralized selector resolution.
        try:
            self.data = (
                data_registry.resolve(sec.get("data")) if data_registry else None
            )
        except KeyError:
            warn("No data found. Use [color_to_mass.data].")
            self.data = None

        # Where to store the calibration results
        self.calibration_folder = _get_key(
            sec, "calibration_folder", required=False, type_=Path
        )
        if not self.calibration_folder:
            assert results is not None
            self.calibration_folder = (
                results / "calibration" / "color_to_mass" / f"from_{self.basis.value}"
            )

        return self
