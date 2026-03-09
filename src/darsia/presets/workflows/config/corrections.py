"""Configuration for corrections."""

from dataclasses import dataclass
from pathlib import Path
from typing import Type, Literal

import numpy as np
from .utils import _get_section_from_toml


@dataclass
class TypeCorrectionConfig:
    """Configuration for type conversion correction.

    Attributes:
        target_type: Target data type for conversion (default: np.float64).

    """

    target_type: Type[np.floating] = np.float64

    def load(self, sec: dict) -> "TypeCorrectionConfig":
        """Load type correction configuration from a dictionary.

        Args:
            sec: Dictionary containing type correction settings.

        Returns:
            self with loaded configuration
        """
        str_target_type = sec.get("type", "float64")
        if str_target_type == "float32":
            self.target_type = np.float32
        elif str_target_type == "float64":
            self.target_type = np.float64
        return self


@dataclass
class ResizeCorrectionConfig:
    """Configuration for resize correction.

    Attributes:
        target_shape: Target shape for resizing (default: None, i.e. no resizing).

    """

    scale: float | None = None
    target_shape: tuple[int, int] | None = None

    def load(self, sec: dict) -> "ResizeCorrectionConfig":
        """Load resize correction configuration from a dictionary.

        Args:
            sec: Dictionary containing resize correction settings.

        Returns:
            self with loaded configuration
        """
        self.scale = sec.get("scale", self.scale)
        self.target_shape = sec.get("target_shape", self.target_shape)

        # Sanity checks.
        assert self.scale is None or self.target_shape is None, (
            "Cannot specify both scale and target_shape for resize correction."
        )
        assert self.scale is not None or self.target_shape is not None, (
            "Must specify either scale or target_shape for resize correction."
        )
        return self


@dataclass
class CurvatureCorrectionConfig:
    """Configuration for curvature correction.

    Attributes:

    """

    # TODO mere with curvature correction config from curvature.py

    config: dict | None = None

    def load(self, sec: dict) -> "CurvatureCorrectionConfig":
        """Load curvature correction configuration from a dictionary.

        Args:
            sec: Dictionary containing curvature correction settings.

        Returns:
            self with loaded configuration
        """
        self.config = sec
        return self


@dataclass
class DriftCorrectionConfig:
    """Configuration for drift correction.

    Attributes:

    """

    colorchecker: (
        Literal["upper_left", "upper_right", "lower_left", "lower_right"] | None
    ) = None

    def load(self, sec: dict) -> "DriftCorrectionConfig":
        """Load drift correction configuration from a dictionary.

        Args:
            sec: Dictionary containing drift correction settings.

        Returns:
            self with loaded configuration
        """
        self.colorchecker = sec.get("colorchecker")

        # Sanity check.
        if self.colorchecker is not None:
            assert self.colorchecker in [
                "upper_left",
                "upper_right",
                "lower_left",
                "lower_right",
            ], (
                """colorchecker must be one of 'upper_left', 'upper_right', """
                """'lower_left', 'lower_right'"""
            )
        return self


@dataclass
class ColorCorrectionConfig:
    """Configuration for color correction.

    Attributes:

    """

    colorchecker: (
        Literal["upper_left", "upper_right", "lower_left", "lower_right"] | None
    ) = None

    def load(self, sec: dict) -> "ColorCorrectionConfig":
        """Load color correction configuration from a dictionary.

        Args:
            sec: Dictionary containing color correction settings.

        Returns:
            self with loaded configuration
        """
        self.colorchecker = sec.get("colorchecker")

        # Sanity check.
        if self.colorchecker is not None:
            assert self.colorchecker in [
                "upper_left",
                "upper_right",
                "lower_left",
                "lower_right",
            ], (
                """colorchecker must be one of 'upper_left', 'upper_right', """
                """'lower_left', 'lower_right'"""
            )
        return self


@dataclass
class CorrectionsConfig:
    """Configuration for image corrections.

    This class manages configuration for various image corrections that can be applied
    during rig setup. Each correction type has its own configuration class that handles
    type-specific settings.

    Attributes:
        type: TypeCorrectionConfig for type conversion correction (default: None).
            Converts image to a specified numpy floating-point type (e.g., float32, float64).
        resize: ResizeCorrectionConfig for resize correction (default: None).
            Resizes images to a target shape or scale.
        drift: DriftCorrectionConfig for drift correction (default: None).
            Corrects color drift based on color checker position.
        curvature: CurvatureCorrectionConfig for curvature correction (default: None).
            Corrects lens distortion based on laser grid configuration.
        color: ColorCorrectionConfig for color correction (default: None).
            Applies color correction based on color checker position.
        relative_color: Enable relative color correction based on color checker
            (default: False). Boolean flag for enabling/disabling.
        illumination: Enable illumination correction based on color checker
            (default: False). Boolean flag for enabling/disabling.

    """

    # Configuration objects for each correction type
    type: TypeCorrectionConfig | None = None
    resize: ResizeCorrectionConfig | None = None
    drift: DriftCorrectionConfig | None = None
    curvature: CurvatureCorrectionConfig | None = None
    color: ColorCorrectionConfig | None = None
    relative_color: bool = False
    illumination: bool = False

    def load(self, path: Path | list[Path]) -> "CorrectionsConfig":
        """Load correction configuration from TOML file.

        Args:
            path: Path to TOML config file
            results: Path to results folder

        Returns:
            self with loaded configuration
        """
        sec = _get_section_from_toml(path, "corrections")

        # Load individual correction settings
        type_sec = sec.get("type")
        if type_sec:
            self.type = TypeCorrectionConfig().load(type_sec)

        resize_sec = sec.get("resize")
        if resize_sec:
            self.resize = ResizeCorrectionConfig().load(resize_sec)

        drift_sec = sec.get("drift")
        if drift_sec:
            self.drift = DriftCorrectionConfig().load(drift_sec)

        curvature_sec = sec.get("curvature")
        if curvature_sec:
            self.curvature = CurvatureCorrectionConfig().load(curvature_sec)

        color_sec = sec.get("color")
        if color_sec:
            self.color = ColorCorrectionConfig().load(color_sec)

        self.relative_color = sec.get("relative_color", self.relative_color)
        if not isinstance(self.relative_color, bool):
            raise NotImplementedError(
                "relative color correction is only implemented as boolean for now."
            )

        self.illumination = sec.get("illumination", self.illumination)
        if not isinstance(self.illumination, bool):
            raise NotImplementedError(
                "illumination correction is only implemented as boolean for now."
            )

        # Identify active corrections
        active_corrections = sec.get("active_corrections", None)
        if active_corrections is not None:
            raise NotImplementedError("active_corrections is not implemented yet.")

        return self
