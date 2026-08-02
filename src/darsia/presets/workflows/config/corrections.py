"""Configuration for corrections."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Type

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
    """Configuration for curvature correction."""

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
    """Configuration for drift correction."""

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
    """Position of color checker for color correction."""

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
class RelativeColorCorrectionConfig:
    """Configuration for relative color correction."""

    path: Path | None = None
    """Path to a precomputed relative color correction file."""
    images: list[Path] = field(default_factory=list)
    """Calibration images used to calibrate relative color correction."""
    interactive: bool = False
    """Whether interactive calibration is allowed."""
    options: dict = field(default_factory=dict)
    """Calibration options forwarded to RelativeColorCorrection."""

    def load(self, sec: dict) -> "RelativeColorCorrectionConfig":
        """Load relative color correction configuration from a dictionary."""
        path = sec.get("path", self.path)
        self.path = Path(path) if path is not None else None
        self.images = [Path(p) for p in sec.get("images", self.images)]
        self.interactive = sec.get("interactive", self.interactive)
        if not isinstance(self.interactive, bool):
            raise ValueError(
                "corrections.relative_color.interactive must be a boolean."
            )

        known_keys = {"path", "images", "interactive"}
        self.options = {
            key: value for key, value in sec.items() if key not in known_keys
        }

        if self.path is None and len(self.images) == 0:
            raise ValueError(
                "corrections.relative_color must define either 'path' or 'images'."
            )

        return self


@dataclass
class IlluminationCorrectionConfig:
    """Configuration for illumination correction."""

    labels: list[int] = field(default_factory=list)
    """List of labels to use for illumination correction. Overrides `label` if not empty."""
    interpolation: Literal["rbf", "quartic", "illumination"] = "illumination"
    """Interpolation method to use for scaling."""
    colorspace: Literal[
        "rgb", "rgb-scalar", "lab", "lab-scalar", "hsl", "hsl-scalar", "gray"
    ] = "hsl-scalar"
    """Color space to use for interpolation."""
    width: int = 100
    """Width of patches to use for interpolation."""
    num_samples: int = 30
    """Number of patches to use for interpolation."""
    seed: int = 42
    """Random seed for patch sampling."""
    sigma: float = 100.0
    """Sigma for Gaussian smoothing of the illumination correction map."""
    outliers: float = 0.1
    """Fraction of outliers to discard when computing the illumination correction map."""
    bounds: tuple[float, float] = (0.5, 2.0)
    """Bounds for the illumination correction factors."""

    def load(self, sec: dict) -> "IlluminationCorrectionConfig":
        """Load illumination correction configuration from a dictionary.

        Args:
            sec: Dictionary containing illumination correction settings.

        Returns:
            self with loaded configuration

        """

        _supported_colorspaces = (
            "rgb",
            "rgb-scalar",
            "lab",
            "lab-scalar",
            "hsl",
            "hsl-scalar",
            "gray",
        )
        self.labels = sec.get("labels", self.labels)
        self.interpolation = sec.get("interpolation", self.interpolation)
        colorspace = sec.get("colorspace", self.colorspace)
        if colorspace not in _supported_colorspaces:
            raise ValueError(
                f"IlluminationCorrectionConfig.colorspace must be one of "
                f"{_supported_colorspaces}, got {colorspace!r}"
            )
        self.colorspace = colorspace
        self.width = sec.get("width", self.width)
        self.num_samples = sec.get("num_samples", self.num_samples)
        self.seed = sec.get("seed", self.seed)
        self.sigma = sec.get("sigma", self.sigma)
        self.bounds = sec.get("bounds", self.bounds)
        self.outliers = sec.get("outliers", self.outliers)
        if not 0.0 <= self.outliers <= 1.0:
            raise ValueError(
                f"IlluminationCorrectionConfig.outliers must be between 0.0 and 1.0, "
                f"got {self.outliers!r}"
            )

        return self


@dataclass
class PatchwiseIlluminationCorrectionConfig:
    """Configuration for patchwise illumination correction."""

    image_path: Path = Path()
    """Path to image for patchwise illumination correction."""
    baseline_paths: list[Path] = field(default_factory=list)
    """Paths to baseline images for patchwise illumination correction."""
    limit: int = 1450
    """Limit in pixels to exclude from top of image for patch sampling."""
    nw: int = 1000
    """Number of patches in width direction for patchwise illumination correction."""
    eps: float = 1e-6
    """Small constant to avoid division by zero in patchwise illumination correction."""

    def load(self, sec: dict) -> "PatchwiseIlluminationCorrectionConfig":
        """Load patchwise illumination correction configuration from a dictionary.

        Args:
            sec: Dictionary containing patchwise illumination correction settings.

         Returns:
            self with loaded configuration

        """
        self.image_path = Path(sec.get("image_path", self.image_path))
        _baseline_paths = sec.get("baseline_paths", self.baseline_paths)
        self.baseline_paths = [Path(p) for p in _baseline_paths]
        self.limit = sec.get("limit", self.limit)
        self.nw = sec.get("nw", self.nw)
        self.eps = sec.get("eps", self.eps)
        return self


_CORRECTION_CLASSES: dict[str, type] = {
    "type": TypeCorrectionConfig,
    "resize": ResizeCorrectionConfig,
    "drift": DriftCorrectionConfig,
    "curvature": CurvatureCorrectionConfig,
    "color": ColorCorrectionConfig,
    "illumination": IlluminationCorrectionConfig,
    "patchwise_illumination": PatchwiseIlluminationCorrectionConfig,
}


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
        illumination: Enable illumination correction.
        patchwise_illumination: PatchwiseIlluminationCorrectionConfig for patchwise
            illumination correction (default: None). Corrects illumination variations
            across the image using patchwise interpolation.

    """

    # Configuration objects for each correction type
    type: TypeCorrectionConfig | None = field(
        default=None, metadata={"active_list_key": "active"}
    )
    resize: ResizeCorrectionConfig | None = field(
        default=None, metadata={"active_list_key": "active"}
    )
    drift: DriftCorrectionConfig | None = field(
        default=None, metadata={"active_list_key": "active"}
    )
    curvature: CurvatureCorrectionConfig | None = field(
        default=None, metadata={"active_list_key": "active"}
    )
    color: ColorCorrectionConfig | None = field(
        default=None, metadata={"active_list_key": "active"}
    )
    relative_color: bool | RelativeColorCorrectionConfig | None = field(
        default=None, metadata={"active_list_key": "active"}
    )
    illumination: IlluminationCorrectionConfig | None = field(
        default=None, metadata={"active_list_key": "active"}
    )
    patchwise_illumination: PatchwiseIlluminationCorrectionConfig | None = field(
        default=None, metadata={"active_list_key": "active"}
    )

    inactive: dict[str, Any] = field(
        default_factory=dict, repr=False, metadata={"hidden": True}
    )
    """Parsed sub-configs for corrections present in the TOML but deactivated via
    `active`. Kept so tuned parameters survive toggling a correction off
    (not consumed by the correction pipeline — see `get_parsed`)."""

    def get_parsed(self, name: str) -> Any | None:
        """Return the parsed sub-config for `name`, active or not (GUI use)."""
        return getattr(self, name, None) or self.inactive.get(name)

    def load(self, path: Path | list[Path]) -> "CorrectionsConfig":
        """Load correction configuration from TOML file.

        Args:
            path: Path to TOML config file

        Returns:
            self with loaded configuration
        """
        sec = _get_section_from_toml(path, "corrections")

        # Parse all correction sub-tables; active list decides exposure
        active = sec.get("active")  # None => all present are active
        for name, cls in _CORRECTION_CLASSES.items():
            sub_sec = sec.get(name)
            if not sub_sec:
                continue
            parsed = cls().load(sub_sec)
            is_active = active is None or name in active
            if is_active:
                setattr(self, name, parsed)
            else:
                self.inactive[name] = parsed

        relative_color_sec = sec.get("relative_color", self.relative_color)
        if isinstance(relative_color_sec, bool):
            self.relative_color = relative_color_sec
        elif isinstance(relative_color_sec, dict):
            self.relative_color = RelativeColorCorrectionConfig().load(
                relative_color_sec
            )
        else:
            raise ValueError(
                "corrections.relative_color must be a boolean or a configuration table."
            )

        return self
