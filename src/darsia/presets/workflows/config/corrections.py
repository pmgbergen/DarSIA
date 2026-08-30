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

    target_type: Type[np.floating] = field(
        default=np.float64,
        metadata={
            "name": "Target type",
            "help": "Target data type for conversion (float32 or float64).",
            "options": ["float32", "float64"],
        },
    )

    def load(self, sec: dict) -> "TypeCorrectionConfig":
        """Load type correction configuration from a dictionary.

        Args:
            sec: Dictionary containing type correction settings.

        Returns:
            self with loaded configuration
        """
        str_target_type = sec.get("target_type", "float64")
        if str_target_type == "float32":
            self.target_type = np.float32
        elif str_target_type == "float64":
            self.target_type = np.float64
        return self


@dataclass
class ResizeCorrectionConfig:
    """Configuration for resize correction.

    Supports two modes: uniform scaling via ``scale`` factor, or explicit target shape
    resizing. Exactly one mode must be configured via the ``mode`` field.

    Attributes:
        mode: Resize mode — either ``"scale"`` (scale all axes uniformly) or
            ``"target_shape"`` (resize to explicit rows/cols).
        scale: Uniform scale factor applied to both spatial axes (mode='scale' only).
        target_shape: Target ``(rows, cols)`` shape to resize to (mode='target_shape' only).

    """

    mode: Literal["scale", "target_shape"] = field(
        default="scale",
        metadata={
            "name": "Resize mode",
            "help": (
                "Whether to resize by a uniform scale factor or to an "
                "explicit target shape."
            ),
            "options": ["scale", "target_shape"],
        },
    )
    """Resize mode: ``"scale"`` (uniform scaling) or ``"target_shape"`` (explicit shape)."""
    scale: float | None = field(
        default=None,
        metadata={
            "name": "Scale factor",
            "help": (
                "Uniform scale factor applied to both axes (mode='scale' only). "
                "For example, 0.5 to halve the image size."
            ),
            "placeholder": "e.g., 0.5 for half size",
            "depends_on": {"field": "mode", "value": "scale"},
        },
    )
    """Uniform scale factor (mode='scale' only)."""
    target_shape: tuple[int, int] | None = field(
        default=None,
        metadata={
            "name": "Target shape",
            "help": (
                "Target (rows, cols) shape to resize to (mode='target_shape' only). "
                "For example, [512, 1024]."
            ),
            "placeholder": "e.g., [512, 1024] for images of 512 rows and 1024 columns",
            "depends_on": {"field": "mode", "value": "target_shape"},
        },
    )
    """Target (rows, cols) shape (mode='target_shape' only)."""

    def load(self, sec: dict) -> "ResizeCorrectionConfig":
        """Load resize correction configuration from a dictionary.

        Args:
            sec: Dictionary containing resize correction settings.

        Returns:
            self with loaded configuration

        Raises:
            ValueError: if mode is invalid, if required field for the mode is missing,
                or if target_shape does not have exactly 2 elements.
        """
        mode = sec.get("mode", self.mode)
        if mode not in ("scale", "target_shape"):
            raise ValueError(
                f"[corrections.resize] mode must be 'scale' or 'target_shape', got {mode!r}"
            )
        self.mode = mode

        self.scale = sec.get("scale", self.scale)
        target_shape_raw = sec.get("target_shape", self.target_shape)
        self.target_shape = (
            tuple(int(v) for v in target_shape_raw)
            if target_shape_raw is not None
            else None
        )

        if mode == "scale" and self.scale is None:
            raise ValueError(
                "[corrections.resize] mode='scale' requires 'scale' to be set."
            )
        if mode == "target_shape" and self.target_shape is None:
            raise ValueError(
                "[corrections.resize] mode='target_shape' requires 'target_shape' to be set."
            )
        if mode == "target_shape" and len(self.target_shape) != 2:
            raise ValueError(
                f"[corrections.resize] target_shape must have exactly 2 elements, "
                f"got {self.target_shape!r}"
            )

        return self

    def resolve(
        self, current_shape: tuple[int, int]
    ) -> tuple[tuple[int, int], float | tuple[float, float]]:
        """Resolve target shape and resize factor for a given current shape.

        Args:
            current_shape: current (rows, cols) shape of the image to be
                resized.

        Returns:
            (target_shape, resize_factor) tuple where:
            - target_shape: resolved (rows, cols) target shape.
            - resize_factor: scalar factor (mode='scale') or (scale_x,
              scale_y) tuple (mode='target_shape') to pass to downstream
              corrections (e.g. CurvatureCorrection) that need to rescale
              pixel-calibrated config values.

        """
        if self.mode == "scale":
            target_shape = tuple(
                int(round(s * self.scale)) for s in current_shape
            )
            resize_factor = self.scale
        else:  # mode == "target_shape"
            target_shape = self.target_shape
            scale_x = target_shape[1] / current_shape[1]
            scale_y = target_shape[0] / current_shape[0]
            resize_factor = (scale_x, scale_y)
        return target_shape, resize_factor


@dataclass
class InitCorrectionConfig:
    """Configuration for initial (pre-bulge) curvature correction stage.

    Attributes:
        horizontal_bulge: Horizontal bulge coefficient (default: 0.0).
        vertical_bulge: Vertical bulge coefficient (default: 0.0).
        horizontal_center_offset: Horizontal offset of bulge center in pixels (default: 0).
        vertical_center_offset: Vertical offset of bulge center in pixels (default: 0).
    """

    horizontal_bulge: float = field(
        default=0.0,
        metadata={"name": "Horizontal bulge", "help": "Horizontal bulge coefficient."},
    )
    vertical_bulge: float = field(
        default=0.0,
        metadata={"name": "Vertical bulge", "help": "Vertical bulge coefficient."},
    )
    horizontal_center_offset: int = field(
        default=0,
        metadata={
            "name": "Horizontal center offset",
            "help": "Horizontal offset of bulge center in pixels.",
        },
    )
    vertical_center_offset: int = field(
        default=0,
        metadata={
            "name": "Vertical center offset",
            "help": "Vertical offset of bulge center in pixels.",
        },
    )

    def load(self, sec: dict) -> "InitCorrectionConfig":
        """Load initial correction configuration from a dictionary."""
        self.horizontal_bulge = float(
            sec.get("horizontal_bulge", self.horizontal_bulge)
        )
        self.vertical_bulge = float(sec.get("vertical_bulge", self.vertical_bulge))
        self.horizontal_center_offset = int(
            sec.get("horizontal_center_offset", self.horizontal_center_offset)
        )
        self.vertical_center_offset = int(
            sec.get("vertical_center_offset", self.vertical_center_offset)
        )
        return self


@dataclass
class CropCorrectionConfig:
    """Configuration for crop curvature correction stage.

    Attributes:
        top_left: Top-left corner as [row, col] in pixels (matrix indexing).
        bottom_left: Bottom-left corner as [row, col] in pixels (matrix indexing).
        bottom_right: Bottom-right corner as [row, col] in pixels (matrix indexing).
        top_right: Top-right corner as [row, col] in pixels (matrix indexing).
        width: Crop width (default: 1.0).
        height: Crop height (default: 1.0).
        in_meters: Whether width/height are in meters (default: True).
    """

    top_left: tuple[int, int] | None = field(
        default=None,
        metadata={
            "name": "Top-left corner",
            "help": (
                "Top-left corner as row, col in pixels. "
                "Row is vertical (0 at top), col is horizontal (0 at left)."
            ),
            "placeholder": "e.g., 47, 415",
        },
    )
    bottom_left: tuple[int, int] | None = field(
        default=None,
        metadata={
            "name": "Bottom-left corner",
            "help": "Bottom-left corner as row, col in pixels.",
            "placeholder": "e.g., 7886, 448",
        },
    )
    bottom_right: tuple[int, int] | None = field(
        default=None,
        metadata={
            "name": "Bottom-right corner",
            "help": "Bottom-right corner as row, col in pixels.",
            "placeholder": "e.g., 7829, 5228",
        },
    )
    top_right: tuple[int, int] | None = field(
        default=None,
        metadata={
            "name": "Top-right corner",
            "help": "Top-right corner as row, col in pixels.",
            "placeholder": "e.g., 110, 5263",
        },
    )
    width: float = field(
        default=1.0,
        metadata={"name": "Width", "help": "Crop width in meters or pixels."},
    )
    height: float = field(
        default=1.0,
        metadata={"name": "Height", "help": "Crop height in meters or pixels."},
    )
    in_meters: bool = field(
        default=True,
        metadata={
            "name": "Width/height in meters",
            "help": "If true, width and height are in meters; otherwise pixels.",
        },
    )

    @property
    def pts_src(self) -> list[list[int]] | None:
        """Get the source points as a list of lists."""
        corners = [self.top_left, self.bottom_left, self.bottom_right, self.top_right]
        if any([c is None for c in corners]):
            return None
        return [list(c) for c in corners]

    def load(self, sec: dict) -> "CropCorrectionConfig":
        """Load crop correction configuration from a dictionary.

        Args:
            sec: Dictionary containing crop correction settings.

        Returns:
            self with loaded configuration
        """
        # Try to load individual corner fields first
        corners = [
            sec.get("top_left"),
            sec.get("bottom_left"),
            sec.get("bottom_right"),
            sec.get("top_right"),
        ]
        make_tuple = lambda c: tuple(c) if c is not None else None
        self.top_left = make_tuple(corners[0])
        self.bottom_left = make_tuple(corners[1])
        self.bottom_right = make_tuple(corners[2])
        self.top_right = make_tuple(corners[3])

        self.width = float(sec.get("width", self.width))
        self.height = float(sec.get("height", self.height))
        self.in_meters = sec.get("in_meters", self.in_meters)
        if not isinstance(self.in_meters, bool):
            self.in_meters = bool(self.in_meters)
        return self


@dataclass
class BulgeCorrectionConfig:
    """Configuration for bulge curvature correction stage.

    Attributes:
        horizontal_bulge: Horizontal bulge coefficient (default: 0.0).
        vertical_bulge: Vertical bulge coefficient (default: 0.0).
        horizontal_center_offset: Horizontal offset of bulge center in pixels (default: 0).
        vertical_center_offset: Vertical offset of bulge center in pixels (default: 0).
    """

    horizontal_bulge: float = field(
        default=0.0,
        metadata={"name": "Horizontal bulge", "help": "Horizontal bulge coefficient."},
    )
    vertical_bulge: float = field(
        default=0.0,
        metadata={"name": "Vertical bulge", "help": "Vertical bulge coefficient."},
    )
    horizontal_center_offset: int = field(
        default=0,
        metadata={
            "name": "Horizontal center offset",
            "help": "Horizontal offset of bulge center in pixels.",
        },
    )
    vertical_center_offset: int = field(
        default=0,
        metadata={
            "name": "Vertical center offset",
            "help": "Vertical offset of bulge center in pixels.",
        },
    )

    def load(self, sec: dict) -> "BulgeCorrectionConfig":
        """Load bulge correction configuration from a dictionary."""
        self.horizontal_bulge = float(
            sec.get("horizontal_bulge", self.horizontal_bulge)
        )
        self.vertical_bulge = float(sec.get("vertical_bulge", self.vertical_bulge))
        self.horizontal_center_offset = int(
            sec.get("horizontal_center_offset", self.horizontal_center_offset)
        )
        self.vertical_center_offset = int(
            sec.get("vertical_center_offset", self.vertical_center_offset)
        )
        return self


@dataclass
class StretchCorrectionConfig:
    """Configuration for stretch curvature correction stage.

    Attributes:
        horizontal_stretch: Horizontal stretch coefficient (default: 0.0).
        vertical_stretch: Vertical stretch coefficient (default: 0.0).
        horizontal_center_offset: Horizontal offset of stretch center in pixels (default: 0).
        vertical_center_offset: Vertical offset of stretch center in pixels (default: 0).
    """

    horizontal_stretch: float = field(
        default=0.0,
        metadata={
            "name": "Horizontal stretch",
            "help": "Horizontal stretch coefficient.",
        },
    )
    vertical_stretch: float = field(
        default=0.0,
        metadata={"name": "Vertical stretch", "help": "Vertical stretch coefficient."},
    )
    horizontal_center_offset: int = field(
        default=0,
        metadata={
            "name": "Horizontal center offset",
            "help": "Horizontal offset of stretch center in pixels.",
        },
    )
    vertical_center_offset: int = field(
        default=0,
        metadata={
            "name": "Vertical center offset",
            "help": "Vertical offset of stretch center in pixels.",
        },
    )

    def load(self, sec: dict) -> "StretchCorrectionConfig":
        """Load stretch correction configuration from a dictionary."""
        self.horizontal_stretch = float(
            sec.get("horizontal_stretch", self.horizontal_stretch)
        )
        self.vertical_stretch = float(
            sec.get("vertical_stretch", self.vertical_stretch)
        )
        self.horizontal_center_offset = int(
            sec.get("horizontal_center_offset", self.horizontal_center_offset)
        )
        self.vertical_center_offset = int(
            sec.get("vertical_center_offset", self.vertical_center_offset)
        )
        return self


@dataclass
class CurvatureCorrectionConfig:
    """Configuration for curvature correction with hierarchical sub-stages.

    Supports 4 independent curvature correction stages applied in order:
    init (pre-bulge) → crop → bulge → stretch. Each stage is independently
    toggleable via the ``active`` list.

    Attributes:
        init: Initial (pre-bulge) correction stage.
        crop: Crop correction stage.
        bulge: Bulge correction stage.
        stretch: Stretch correction stage.
        inactive: Parsed but deactivated stage configs (preserved when toggled off).
    """

    init: InitCorrectionConfig | None = field(
        default=None,
        metadata={
            "name": "Initial bulge correction",
            "active_list_key": "active",
        },
    )
    crop: CropCorrectionConfig | None = field(
        default=None,
        metadata={
            "name": "Crop",
            "active_list_key": "active",
        },
    )
    bulge: BulgeCorrectionConfig | None = field(
        default=None,
        metadata={
            "name": "Bulge correction",
            "active_list_key": "active",
        },
    )
    stretch: StretchCorrectionConfig | None = field(
        default=None,
        metadata={
            "name": "Stretch correction",
            "active_list_key": "active",
        },
    )
    inactive: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        metadata={"hidden": True},
    )
    """Parsed sub-configs for stages present in TOML but deactivated via ``active``.
    Kept so tuned parameters survive toggling a stage off."""

    def load(self, sec: dict) -> "CurvatureCorrectionConfig":
        """Load curvature correction configuration from a dictionary.

        Args:
            sec: Dictionary containing curvature correction settings.

        Returns:
            self with loaded configuration
        """
        # Mapping of stage names to their config classes
        _STAGE_CLASSES = {
            "init": InitCorrectionConfig,
            "crop": CropCorrectionConfig,
            "bulge": BulgeCorrectionConfig,
            "stretch": StretchCorrectionConfig,
        }

        # Parse active list (None = all present stages active)
        active = sec.get("active")

        # Parse all stages; active list decides exposure
        for stage_name, stage_cls in _STAGE_CLASSES.items():
            stage_sec = sec.get(stage_name)
            if not stage_sec:
                continue
            parsed = stage_cls().load(stage_sec)
            is_active = active is None or stage_name in active
            if is_active:
                setattr(self, stage_name, parsed)
            else:
                self.inactive[stage_name] = parsed

        return self

    def to_dict(self) -> dict:
        """Convert to flat dict format expected by CurvatureCorrection engine.

        Returns a dict with keys for active stages only, suitable for passing to
        darsia.CurvatureCorrection(config=...).
        """
        result = {}
        for stage_name in ["init", "crop", "bulge", "stretch"]:
            stage_config = getattr(self, stage_name, None)
            if stage_config is None:
                continue

            # Build stage dict based on type
            if isinstance(stage_config, InitCorrectionConfig):
                result["init"] = {
                    "horizontal_bulge": stage_config.horizontal_bulge,
                    "vertical_bulge": stage_config.vertical_bulge,
                    "horizontal_center_offset": stage_config.horizontal_center_offset,
                    "vertical_center_offset": stage_config.vertical_center_offset,
                }
            elif isinstance(stage_config, CropCorrectionConfig):
                result["crop"] = {
                    "top_left": stage_config.top_left,
                    "bottom_left": stage_config.bottom_left,
                    "bottom_right": stage_config.bottom_right,
                    "top_right": stage_config.top_right,
                    "width": stage_config.width,
                    "height": stage_config.height,
                    "in_meters": stage_config.in_meters,
                }
            elif isinstance(stage_config, BulgeCorrectionConfig):
                result["bulge"] = {
                    "horizontal_bulge": stage_config.horizontal_bulge,
                    "vertical_bulge": stage_config.vertical_bulge,
                    "horizontal_center_offset": stage_config.horizontal_center_offset,
                    "vertical_center_offset": stage_config.vertical_center_offset,
                }
            elif isinstance(stage_config, StretchCorrectionConfig):
                result["stretch"] = {
                    "horizontal_stretch": stage_config.horizontal_stretch,
                    "vertical_stretch": stage_config.vertical_stretch,
                    "horizontal_center_offset": stage_config.horizontal_center_offset,
                    "vertical_center_offset": stage_config.vertical_center_offset,
                }

        return result


@dataclass
class DriftCorrectionConfig:
    """Configuration for drift correction."""

    colorchecker: (
        Literal["upper_left", "upper_right", "lower_left", "lower_right"] | None
    ) = field(
        default=None,
        metadata={
            "name": "Color checker position",
            "help": (
                "Position of color checker (upper/lower, left/right). "
                "Leave empty to disable drift correction."
            ),
            "options": ["upper_left", "upper_right", "lower_left", "lower_right"],
        },
    )

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
        colorchecker: Position of color checker for color correction.

    """

    colorchecker: (
        Literal["upper_left", "upper_right", "lower_left", "lower_right"] | None
    ) = field(
        default=None,
        metadata={
            "name": "Color checker position",
            "help": (
                "Position of color checker (upper/lower, left/right). "
                "Leave empty to disable color correction."
            ),
            "options": ["upper_left", "upper_right", "lower_left", "lower_right"],
        },
    )
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
    """Configuration for relative color correction.

    Relative color correction calibrates color-drift correction by comparing
    baseline and calibration images, optionally using user-selected color
    samples or tensorial calibration, then builds a polynomial/linear
    approximation over the image domain.
    """

    path: Path | None = field(
        default=None,
        metadata={
            "name": "Precomputed correction path",
            "help": "Path to a precomputed relative color correction file (.npz). "
            "If provided, interactive calibration is skipped.",
        },
    )
    """Path to a precomputed relative color correction file."""
    images: list[Path] = field(
        default_factory=list,
        metadata={
            "name": "Calibration images",
            "help": (
                "List of paths to calibration images used to calibrate "
                "relative color correction. Used only if 'path' is not provided; "
                "requires interactive=true."
            ),
            "widget": "multi_file",
        },
    )
    """Calibration images used to calibrate relative color correction."""
    interactive: bool = field(
        default=False,
        metadata={
            "name": "Interactive calibration",
            "help": "Enable interactive calibration mode using the GUI assistant. "
            "Required when calibrating from images (when 'path' is not provided).",
        },
    )
    """Whether interactive calibration is allowed."""
    mode: Literal["custom", "tensorial"] = field(
        default="custom",
        metadata={
            "name": "Calibration mode",
            "help": (
                "'custom': Manual selection of similar colors and reference colors via GUI. "
                "'tensorial': Automatic tensorial calibration without manual selection."
            ),
            "options": ["custom", "tensorial"],
        },
    )
    """Mode for color sample selection during interactive calibration."""
    method: Literal["polynomial"] = field(
        default="polynomial",
        metadata={
            "name": "Approximation method",
            "help": (
                "Method for spatial color-correction approximation "
                "(polynomial is currently the only supported method)."
            ),
            "options": ["polynomial"],
        },
    )
    """Approximation method for spatial color correction."""
    degree: int = field(
        default=2,
        metadata={
            "name": "Polynomial degree",
            "help": (
                "Degree of the polynomial approximation space "
                "(used with method='polynomial'). Higher degrees fit finer spatial "
                "variations but risk overfitting."
            ),
            "placeholder": "e.g., 1, 2, 3",
        },
    )
    """Degree of polynomial approximation."""
    sample_size: int = field(
        default=50,
        metadata={
            "name": "Sample box size",
            "help": (
                "Pixel width of the box-selection assistant during "
                "interactive calibration."
            ),
            "placeholder": "e.g., 30, 50, 100",
        },
    )
    """Pixel width of color-sample selection boxes."""
    debug: bool = field(
        default=False,
        metadata={
            "name": "Debug output",
            "help": "Enable debug-level output during interactive color-sample selection.",
        },
    )
    """Enable debug output during interactive calibration."""

    def load(self, sec: dict) -> "RelativeColorCorrectionConfig":
        """Load relative color correction configuration from a dictionary."""
        if not isinstance(sec, dict):
            raise ValueError(
                "corrections.relative_color must be a configuration table, not a boolean."
            )

        path = sec.get("path", self.path)
        self.path = Path(path) if path is not None else None
        self.images = [Path(p) for p in sec.get("images", self.images)]
        self.interactive = sec.get("interactive", self.interactive)
        if not isinstance(self.interactive, bool):
            raise ValueError(
                "corrections.relative_color.interactive must be a boolean."
            )

        self.mode = sec.get("mode", self.mode)
        self.method = sec.get("method", self.method)
        self.degree = sec.get("degree", self.degree)
        self.sample_size = sec.get("sample_size", self.sample_size)
        self.debug = sec.get("debug", self.debug)

        if self.path is None and len(self.images) == 0:
            raise ValueError(
                "corrections.relative_color must define either 'path' or 'images'."
            )

        return self

    def to_options_dict(self) -> dict:
        """Convert explicit config fields to the plain dict format expected by
        RelativeColorCorrection.

        Returns:
            dict with keys: mode, method, degree, sample_size, debug
        """
        return {
            "mode": self.mode,
            "method": self.method,
            "degree": self.degree,
            "sample_size": self.sample_size,
            "debug": self.debug,
        }


@dataclass
class IlluminationCorrectionConfig:
    """Configuration for illumination correction.

    Attributes:
        labels: List of label IDs to use for illumination correction (default: []).
        interpolation: Interpolation method for scaling ("rbf", "quartic", "illumination";
            default: "illumination").
        colorspace: Color space for interpolation ("rgb", "rgb-scalar", "lab", "lab-scalar",
            "hsl", "hsl-scalar", "gray"; default: "hsl-scalar").
        width: Width of patches used for interpolation in pixels (default: 100).
        num_samples: Number of sample patches to use (default: 30).
        seed: Random seed for reproducible patch sampling (default: 42).
        sigma: Sigma for Gaussian smoothing of the illumination map in pixels (default: 100.0).
        outliers: Fraction of outliers [0.0–1.0] to discard (default: 0.1).
        bounds: [min, max] bounds for illumination correction factors (default: [0.5, 2.0]).
    """

    labels: list[int] = field(
        default_factory=list,
        metadata={
            "name": "Labels",
            "help": "List of label IDs to use for illumination correction.",
            "placeholder": "e.g., [4, 5, 6]",
        },
    )
    """List of labels to use for illumination correction. Overrides `label` if not empty."""
    interpolation: Literal["rbf", "quartic", "illumination"] = field(
        default="illumination",
        metadata={
            "name": "Interpolation method",
            "help": "Interpolation method to use for scaling.",
            "options": ["rbf", "quartic", "illumination"],
        },
    )
    """Interpolation method to use for scaling."""
    colorspace: Literal[
        "rgb", "rgb-scalar", "lab", "lab-scalar", "hsl", "hsl-scalar", "gray"
    ] = field(
        default="hsl-scalar",
        metadata={
            "name": "Color space",
            "help": "Color space to use for interpolation.",
            "options": [
                "rgb",
                "rgb-scalar",
                "lab",
                "lab-scalar",
                "hsl",
                "hsl-scalar",
                "gray",
            ],
        },
    )
    """Color space to use for interpolation."""
    width: int = field(
        default=100,
        metadata={
            "name": "Patch width",
            "help": "Width of patches to use for interpolation in pixels.",
        },
    )
    """Width of patches to use for interpolation."""
    num_samples: int = field(
        default=30,
        metadata={
            "name": "Number of samples",
            "help": "Number of sample patches to use for interpolation.",
        },
    )
    """Number of patches to use for interpolation."""
    seed: int = field(
        default=42,
        metadata={
            "name": "Random seed",
            "help": "Random seed for reproducible patch sampling.",
        },
    )
    """Random seed for patch sampling."""
    sigma: float = field(
        default=100.0,
        metadata={
            "name": "Gaussian sigma",
            "help": (
                "Sigma for Gaussian smoothing of the illumination "
                "correction map in pixels."
            ),
        },
    )
    """Sigma for Gaussian smoothing of the illumination correction map."""
    outliers: float = field(
        default=0.1,
        metadata={
            "name": "Outlier fraction",
            "help": (
                "Fraction of outliers [0.0–1.0] to discard when computing "
                "the correction map."
            ),
        },
    )
    """Fraction of outliers to discard when computing the illumination correction map."""
    bounds: tuple[float, float] = field(
        default=(0.5, 2.0),
        metadata={
            "name": "Correction bounds",
            "help": "Min and max bounds [min, max] for illumination correction factors.",
            "placeholder": "e.g., [0.5, 2.0]",
        },
    )
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
    """Configuration for patchwise illumination correction.

    Attributes:
        baseline_paths: Paths to baseline images (not yet GUI-editable; edit via TOML).
        nw: Number of patches in the width direction (default: 1000).
        eps: Small constant to avoid division by zero (default: 1e-6).
    """

    baseline_paths: list[Path] = field(
        default_factory=list,
        metadata={
            "name": "Baseline image paths",
            "help": (
                "List of paths to baseline images for patchwise illumination correction. "
                "Not yet GUI-editable (requires multi-file picker). "
                'Edit via TOML: baseline_paths = ["path1.jpg", "path2.jpg", ...]'
            ),
            "widget": "multi_file",
        },
    )
    """Paths to baseline images for patchwise illumination correction."""
    nw: int = field(
        default=1000,
        metadata={
            "name": "Number of patches (width)",
            "help": "Number of patches in the width direction.",
            "placeholder": "e.g., 1000",
            "group": "Options",
        },
    )
    """Number of patches in width direction for patchwise illumination correction."""
    eps: float = field(
        default=1e-6,
        metadata={
            "name": "Epsilon (division safety)",
            "help": "Small constant to avoid division by zero.",
            "placeholder": "e.g., 1e-6",
            "group": "Options",
        },
    )
    """Small constant to avoid division by zero in patchwise illumination correction."""

    def load(self, sec: dict) -> "PatchwiseIlluminationCorrectionConfig":
        """Load patchwise illumination correction configuration from a dictionary.

        Args:
            sec: Dictionary containing patchwise illumination correction settings.

         Returns:
            self with loaded configuration

        """
        _baseline_paths = sec.get("baseline_paths", self.baseline_paths)
        self.baseline_paths = [Path(p) for p in _baseline_paths]
        self.nw = sec.get("nw", self.nw)
        self.eps = sec.get("eps", self.eps)
        return self


_CORRECTION_CLASSES: dict[str, type] = {
    "type": TypeCorrectionConfig,
    "resize": ResizeCorrectionConfig,
    "drift": DriftCorrectionConfig,
    "curvature": CurvatureCorrectionConfig,
    "color": ColorCorrectionConfig,
    "relative_color": RelativeColorCorrectionConfig,
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
        default=None, metadata={"name": "Type correction", "active_list_key": "active"}
    )
    resize: ResizeCorrectionConfig | None = field(
        default=None,
        metadata={"name": "Resize correction", "active_list_key": "active"},
    )
    drift: DriftCorrectionConfig | None = field(
        default=None, metadata={"name": "Drift correction", "active_list_key": "active"}
    )
    curvature: CurvatureCorrectionConfig | None = field(
        default=None,
        metadata={
            "name": "Curvature correction",
            "active_list_key": "active",
            "loadable": "curvature",
        },
    )
    color: ColorCorrectionConfig | None = field(
        default=None, metadata={"name": "Color correction", "active_list_key": "active"}
    )
    relative_color: RelativeColorCorrectionConfig | None = field(
        default=None,
        metadata={"name": "Relative color correction", "active_list_key": "active"},
    )
    illumination: IlluminationCorrectionConfig | None = field(
        default=None,
        metadata={"name": "Illumination correction", "active_list_key": "active"},
    )
    patchwise_illumination: PatchwiseIlluminationCorrectionConfig | None = field(
        default=None,
        metadata={
            "name": "Patchwise illumination correction",
            "active_list_key": "active",
        },
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

        patchwise_illumination_sec = sec.get("patchwise_illumination")
        if patchwise_illumination_sec:
            self.patchwise_illumination = PatchwiseIlluminationCorrectionConfig().load(
                patchwise_illumination_sec
            )

        # Identify active corrections
        active_corrections = sec.get("active_corrections", None)
        if active_corrections is not None:
            raise NotImplementedError("active_corrections is not implemented yet.")

        return self
