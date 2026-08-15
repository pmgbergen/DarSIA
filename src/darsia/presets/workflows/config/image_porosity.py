"""Configuration for image porosity setup."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .utils import _get_section_from_toml


@dataclass
class ImagePorosityConfig:
    """Configuration for image-porosity workflow in :class:`darsia.Rig`.

    Config section (TOML)::

        [image_porosity]
        mode = "full"   # or "from_image"
        tol  = 0.9

        # Additional options for from_image mode:
        patches              = [1, 1]
        num_clusters         = 5
        sample_width         = 50
        tol_color_distance   = 0.1
        tol_color_gradient   = 0.02

    """

    active: bool = field(
        default=True,
        metadata={
            "name": "Enable image porosity",
            "help": "When unchecked, image porosity is skipped (equivalent to omitting [image_porosity] from the config). Other values below are preserved even when unchecked.",
            "section_active": True,
            "hidden": True,
        },
    )
    """Whether to enable image porosity setup. When False, behaves like the section is absent."""
    mode: Literal["full", "from_image"] = field(
        default="full",
        metadata={
            "name": "Mode",
            "help": "'full' produces constant full porosity (1 everywhere); 'from_image' derives porosity from the baseline image.",
            "options": ["full", "from_image"],
        },
    )
    """Porosity mode: ``"full"`` (constant 1) or ``"from_image"`` (image-derived)."""
    tol: float = field(
        default=0.9,
        metadata={
            "name": "Boolean threshold",
            "help": "Threshold to binarize the continuous porosity map (mode='from_image' only).",
        },
    )
    """Threshold for boolean image porosity (only used in ``"from_image"`` mode)."""
    patches: tuple[int, int] = field(
        default=(1, 1),
        metadata={
            "name": "Patches",
            "help": "Number of patches (rows, cols) for patched porosity analysis, e.g. [10, 10] (mode='from_image' only).",
        },
    )
    """Number of patches ``(rows, cols)`` for patched porosity analysis."""
    num_clusters: int = field(
        default=5,
        metadata={
            "name": "Number of clusters",
            "help": "Number of k-means clusters for the porosity analysis (mode='from_image' only).",
        },
    )
    """Number of k-means clusters for the porosity analysis."""
    sample_width: int = field(
        default=50,
        metadata={
            "name": "Sample width",
            "help": "Width of random samples (pixels) for the porosity analysis (mode='from_image' only).",
        },
    )
    """Width of random samples (pixels) for the porosity analysis."""
    tol_color_distance: float = field(
        default=0.1,
        metadata={
            "name": "Tolerance for color distance",
            "help": "Tolerance for colour-distance filtering in the porosity analysis (mode='from_image' only).",
        },
    )
    """Tolerance for colour-distance filtering in the porosity analysis."""
    tol_color_gradient: float = field(
        default=0.02,
        metadata={
            "name": "Tolerance for color gradient",
            "help": "Tolerance for colour-gradient filtering in the porosity analysis (mode='from_image' only).",
        },
    )
    """Tolerance for colour-gradient filtering in the porosity analysis."""

    def load(self, path: Path | list[Path]) -> "ImagePorosityConfig":
        """Populate from a TOML config file.

        Reads the ``[image_porosity]`` section and updates the instance in-place.

        Args:
            path: Path (or list of paths) to the TOML config file(s).

        Returns:
            self – updated in-place and returned for chaining.

        Raises:
            KeyError: if the ``[image_porosity]`` section is absent.
            ValueError: if ``mode`` is not one of the supported values.
            ValueError: if ``tol`` is not a float in ``(0, 1]``.
            ValueError: if ``patches`` does not have exactly 2 elements.
        """
        sec = _get_section_from_toml(path, "image_porosity")
        return self._load_dict(sec)

    def _load_dict(self, sec: dict) -> "ImagePorosityConfig":
        """Populate from a plain dictionary (e.g. a parsed TOML section).

        Args:
            sec: Dictionary for the ``[image_porosity]`` section.

        Returns:
            self – updated in-place and returned for chaining.

        Raises:
            ValueError: if ``mode`` is not one of the supported values.
            ValueError: if ``tol`` is not a float in ``(0, 1]``.
            ValueError: if ``patches`` does not have exactly 2 elements.
        """
        self.active = bool(sec.get("active", self.active))

        mode = sec.get("mode", self.mode)
        if mode not in ("full", "from_image"):
            raise ValueError(
                f"[image_porosity] mode must be 'full' or 'from_image', got {mode!r}"
            )
        self.mode = mode

        tol = float(sec.get("tol", self.tol))
        if not (0.0 < tol <= 1.0):
            raise ValueError(f"[image_porosity] tol must be in (0, 1], got {tol!r}")
        self.tol = tol

        patches_raw = sec.get("patches", list(self.patches))
        if len(patches_raw) != 2:
            raise ValueError(
                f"[image_porosity] patches must be a list of 2 integers, got {patches_raw!r}"
            )
        self.patches = (int(patches_raw[0]), int(patches_raw[1]))

        self.num_clusters = int(sec.get("num_clusters", self.num_clusters))
        self.sample_width = int(sec.get("sample_width", self.sample_width))
        self.tol_color_distance = float(
            sec.get("tol_color_distance", self.tol_color_distance)
        )
        self.tol_color_gradient = float(
            sec.get("tol_color_gradient", self.tol_color_gradient)
        )

        return self
