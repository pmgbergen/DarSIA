"""Configuration for image porosity setup."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ImagePorosityConfig:
    """Configuration for image-porosity workflow in :class:`darsia.Rig`.

    Attributes:
        mode: ``"full"`` produces constant full porosity (value ``1`` everywhere);
            ``"from_image"`` derives the porosity from the baseline image using
            :func:`~darsia.patched_porosity_analysis`.  Default is ``"full"``.
        tol: Threshold used by :meth:`~darsia.Rig.setup_boolean_image_porosity` when
            ``mode="from_image"`` to binarise the continuous porosity map.  Ignored in
            ``"full"`` mode (the boolean mask is always all-``True``).  Default ``0.9``.
        patches: Number of patches ``(rows, cols)`` for the patched porosity analysis.
            Only used when ``mode="from_image"``.  Default ``(1, 1)``.
        num_clusters: Number of k-means clusters for the porosity analysis.
            Only used when ``mode="from_image"``.  Default ``5``.
        sample_width: Width of random samples in pixels for the porosity analysis.
            Only used when ``mode="from_image"``.  Default ``50``.
        tol_color_distance: Tolerance for colour-distance filtering in the porosity
            analysis.  Only used when ``mode="from_image"``.  Default ``0.1``.
        tol_color_gradient: Tolerance for colour-gradient filtering in the porosity
            analysis.  Only used when ``mode="from_image"``.  Default ``0.02``.

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

    mode: Literal["full", "from_image"] = "full"
    """Porosity mode: ``"full"`` (constant 1) or ``"from_image"`` (image-derived)."""
    tol: float = 0.9
    """Threshold for boolean image porosity (only used in ``"from_image"`` mode)."""
    patches: tuple[int, int] = (1, 1)
    """Number of patches ``(rows, cols)`` for patched porosity analysis."""
    num_clusters: int = 5
    """Number of k-means clusters for the porosity analysis."""
    sample_width: int = 50
    """Width of random samples (pixels) for the porosity analysis."""
    tol_color_distance: float = 0.1
    """Tolerance for colour-distance filtering in the porosity analysis."""
    tol_color_gradient: float = 0.02
    """Tolerance for colour-gradient filtering in the porosity analysis."""

    def load(self, sec: dict) -> "ImagePorosityConfig":
        """Populate from a TOML-section dictionary.

        Args:
            sec: Dictionary for the ``[image_porosity]`` section.

        Returns:
            self – updated in-place and returned for chaining.

        Raises:
            ValueError: if ``mode`` is not one of the supported values.
            ValueError: if ``tol`` is not a float in ``(0, 1]``.
        """
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
        self.patches = tuple(int(x) for x in patches_raw)  # type: ignore[assignment]

        self.num_clusters = int(sec.get("num_clusters", self.num_clusters))
        self.sample_width = int(sec.get("sample_width", self.sample_width))
        self.tol_color_distance = float(
            sec.get("tol_color_distance", self.tol_color_distance)
        )
        self.tol_color_gradient = float(
            sec.get("tol_color_gradient", self.tol_color_gradient)
        )

        return self
