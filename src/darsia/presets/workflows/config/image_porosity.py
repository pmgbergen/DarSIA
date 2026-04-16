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

    Config section (TOML)::

        [image_porosity]
        mode = "full"   # or "from_image"
        tol  = 0.9

    """

    mode: Literal["full", "from_image"] = "full"
    """Porosity mode: ``"full"`` (constant 1) or ``"from_image"`` (image-derived)."""
    tol: float = 0.9
    """Threshold for boolean image porosity (only used in ``"from_image"`` mode)."""

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

        return self
