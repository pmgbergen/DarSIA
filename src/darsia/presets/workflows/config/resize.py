"""Configuration of a resize, shared by the corrections and the restoration."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ResizeConfig:
    """Configuration of a resize (used by the corrections and the restoration).

    Supports two modes: uniform scaling via ``scale`` factor, or explicit target shape
    resizing. Exactly one mode must be configured via the ``mode`` field.

    Attributes
    ----------
    mode
        Resize mode — either ``"scale"`` (scale all axes uniformly) or
        ``"target_shape"`` (resize to explicit rows/cols).
    scale
        Uniform scale factor applied to both spatial axes (mode='scale' only).
    target_shape
        Target ``(rows, cols)`` shape to resize to (mode='target_shape' only).
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

    def load(self, sec: dict, context: str) -> "ResizeConfig":
        """Load the resize configuration from a dictionary.

        Parameters
        ----------
        sec : dict
            Dictionary containing the resize settings.
        context : str
            Location of the settings in the config (e.g. ``corrections.resize``),
            for error messages.

        Returns
        -------
            Self with loaded configuration.

        Raises
        ------
        ValueError
            If mode is invalid, if required field for the mode is missing,
            or if target_shape does not have exactly 2 elements.
        """
        mode = sec.get("mode", self.mode)
        if mode not in ("scale", "target_shape"):
            raise ValueError(
                f"[{context}] mode must be 'scale' or 'target_shape', got {mode!r}"
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
            raise ValueError(f"[{context}] mode='scale' requires 'scale' to be set.")
        if mode == "target_shape" and self.target_shape is None:
            raise ValueError(
                f"[{context}] mode='target_shape' requires 'target_shape' to be set."
            )
        if mode == "target_shape" and len(self.target_shape) != 2:
            raise ValueError(
                f"[{context}] target_shape must have exactly 2 elements, "
                f"got {self.target_shape!r}"
            )

        return self

    def resolve(
        self, current_shape: tuple[int, int]
    ) -> tuple[tuple[int, int], float | tuple[float, float]]:
        """Resolve target shape and resize factor for a given current shape.

        Parameters
        ----------
        current_shape : tuple[int, int]
            Current (rows, cols) shape of the image to be
            resized.

        Returns
        -------
            (target_shape, resize_factor) tuple where:
            - target_shape: resolved (rows, cols) target shape.
            - resize_factor: scalar factor (mode='scale') or (scale_x,
            Scale_y) tuple (mode='target_shape') to pass to downstream
            corrections (e.g. CurvatureCorrection) that need to rescale
            pixel-calibrated config values.
        """
        if self.mode == "scale":
            target_shape = tuple(int(round(s * self.scale)) for s in current_shape)
            resize_factor = self.scale
        else:  # mode == "target_shape"
            target_shape = self.target_shape
            scale_x = target_shape[1] / current_shape[1]
            scale_y = target_shape[0] / current_shape[0]
            resize_factor = (scale_x, scale_y)
        return target_shape, resize_factor
