"""Interface to configure restoration methods."""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

from .utils import _get_key, _get_section_from_toml


@dataclass
class VolumeAveragingConfig:
    rev_size: float = field(
        default=0.005,
        metadata={
            "name": "REV size",
            "help": "Size of the representative elementary volume (REV) in meters.",
        },
    )

    def load(self, sec: dict) -> "VolumeAveragingConfig":
        self.rev_size = _get_key(
            sec, "rev_size", self.rev_size, required=False, type_=float
        )
        return self


@dataclass
class TVDConfig:
    """Configuration for TVD (Total Variation Denoising) restoration.

    Attributes:
        method: TVD solver method. One of "chambolle", "anisotropic bregman",
            "isotropic bregman", "heterogeneous bregman".
        weight: Regularization weight. Either a float or one of the strings
            "porosity" (use fluidflower.image_porosity as heterogeneous weight)
            or "boolean_porosity" (use fluidflower.boolean_porosity as heterogeneous
            weight). When a string value is provided, "heterogeneous bregman" is
            automatically selected as the TVD method.
        max_num_iter: Maximum number of iterations.
        eps: Convergence tolerance.
        omega: Data fidelity weight (only for "heterogeneous bregman").
        regularization: Regularization parameter (only for "heterogeneous bregman").

    """

    method: Literal[
        "chambolle", "anisotropic bregman", "isotropic bregman", "heterogeneous bregman"
    ] = field(
        default="chambolle",
        metadata={
            "name": "TVD solver method",
            "help": (
                "Solver method for Total Variation Denoising. "
                "'heterogeneous bregman' is automatically selected when weight is a "
                "string (image_porosity/boolean_porosity)."
            ),
            "options": [
                "chambolle",
                "anisotropic bregman",
                "isotropic bregman",
                "heterogeneous bregman",
            ],
        },
    )
    weight: Union[float, Literal["image_porosity", "boolean_porosity"]] = field(
        default=0.1,
        metadata={
            "name": "Regularization weight",
            "help": (
                "Regularization weight (float value) or special string: 'image_porosity' "
                "(use fluidflower.image_porosity) or 'boolean_porosity' "
                "(use fluidflower.boolean_porosity). String values automatically "
                "select 'heterogeneous bregman' method."
            ),
            "placeholder": "e.g., 0.1 or 'image_porosity'",
        },
    )
    max_num_iter: int = field(
        default=200,
        metadata={
            "name": "Maximum iterations",
            "help": "Maximum number of iterations for the TVD solver.",
        },
    )
    eps: float = field(
        default=2e-4,
        metadata={
            "name": "Convergence tolerance",
            "help": "Convergence tolerance (epsilon) for the TVD solver.",
        },
    )
    omega: float = field(
        default=1.0,
        metadata={
            "name": "Data fidelity weight",
            "help": "Data fidelity weight (only used for 'heterogeneous bregman' method).",
        },
    )
    regularization: float = field(
        default=1.0,
        metadata={
            "name": "Regularization parameter",
            "help": "Regularization parameter (only used for 'heterogeneous bregman' method).",
        },
    )
    kwargs: dict = field(
        default_factory=dict,
        metadata={
            "name": "Additional options",
            "help": "Additional keyword arguments passed to the TVD solver.",
            "hidden": True,
        },
    )

    def load(self, sec: dict) -> "TVDConfig":
        self.method = _get_key(sec, "method", self.method, required=False, type_=str)
        # weight can be float or special string ("porosity" / "boolean-porosity")
        raw_weight = _get_key(sec, "weight", self.weight, required=False)
        if isinstance(raw_weight, str):
            self.weight = raw_weight
        else:
            self.weight = float(raw_weight)
        self.max_num_iter = _get_key(
            sec, "max_num_iter", self.max_num_iter, required=False, type_=int
        )
        self.eps = _get_key(sec, "eps", self.eps, required=False, type_=float)
        self.omega = _get_key(sec, "omega", self.omega, required=False, type_=float)
        self.regularization = _get_key(
            sec, "regularization", self.regularization, required=False, type_=float
        )
        # Collect any remaining keys as extra kwargs, excluding known dataclass fields
        known_keys = {f.name for f in dataclasses.fields(self)} - {"kwargs"}
        self.kwargs = {k: v for k, v in sec.items() if k not in known_keys}
        return self


@dataclass
class RestorationConfig:
    active: bool = field(
        default=True,
        metadata={
            "name": "Activate restoration",
            "help": (
                "When unchecked, restoration is skipped (equivalent to omitting "
                "[restoration] from the config). Other values below are preserved even "
                "when unchecked."
            ),
            "section_active": True,
            "hidden": True,
        },
    )
    """Whether to enable restoration processing."""
    method: Literal["volume_average", "tvd"] | None = field(
        default="volume_average",
        metadata={
            "name": "Restoration method",
            "help": (
                "Restoration method to apply: 'volume_average' (simple averaging) or "
                "'tvd' (Total Variation Denoising)."
            ),
            "options": ["volume_average", "tvd"],
        },
    )
    volume_averaging_options: VolumeAveragingConfig | None = field(
        default=None,
        metadata={
            "name": "Options",
            "toml_key": "options",
            "depends_on": {"field": "method", "value": "volume_average"},
        },
    )
    tvd_options: TVDConfig | None = field(
        default=None,
        metadata={
            "name": "Options",
            "toml_key": "options",
            "depends_on": {"field": "method", "value": "tvd"},
        },
    )
    ignore: list[str] = field(
        default_factory=list,
        metadata={
            "name": "Ignore regions",
            "help": "List of region/label names to exclude from restoration.",
            "placeholder": "e.g., label1, label2",
        },
    )

    @property
    def options(self) -> VolumeAveragingConfig | TVDConfig | None:
        """Backward-compatibility property: returns the active options (VA or TVD)."""
        if self.method == "volume_average":
            return self.volume_averaging_options
        elif self.method == "tvd":
            return self.tvd_options
        return None

    def load(self, path: Path | dict) -> "RestorationConfig":
        if isinstance(path, dict):
            sec = path
        else:
            sec = _get_section_from_toml(path, "restoration")

        # Check if restoration is active.
        self.active = bool(sec.get("active", self.active))

        # Select and validate the restoration method.
        method_str = _get_key(
            sec, "method", default=self.method, required=False, type_=str
        )
        if method_str is not None:
            self.method = method_str.lower()
        else:
            self.method = None
        if self.method == "none":
            self.method = None
        elif self.method is not None and self.method not in ["volume_average", "tvd"]:
            raise NotImplementedError(f"Invalid restoration method: {self.method}")

        # Allow to mask out certain regions from restoration.
        self.ignore = _get_key(sec, "ignore", default=[], required=False, type_=list)
        if not all(isinstance(entry, str) for entry in self.ignore):
            raise ValueError("restoration.ignore must be a list of strings.")

        # Allow for method-specific options under an "options" subsection.
        options_sec = sec.get("options", {})
        if self.method is None:
            pass
        elif self.method == "volume_average":
            self.volume_averaging_options = VolumeAveragingConfig().load(options_sec)
        elif self.method == "tvd":
            self.tvd_options = TVDConfig().load(options_sec)

        return self
