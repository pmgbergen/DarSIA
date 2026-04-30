"""Interface to configure restoration methods."""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

from .utils import _get_key, _get_section_from_toml


@dataclass
class VolumeAveragingConfig:
    rev_size: int = 3

    def load(self, sec: dict) -> "VolumeAveragingConfig":
        self.rev_size = _get_key(sec, "rev_size", self.rev_size, required=False)
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
    ] = "chambolle"
    weight: Union[float, Literal["image_porosity", "boolean_porosity"]] = 0.1
    max_num_iter: int = 200
    eps: float = 2e-4
    omega: float = 1.0
    regularization: float = 1.0
    kwargs: dict = field(default_factory=dict)

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
    method: Literal["volume_average", "tvd"] | None = "volume_average"
    options: VolumeAveragingConfig | TVDConfig | None = None
    ignore: list[str] = field(default_factory=list)

    def load(self, path: Path) -> "RestorationConfig":
        sec = _get_section_from_toml(path, "restoration")

        # Select and validate the restoration method.
        self.method = _get_key(sec, "method", required=True, type_=str).lower()
        if self.method == "none":
            self.method = None
        if self.method not in ["volume_average", "tvd", None]:
            raise ValueError(f"Invalid restoration method: {self.method}")

        # Allow to mask out certain regions from restoration.
        self.ignore = _get_key(sec, "ignore", default=[], required=False, type_=list)
        if not all(isinstance(entry, str) for entry in self.ignore):
            raise ValueError("restoration.ignore must be a list of strings.")

        # Allow for method-specific options under an "options" subsection.
        options_sec = sec.get("options", {})
        if self.method is None:
            pass
        elif self.method == "volume_average":
            self.options = VolumeAveragingConfig().load(options_sec)
        elif self.method == "tvd":
            self.options = TVDConfig().load(options_sec)

        return self
