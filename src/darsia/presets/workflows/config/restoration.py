"""Interface to configure restoration methods."""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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
        weight: Regularization weight.
        max_num_iter: Maximum number of iterations.
        eps: Convergence tolerance.
        omega: Data fidelity weight (only for "heterogeneous bregman").
        regularization: Regularization parameter (only for "heterogeneous bregman").

    """

    method: Literal[
        "chambolle", "anisotropic bregman", "isotropic bregman", "heterogeneous bregman"
    ] = "chambolle"
    weight: float = 0.1
    max_num_iter: int = 200
    eps: float = 2e-4
    omega: float = 1.0
    regularization: float = 1.0
    kwargs: dict = field(default_factory=dict)

    def load(self, sec: dict) -> "TVDConfig":
        self.method = _get_key(sec, "method", self.method, required=False, type_=str)
        self.weight = _get_key(sec, "weight", self.weight, required=False, type_=float)
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
    method: Literal["none", "volume_average", "tvd"] | None = "volume_average"
    options: VolumeAveragingConfig | TVDConfig | None = None

    def load(self, path: Path) -> "RestorationConfig":
        sec = _get_section_from_toml(path, "restoration")
        self.method = _get_key(sec, "method", required=True, type_=str)

        options_sec = sec.get("options", {})
        if self.method == "none":
            self.options = None
        elif self.method == "volume_average":
            self.options = VolumeAveragingConfig().load(options_sec)
        elif self.method == "tvd":
            self.options = TVDConfig().load(options_sec)
        else:
            raise NotImplementedError(
                f"Restoration method {self.method} not supported."
            )
        return self
