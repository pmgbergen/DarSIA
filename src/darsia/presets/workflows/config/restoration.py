"""Interface to configure restoration methods."""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

from .image_porosity import POROSITY_WEIGHTS, load_porosity_mask
from .resize import ResizeConfig
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
    mask: Literal["none", "image_porosity", "boolean_porosity"] = field(
        default="image_porosity",
        metadata={
            "name": "Mask",
            "help": (
                "'image_porosity': average weighted by the image porosity of Setup > "
                "Image porosity (pixels with porosity 0, e.g. the boundary band, do not "
                "contribute); 'boolean_porosity': average over the pixels of the "
                "boolean porosity, equally weighted; 'none': plain average."
            ),
            "options": list(POROSITY_WEIGHTS),
        },
    )

    def load(self, sec: dict) -> "VolumeAveragingConfig":
        self.rev_size = _get_key(
            sec, "rev_size", self.rev_size, required=False, type_=float
        )
        self.mask = load_porosity_mask(
            sec, self.mask, "volume_average", options=POROSITY_WEIGHTS
        )
        return self


@dataclass
class TVDConfig:
    """Configuration for TVD (Total Variation Denoising) restoration.

    Attributes
    ----------
    method
        TVD solver method. One of "chambolle", "anisotropic bregman",
        "isotropic bregman", "heterogeneous bregman".
    weight
        Regularization weight. Either a float or one of the strings
        "image_porosity" (the stored image porosity of the setup step "Image
        porosity" as heterogeneous weight) or "boolean_porosity" (its boolean
        porosity). When a string value is provided, "heterogeneous bregman" is
        automatically selected as the TVD method.
    max_num_iter
        Maximum number of iterations.
    eps
        Convergence tolerance.
    omega
        Data fidelity weight (only for "heterogeneous bregman").
    regularization
        Regularization parameter (only for "heterogeneous bregman").
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
                "(the image porosity of Setup > Image porosity) or 'boolean_porosity' "
                "(its boolean porosity). String values automatically "
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


RESTORATION_METHODS = ("resize", "volume_average", "tvd")
"""Methods of a restoration step."""


@dataclass
class RestorationStepConfig:
    """One restoration step: a method and its options.

    A ``resize`` step changes the grid of the images (always by area averaging); all
    later steps and the rest of the analysis work on the new grid.

    """

    active: bool = field(
        default=True,
        metadata={
            "name": "Active",
            "help": "When unchecked, the step is skipped; its values are kept.",
            "section_active": True,
            "hidden": True,
        },
    )
    """Whether the step is applied (the checkbox of the step in the GUI)."""
    method: Literal["resize", "volume_average", "tvd"] = field(
        default="volume_average",
        metadata={
            "name": "Method",
            "help": (
                "'resize' (change the resolution by area averaging), "
                "'volume_average' (average over a representative elementary volume) "
                "or 'tvd' (total variation denoising)."
            ),
            "options": list(RESTORATION_METHODS),
        },
    )
    """Restoration method of the step."""
    resize_options: ResizeConfig | None = field(
        default=None,
        metadata={
            "name": "Options",
            "toml_key": "options",
            "depends_on": {"field": "method", "value": "resize"},
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

    def load(self, sec: dict, context: str = "restoration") -> "RestorationStepConfig":
        """Load a step from its table.

        Parameters
        ----------
        sec : dict
            Table with ``method``, optional ``options`` and optional ``active``.
        context : str
            Location of the step in the config, for error messages.

        Returns
        -------
        RestorationStepConfig
            The loaded step.

        Raises
        ------
        ValueError
            For unknown keys, a missing method, or options which are not a table.
        NotImplementedError
            For an unknown method.

        """
        unknown = set(sec) - {"active", "method", "options"}
        if unknown:
            raise ValueError(
                f"{context}: unknown keys {sorted(unknown)}; a step has 'method', "
                "'options' and 'active'."
            )
        if "method" not in sec:
            raise ValueError(f"{context}: a step needs a 'method'.")
        method = str(sec["method"]).lower()
        if method not in RESTORATION_METHODS:
            raise NotImplementedError(
                f"{context}: invalid restoration method {sec['method']!r}; "
                f"methods are {list(RESTORATION_METHODS)}."
            )
        self.method = method  # type: ignore[assignment]
        self.active = bool(sec.get("active", True))
        options = sec.get("options", {})
        if not isinstance(options, dict):
            raise ValueError(f"{context}.options must be a table.")
        if method == "resize":
            self.resize_options = ResizeConfig().load(
                options, context=f"{context}.options"
            )
        elif method == "volume_average":
            self.volume_averaging_options = VolumeAveragingConfig().load(options)
        else:
            self.tvd_options = TVDConfig().load(options)
        return self


@dataclass
class RestorationConfig:
    """Restoration: steps applied in order to the signal.

    The steps are given as ``[[restoration.steps]]``. The form with a single method,
    ``[restoration] method = ...`` and ``[restoration.options]``, is read as one step.

    """

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
    steps: list[RestorationStepConfig] = field(
        default_factory=list,
        metadata={
            "name": "Steps",
            "help": (
                "Restoration steps, applied in this order. A resize step changes the "
                "resolution for all later steps and the rest of the analysis."
            ),
            "widget": "dataclass_list",
            "entry_title": "Step",
            "single_entry_keys": ["method", "options"],
        },
    )
    """Restoration steps, in the order of application."""
    ignore: list[str] = field(
        default_factory=list,
        metadata={
            "name": "Ignore regions",
            "help": (
                "Regions excluded from all averaging and denoising steps: "
                "'image_porosity', 'boolean_porosity', 'inner_labels'."
            ),
            "placeholder": "e.g., boolean_porosity",
        },
    )

    def load(self, path: Path | dict) -> "RestorationConfig":
        if isinstance(path, dict):
            sec = path
        else:
            sec = _get_section_from_toml(path, "restoration")

        # Check if restoration is active.
        self.active = bool(sec.get("active", self.active))

        # Allow to mask out certain regions from restoration.
        self.ignore = _get_key(sec, "ignore", default=[], required=False, type_=list)
        if not all(isinstance(entry, str) for entry in self.ignore):
            raise ValueError("restoration.ignore must be a list of strings.")

        # Steps, or a single method (read as one step)
        if "steps" in sec:
            if "method" in sec or "options" in sec:
                raise ValueError(
                    "restoration: give either [[restoration.steps]] or a single "
                    "'method' with [restoration.options], not both."
                )
            raw_steps = sec["steps"]
            if not isinstance(raw_steps, list) or not all(
                isinstance(step, dict) for step in raw_steps
            ):
                raise ValueError("restoration.steps must be an array of tables.")
        elif sec.get("method") is None or str(sec["method"]).lower() == "none":
            raw_steps = []
        else:
            raw_steps = [{"method": sec["method"], "options": sec.get("options", {})}]
        self.steps = [
            RestorationStepConfig().load(step, context=f"restoration.steps[{index}]")
            for index, step in enumerate(raw_steps)
        ]
        return self
