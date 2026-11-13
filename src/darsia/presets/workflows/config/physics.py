"""Configuration of the physical model underlying the analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from darsia.pvt import FluidSystem

from .utils import _get_key, _get_section_from_toml

SUPPORTED_FLUIDS = ("co2", "tracer")
"""Supported analysis types: multiphase CO2 or single-phase tracer."""


@dataclass(frozen=True)
class Interpretation:
    """How a calibrated color path signal is understood.

    The `quantity` names the calibration folder below `interpolation`. The `closure`
    maps the signal to the fields (aqueous concentration, gas saturation).
    """

    quantity: str
    """Quantity of the calibration, e.g. 'mass' or 'concentration_aq'."""
    closure: str
    """Closure from signal to fields: 'flash' (thresholds) or 'identity' (no gas)."""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Interpretation":
        # Unknown keys (e.g. a stored 'anchor' of earlier calibrations) are ignored.
        return cls(quantity=data["quantity"], closure=data["closure"])


INTERPRETATIONS = {
    "co2": Interpretation(quantity="mass", closure="flash"),
    "tracer": Interpretation(quantity="concentration_aq", closure="identity"),
}
"""Default interpretation (preset) of each supported fluid."""

GAS_MODES = ("saturation_g", "rescaled_saturation_g", "mass_g")
"""Analysis modes that only exist with a gas phase."""

GAS_EXPERT_KNOWLEDGE_KEYS = ("saturation_g", "no_saturation_g")
"""Keys of [expert_knowledge] constraining the gas saturation."""


def check_fluid_modes(config, step: str, modes: dict[str, str | None]) -> None:
    """Raise if a mode is not available for the configured fluid.

    Parameters
    ----------
    config : FluidFlowerConfig
        Configuration; ignored if it has no `[physics]` (for example test doubles).
    step : str
        Name of the step, for the error message.
    modes : dict[str, str | None]
        Configured modes by the key they were configured at, for example
        `{"analysis.thresholding.layers.plume.mode": "saturation_g"}`.
    """
    physics = getattr(config, "physics", None)
    if physics is not None:
        physics.check_modes(step, modes)


@dataclass
class PhysicsConfig:
    """Physical model of the experiment, selecting the color-to-mass model.

    `co2` is the multiphase analysis (gas saturation, dissolved concentration, flash).
    `tracer` is single-phase: no gas, saturation_g = mass_g = 0 and all mass is
    aqueous.
    """

    fluid: str = field(
        default="co2",
        metadata={
            "name": "Fluid",
            "help": "'co2' for multiphase CO2 analysis (gas saturation, dissolved "
            "concentration, flash). 'tracer' for single-phase tracer analysis "
            "(no gas phase, all mass is aqueous).",
            "options": list(SUPPORTED_FLUIDS),
        },
    )

    @property
    def system(self) -> FluidSystem:
        """Fluids of the configured fluid: two-phase water and CO2, or water only."""
        if self.fluid == "co2":
            return FluidSystem.two_phase(aqueous="water", gas="co2")
        if self.fluid == "tracer":
            return FluidSystem.single_phase(aqueous="water")
        raise ValueError(f"No fluid system for [physics] fluid = '{self.fluid}'.")

    @property
    def interpretation(self) -> Interpretation:
        """Interpretation of the color path signal for the configured fluid."""
        return INTERPRETATIONS[self.fluid]

    @property
    def unavailable_modes(self) -> tuple[str, ...]:
        """Analysis modes that are not available for the configured fluid."""
        return () if self.system.is_multiphase else GAS_MODES

    def check_modes(self, step: str, modes: dict[str, str | None]) -> None:
        """Raise if a configured mode is not available for the configured fluid.

        Parameters
        ----------
        step : str
            Name of the step, for the error message.
        modes : dict[str, str | None]
            Configured modes by the key they were configured at.
        """
        unavailable = self.unavailable_modes
        offending = {key: mode for key, mode in modes.items() if mode in unavailable}
        if offending:
            listed = ", ".join(f"{key} = '{mode}'" for key, mode in offending.items())
            raise ValueError(
                f"{step}: {listed} not available for [physics] fluid = "
                f"'{self.fluid}' (no gas phase). Use a concentration or mass mode, "
                "for example 'concentration_aq' or 'rescaled_concentration_aq'."
            )

    def check_expert_knowledge(self, expert_knowledge) -> None:
        """Raise if [expert_knowledge] constrains the gas saturation of a single phase.

        Parameters
        ----------
        expert_knowledge : ExpertKnowledgeConfig | None
            Loaded [expert_knowledge] section.
        """
        if self.system.is_multiphase or expert_knowledge is None:
            return
        offending = [
            key
            for key in GAS_EXPERT_KNOWLEDGE_KEYS
            if getattr(expert_knowledge, key, None)
        ]
        if offending:
            raise ValueError(
                f"[expert_knowledge] {', '.join(offending)} constrain the gas "
                f"saturation, which does not exist for [physics] fluid = "
                f"'{self.fluid}' (no gas phase). Remove the key(s)."
            )

    def require(self, step: str, supported: tuple[str, ...] = ("co2",)) -> None:
        """Raise if `step` is not implemented for the configured fluid.

        This is the single place where workflows decide on the physics; calibration
        and analysis of the mass model must both go through it.
        """
        if self.fluid not in supported:
            raise NotImplementedError(
                f"{step} is not available for [physics] fluid = '{self.fluid}' "
                f"(supported: {', '.join(supported)})."
            )

    def load(self, path: Path | list[Path]) -> "PhysicsConfig":
        try:
            sec = _get_section_from_toml(path, "physics")
        except KeyError:
            # Missing section: keep the multiphase CO2 default.
            sec = {}
        self.fluid = _get_key(
            sec, "fluid", default="co2", required=False, type_=str
        ).lower()
        if self.fluid not in SUPPORTED_FLUIDS:
            raise ValueError(
                f"physics.fluid must be one of {SUPPORTED_FLUIDS}, got '{self.fluid}'."
            )
        return self
