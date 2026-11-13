"""Fluid of the loaded config (`[physics] fluid`), for fluid-aware GUI fields."""

from darsia.presets.workflows.config.physics import SUPPORTED_FLUIDS

DEFAULT_FLUID = "co2"
"""Fluid of a config without `[physics]` (multiphase CO2)."""


def current_fluid(config_dict: dict | None) -> str:
    """Return the configured fluid, falling back to CO2 for missing or unknown values.

    Parameters
    ----------
    config_dict : dict | None
        Loaded TOML config.
    """
    physics = (config_dict or {}).get("physics", {})
    fluid = physics.get("fluid", DEFAULT_FLUID) if isinstance(physics, dict) else None
    fluid = str(fluid).lower() if fluid else DEFAULT_FLUID
    return fluid if fluid in SUPPORTED_FLUIDS else DEFAULT_FLUID


def is_visible_for_fluid(setting: dict, fluid: str) -> bool:
    """Whether a setting is shown for the fluid (metadata `fluids`, default: all)."""
    fluids = setting.get("fluids")
    return fluids is None or fluid in fluids
