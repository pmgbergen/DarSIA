"""Shared scalar product utilities for workflow analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

import darsia

if TYPE_CHECKING:
    from darsia.presets.workflows.analysis.expert_knowledge import (
        ExpertKnowledgeAdapter,
    )

# Guard threshold preventing division by (near) zero detected mass.
EPSILON = 1e-12

RESCALED_MODES = {
    "rescaled_mass",
    "rescaled_mass_g",
    "rescaled_mass_aq",
    "rescaled_concentration_aq",
    "rescaled_saturation_g",
}


@dataclass
class RescaledMassProducts:
    """Rescaled mass products and scaling metadata."""

    rescaled_mass: darsia.Image
    rescaled_concentration_aq: darsia.Image
    rescaled_saturation_g: darsia.Image | None
    """None without a gas phase."""
    mass_scaling_factor: float
    detected_mass_total: float
    exact_mass_total: float
    detected_volume_total: float | None = None
    """Integrated aqueous concentration (volume basis only)."""
    exact_volume_total: float | None = None
    """Injected volume (volume basis only)."""
    basis: str = "mass"
    """Quantity the rescaling matches the injection protocol on: 'mass' or 'volume'."""
    rescaled_mass_g: darsia.Image | None = None
    """Mass in the gas phase; only if requested (and with a gas phase)."""
    rescaled_mass_aq: darsia.Image | None = None
    """Mass in the aqueous phase; only if requested."""
    owned: bool = False
    """Whether the rescaled fields are new arrays, which may be modified in place."""


def rescale_basis(fluid_model: Any) -> str:
    """Quantity the rescaling matches the injection protocol on.

    A single-phase fluid model (for example a tracer) is rescaled on the injected
    volume, a multiphase one on the injected mass. Models without a fluid system
    (for example test doubles) use the mass.

    Parameters
    ----------
    fluid_model : Any
        Fluid model, optionally with a `system` of type `darsia.FluidSystem`.

    Returns
    -------
        "volume" or "mass".
    """
    system = getattr(fluid_model, "system", None)
    if system is not None and not system.is_multiphase:
        return "volume"
    return "mass"


def check_rescaling_possible(
    fluid_model: Any, injection_protocol: Any, step: str
) -> None:
    """Raise early if rescaling needs a volumetric injection rate that is missing.

    A single-phase fluid is rescaled on the injected volume.

    Parameters
    ----------
    fluid_model : Any
        Fluid model.
    injection_protocol : Any
        Injection protocol; protocols without `has_volume` (test doubles) pass.
    step : str
        Name of the step, for the error message.
    """
    if rescale_basis(fluid_model) == "volume" and not getattr(
        injection_protocol, "has_volume", True
    ):
        raise ValueError(
            f"{step}: rescaled products of a single-phase fluid need a volumetric "
            "injection rate (rate_mL/s, rate_mL/min or rate_mL/hr) in the injection "
            "protocol."
        )


def _apply_expert_knowledge_constraints(
    products: dict[str, darsia.Image],
    expert_knowledge_adapter: "ExpertKnowledgeAdapter | None",
) -> dict[str, darsia.Image]:
    """Apply expert-knowledge constraints to supported scalar products."""
    if expert_knowledge_adapter is None:
        return products

    constrained = dict(products)
    mode_map = {
        "concentration_aq": "concentration_aq",
        "saturation_g": "saturation_g",
        "rescaled_concentration_aq": "concentration_aq",
        "rescaled_saturation_g": "saturation_g",
    }
    for key, mode in mode_map.items():
        if key in constrained:
            constrained[key] = expert_knowledge_adapter.apply(constrained[key], mode)
    return constrained


def _constrain_rescaled(
    rescaled: RescaledMassProducts,
    expert_knowledge_adapter: "ExpertKnowledgeAdapter | None",
) -> None:
    """Expert-knowledge constraints of the rescaled concentration and saturation.

    Fields owned by `rescaled` are constrained in place, others replaced by constrained
    copies (`ExpertKnowledgeAdapter.apply`); the values are the same.
    """
    if expert_knowledge_adapter is None:
        return
    for attribute, mode in (
        ("rescaled_concentration_aq", "concentration_aq"),
        ("rescaled_saturation_g", "saturation_g"),
    ):
        image = getattr(rescaled, attribute)
        if image is None:
            continue
        if rescaled.owned:
            expert_knowledge_adapter.constrain(image, mode)
        else:
            setattr(rescaled, attribute, expert_knowledge_adapter.apply(image, mode))


def requires_rescaled_modes(modes: set[str] | list[str] | tuple[str, ...]) -> bool:
    """Check if any requested mode requires mass rescaling."""
    return any(mode in RESCALED_MODES for mode in modes)


def compute_rescaled_mass_products(
    *,
    mass_analysis_result: darsia.SimpleMassAnalysisResults,
    geometry: darsia.Geometry,
    injection_protocol: Any,
    fluid_model: darsia.FluidMassAnalysis,
    date: Any = None,
    epsilon: float = EPSILON,
    basis: str | None = None,
    phase_masses: bool = False,
) -> RescaledMassProducts:
    """Compute rescaled mass products and scaling metadata.

    The mass basis matches the integrated mass to the injected mass. The volume basis
    (default for single-phase fluids) matches the integrated aqueous concentration to
    the injected volume; for a tracer this gives the same factor as the mass, since the
    density cancels, and the exact mass is the injected volume times the model density.

    Parameters
    ----------
    basis : str, optional
        "mass" or "volume"; by default derived from the fluid model.
    phase_masses : bool
        Whether to also compute the rescaled masses in the gas and the aqueous phase.
    """
    basis = rescale_basis(fluid_model) if basis is None else basis
    if basis not in ("mass", "volume"):
        raise ValueError(f"Unknown rescaling basis '{basis}', use 'mass' or 'volume'.")

    detected_mass_total = geometry.integrate(mass_analysis_result.mass)
    detected_volume_total = None
    exact_volume_total = None
    if basis == "mass":
        exact_mass_total = injection_protocol.injected_mass(date=date)
        detected_total, exact_total = detected_mass_total, exact_mass_total
    else:
        detected_volume_total = float(
            geometry.integrate(mass_analysis_result.concentration_aq)
        )
        exact_volume_total = float(injection_protocol.injected_volume(date=date))
        exact_mass_total = exact_volume_total * fluid_model.scale
        detected_total, exact_total = detected_volume_total, exact_volume_total

    mass_scaling_factor = (
        exact_total / detected_total if np.abs(detected_total) > epsilon else 1.0
    )
    inverse_mass_fields = getattr(fluid_model, "inverse_mass_fields", None)
    if inverse_mass_fields is not None:
        fields = inverse_mass_fields(
            mass_analysis_result.mass, mass_scaling_factor, phase_masses=phase_masses
        )
        owned = True
    else:
        rescaled_mass = darsia.weight(mass_analysis_result.mass, mass_scaling_factor)
        result = fluid_model.inverse_mass_analysis(rescaled_mass)
        fields = {
            "mass": result.mass,
            "concentration_aq": result.concentration_aq,
            "saturation_g": getattr(result, "saturation_g", None),
            "mass_g": getattr(result, "mass_g", None) if phase_masses else None,
            "mass_aq": getattr(result, "mass_aq", None) if phase_masses else None,
        }
        owned = False
    return RescaledMassProducts(
        rescaled_mass=fields["mass"],
        rescaled_concentration_aq=fields["concentration_aq"],
        rescaled_saturation_g=fields["saturation_g"],
        rescaled_mass_g=fields["mass_g"],
        rescaled_mass_aq=fields["mass_aq"],
        owned=owned,
        mass_scaling_factor=mass_scaling_factor,
        detected_mass_total=float(detected_mass_total),
        exact_mass_total=float(exact_mass_total),
        detected_volume_total=detected_volume_total,
        exact_volume_total=exact_volume_total,
        basis=basis,
    )


def analysis_scalar_products(
    *,
    mass_analysis_result: darsia.SimpleMassAnalysisResults,
    requested_modes: set[str] | list[str] | tuple[str, ...] | None = None,
    geometry: darsia.Geometry | None = None,
    injection_protocol: Any | None = None,
    fluid_model: darsia.FluidMassAnalysis | None = None,
    date: Any = None,
    expert_knowledge_adapter: "ExpertKnowledgeAdapter | None" = None,
    basis: str | None = None,
) -> tuple[dict[str, darsia.Image], RescaledMassProducts | None]:
    """Map workflow analysis mode keys to scalar products.

    Returns all base products and computes rescaled products only when requested. The
    rescaling basis ("mass" or "volume") is derived from the fluid model unless given.
    """
    products: dict[str, darsia.Image] = {
        "concentration_aq": mass_analysis_result.concentration_aq,
        "saturation_g": mass_analysis_result.saturation_g,
        "mass_total": mass_analysis_result.mass,
        "mass": mass_analysis_result.mass,  # backward-compatible alias
        "mass_g": mass_analysis_result.mass_g,
        "mass_aq": mass_analysis_result.mass_aq,
    }
    # A single-phase fluid has no gas: there are no gas products
    single_phase = rescale_basis(fluid_model) == "volume"
    if single_phase:
        del products["saturation_g"]
        del products["mass_g"]
    products = _apply_expert_knowledge_constraints(products, expert_knowledge_adapter)

    requested_modes = set(requested_modes or [])
    if not requires_rescaled_modes(requested_modes):
        return products, None

    if geometry is None or injection_protocol is None or fluid_model is None:
        raise ValueError(
            "Rescaled modes requested but missing geometry/injection_protocol/"
            "fluid_model."
        )

    rescaled = compute_rescaled_mass_products(
        mass_analysis_result=mass_analysis_result,
        geometry=geometry,
        injection_protocol=injection_protocol,
        fluid_model=fluid_model,
        date=date,
        basis=basis,
        phase_masses=bool(requested_modes & {"rescaled_mass_g", "rescaled_mass_aq"}),
    )
    _constrain_rescaled(rescaled, expert_knowledge_adapter)
    products["rescaled_mass"] = rescaled.rescaled_mass
    if "rescaled_mass_aq" in requested_modes:
        products["rescaled_mass_aq"] = rescaled.rescaled_mass_aq
    if not single_phase:
        products["rescaled_saturation_g"] = rescaled.rescaled_saturation_g
        if "rescaled_mass_g" in requested_modes:
            products["rescaled_mass_g"] = rescaled.rescaled_mass_g
    products["rescaled_concentration_aq"] = rescaled.rescaled_concentration_aq
    return products, rescaled
