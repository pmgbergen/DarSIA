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
    "rescaled_concentration_aq",
    "rescaled_saturation_g",
}


@dataclass
class RescaledMassProducts:
    """Rescaled mass products and scaling metadata."""

    rescaled_result: darsia.SimpleMassAnalysisResults
    mass_scaling_factor: float
    detected_mass_total: float
    exact_mass_total: float
    detected_volume_total: float | None = None
    """Integrated aqueous concentration (volume basis only)."""
    exact_volume_total: float | None = None
    """Injected volume (volume basis only)."""
    basis: str = "mass"
    """Quantity the rescaling matches the injection protocol on: 'mass' or 'volume'."""


def rescale_basis(co2_mass_analysis: Any) -> str:
    """Quantity the rescaling matches the injection protocol on.

    A single-phase fluid model (for example a tracer) is rescaled on the injected
    volume, a multiphase one on the injected mass. Models without a fluid system
    (for example test doubles) use the mass.

    Parameters
    ----------
    co2_mass_analysis : Any
        Fluid model, optionally with a `system` of type `darsia.FluidSystem`.

    Returns
    -------
        "volume" or "mass".
    """
    system = getattr(co2_mass_analysis, "system", None)
    if system is not None and not system.is_multiphase:
        return "volume"
    return "mass"


def check_rescaling_possible(
    co2_mass_analysis: Any, injection_protocol: Any, step: str
) -> None:
    """Raise early if rescaling needs a volumetric injection rate that is missing.

    A single-phase fluid is rescaled on the injected volume.

    Parameters
    ----------
    co2_mass_analysis : Any
        Fluid model.
    injection_protocol : Any
        Injection protocol; protocols without `has_volume` (test doubles) pass.
    step : str
        Name of the step, for the error message.
    """
    if rescale_basis(co2_mass_analysis) == "volume" and not getattr(
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


def requires_rescaled_modes(modes: set[str] | list[str] | tuple[str, ...]) -> bool:
    """Check if any requested mode requires mass rescaling."""
    return any(mode in RESCALED_MODES for mode in modes)


def compute_rescaled_mass_products(
    *,
    mass_analysis_result: darsia.SimpleMassAnalysisResults,
    geometry: darsia.Geometry,
    injection_protocol: Any,
    co2_mass_analysis: darsia.FluidMassAnalysis,
    date: Any = None,
    epsilon: float = EPSILON,
    basis: str | None = None,
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
    """
    basis = rescale_basis(co2_mass_analysis) if basis is None else basis
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
        exact_mass_total = exact_volume_total * co2_mass_analysis.scale
        detected_total, exact_total = detected_volume_total, exact_volume_total

    mass_scaling_factor = (
        exact_total / detected_total if np.abs(detected_total) > epsilon else 1.0
    )
    rescaled_mass = darsia.weight(mass_analysis_result.mass, mass_scaling_factor)
    rescaled_result = co2_mass_analysis.inverse_mass_analysis(rescaled_mass)
    return RescaledMassProducts(
        rescaled_result=rescaled_result,
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
    co2_mass_analysis: darsia.FluidMassAnalysis | None = None,
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
    single_phase = rescale_basis(co2_mass_analysis) == "volume"
    if single_phase:
        del products["saturation_g"]
        del products["mass_g"]
    products = _apply_expert_knowledge_constraints(products, expert_knowledge_adapter)

    requested_modes = set(requested_modes or [])
    if not requires_rescaled_modes(requested_modes):
        return products, None

    if geometry is None or injection_protocol is None or co2_mass_analysis is None:
        raise ValueError(
            "Rescaled modes requested but missing geometry/injection_protocol/"
            "co2_mass_analysis."
        )

    rescaled = compute_rescaled_mass_products(
        mass_analysis_result=mass_analysis_result,
        geometry=geometry,
        injection_protocol=injection_protocol,
        co2_mass_analysis=co2_mass_analysis,
        date=date,
        basis=basis,
    )
    products["rescaled_mass"] = rescaled.rescaled_result.mass
    if not single_phase:
        products["rescaled_saturation_g"] = rescaled.rescaled_result.saturation_g
    products["rescaled_concentration_aq"] = rescaled.rescaled_result.concentration_aq
    products = _apply_expert_knowledge_constraints(products, expert_knowledge_adapter)
    return products, rescaled
