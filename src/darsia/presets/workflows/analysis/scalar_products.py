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
    co2_mass_analysis: darsia.CO2MassAnalysis,
    date: Any = None,
    epsilon: float = EPSILON,
) -> RescaledMassProducts:
    """Compute rescaled mass products and scaling metadata."""
    detected_mass_total = geometry.integrate(mass_analysis_result.mass)
    exact_mass_total = injection_protocol.injected_mass(date=date)
    mass_scaling_factor = (
        exact_mass_total / detected_mass_total
        if np.abs(detected_mass_total) > epsilon
        else 1.0
    )
    rescaled_mass = darsia.weight(mass_analysis_result.mass, mass_scaling_factor)
    rescaled_result = co2_mass_analysis.inverse_mass_analysis(rescaled_mass)
    return RescaledMassProducts(
        rescaled_result=rescaled_result,
        mass_scaling_factor=mass_scaling_factor,
        detected_mass_total=float(detected_mass_total),
        exact_mass_total=float(exact_mass_total),
    )


def analysis_scalar_products(
    *,
    mass_analysis_result: darsia.SimpleMassAnalysisResults,
    requested_modes: set[str] | list[str] | tuple[str, ...] | None = None,
    geometry: darsia.Geometry | None = None,
    injection_protocol: Any | None = None,
    co2_mass_analysis: darsia.CO2MassAnalysis | None = None,
    date: Any = None,
    expert_knowledge_adapter: "ExpertKnowledgeAdapter | None" = None,
) -> tuple[dict[str, darsia.Image], RescaledMassProducts | None]:
    """Map workflow analysis mode keys to scalar products.

    Returns all base products and computes rescaled products only when requested.
    """
    products: dict[str, darsia.Image] = {
        "concentration_aq": mass_analysis_result.concentration_aq,
        "saturation_g": mass_analysis_result.saturation_g,
        "mass_total": mass_analysis_result.mass,
        "mass": mass_analysis_result.mass,  # backward-compatible alias
        "mass_g": mass_analysis_result.mass_g,
        "mass_aq": mass_analysis_result.mass_aq,
    }
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
    )
    products["rescaled_mass"] = rescaled.rescaled_result.mass
    products["rescaled_saturation_g"] = rescaled.rescaled_result.saturation_g
    products["rescaled_concentration_aq"] = rescaled.rescaled_result.concentration_aq
    products = _apply_expert_knowledge_constraints(products, expert_knowledge_adapter)
    return products, rescaled
