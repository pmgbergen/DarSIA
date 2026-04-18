from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import darsia
from darsia import make_coordinate
from darsia.presets.workflows.analysis.expert_knowledge import ExpertKnowledgeAdapter
from darsia.presets.workflows.analysis.scalar_products import analysis_scalar_products
from darsia.presets.workflows.config.roi import RoiConfig


def _mass_result(mass_value: float) -> darsia.SimpleMassAnalysisResults:
    """Create a compact synthetic mass result for scalar-product plumbing tests.

    The fields intentionally reuse the same scalar values across products since these
    tests only verify product selection/rescaling orchestration.
    """
    mass = darsia.ScalarImage(np.full((2, 2), mass_value, dtype=float), dimensions=[1.0, 1.0])
    return darsia.SimpleMassAnalysisResults(
        name="img",
        date=None,
        time=0.0,
        mass=mass,
        mass_g=mass.copy(),
        mass_aq=mass.copy(),
        saturation_g=mass.copy(),
        color_signal=mass.copy(),
        concentration_aq=mass.copy(),
    )


def test_analysis_scalar_products_skips_rescaling_if_not_requested() -> None:
    class _Co2Mass:
        def __init__(self) -> None:
            self.calls = 0

        def inverse_mass_analysis(self, mass):
            self.calls += 1
            return _mass_result(float(np.mean(mass.img)))

    co2 = _Co2Mass()
    products, rescaled = analysis_scalar_products(
        mass_analysis_result=_mass_result(0.25),
        requested_modes={"mass_total"},
        geometry=darsia.Geometry(space_dim=2, num_voxels=(2, 2), dimensions=[1.0, 1.0]),
        injection_protocol=SimpleNamespace(injected_mass=lambda **kwargs: 10.0),
        co2_mass_analysis=co2,
    )

    assert rescaled is None
    assert "rescaled_mass" not in products
    assert co2.calls == 0


def test_analysis_scalar_products_zero_detected_mass_uses_unit_scaling() -> None:
    class _Co2Mass:
        def __init__(self) -> None:
            self.calls = 0

        def inverse_mass_analysis(self, mass):
            self.calls += 1
            return darsia.SimpleMassAnalysisResults(
                name=mass.name,
                date=mass.date,
                time=mass.time,
                mass=mass.copy(),
                mass_g=mass.copy(),
                mass_aq=mass.copy(),
                saturation_g=mass.copy(),
                color_signal=mass.copy(),
                concentration_aq=mass.copy(),
            )

    co2 = _Co2Mass()
    products, rescaled = analysis_scalar_products(
        mass_analysis_result=_mass_result(0.0),
        requested_modes={"rescaled_mass"},
        geometry=darsia.Geometry(space_dim=2, num_voxels=(2, 2), dimensions=[1.0, 1.0]),
        injection_protocol=SimpleNamespace(injected_mass=lambda **kwargs: 10.0),
        co2_mass_analysis=co2,
    )

    assert rescaled is not None
    assert rescaled.mass_scaling_factor == 1.0
    assert np.allclose(products["rescaled_mass"].img, 0.0)
    assert co2.calls == 1


def test_analysis_scalar_products_applies_expert_knowledge_to_rescaled_fields() -> None:
    class _Co2Mass:
        def inverse_mass_analysis(self, mass):
            return darsia.SimpleMassAnalysisResults(
                name=mass.name,
                date=mass.date,
                time=mass.time,
                mass=mass.copy(),
                mass_g=mass.copy(),
                mass_aq=mass.copy(),
                saturation_g=mass.copy(),
                color_signal=mass.copy(),
                concentration_aq=mass.copy(),
            )

    roi = RoiConfig()
    roi.roi = make_coordinate([[0.0, 0.0], [0.5, 1.0]])
    roi.name = "left_half"
    adapter = ExpertKnowledgeAdapter(
        saturation_g_rois={"left_half": roi},
        concentration_aq_rois={"left_half": roi},
    )

    products, rescaled = analysis_scalar_products(
        mass_analysis_result=_mass_result(1.0),
        requested_modes={"rescaled_saturation_g", "rescaled_concentration_aq"},
        geometry=darsia.Geometry(space_dim=2, num_voxels=(2, 2), dimensions=[1.0, 1.0]),
        injection_protocol=SimpleNamespace(injected_mass=lambda **kwargs: 1.0),
        co2_mass_analysis=_Co2Mass(),
        expert_knowledge_adapter=adapter,
    )

    assert rescaled is not None
    assert np.any(products["rescaled_saturation_g"].img == 0.0)
    assert np.any(products["rescaled_concentration_aq"].img == 0.0)
