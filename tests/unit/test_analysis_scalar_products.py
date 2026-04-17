from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import darsia
from darsia.presets.workflows.analysis.scalar_products import analysis_scalar_products


def _mass_result(mass_value: float) -> darsia.SimpleMassAnalysisResults:
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
