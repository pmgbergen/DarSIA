"""Frozen flash support must not drift when the reporting domain grows."""

from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.mass import n1_models
from darsia.fluidflower.mass.production_precision import PixelMath


@pytest.fixture
def response_inputs(monkeypatch):
    """Keep optics and thermodynamics controlled while using the real flash."""
    shape = (2, 2)
    support = np.ones(shape, bool)
    frame = SimpleNamespace(
        domain=support,
        known_physical_domain=support,
        supported_mask=support,
        clear_water_supported_mask=support,
        current_rgb=np.full((*shape, 3), .8, np.float32),
        baseline_rgb=np.ones((*shape, 3), np.float32),
        clear_water_rgb=np.ones((*shape, 3), np.float32),
        phase_labels=np.ones(shape, np.uint8),
        facies=np.ones(shape, np.uint8),
        elapsed_h=1., pressure_bar=1., temperature_c=20.,
        pressure_gradient_bar_per_m=0., temperature_gradient_c_per_m=0.,
        recipe=dict(alkalinity_M=.001, btb_M=.0001),
    )

    class Analysis:
        def __init__(self, *args):
            pass

        def scalar(self, array, elapsed_h):
            return SimpleNamespace(img=array)

        def interpretation(self, array):
            return SimpleNamespace(img=np.full(shape, .5, np.float32))

    class Physics:
        def __init__(self, *args, **kwargs):
            self.solubility_co2 = np.asarray([[1., 2.], [3., 8.]], np.float32)

    monkeypatch.setattr(n1_models, "NativeAnalysis", Analysis)
    monkeypatch.setattr(n1_models, "TypedPhysics", Physics)
    models = dict(metadata=dict(ignore_labels=[], flash_bounds=dict(
        min_value_aq=0., max_value_aq=1., min_value_g=0., max_value_g=1.)))
    return frame, models, PixelMath("float32"), {1: .2}


def test_explicit_flash_basis_is_independent_of_reporting_support(response_inputs):
    """A larger mass domain cannot change the frozen colour-to-CO2 flash."""
    frame, models, math, swi = response_inputs
    basis = np.asarray([[True, True], [False, False]])
    reference = n1_models.prepare_response(frame, models, math, swi,
                                          solubility_domain=basis)
    assert reference["flash"].titration_params["co2_sat_M"] == 1.5 / 44.0095
    original = n1_models.prepare_response(frame, models, math, swi)
    assert original["flash"].titration_params["co2_sat_M"] == 2.5 / 44.0095
    # The flash basis is not a reporting mask: all pixels retain their colour path.
    assert reference["quantifiable"].all()
    assert frame.domain.all()
    assert np.array_equal(basis, [[True, True], [False, False]])


@pytest.mark.parametrize("bad", [np.zeros((2, 2), bool), np.ones((2, 2)),
                                np.ones((1, 2), bool)])
def test_flash_basis_rejects_implicit_or_missing_support(response_inputs, bad):
    """Invalid reference support cannot silently adopt the reporting domain."""
    with pytest.raises(ValueError, match="Flash solubility support"):
        n1_models.prepare_response(*response_inputs, solubility_domain=bad)
