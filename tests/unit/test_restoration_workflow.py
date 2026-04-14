from types import SimpleNamespace

import numpy as np
import pytest

from darsia.presets.workflows.config.restoration import RestorationConfig, TVDConfig
from darsia.presets.workflows.restoration import IgnoreMaskedRestoration, build_restoration


def test_build_restoration_applies_boolean_porosity_ignore_mask(monkeypatch):
    class _FakeTVD:
        def __init__(self, **kwargs):
            pass

        def __call__(self, img):
            return np.full_like(img, fill_value=-1.0, dtype=float)

    monkeypatch.setattr("darsia.presets.workflows.restoration.darsia.TVD", _FakeTVD)

    restoration_config = RestorationConfig(
        method="tvd",
        options=TVDConfig(weight=0.1),
        ignore=["boolean_porosity"],
    )
    fluidflower = SimpleNamespace(
        boolean_porosity=SimpleNamespace(
            img=np.array([[True, False], [False, True]], dtype=bool)
        )
    )

    restoration = build_restoration(restoration_config, fluidflower)
    assert isinstance(restoration, IgnoreMaskedRestoration)

    signal = np.array([[1.0, 2.0], [3.0, 4.0]])
    restored = restoration(signal)

    np.testing.assert_allclose(restored, np.array([[-1.0, 2.0], [3.0, -1.0]]))


def test_build_restoration_unknown_ignore_mask_raises():
    restoration_config = RestorationConfig(
        method="tvd",
        options=TVDConfig(weight=0.1),
        ignore=["unknown-mask"],
    )
    fluidflower = SimpleNamespace(
        boolean_porosity=SimpleNamespace(img=np.ones((2, 2), dtype=bool))
    )

    with pytest.raises(ValueError, match="Unknown restoration ignore mask"):
        build_restoration(restoration_config, fluidflower)
