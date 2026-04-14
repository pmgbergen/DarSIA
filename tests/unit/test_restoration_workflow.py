from types import SimpleNamespace

import numpy as np
import pytest

import darsia
from darsia.presets.workflows.config.restoration import (
    RestorationConfig,
    TVDConfig,
    VolumeAveragingConfig,
)
from darsia.presets.workflows.restoration import build_restoration


def test_build_restoration_applies_boolean_porosity_ignore_mask_to_volume_averaging(
    monkeypatch,
):
    captured = {}

    class _FakeVolumeAveraging:
        def __init__(self, rev, mask):
            captured["mask"] = mask

        def __call__(self, img):
            return img

    monkeypatch.setattr(
        "darsia.presets.workflows.restoration.darsia.VolumeAveraging",
        _FakeVolumeAveraging,
    )
    monkeypatch.setattr(
        "darsia.presets.workflows.restoration.darsia.REV",
        lambda size, img: SimpleNamespace(size=size, img=img),
    )

    restoration_config = RestorationConfig(
        method="volume_average",
        options=VolumeAveragingConfig(rev_size=3),
        ignore=["boolean_porosity"],
    )
    baseline = darsia.ScalarImage(np.zeros((2, 2), dtype=float), space_dim=2)
    image_porosity = darsia.ScalarImage(
        np.array([[0.2, 0.4], [0.6, 0.8]], dtype=float), space_dim=2
    )
    boolean_porosity = darsia.ScalarImage(
        np.array([[True, False], [True, True]], dtype=bool), space_dim=2
    )
    fluidflower = SimpleNamespace(
        baseline=baseline,
        image_porosity=image_porosity,
        boolean_porosity=boolean_porosity,
    )

    build_restoration(restoration_config, fluidflower)

    np.testing.assert_allclose(
        captured["mask"].img,
        np.array([[0.2, 0.0], [0.6, 0.8]], dtype=float),
    )


def test_build_restoration_applies_boolean_porosity_ignore_mask_to_tvd_weight(
    monkeypatch,
):
    captured = {}

    class _FakeTVD:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __call__(self, img):
            return img

    monkeypatch.setattr("darsia.presets.workflows.restoration.darsia.TVD", _FakeTVD)

    restoration_config = RestorationConfig(
        method="tvd",
        options=TVDConfig(method="chambolle", weight=2.0),
        ignore=["boolean_porosity"],
    )
    baseline = darsia.ScalarImage(np.zeros((2, 2), dtype=float), space_dim=2)
    image_porosity = darsia.ScalarImage(np.ones((2, 2), dtype=float), space_dim=2)
    boolean_porosity = darsia.ScalarImage(
        np.array([[True, False], [False, True]], dtype=bool), space_dim=2
    )
    fluidflower = SimpleNamespace(
        baseline=baseline,
        image_porosity=image_porosity,
        boolean_porosity=boolean_porosity,
    )

    build_restoration(restoration_config, fluidflower)

    np.testing.assert_allclose(
        captured["weight"],
        np.array([[2.0, 0.0], [0.0, 2.0]], dtype=float),
    )
    assert captured["method"] == "heterogeneous bregman"


def test_build_restoration_unknown_ignore_mask_raises():
    restoration_config = RestorationConfig(
        method="tvd",
        options=TVDConfig(weight=0.1),
        ignore=["unknown-mask"],
    )
    fluidflower = SimpleNamespace(
        image_porosity=SimpleNamespace(img=np.ones((2, 2), dtype=float)),
        boolean_porosity=SimpleNamespace(img=np.ones((2, 2), dtype=bool)),
    )

    with pytest.raises(ValueError, match="Unknown restoration ignore mask"):
        build_restoration(restoration_config, fluidflower)
