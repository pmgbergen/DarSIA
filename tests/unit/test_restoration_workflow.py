from types import SimpleNamespace

import numpy as np
import pytest

import darsia
from darsia.presets.workflows.config.resize import ResizeConfig
from darsia.presets.workflows.config.restoration import (
    RestorationConfig,
    RestorationStepConfig,
    TVDConfig,
    VolumeAveragingConfig,
)
from darsia.presets.workflows.restoration import RestorationSteps, build_restoration

META = {"space_dim": 2, "dimensions": [1.0, 2.0]}


def _config(*steps: RestorationStepConfig, ignore=None) -> RestorationConfig:
    return RestorationConfig(steps=list(steps), ignore=ignore or [])


def _resize(scale: float) -> RestorationStepConfig:
    return RestorationStepConfig(
        method="resize", resize_options=ResizeConfig(scale=scale)
    )


def _rig(shape=(2, 2), image_porosity=None, boolean_porosity=None):
    image_porosity = np.ones(shape) if image_porosity is None else image_porosity
    boolean_porosity = (
        np.ones(shape, dtype=bool) if boolean_porosity is None else boolean_porosity
    )
    return SimpleNamespace(
        baseline=darsia.ScalarImage(np.zeros(shape), **META),
        image_porosity=darsia.ScalarImage(np.asarray(image_porosity, float), **META),
        boolean_porosity=darsia.ScalarImage(np.asarray(boolean_porosity), **META),
    )


class _Porosity:
    """Stand-in for `processing.image_porosity.ImagePorosity`: the maps of a fake rig."""

    def __init__(self, rig):
        self.rig = rig

    def map_on(self, grid):
        return darsia.transfer(self.rig.image_porosity, grid)

    def boolean_on(self, grid):
        return darsia.transfer(self.rig.boolean_porosity, grid)


def _build(restoration_config, rig):
    return build_restoration(restoration_config, rig, _Porosity(rig))


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

    restoration_config = _config(
        RestorationStepConfig(
            method="volume_average",
            volume_averaging_options=VolumeAveragingConfig(rev_size=0.005),
        ),
        ignore=["boolean_porosity"],
    )
    fluidflower = _rig(
        image_porosity=[[0.2, 0.4], [0.6, 0.8]],
        boolean_porosity=[[True, False], [True, True]],
    )

    _build(restoration_config, fluidflower)

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

    restoration_config = _config(
        RestorationStepConfig(
            method="tvd", tvd_options=TVDConfig(method="chambolle", weight=2.0)
        ),
        ignore=["boolean_porosity"],
    )
    fluidflower = _rig(boolean_porosity=[[True, False], [False, True]])

    _build(restoration_config, fluidflower)

    np.testing.assert_allclose(
        captured["weight"],
        np.array([[2.0, 0.0], [0.0, 2.0]], dtype=float),
    )
    assert captured["method"] == "heterogeneous bregman"


def test_build_restoration_unknown_ignore_mask_raises():
    restoration_config = _config(
        RestorationStepConfig(method="tvd", tvd_options=TVDConfig(weight=0.1)),
        ignore=["unknown-mask"],
    )

    with pytest.raises(ValueError, match="Unknown restoration ignore mask"):
        _build(restoration_config, _rig())


def test_build_restoration_volume_averaging_with_default_rev_size():
    """Regression test: VolumeAveragingConfig default rev_size works with real REV."""
    restoration_config = _config(
        RestorationStepConfig(
            method="volume_average",
            volume_averaging_options=VolumeAveragingConfig(),  # default rev_size
        )
    )

    # Should not raise TypeError about int not being subscriptable
    restoration = _build(restoration_config, _rig(shape=(10, 10)))
    assert isinstance(restoration, RestorationSteps)
    assert isinstance(restoration.models[0], darsia.VolumeAveraging)


@pytest.mark.parametrize("active", [[], [False, False]])
def test_build_restoration_without_active_steps_is_none(active):
    steps = [_resize(0.5) for _ in active]
    for step, is_active in zip(steps, active):
        step.active = is_active

    assert _build(_config(*steps), _rig()) is None


def test_build_restoration_skips_inactive_steps():
    inactive = _resize(0.5)
    inactive.active = False
    restoration_config = _config(
        inactive, RestorationStepConfig(method="tvd", tvd_options=TVDConfig())
    )

    restoration = _build(restoration_config, _rig(shape=(8, 8)))

    assert len(restoration.models) == 1
    assert isinstance(restoration.models[0], darsia.TVD)
    assert restoration.grid.img.shape == (8, 8)


def test_build_restoration_tracks_the_grid_of_resize_steps():
    restoration_config = _config(
        _resize(0.5),
        RestorationStepConfig(
            method="resize",
            resize_options=ResizeConfig(mode="target_shape", target_shape=(3, 5)),
        ),
    )

    restoration = _build(restoration_config, _rig(shape=(12, 20)))

    assert restoration.grid.img.shape[:2] == (3, 5)
    assert np.allclose(restoration.grid.dimensions, META["dimensions"])
    signal = np.random.default_rng(0).random((12, 20))
    assert restoration(signal).shape == (3, 5)


def test_build_restoration_averages_by_area_like_the_legacy_restoration():
    signal = np.random.default_rng(1).random((10, 15))

    restoration = _build(_config(_resize(0.2)), _rig(shape=(10, 15)))

    legacy = darsia.Resize(fx=0.2, fy=0.2, interpolation="inter_area")(signal)
    assert np.array_equal(restoration(signal), legacy)


def test_build_restoration_takes_rig_data_on_the_grid_after_a_resize(monkeypatch):
    """Porosity and ignore masks of later steps are on the grid of their input."""
    captured = {}

    class _FakeTVD:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __call__(self, img):
            return img

    monkeypatch.setattr("darsia.presets.workflows.restoration.darsia.TVD", _FakeTVD)
    boolean_porosity = np.ones((10, 10), dtype=bool)
    boolean_porosity[:5] = False  # upper half without porosity
    image_porosity = np.full((10, 10), 0.4)
    image_porosity[:, :5] = 0.2  # left half with lower porosity
    restoration_config = _config(
        _resize(0.2),
        RestorationStepConfig(
            method="tvd", tvd_options=TVDConfig(weight="image_porosity")
        ),
        ignore=["boolean_porosity"],
    )

    _build(
        restoration_config,
        _rig(
            shape=(10, 10),
            image_porosity=image_porosity,
            boolean_porosity=boolean_porosity,
        ),
    )

    # Porosity averaged onto the 2 x 2 grid, the ignored upper half (center sampled)
    np.testing.assert_allclose(captured["weight"], [[0.0, 0.0], [0.2, 0.4]])


def test_build_restoration_volume_averaging_after_a_resize_uses_the_coarse_grid():
    restoration_config = _config(
        _resize(0.5),
        RestorationStepConfig(
            method="volume_average",
            volume_averaging_options=VolumeAveragingConfig(rev_size=0.2),
        ),
    )

    restoration = _build(restoration_config, _rig(shape=(20, 40)))

    averaging = restoration.models[1]
    assert averaging.mask.img.shape == (10, 20)
    # 0.2 m are 2 pixels of 0.1 m on the coarse grid (not 4 of the rig grid)
    assert averaging.rev_size == 2
