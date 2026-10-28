"""Prerequisite compatibility checks for the original preset comparison route."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from darsia.presets.workflows.config.multi_fluidflower_config import (
    MultiFluidFlowerConfig,
)
from darsia.presets.workflows.config.roi import MultiRoiConfig, RoiConfig
from darsia.presets.workflows.config.wasserstein import WassersteinDistancesConfig


def _configuration(tmp_path, *, times="", selected="whole", extra_roi=""):
    path = tmp_path / "comparison.toml"
    path.write_text(
        f'''[[roi]]
name = "whole"
corner_1 = [0.1, 1.7]
corner_2 = [2.4, 0.2]
{extra_roi}
[data]
results = "{(tmp_path / 'results').as_posix()}"
[wasserstein]
runs = ["run-a", "run-b"]
roi = ["{selected}"]
resize = 0.25
relative_tol = 0.01
{times}
''',
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("times", "expected"),
    [
        ("", []),
        (
            '[wasserstein.data.time.sample]\n'
            'times = [2.0, "01:00", 1.0]\ntol = "00:15"',
            [(1.0, 0.25), (2.0, 0.25)],
        ),
        (
            '[wasserstein.data.interval.sample]\n'
            'start = "00:00"\nend = "02:00"\nnum = 3\ntol = 0.25',
            [(0.0, 0.25), (1.0, 0.25), (2.0, 0.25)],
        ),
        (
            '[wasserstein.data.time.sample]\n'
            'times = [2.0, "01:00", 1.0]\ntol = "00:15"\n'
            '[wasserstein.data.interval.sample]\n'
            'start = "00:00"\nend = "02:00"\nstep = "01:00"\ntol = 0.25',
            [(0.0, 0.25), (1.0, 0.25), (2.0, 0.25)],
        ),
    ],
)
def test_original_multi_config_reaches_wasserstein_with_current_roi_and_times(
    tmp_path, times, expected
):
    path = _configuration(tmp_path, times=times)

    config = MultiFluidFlowerConfig(path, require_data=False, require_results=True)

    assert config.wasserstein is not None
    assert config.wasserstein.times == expected
    assert config.wasserstein.runs == ["run-a", "run-b"]
    assert config.wasserstein.resize_factor == 0.25
    # This repair must not enable the previously unloaded scientific threshold.
    assert config.wasserstein.relative_tol is None
    roi = config.wasserstein.roi["whole"]
    assert isinstance(roi, RoiConfig)
    assert roi is config.roi.roi["whole"]
    assert roi.corner_1 == (0.1, 1.7)
    assert roi.corner_2 == (2.4, 0.2)
    assert roi.label is None


def test_comparison_refuses_a_selected_label_restriction(tmp_path):
    path = _configuration(
        tmp_path,
        selected="sand",
        extra_roi='''[[roi]]
name = "sand"
corner_1 = [0.0, 1.0]
corner_2 = [1.0, 0.0]
label = 4
''',
    )
    roi = MultiRoiConfig().load(path)

    with pytest.raises(ValueError, match="label-restricted ROIs: sand"):
        WassersteinDistancesConfig().load(path, tmp_path / "results", roi)


def test_unselected_label_restriction_does_not_change_plain_roi(tmp_path):
    path = _configuration(
        tmp_path,
        extra_roi='''[[roi]]
name = "sand"
corner_1 = [0.0, 1.0]
corner_2 = [1.0, 0.0]
label = 4
''',
    )

    config = MultiFluidFlowerConfig(path, require_data=False, require_results=True)

    assert config.roi.roi["sand"].label == 4
    assert config.wasserstein is not None
    assert list(config.wasserstein.roi) == ["whole"]


def test_multi_roi_keeps_registry_refusal_of_obsolete_nested_tables(tmp_path):
    path = tmp_path / "comparison.toml"
    path.write_text(
        '[roi.whole]\nname = "whole"\n'
        'corner_1 = [0.0, 1.0]\ncorner_2 = [1.0, 0.0]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="array-of-tables"):
        MultiRoiConfig().load(path)


def test_missing_mass_returns_none_before_weight_or_resize(monkeypatch):
    original = import_module(
        "darsia.presets.workflows.comparison.comparison_wasserstein"
    )
    run_config = object()
    calls = []

    def missing_mass(config, **kwargs):
        calls.append((config, kwargs))
        return None

    def unexpected_resize(*args, **kwargs):
        pytest.fail("Missing mass must not enter image processing")

    monkeypatch.setattr(original, "load_data", missing_mass)
    monkeypatch.setattr(original.darsia, "resize", unexpected_resize)
    config = SimpleNamespace(runs=SimpleNamespace(config={"run-a": run_config}))

    result = original._load_and_process_mass(
        "run-a", config, time=1.5, uncertainty=0.25, porosity_times_depth=object()
    )

    assert result is None
    assert calls == [(run_config, {"data": "mass", "time": 1.5, "tol": 0.25})]
