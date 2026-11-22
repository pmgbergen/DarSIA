"""Unit tests for RestorationConfig parsing: steps, and the single method form."""

from pathlib import Path

import pytest

from darsia.presets.workflows.config.resize import ResizeConfig
from darsia.presets.workflows.config.restoration import (
    RestorationConfig,
    TVDConfig,
    VolumeAveragingConfig,
)


def _write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(content)
    return p


def _single_step(cfg: RestorationConfig):
    assert len(cfg.steps) == 1
    return cfg.steps[0]


# ! ---- Single method form (one step) ----


def test_restoration_config_volume_average_defaults(tmp_path):
    cfg_path = _write_toml(tmp_path, '[restoration]\nmethod = "volume_average"\n')
    step = _single_step(RestorationConfig().load(cfg_path))
    assert step.method == "volume_average"
    assert step.active
    assert isinstance(step.volume_averaging_options, VolumeAveragingConfig)
    assert step.volume_averaging_options.rev_size == pytest.approx(0.005)  # default


def test_restoration_config_volume_average_custom_rev_size(tmp_path):
    cfg_path = _write_toml(
        tmp_path,
        '[restoration]\nmethod = "volume_average"\n\n[restoration.options]\nrev_size = 0.01\n',
    )
    step = _single_step(RestorationConfig().load(cfg_path))
    assert step.method == "volume_average"
    assert step.volume_averaging_options.rev_size == pytest.approx(0.01)


def test_restoration_config_tvd_defaults(tmp_path):
    cfg_path = _write_toml(tmp_path, '[restoration]\nmethod = "tvd"\n')
    step = _single_step(RestorationConfig().load(cfg_path))
    assert step.method == "tvd"
    options = step.tvd_options
    assert isinstance(options, TVDConfig)
    # Check defaults
    assert options.method == "chambolle"
    assert options.weight == pytest.approx(0.1)
    assert options.max_num_iter == 200
    assert options.eps == pytest.approx(2e-4)
    assert options.omega == pytest.approx(1.0)
    assert options.regularization == pytest.approx(1.0)
    assert options.kwargs == {}


def test_restoration_config_tvd_custom_options(tmp_path):
    cfg_path = _write_toml(
        tmp_path,
        (
            '[restoration]\nmethod = "tvd"\n\n'
            "[restoration.options]\n"
            'method = "isotropic bregman"\n'
            "weight = 0.05\n"
            "max_num_iter = 100\n"
            "eps = 1e-3\n"
            "omega = 2.0\n"
            "regularization = 0.5\n"
        ),
    )
    options = _single_step(RestorationConfig().load(cfg_path)).tvd_options
    assert options.method == "isotropic bregman"
    assert options.weight == pytest.approx(0.05)
    assert options.max_num_iter == 100
    assert options.eps == pytest.approx(1e-3)
    assert options.omega == pytest.approx(2.0)
    assert options.regularization == pytest.approx(0.5)


@pytest.mark.parametrize("weight", ["image_porosity", "boolean_porosity"])
def test_restoration_config_tvd_porosity_weight(tmp_path, weight):
    cfg_path = _write_toml(
        tmp_path,
        f'[restoration]\nmethod = "tvd"\n\n[restoration.options]\nweight = "{weight}"\n',
    )
    options = _single_step(RestorationConfig().load(cfg_path)).tvd_options
    assert options.weight == weight


@pytest.mark.parametrize(
    "content", ['[restoration]\nmethod = "none"\n', "[restoration]"]
)
def test_restoration_config_without_method_has_no_steps(tmp_path, content):
    cfg = RestorationConfig().load(_write_toml(tmp_path, content))
    assert cfg.steps == []


def test_restoration_config_unsupported_method_raises(tmp_path):
    cfg_path = _write_toml(tmp_path, '[restoration]\nmethod = "unknown_method"\n')
    with pytest.raises(NotImplementedError):
        RestorationConfig().load(cfg_path)


# ! ---- Steps ----


def test_restoration_config_steps_keep_their_order(tmp_path):
    cfg_path = _write_toml(
        tmp_path,
        """
[restoration]
ignore = ["boolean_porosity"]

[[restoration.steps]]
method = "resize"
options = { mode = "scale", scale = 0.2 }

[[restoration.steps]]
method = "tvd"
options = { weight = 0.3 }

[[restoration.steps]]
method = "resize"
active = false
options = { mode = "target_shape", target_shape = [10, 20] }
""",
    )
    cfg = RestorationConfig().load(cfg_path)

    assert [step.method for step in cfg.steps] == ["resize", "tvd", "resize"]
    assert [step.active for step in cfg.steps] == [True, True, False]
    assert cfg.steps[0].resize_options == ResizeConfig(mode="scale", scale=0.2)
    assert cfg.steps[1].tvd_options.weight == pytest.approx(0.3)
    assert cfg.steps[2].resize_options.target_shape == (10, 20)
    assert cfg.ignore == ["boolean_porosity"]


def test_restoration_config_rejects_steps_and_a_single_method(tmp_path):
    cfg_path = _write_toml(
        tmp_path,
        '[restoration]\nmethod = "tvd"\n\n[[restoration.steps]]\nmethod = "tvd"\n',
    )
    with pytest.raises(ValueError, match="not both"):
        RestorationConfig().load(cfg_path)


def test_restoration_config_rejects_unknown_step_keys(tmp_path):
    cfg_path = _write_toml(
        tmp_path, '[[restoration.steps]]\nmethod = "resize"\nscale = 0.2\n'
    )
    with pytest.raises(ValueError, match=r"restoration.steps\[0\]: unknown keys"):
        RestorationConfig().load(cfg_path)


def test_restoration_config_rejects_a_step_without_method(tmp_path):
    cfg_path = _write_toml(tmp_path, "[[restoration.steps]]\nactive = true\n")
    with pytest.raises(ValueError, match="needs a 'method'"):
        RestorationConfig().load(cfg_path)


def test_restoration_config_resize_step_needs_its_options(tmp_path):
    cfg_path = _write_toml(tmp_path, '[[restoration.steps]]\nmethod = "resize"\n')
    with pytest.raises(ValueError, match=r"restoration.steps\[0\].options\] mode"):
        RestorationConfig().load(cfg_path)
