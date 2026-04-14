"""Unit tests for RestorationConfig parsing (including TVD method)."""

from pathlib import Path

import pytest

from darsia.presets.workflows.config.restoration import (
    RestorationConfig,
    TVDConfig,
    VolumeAveragingConfig,
)


def _write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(content)
    return p


def test_restoration_config_volume_average_defaults(tmp_path):
    cfg_path = _write_toml(tmp_path, '[restoration]\nmethod = "volume_average"\n')
    cfg = RestorationConfig().load(cfg_path)
    assert cfg.method == "volume_average"
    assert isinstance(cfg.options, VolumeAveragingConfig)
    assert cfg.options.rev_size == 3  # default


def test_restoration_config_volume_average_custom_rev_size(tmp_path):
    cfg_path = _write_toml(
        tmp_path,
        '[restoration]\nmethod = "volume_average"\n\n[restoration.options]\nrev_size = 5\n',
    )
    cfg = RestorationConfig().load(cfg_path)
    assert cfg.method == "volume_average"
    assert isinstance(cfg.options, VolumeAveragingConfig)
    assert cfg.options.rev_size == 5


def test_restoration_config_tvd_defaults(tmp_path):
    cfg_path = _write_toml(tmp_path, '[restoration]\nmethod = "tvd"\n')
    cfg = RestorationConfig().load(cfg_path)
    assert cfg.method == "tvd"
    assert isinstance(cfg.options, TVDConfig)
    # Check defaults
    assert cfg.options.method == "chambolle"
    assert cfg.options.weight == pytest.approx(0.1)
    assert cfg.options.max_num_iter == 200
    assert cfg.options.eps == pytest.approx(2e-4)
    assert cfg.options.omega == pytest.approx(1.0)
    assert cfg.options.regularization == pytest.approx(1.0)
    assert cfg.options.kwargs == {}


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
    cfg = RestorationConfig().load(cfg_path)
    assert cfg.method == "tvd"
    assert isinstance(cfg.options, TVDConfig)
    assert cfg.options.method == "isotropic bregman"
    assert cfg.options.weight == pytest.approx(0.05)
    assert cfg.options.max_num_iter == 100
    assert cfg.options.eps == pytest.approx(1e-3)
    assert cfg.options.omega == pytest.approx(2.0)
    assert cfg.options.regularization == pytest.approx(0.5)


def test_restoration_config_tvd_porosity_weight(tmp_path):
    cfg_path = _write_toml(
        tmp_path,
        '[restoration]\nmethod = "tvd"\n\n[restoration.options]\nweight = "porosity"\n',
    )
    cfg = RestorationConfig().load(cfg_path)
    assert cfg.method == "tvd"
    assert isinstance(cfg.options, TVDConfig)
    assert cfg.options.weight == "porosity"


def test_restoration_config_tvd_boolean_porosity_weight(tmp_path):
    cfg_path = _write_toml(
        tmp_path,
        '[restoration]\nmethod = "tvd"\n\n[restoration.options]\nweight = "boolean-porosity"\n',
    )
    cfg = RestorationConfig().load(cfg_path)
    assert cfg.method == "tvd"
    assert isinstance(cfg.options, TVDConfig)
    assert cfg.options.weight == "boolean-porosity"


def test_restoration_config_ignore_masks(tmp_path):
    cfg_path = _write_toml(
        tmp_path,
        '[restoration]\nmethod = "tvd"\nignore = ["boolean_porosity"]\n',
    )
    cfg = RestorationConfig().load(cfg_path)
    assert cfg.ignore == ["boolean_porosity"]


def test_restoration_config_ignore_masks_must_be_strings(tmp_path):
    cfg_path = _write_toml(
        tmp_path,
        '[restoration]\nmethod = "tvd"\nignore = ["boolean_porosity", 1]\n',
    )
    with pytest.raises(
        ValueError, match="restoration.ignore must be a list of strings"
    ):
        RestorationConfig().load(cfg_path)


def test_restoration_config_none_method_raises(tmp_path):
    """'none' is no longer a valid method string; omit the section instead."""
    cfg_path = _write_toml(tmp_path, '[restoration]\nmethod = "none"\n')
    with pytest.raises(NotImplementedError):
        RestorationConfig().load(cfg_path)


def test_restoration_config_unsupported_method_raises(tmp_path):
    cfg_path = _write_toml(tmp_path, '[restoration]\nmethod = "unknown_method"\n')
    with pytest.raises(NotImplementedError):
        RestorationConfig().load(cfg_path)
