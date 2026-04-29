from pathlib import Path

import pytest

from darsia.presets.workflows.config.data_registry import DataRegistry
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.config.format_registry import FormatRegistry
from darsia.presets.workflows.config.helper import HelperConfig
from darsia.presets.workflows.config.roi_registry import RoiRegistry


def _write(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


def test_helper_roi_config_loads_defaults_and_registry_data(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[helper.roi]
data = "sel"
""".strip(),
    )
    data_registry = DataRegistry().load(
        {"time": {"sel": {"times": ["01:00:00"], "tol": "00:05:00"}}}
    )

    config = HelperConfig().load(
        config_path,
        data=tmp_path,
        data_registry=data_registry,
    )

    assert config.roi is not None
    assert config.roi.mode == "none"
    assert config.roi.data is not None
    assert config.roi.data.image_times == pytest.approx([1.0])


def test_helper_roi_config_rejects_invalid_mode(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[helper.roi]
mode = "invalid_mode"
data = "sel"
""".strip(),
    )
    data_registry = DataRegistry().load(
        {"time": {"sel": {"times": ["01:00:00"], "tol": "00:05:00"}}}
    )

    with pytest.raises(ValueError, match=r"Unsupported helper\.roi\.mode"):
        HelperConfig().load(
            config_path,
            data=tmp_path,
            data_registry=data_registry,
        )


def test_fluidflower_config_loads_helper_config(tmp_path: Path) -> None:
    data_folder = tmp_path / "data"
    data_folder.mkdir(parents=True, exist_ok=True)
    (data_folder / "baseline.jpg").touch()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[data]
folder = "{data_folder.as_posix()}"
baseline = "baseline.jpg"
results = "{(tmp_path / "results").as_posix()}"

[data.time.analysis_set]
times = ["01:00:00"]
tol = "00:05:00"

[helper.roi]
data = "analysis_set"
"""
    )

    config = FluidFlowerConfig(config_path, require_data=False, require_results=False)
    assert config.helper is not None
    assert config.helper.roi is not None
    assert config.helper.roi.mode == "none"


def test_helper_roi_viewer_config_loads_nested_data_selector(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[helper.roi_viewer]
data = "sel"
""".strip(),
    )
    data_registry = DataRegistry().load(
        {"time": {"sel": {"times": ["01:00:00"], "tol": "00:05:00"}}}
    )

    config = HelperConfig().load(
        config_path,
        data=tmp_path,
        data_registry=data_registry,
    )

    assert config.roi_viewer is not None
    assert config.roi_viewer.data is not None
    assert config.roi_viewer.data.image_times == pytest.approx([1.0])


def test_helper_roi_viewer_config_loads_shorthand_helper_data(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[helper]
data = "sel"
""".strip(),
    )
    data_registry = DataRegistry().load(
        {"time": {"sel": {"times": ["01:00:00"], "tol": "00:05:00"}}}
    )

    config = HelperConfig().load(
        config_path,
        data=tmp_path,
        data_registry=data_registry,
    )

    assert config.roi_viewer is not None
    assert config.roi_viewer.data is not None
    assert config.roi_viewer.data.image_times == pytest.approx([1.0])


def test_helper_results_config_loads_with_registry_validation(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[format.csv.csv_default]
name = "stem"

[roi.main]
name = "main"
corner_1 = [0.0, 0.0]
corner_2 = [1.0, 1.0]

[helper.results]
data = "sel"
mode = "rescaled_mass"
format = "csv_default"
cmap = "matplotlib.viridis"
roi = ["main"]
""".strip(),
    )
    data_registry = DataRegistry().load(
        {"time": {"sel": {"times": ["01:00:00"], "tol": "00:05:00"}}}
    )
    format_registry = FormatRegistry().load(config_path)
    roi_registry = RoiRegistry().load(config_path)

    config = HelperConfig().load(
        config_path,
        data=tmp_path,
        data_registry=data_registry,
        format_registry=format_registry,
        roi_registry=roi_registry,
    )

    assert config.results is not None
    assert config.results.mode == "rescaled_mass"
    assert config.results.format == "csv_default"
    assert config.results.cmap == "matplotlib.viridis"
    assert config.results.roi == ["main"]


def test_helper_results_config_rejects_unsupported_format(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[format.jpg.preview]
name = "stem"

[helper.results]
data = "sel"
mode = "rescaled_mass"
format = "preview"
""".strip(),
    )
    data_registry = DataRegistry().load(
        {"time": {"sel": {"times": ["01:00:00"], "tol": "00:05:00"}}}
    )
    format_registry = FormatRegistry().load(config_path)

    with pytest.raises(ValueError, match="helper\\.results\\.format"):
        HelperConfig().load(
            config_path,
            data=tmp_path,
            data_registry=data_registry,
            format_registry=format_registry,
            roi_registry=None,
        )
