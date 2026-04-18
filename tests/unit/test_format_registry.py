from pathlib import Path

import pytest

from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.config.format_registry import FormatRegistry


def _write(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


def test_format_registry_loads_named_entries(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[format.jpg.4k]
resolution = [2160, 4096]
cmap = "matplotlib.viridis"
name = "time_HH:MM"
quality = 77

[format.npz.my_npz]
resolution = [500, 1000]
keep_ratio = true
dtype = "np.float32"

[format.csv.my_csv]
delimiter = ";"
header = "h1;h2"
float_format = "{:.6g}"
""".strip(),
    )

    registry = FormatRegistry().load(config_path)
    assert set(registry.keys()) == {"4k", "my_npz", "my_csv"}
    specs = {s.identifier: s for s in registry.resolve(["4k", "my_npz", "my_csv"])}
    assert specs["4k"].type == "jpg"
    assert specs["4k"].resolution == (2160, 4096)
    assert specs["4k"].cmap == "matplotlib.viridis"
    assert specs["4k"].name == "time_HH:MM"
    assert specs["4k"].quality == 77
    assert specs["my_npz"].keep_ratio is True
    assert specs["my_npz"].dtype == "np.float32"
    assert specs["my_csv"].delimiter == ";"
    assert specs["my_csv"].float_format == "{:.6g}"


def test_format_registry_rejects_duplicate_identifiers(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[format.jpg.shared]

[format.npz.shared]
""".strip(),
    )
    with pytest.raises(ValueError, match="duplicated"):
        FormatRegistry().load(config_path)


def test_format_registry_rejects_unsupported_name(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[format.jpg.preview]
name = "unknown_option"
""".strip(),
    )
    with pytest.raises(ValueError, match="Unsupported name option"):
        FormatRegistry().load(config_path)


def test_format_registry_rejects_removed_name(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[format.jpg.preview]
name = "name_stem"
""".strip(),
    )
    with pytest.raises(ValueError, match="Removed options"):
        FormatRegistry().load(config_path)


def test_fluidflower_config_loads_format_registry(tmp_path: Path) -> None:
    data_folder = tmp_path / "data"
    data_folder.mkdir(parents=True, exist_ok=True)
    (data_folder / "baseline.jpg").touch()
    config_path = _write(
        tmp_path / "config.toml",
        f"""
[data]
folder = "{data_folder}"
baseline = "baseline.jpg"
results = "{tmp_path / "results"}"

[format.npy.my_npy]
dtype = "np.float32"
""".strip(),
    )

    config = FluidFlowerConfig(config_path, require_data=False, require_results=False)
    assert config.format_registry is not None
    assert config.format_registry.keys() == ["my_npy"]
