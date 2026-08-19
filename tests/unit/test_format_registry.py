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
[[format]]
type = "jpg"
name = "4k"
filename_pattern = "time_HH:MM"
resolution = [2160, 4096]
cmap = "matplotlib.viridis"
quality = 77

[[format]]
type = "npz"
name = "my_npz"
filename_pattern = "stem"
resolution = [500, 1000]
keep_ratio = true
dtype = "np.float32"

[[format]]
type = "csv"
name = "my_csv"
filename_pattern = "stem"
delimiter = ";"
header = "h1;h2"
float_format = "{:.6g}"
""".strip(),
    )

    registry = FormatRegistry().load(config_path)
    assert set(registry.keys()) == {"4k", "my_npz", "my_csv"}
    specs = {s.name: s for s in registry.resolve(["4k", "my_npz", "my_csv"])}
    assert specs["4k"].type == "jpg"
    assert specs["4k"].resolution == (2160, 4096)
    assert specs["4k"].cmap == "matplotlib.viridis"
    assert specs["4k"].filename_pattern == "time_HH:MM"
    assert specs["4k"].quality == 77
    assert specs["my_npz"].keep_ratio is True
    assert specs["my_npz"].dtype == "np.float32"
    assert specs["my_csv"].delimiter == ";"
    assert specs["my_csv"].float_format == "{:.6g}"


def test_format_registry_rejects_duplicate_identifiers(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[[format]]
type = "jpg"
name = "shared"
filename_pattern = "stem"

[[format]]
type = "npz"
name = "shared"
filename_pattern = "stem"
""".strip(),
    )
    with pytest.raises(ValueError, match="duplicated"):
        FormatRegistry().load(config_path)


def test_format_registry_rejects_unsupported_name(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[[format]]
type = "jpg"
name = "preview"
filename_pattern = "unknown_option"
""".strip(),
    )
    with pytest.raises(ValueError, match="Unsupported name option"):
        FormatRegistry().load(config_path)


def test_fluidflower_config_loads_format_registry(tmp_path: Path) -> None:
    data_folder = tmp_path / "data"
    data_folder.mkdir(parents=True, exist_ok=True)
    (data_folder / "baseline.jpg").touch()
    config_path = _write(
        tmp_path / "config.toml",
        f"""
[data]
folder = "{data_folder.as_posix()}"
baseline = "baseline.jpg"
results = "{(tmp_path / "results").as_posix()}"

[[format]]
type = "npy"
name = "my_npy"
filename_pattern = "stem"
dtype = "np.float32"
""".strip(),
    )

    config = FluidFlowerConfig(config_path, require_data=False, require_results=False)
    assert config.format_registry is not None
    assert config.format_registry.keys() == ["my_npy"]
