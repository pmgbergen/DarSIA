from pathlib import Path

from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.config.workflow_utils import WorkflowUtilsConfig


def test_workflow_utils_config_load_flat_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[utils]
export_calibration_bundle = "/tmp/export.zip"
import_calibration_bundle = "/tmp/import.zip"
"""
    )

    config = WorkflowUtilsConfig().load(config_path)
    assert config.export_calibration_bundle == Path("/tmp/export.zip")
    assert config.import_calibration_bundle == Path("/tmp/import.zip")


def test_workflow_utils_config_load_nested_keys_prefer_nested(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[utils]
export_calibration_bundle = "/tmp/flat_export.zip"
import_calibration_bundle = "/tmp/flat_import.zip"

[utils.calibration]
export_bundle = "/tmp/nested_export.zip"
import_bundle = "/tmp/nested_import.zip"
"""
    )

    config = WorkflowUtilsConfig().load(config_path)
    assert config.export_calibration_bundle == Path("/tmp/nested_export.zip")
    assert config.import_calibration_bundle == Path("/tmp/nested_import.zip")


def test_fluidflower_config_loads_workflow_utils(tmp_path: Path) -> None:
    data_folder = tmp_path / "data"
    data_folder.mkdir(parents=True, exist_ok=True)
    (data_folder / "baseline.jpg").touch()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[data]
folder = "{data_folder}"
baseline = "baseline.jpg"
results = "{tmp_path / "results"}"

[utils.calibration]
export_bundle = "{tmp_path / "bundle_out.zip"}"
import_bundle = "{tmp_path / "bundle_in.zip"}"
"""
    )

    config = FluidFlowerConfig(config_path, require_data=False, require_results=False)
    assert config.workflow_utils is not None
    assert (
        config.workflow_utils.export_calibration_bundle == tmp_path / "bundle_out.zip"
    )
    assert config.workflow_utils.import_calibration_bundle == tmp_path / "bundle_in.zip"


def test_fluidflower_config_loads_colorchannel_registry(tmp_path: Path) -> None:
    data_folder = tmp_path / "data"
    data_folder.mkdir(parents=True, exist_ok=True)
    (data_folder / "baseline.jpg").touch()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[data]
folder = "{data_folder}"
baseline = "baseline.jpg"
results = "{tmp_path / "results"}"

[colorchannel.red_channel]
color_space = "RGB"
channel = "r"
"""
    )

    config = FluidFlowerConfig(config_path, require_data=False, require_results=False)
    assert config.colorchannel is not None
    resolved = config.colorchannel.resolve("red_channel")
    assert resolved["red_channel"].color_space == "RGB"
    assert resolved["red_channel"].channel == "r"
