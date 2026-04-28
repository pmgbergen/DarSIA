"""Unit tests for ColorToMassConfig."""

import textwrap
from pathlib import Path

from darsia.presets.workflows.config.color_to_mass import ColorToMassConfig
from darsia.presets.workflows.config.data_registry import DataRegistry
from darsia.presets.workflows.config.roi import RoiConfig
from darsia.presets.workflows.config.roi_registry import RoiRegistry


def _write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(textwrap.dedent(content))
    return p


def _make_registry_with_roi(name: str = "my_roi") -> RoiRegistry:
    reg = RoiRegistry()
    roi = RoiConfig()
    roi.load({"name": name, "corner_1": [0.1, 0.2], "corner_2": [0.8, 0.9]})
    reg.register(name, roi)
    return reg


def _make_data_registry(tmp_path: Path) -> DataRegistry:
    dummy = tmp_path / "dummy.jpg"
    dummy.touch()
    sec = {
        "path": {
            "cal_imgs": {"paths": ["dummy.jpg"]},
        }
    }
    return DataRegistry().load(sec, data_folder=tmp_path)


class TestColorToMassConfig:
    def _load_cfg(self, tmp_path: Path, extra: str = "") -> ColorToMassConfig:
        toml_path = _write_toml(
            tmp_path,
            textwrap.dedent(
                f"""
                [color_to_mass]
                data = "cal_imgs"
                {extra}
                """
            ),
        )
        data_reg = _make_data_registry(tmp_path)
        cfg = ColorToMassConfig()
        cfg.load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
        )
        return cfg

    def test_color_to_mass_default_folder_uses_basis(self, tmp_path):
        dummy = tmp_path / "dummy.jpg"
        dummy.touch()
        data_reg = DataRegistry().load(
            {"path": {"cal_imgs": {"paths": ["dummy.jpg"]}}},
            data_folder=tmp_path,
        )

        cfg_path = _write_toml(
            tmp_path,
            """
            [color_to_mass]
            basis = "labels"
            data = "cal_imgs"
            """,
        )
        cfg = ColorToMassConfig().load(
            path=cfg_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
        )
        assert cfg.basis.value == "labels"
        assert (
            cfg.calibration_folder
            == tmp_path / "calibration" / "color_to_mass" / "from_labels"
        )
        assert cfg.threshold == 0.2

    def test_color_to_mass_threshold_can_be_overridden(self, tmp_path):
        dummy = tmp_path / "dummy.jpg"
        dummy.touch()
        data_reg = DataRegistry().load(
            {"path": {"cal_imgs": {"paths": ["dummy.jpg"]}}},
            data_folder=tmp_path,
        )

        cfg_path = _write_toml(
            tmp_path,
            """
            [color_to_mass]
            data = "cal_imgs"
            threshold = 0.35
            """,
        )
        cfg = ColorToMassConfig().load(
            path=cfg_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
        )
        assert cfg.threshold == 0.35

    def test_color_to_mass_absent_rois_defaults_to_empty_list(self, tmp_path):
        dummy = tmp_path / "dummy.jpg"
        dummy.touch()
        data_reg = DataRegistry().load(
            {"path": {"cal_imgs": {"paths": ["dummy.jpg"]}}},
            data_folder=tmp_path,
        )

        cfg_path = _write_toml(
            tmp_path,
            """
            [color_to_mass]
            data = "cal_imgs"
            """,
        )
        cfg = ColorToMassConfig().load(
            path=cfg_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
        )
        assert cfg.rois == []

    def test_color_to_mass_rois_list_is_stored(self, tmp_path):
        dummy = tmp_path / "dummy.jpg"
        dummy.touch()
        data_reg = DataRegistry().load(
            {"path": {"cal_imgs": {"paths": ["dummy.jpg"]}}},
            data_folder=tmp_path,
        )
        roi_registry = _make_registry_with_roi("my_roi")

        cfg_path = _write_toml(
            tmp_path,
            """
            [color_to_mass]
            data = "cal_imgs"
            rois = ["my_roi"]
            """,
        )
        cfg = ColorToMassConfig().load(
            path=cfg_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
            roi_registry=roi_registry,
        )
        assert cfg.rois == ["my_roi"]

    def test_color_to_mass_inline_roi_is_registered_and_added(self, tmp_path):
        dummy = tmp_path / "dummy.jpg"
        dummy.touch()
        data_reg = DataRegistry().load(
            {"path": {"cal_imgs": {"paths": ["dummy.jpg"]}}},
            data_folder=tmp_path,
        )
        roi_registry = RoiRegistry()

        cfg_path = _write_toml(
            tmp_path,
            """
            [color_to_mass]
            data = "cal_imgs"

            [color_to_mass.roi.calib_roi]
            name = "calib_roi"
            corner_1 = [0.1, 0.2]
            corner_2 = [0.7, 0.8]
            """,
        )
        cfg = ColorToMassConfig().load(
            path=cfg_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
            roi_registry=roi_registry,
        )
        assert cfg.rois == ["calib_roi"]
        assert "calib_roi" in roi_registry.keys()
