"""Unit tests for CalibrationColorConfig and CalibrationMassConfig."""

import textwrap
from pathlib import Path

import pytest

from darsia.presets.workflows.config.calibration import (
    CalibrationColorConfig,
    CalibrationMassConfig,
)
from darsia.presets.workflows.config.color_embedding_registry import (
    ColorEmbeddingRegistry,
)
from darsia.presets.workflows.config.roi import RoiConfig
from darsia.presets.workflows.config.roi_registry import RoiRegistry


def _write_toml(tmp_path: Path, content: str, filename: str = "config.toml") -> Path:
    """Write TOML content to a temp file and return its path."""
    p = tmp_path / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content))
    return p


def _make_color_embedding_registry(tmp_path: Path) -> ColorEmbeddingRegistry:
    """Create a ColorEmbeddingRegistry with one minimal [[color_path]] entry."""
    toml_path = _write_toml(
        tmp_path,
        """
        [[color_path]]
        name = "color_path"
        """,
    )
    return ColorEmbeddingRegistry().load(
        path=toml_path,
        data=None,
        results=None,
    )


def _make_registry_with_roi(name: str = "test_roi") -> RoiRegistry:
    """Create a RoiRegistry with a single ROI for testing."""
    reg = RoiRegistry()
    roi = RoiConfig()
    roi.load({"name": name, "corner_1": [0.1, 0.2], "corner_2": [0.8, 0.9]})
    reg.register(name, roi)
    return reg


class TestCalibrationColorConfig:
    """Test CalibrationColorConfig.load() and eager ROI validation."""

    def test_unknown_roi_key_raises_at_load_time(self, tmp_path):
        """Unknown ROI key in 'rois' list raises KeyError at .load()."""
        color_registry = _make_color_embedding_registry(tmp_path)
        roi_registry = _make_registry_with_roi("valid_roi")
        sec = {
            "embedding": "color_path",
            "rois": ["unknown_roi"],
        }
        with pytest.raises(KeyError, match="unknown_roi.*not found"):
            CalibrationColorConfig().load(
                sec,
                color_embedding_registry=color_registry,
                roi_registry=roi_registry,
            )

    def test_rois_list_is_stored(self, tmp_path):
        """A ``rois`` list is stored in the config."""
        color_registry = _make_color_embedding_registry(tmp_path)
        roi_registry = _make_registry_with_roi("my_roi")
        sec = {
            "embedding": "color_path",
            "rois": ["my_roi"],
        }
        cfg = CalibrationColorConfig().load(
            sec,
            color_embedding_registry=color_registry,
            roi_registry=roi_registry,
        )
        assert cfg.rois == ["my_roi"]

    def test_no_rois_absent_defaults_to_empty_list(self, tmp_path):
        """``no_rois`` defaults to an empty list when absent."""
        color_registry = _make_color_embedding_registry(tmp_path)
        sec = {"embedding": "color_path"}
        cfg = CalibrationColorConfig().load(
            sec,
            color_embedding_registry=color_registry,
            roi_registry=None,
        )
        assert cfg.no_rois == []

    def test_no_rois_list_is_stored(self, tmp_path):
        """A ``no_rois`` list is stored in the config, independent of ``rois``."""
        color_registry = _make_color_embedding_registry(tmp_path)
        roi_registry = _make_registry_with_roi("my_roi")
        roi_registry.register(
            "water_pocket",
            RoiConfig().load(
                {"name": "water_pocket", "corner_1": [0.0, 0.0], "corner_2": [0.1, 0.1]}
            ),
        )
        sec = {
            "embedding": "color_path",
            "rois": ["my_roi"],
            "no_rois": ["water_pocket"],
        }
        cfg = CalibrationColorConfig().load(
            sec,
            color_embedding_registry=color_registry,
            roi_registry=roi_registry,
        )
        assert cfg.rois == ["my_roi"]
        assert cfg.no_rois == ["water_pocket"]

    def test_unknown_no_roi_key_raises_at_load_time(self, tmp_path):
        """Unknown ROI key in 'no_rois' list raises KeyError at .load()."""
        color_registry = _make_color_embedding_registry(tmp_path)
        roi_registry = _make_registry_with_roi("valid_roi")
        sec = {
            "embedding": "color_path",
            "no_rois": ["unknown_roi"],
        }
        with pytest.raises(KeyError, match="unknown_roi.*not found"):
            CalibrationColorConfig().load(
                sec,
                color_embedding_registry=color_registry,
                roi_registry=roi_registry,
            )


class TestCalibrationMassConfig:
    """Test CalibrationMassConfig.load() and eager ROI validation."""

    def test_unknown_roi_key_raises_at_load_time(self, tmp_path):
        """Unknown ROI key in 'rois' list raises KeyError at .load()."""
        color_registry = _make_color_embedding_registry(tmp_path)
        roi_registry = _make_registry_with_roi("valid_roi")
        sec = {
            "embedding": "color_path",
            "rois": ["unknown_roi"],
        }
        with pytest.raises(KeyError, match="unknown_roi.*not found"):
            CalibrationMassConfig().load(
                sec,
                data=None,
                color_embedding_registry=color_registry,
                roi_registry=roi_registry,
            )

    def test_rois_list_is_stored(self, tmp_path):
        """A ``rois`` list is stored in the config."""
        color_registry = _make_color_embedding_registry(tmp_path)
        roi_registry = _make_registry_with_roi("my_roi")
        sec = {
            "embedding": "color_path",
            "rois": ["my_roi"],
        }
        cfg = CalibrationMassConfig().load(
            sec,
            data=None,
            color_embedding_registry=color_registry,
            roi_registry=roi_registry,
        )
        assert cfg.rois == ["my_roi"]
