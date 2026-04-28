"""Unit tests for color path embedding configuration via ColorEmbeddingRegistry."""

import textwrap
from pathlib import Path

import pytest

from darsia.presets.workflows.config.color_embedding_registry import (
    ColorEmbeddingRegistry,
)
from darsia.presets.workflows.config.data_registry import DataRegistry
from darsia.presets.workflows.config.roi import RoiAndLabelConfig, RoiConfig
from darsia.presets.workflows.config.roi_registry import RoiRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_toml(tmp_path: Path, content: str) -> Path:
    """Write *content* to a ``config.toml`` file and return its path."""
    p = tmp_path / "config.toml"
    p.write_text(textwrap.dedent(content))
    return p


def _make_registry_with_roi(name: str = "my_roi") -> RoiRegistry:
    """Return a :class:`RoiRegistry` with a single plain :class:`RoiConfig`."""
    reg = RoiRegistry()
    roi = RoiConfig()
    roi.load({"name": name, "corner_1": [0.1, 0.2], "corner_2": [0.8, 0.9]})
    reg.register(name, roi)
    return reg


def _make_data_registry(tmp_path: Path) -> DataRegistry:
    """Return a DataRegistry with dummy baseline and calibration path entries.

    Creates one dummy image file so ImagePathData validation passes.
    """
    dummy = tmp_path / "dummy.jpg"
    dummy.touch()
    sec = {
        "path": {
            "baseline_imgs": {"paths": ["dummy.jpg"]},
            "cal_imgs": {"paths": ["dummy.jpg"]},
        }
    }
    return DataRegistry().load(sec, data_folder=tmp_path)


def _minimal_color_path_toml(extra: str = "", rois_line: str = "") -> str:
    """Return a minimal [color.path.*] TOML section using registry references.

    Args:
        extra: Additional TOML lines to insert inside [color.path.*].
        rois_line: A ``rois = ...`` line to inject (empty → key absent).
    """
    return textwrap.dedent(
        f"""
        [color.path.default]
        baseline = "baseline_imgs"
        data     = "cal_imgs"
        {rois_line}
        {extra}
        """
    )


# ---------------------------------------------------------------------------
# Registry-reference rois
# ---------------------------------------------------------------------------


class TestColorEmbeddingRegistryRoisFromRegistry:
    def test_rois_list_is_stored(self, tmp_path):
        """A ``rois = [...]`` key is stored verbatim in embedding.rois."""
        toml_path = _write_toml(
            tmp_path,
            _minimal_color_path_toml(rois_line='rois = ["my_roi"]'),
        )
        data_reg = _make_data_registry(tmp_path)
        roi_registry = _make_registry_with_roi("my_roi")
        registry = ColorEmbeddingRegistry().load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
            roi_registry=roi_registry,
        )
        embedding = registry.resolve("default")
        assert embedding.rois == ["my_roi"]


# ---------------------------------------------------------------------------
# Inline ROI sub-sections
# ---------------------------------------------------------------------------


class TestColorEmbeddingRegistryInlineRoi:
    def test_inline_roi_injected_into_registry(self, tmp_path):
        """Inline ``[color.path.<id>.roi.*]`` entries are added to the registry."""
        toml_path = _write_toml(
            tmp_path,
            _minimal_color_path_toml(
                extra="""
                [color.path.default.roi.box1]
                name     = "box1"
                corner_1 = [0.1, 0.2]
                corner_2 = [0.5, 0.6]
                """
            ),
        )
        data_reg = _make_data_registry(tmp_path)
        registry = RoiRegistry()
        color_registry = ColorEmbeddingRegistry().load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
            roi_registry=registry,
        )
        embedding = color_registry.resolve("default")
        assert "box1" in registry.keys()
        assert isinstance(registry.resolve("box1")["box1"], RoiConfig)
        assert "box1" in embedding.rois

    def test_inline_roi_and_label_injected_into_registry(self, tmp_path):
        """Inline ROIs with a ``label`` key become :class:`RoiAndLabelConfig`."""
        toml_path = _write_toml(
            tmp_path,
            _minimal_color_path_toml(
                extra="""
                [color.path.default.roi.sand]
                name     = "sand"
                corner_1 = [0.0, 0.0]
                corner_2 = [1.0, 1.0]
                label    = 3
                """
            ),
        )
        data_reg = _make_data_registry(tmp_path)
        registry = RoiRegistry()
        color_registry = ColorEmbeddingRegistry().load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
            roi_registry=registry,
        )
        embedding = color_registry.resolve("default")
        resolved = registry.resolve("sand")
        assert isinstance(resolved["sand"], RoiAndLabelConfig)
        assert resolved["sand"].label == 3
        assert "sand" in embedding.rois

    def test_inline_roi_not_duplicated_in_rois_list(self, tmp_path):
        """Inline ROIs listed in ``rois`` appear only once in embedding.rois."""
        toml_path = _write_toml(
            tmp_path,
            _minimal_color_path_toml(
                rois_line='rois = ["box1"]',
                extra="""
                [color.path.default.roi.box1]
                name     = "box1"
                corner_1 = [0.1, 0.2]
                corner_2 = [0.5, 0.6]
                """,
            ),
        )
        data_reg = _make_data_registry(tmp_path)
        registry = RoiRegistry()
        color_registry = ColorEmbeddingRegistry().load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
            roi_registry=registry,
        )
        embedding = color_registry.resolve("default")
        assert embedding.rois.count("box1") == 1

    def test_inline_roi_without_registry_does_not_crash(self, tmp_path):
        """Inline ROIs are ignored when no registry is provided."""
        toml_path = _write_toml(
            tmp_path,
            _minimal_color_path_toml(
                extra="""
                [color.path.default.roi.box1]
                name     = "box1"
                corner_1 = [0.1, 0.2]
                corner_2 = [0.5, 0.6]
                """
            ),
        )
        data_reg = _make_data_registry(tmp_path)
        color_registry = ColorEmbeddingRegistry().load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
            roi_registry=None,
        )
        embedding = color_registry.resolve("default")
        assert embedding.rois == []


# ---------------------------------------------------------------------------
# Selector resolution and validation
# ---------------------------------------------------------------------------


class TestColorEmbeddingRegistrySelectors:
    def test_inline_selectors_emit_deprecation_warning(self, tmp_path):
        dummy = tmp_path / "dummy.jpg"
        dummy.touch()
        toml_path = _write_toml(
            tmp_path,
            """
            [color.path.default]
            [color.path.default.baseline.path.baseline]
            paths = ["dummy.jpg"]
            [color.path.default.data.time.calibration]
            times = ["01:00:00"]
            tol = "00:05:00"
            """,
        )
        with pytest.warns(DeprecationWarning) as warnings:
            registry = ColorEmbeddingRegistry().load(
                path=toml_path,
                data=tmp_path,
                results=tmp_path,
            )
        messages = [str(item.message) for item in warnings]
        assert any("color.path.default.baseline" in msg for msg in messages)
        assert any("color.path.default.data" in msg for msg in messages)
        embedding = registry.resolve("default")
        assert embedding.data is not None
        assert embedding.baseline_data is not None
        assert embedding.data.image_times == pytest.approx([1.0])
        assert embedding.baseline_data.image_paths == [dummy]

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("ignore_baseline_spectrum", "bad_value"),
            ("histogram_weighting", "bad_value"),
            ("calibration_mode", "bad_value"),
        ],
    )
    def test_invalid_color_path_values_raise(self, tmp_path, key, value):
        toml_path = _write_toml(
            tmp_path,
            _minimal_color_path_toml(extra=f'{key} = "{value}"'),
        )
        data_reg = _make_data_registry(tmp_path)
        with pytest.raises(ValueError, match=key):
            ColorEmbeddingRegistry().load(
                path=toml_path,
                data=tmp_path,
                results=tmp_path,
                data_registry=data_reg,
            )
