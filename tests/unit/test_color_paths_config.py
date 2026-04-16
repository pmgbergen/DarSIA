"""Unit tests for ColorPathsConfig ROI support."""

import textwrap
from pathlib import Path

import pytest

from darsia.presets.workflows.config.color_paths import ColorPathsConfig
from darsia.presets.workflows.config.color_to_mass import ColorToMassConfig
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


def _minimal_color_paths_toml(extra: str = "", rois_line: str = "") -> str:
    """Return a minimal [color_paths] TOML section using registry references.

    Args:
        extra: Additional TOML lines to insert inside [color_paths].
        rois_line: A ``rois = ...`` line to inject (empty → key absent).
    """
    return textwrap.dedent(
        f"""
        [color_paths]
        baseline = "baseline_imgs"
        data     = "cal_imgs"
        {rois_line}
        {extra}
        """
    )


# ---------------------------------------------------------------------------
# Default behaviour (no rois key)
# ---------------------------------------------------------------------------


class TestColorPathsConfigNoRois:
    def test_rois_defaults_to_empty_list(self, tmp_path):
        toml_path = _write_toml(
            tmp_path,
            _minimal_color_paths_toml(rois_line="rois = []"),
        )
        data_reg = _make_data_registry(tmp_path)
        cfg = ColorPathsConfig()
        cfg.load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
        )
        assert cfg.rois == []

    def test_absent_rois_key_defaults_to_empty_list(self, tmp_path):
        toml_path = _write_toml(tmp_path, _minimal_color_paths_toml())
        data_reg = _make_data_registry(tmp_path)
        cfg = ColorPathsConfig()
        cfg.load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
        )
        assert cfg.rois == []


# ---------------------------------------------------------------------------
# Registry-reference rois
# ---------------------------------------------------------------------------


class TestColorPathsConfigRoisFromRegistry:
    def test_rois_list_is_stored(self, tmp_path):
        """A ``rois = [...]`` key is stored verbatim in cfg.rois."""
        toml_path = _write_toml(
            tmp_path,
            _minimal_color_paths_toml(rois_line='rois = ["my_roi"]'),
        )
        data_reg = _make_data_registry(tmp_path)
        roi_registry = _make_registry_with_roi("my_roi")
        cfg = ColorPathsConfig()
        cfg.load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
            roi_registry=roi_registry,
        )
        assert cfg.rois == ["my_roi"]

    def test_multiple_rois_stored(self, tmp_path):
        toml_path = _write_toml(
            tmp_path,
            _minimal_color_paths_toml(rois_line='rois = ["roi_a", "roi_b"]'),
        )
        data_reg = _make_data_registry(tmp_path)
        reg = RoiRegistry()
        for name in ("roi_a", "roi_b"):
            roi = RoiConfig()
            roi.load({"name": name, "corner_1": [0.0, 0.0], "corner_2": [1.0, 1.0]})
            reg.register(name, roi)

        cfg = ColorPathsConfig()
        cfg.load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
            roi_registry=reg,
        )
        assert cfg.rois == ["roi_a", "roi_b"]

    def test_rois_without_registry_still_stored(self, tmp_path):
        """rois list is stored even when no roi_registry is provided."""
        toml_path = _write_toml(
            tmp_path,
            _minimal_color_paths_toml(rois_line='rois = ["some_roi"]'),
        )
        data_reg = _make_data_registry(tmp_path)
        cfg = ColorPathsConfig()
        cfg.load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
            roi_registry=None,
        )
        assert cfg.rois == ["some_roi"]


# ---------------------------------------------------------------------------
# Inline ROI sub-sections
# ---------------------------------------------------------------------------


class TestColorPathsConfigInlineRoi:
    def test_inline_roi_injected_into_registry(self, tmp_path):
        """Inline ``[color_paths.roi.*]`` entries are added to the registry."""
        toml_path = _write_toml(
            tmp_path,
            _minimal_color_paths_toml(
                extra="""
                [color_paths.roi.box1]
                name     = "box1"
                corner_1 = [0.1, 0.2]
                corner_2 = [0.5, 0.6]
                """
            ),
        )
        data_reg = _make_data_registry(tmp_path)
        registry = RoiRegistry()
        cfg = ColorPathsConfig()
        cfg.load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
            roi_registry=registry,
        )
        assert "box1" in registry.keys()
        assert isinstance(registry.resolve("box1")["box1"], RoiConfig)
        assert "box1" in cfg.rois

    def test_inline_roi_and_label_injected_into_registry(self, tmp_path):
        """Inline ROIs with a ``label`` key become :class:`RoiAndLabelConfig`."""
        toml_path = _write_toml(
            tmp_path,
            _minimal_color_paths_toml(
                extra="""
                [color_paths.roi.sand]
                name     = "sand"
                corner_1 = [0.0, 0.0]
                corner_2 = [1.0, 1.0]
                label    = 3
                """
            ),
        )
        data_reg = _make_data_registry(tmp_path)
        registry = RoiRegistry()
        cfg = ColorPathsConfig()
        cfg.load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
            roi_registry=registry,
        )
        resolved = registry.resolve("sand")
        assert isinstance(resolved["sand"], RoiAndLabelConfig)
        assert resolved["sand"].label == 3
        assert "sand" in cfg.rois

    def test_inline_roi_not_duplicated_in_rois_list(self, tmp_path):
        """If the user also lists the inline key in ``rois``, it appears only once."""
        toml_path = _write_toml(
            tmp_path,
            _minimal_color_paths_toml(
                rois_line='rois = ["box1"]',
                extra="""
                [color_paths.roi.box1]
                name     = "box1"
                corner_1 = [0.1, 0.2]
                corner_2 = [0.5, 0.6]
                """,
            ),
        )
        data_reg = _make_data_registry(tmp_path)
        registry = RoiRegistry()
        cfg = ColorPathsConfig()
        cfg.load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
            roi_registry=registry,
        )
        # "box1" should appear exactly once in cfg.rois
        assert cfg.rois.count("box1") == 1

    def test_inline_roi_without_registry_does_not_crash(self, tmp_path):
        """Inline ROIs are silently ignored when no registry is provided."""
        toml_path = _write_toml(
            tmp_path,
            _minimal_color_paths_toml(
                extra="""
                [color_paths.roi.box1]
                name     = "box1"
                corner_1 = [0.1, 0.2]
                corner_2 = [0.5, 0.6]
                """
            ),
        )
        data_reg = _make_data_registry(tmp_path)
        cfg = ColorPathsConfig()
        # Must not raise
        cfg.load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
            roi_registry=None,
        )
        # Without a registry, inline ROIs cannot be stored; rois stays empty
        assert cfg.rois == []


# ---------------------------------------------------------------------------
# RoiRegistry.register()
# ---------------------------------------------------------------------------


class TestRoiRegistryRegister:
    def test_register_new_key(self):
        reg = RoiRegistry()
        roi = RoiConfig()
        roi.load({"name": "a", "corner_1": [0.0, 0.0], "corner_2": [1.0, 1.0]})
        reg.register("a", roi)
        assert "a" in reg.keys()
        assert reg.resolve("a")["a"] is roi

    def test_register_duplicate_key_raises(self):
        reg = RoiRegistry()
        roi = RoiConfig()
        roi.load({"name": "a", "corner_1": [0.0, 0.0], "corner_2": [1.0, 1.0]})
        reg.register("a", roi)
        with pytest.raises(KeyError, match="already registered"):
            reg.register("a", roi)

    def test_register_does_not_affect_loaded_entries(self):
        """Entries added via register() coexist with entries from load()."""
        reg = RoiRegistry()
        # Manually seed the registry as if load() had been called
        roi_loaded = RoiConfig()
        roi_loaded.load(
            {"name": "loaded", "corner_1": [0.0, 0.0], "corner_2": [0.5, 0.5]}
        )
        reg._registry["loaded"] = roi_loaded

        roi_new = RoiConfig()
        roi_new.load({"name": "new", "corner_1": [0.5, 0.5], "corner_2": [1.0, 1.0]})
        reg.register("new", roi_new)
        assert set(reg.keys()) == {"loaded", "new"}


# ---------------------------------------------------------------------------
# ignore_baseline_spectrum config key
# ---------------------------------------------------------------------------


class TestIgnoreBaselineSpectrum:
    """Tests for the ``ignore_baseline_spectrum`` configuration key."""

    def _load_cfg(self, tmp_path: Path, extra: str = "") -> ColorPathsConfig:
        toml_path = _write_toml(tmp_path, _minimal_color_paths_toml(extra=extra))
        data_reg = _make_data_registry(tmp_path)
        cfg = ColorPathsConfig()
        cfg.load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
        )
        return cfg

    def test_default_is_expanded(self, tmp_path):
        """When the key is absent the default must be ``'expanded'``."""
        cfg = self._load_cfg(tmp_path)
        assert cfg.ignore_baseline_spectrum == "expanded"

    def test_explicit_expanded(self, tmp_path):
        cfg = self._load_cfg(tmp_path, 'ignore_baseline_spectrum = "expanded"')
        assert cfg.ignore_baseline_spectrum == "expanded"

    def test_baseline_value(self, tmp_path):
        cfg = self._load_cfg(tmp_path, 'ignore_baseline_spectrum = "baseline"')
        assert cfg.ignore_baseline_spectrum == "baseline"

    def test_none_value(self, tmp_path):
        cfg = self._load_cfg(tmp_path, 'ignore_baseline_spectrum = "none"')
        assert cfg.ignore_baseline_spectrum == "none"

    def test_invalid_value_raises(self, tmp_path):
        """An unrecognised value must raise a ``ValueError``."""
        with pytest.raises(ValueError, match="ignore_baseline_spectrum"):
            self._load_cfg(tmp_path, 'ignore_baseline_spectrum = "bad_value"')

    def test_dataclass_default(self):
        """The dataclass default must be ``'expanded'`` without loading any TOML."""
        cfg = ColorPathsConfig()
        assert cfg.ignore_baseline_spectrum == "expanded"


class TestBasisAwareCalibrationPaths:
    def test_color_paths_default_folder_uses_labels_basis(self, tmp_path):
        toml_path = _write_toml(tmp_path, _minimal_color_paths_toml())
        data_reg = _make_data_registry(tmp_path)
        cfg = ColorPathsConfig().load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
        )
        assert cfg.basis.value == "labels"
        assert (
            cfg.calibration_file
            == tmp_path / "calibration" / "color_paths" / "from_labels"
        )

    def test_color_paths_default_folder_uses_explicit_labels_basis(self, tmp_path):
        toml_path = _write_toml(
            tmp_path,
            _minimal_color_paths_toml(extra='basis = "labels"'),
        )
        data_reg = _make_data_registry(tmp_path)
        cfg = ColorPathsConfig().load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
        )
        assert cfg.basis.value == "labels"
        assert (
            cfg.calibration_file
            == tmp_path / "calibration" / "color_paths" / "from_labels"
        )


# ---------------------------------------------------------------------------
# histogram_weighting config key
# ---------------------------------------------------------------------------


class TestHistogramWeighting:
    """Tests for the ``histogram_weighting`` configuration key."""

    def _load_cfg(self, tmp_path: Path, extra: str = "") -> ColorPathsConfig:
        toml_path = _write_toml(tmp_path, _minimal_color_paths_toml(extra=extra))
        data_reg = _make_data_registry(tmp_path)
        cfg = ColorPathsConfig()
        cfg.load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
        )
        assert cfg.basis.value == "labels"
        assert (
            cfg.calibration_file
            == tmp_path / "calibration" / "color_paths" / "from_labels"
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

    def test_default_is_threshold(self, tmp_path):
        """When the key is absent the default must be ``'threshold'``."""
        cfg = self._load_cfg(tmp_path)
        assert cfg.histogram_weighting == "threshold"

    def test_explicit_threshold(self, tmp_path):
        cfg = self._load_cfg(tmp_path, 'histogram_weighting = "threshold"')
        assert cfg.histogram_weighting == "threshold"

    def test_wls_value(self, tmp_path):
        cfg = self._load_cfg(tmp_path, 'histogram_weighting = "wls"')
        assert cfg.histogram_weighting == "wls"

    def test_wls_sqrt_value(self, tmp_path):
        cfg = self._load_cfg(tmp_path, 'histogram_weighting = "wls_sqrt"')
        assert cfg.histogram_weighting == "wls_sqrt"

    def test_wls_log_value(self, tmp_path):
        cfg = self._load_cfg(tmp_path, 'histogram_weighting = "wls_log"')
        assert cfg.histogram_weighting == "wls_log"

    def test_invalid_value_raises(self, tmp_path):
        """An unrecognised value must raise a ``ValueError``."""
        with pytest.raises(ValueError, match="histogram_weighting"):
            self._load_cfg(tmp_path, 'histogram_weighting = "bad_value"')

    def test_dataclass_default(self):
        """The dataclass default must be ``'threshold'`` without loading any TOML."""
        cfg = ColorPathsConfig()
        assert cfg.histogram_weighting == "threshold"


# ---------------------------------------------------------------------------
# mode config key
# ---------------------------------------------------------------------------


class TestColorPathMode:
    """Tests for the ``mode`` configuration key in ``[color_paths]``."""

    def _load_cfg(self, tmp_path: Path, extra: str = "") -> ColorPathsConfig:
        toml_path = _write_toml(tmp_path, _minimal_color_paths_toml(extra=extra))
        data_reg = _make_data_registry(tmp_path)
        cfg = ColorPathsConfig()
        cfg.load(
            path=toml_path,
            data=tmp_path,
            results=tmp_path,
            data_registry=data_reg,
        )
        return cfg

    def test_default_is_auto(self, tmp_path):
        cfg = self._load_cfg(tmp_path)
        assert cfg.mode == "auto"

    def test_manual_value(self, tmp_path):
        cfg = self._load_cfg(tmp_path, 'mode = "manual"')
        assert cfg.mode == "manual"

    def test_auto_value(self, tmp_path):
        cfg = self._load_cfg(tmp_path, 'mode = "auto"')
        assert cfg.mode == "auto"

    def test_invalid_value_raises(self, tmp_path):
        with pytest.raises(ValueError, match="for 'mode'"):
            self._load_cfg(tmp_path, 'mode = "bad_value"')

    def test_dataclass_default(self):
        cfg = ColorPathsConfig()
        assert cfg.mode == "auto"
