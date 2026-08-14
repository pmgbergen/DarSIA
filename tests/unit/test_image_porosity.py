"""Unit tests for ImagePorosityConfig and Rig image-porosity workflow."""

from pathlib import Path

import numpy as np
import pytest

import darsia
from darsia.presets.workflows.config.image_porosity import ImagePorosityConfig
from darsia.presets.workflows.rig import Rig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rig(shape=(10, 20)):
    """Return a minimal Rig with a synthetic baseline and labels."""
    rig = Rig()
    rig.baseline = darsia.Image(np.ones((*shape, 3), dtype=np.float32))
    rig.labels = darsia.Image(np.zeros(shape, dtype=np.uint8))
    return rig


# ---------------------------------------------------------------------------
# ImagePorosityConfig – unit tests
# ---------------------------------------------------------------------------


class TestImagePorosityConfig:
    def test_defaults(self):
        cfg = ImagePorosityConfig()
        assert cfg.mode == "full"
        assert cfg.tol == pytest.approx(0.9)
        assert cfg.patches == (1, 1)
        assert cfg.num_clusters == 5
        assert cfg.sample_width == 50
        assert cfg.tol_color_distance == pytest.approx(0.1)
        assert cfg.tol_color_gradient == pytest.approx(0.02)

    def test_load_full_mode(self):
        cfg = ImagePorosityConfig()._load_dict({"mode": "full", "tol": 0.8})
        assert cfg.mode == "full"
        assert cfg.tol == pytest.approx(0.8)

    def test_load_from_image_mode(self):
        cfg = ImagePorosityConfig()._load_dict({"mode": "from_image"})
        assert cfg.mode == "from_image"
        assert cfg.tol == pytest.approx(0.9)  # default unchanged

    def test_load_from_image_options(self):
        cfg = ImagePorosityConfig()._load_dict(
            {
                "mode": "from_image",
                "patches": [2, 3],
                "num_clusters": 8,
                "sample_width": 100,
                "tol_color_distance": 0.05,
                "tol_color_gradient": 0.01,
            }
        )
        assert cfg.patches == (2, 3)
        assert cfg.num_clusters == 8
        assert cfg.sample_width == 100
        assert cfg.tol_color_distance == pytest.approx(0.05)
        assert cfg.tol_color_gradient == pytest.approx(0.01)

    def test_load_missing_section_uses_defaults(self):
        # Empty dict → all defaults
        cfg = ImagePorosityConfig()._load_dict({})
        assert cfg.mode == "full"
        assert cfg.tol == pytest.approx(0.9)
        assert cfg.patches == (1, 1)

    def test_load_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            ImagePorosityConfig()._load_dict({"mode": "bad_mode"})

    def test_load_invalid_tol_raises(self):
        with pytest.raises(ValueError, match="tol must be"):
            ImagePorosityConfig()._load_dict({"tol": 0.0})

    def test_load_tol_gt_one_raises(self):
        with pytest.raises(ValueError, match="tol must be"):
            ImagePorosityConfig()._load_dict({"tol": 1.5})

    def test_load_from_toml_path(self, tmp_path: Path):
        """load(path) reads [image_porosity] from a TOML file."""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            '[image_porosity]\nmode = "from_image"\ntol = 0.75\nnum_clusters = 7\n'
        )
        cfg = ImagePorosityConfig().load(toml_file)
        assert cfg.mode == "from_image"
        assert cfg.tol == pytest.approx(0.75)
        assert cfg.num_clusters == 7

    def test_load_missing_section_raises(self, tmp_path: Path):
        """load(path) raises KeyError when section is absent."""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[data]\nfoo = 1\n")
        with pytest.raises(KeyError):
            ImagePorosityConfig().load(toml_file)


# ---------------------------------------------------------------------------
# Rig.setup_image_porosity – unit tests
# ---------------------------------------------------------------------------


class TestSetupImagePorosity:
    def test_default_no_config_produces_full_porosity(self):
        """Missing config → constant 1 porosity (backward compatibility)."""
        rig = _make_rig()
        rig.setup_image_porosity()
        assert np.all(rig.image_porosity.img == pytest.approx(1.0))

    def test_full_mode_produces_full_porosity(self):
        rig = _make_rig()
        rig.setup_image_porosity(config=ImagePorosityConfig(mode="full"))
        assert np.all(rig.image_porosity.img == pytest.approx(1.0))

    def test_from_image_mode_returns_image_shaped_porosity(self):
        rig = _make_rig(shape=(8, 12))
        cfg = ImagePorosityConfig(mode="from_image")
        rig.setup_image_porosity(config=cfg)
        assert rig.image_porosity.img.shape == (8, 12)

    def test_from_image_values_in_unit_interval(self):
        rig = _make_rig(shape=(6, 8))
        cfg = ImagePorosityConfig(mode="from_image")
        rig.setup_image_porosity(config=cfg)
        img = rig.image_porosity.img
        assert np.all(img >= 0.0)
        assert np.all(img <= 1.0)

    def test_config_stored_on_rig(self):
        rig = _make_rig()
        cfg = ImagePorosityConfig(mode="from_image", tol=0.7)
        rig.setup_image_porosity(config=cfg)
        assert rig._image_porosity_config is cfg

    def test_log_saves_jpg(self, tmp_path: Path):
        """Image porosity JPG is stored when log is provided."""
        rig = _make_rig()
        rig.setup_image_porosity(config=ImagePorosityConfig(mode="full"), log=tmp_path)
        jpg = tmp_path / "image_porosity" / "image_porosity.jpg"
        assert jpg.exists()
        assert jpg.stat().st_size > 0

    def test_from_image_log_saves_jpg(self, tmp_path: Path):
        """from_image mode also saves JPG when log is provided."""
        rig = _make_rig(shape=(6, 8))
        cfg = ImagePorosityConfig(mode="from_image")
        rig.setup_image_porosity(config=cfg, log=tmp_path)
        jpg = tmp_path / "image_porosity" / "image_porosity.jpg"
        assert jpg.exists()
        assert jpg.stat().st_size > 0


# ---------------------------------------------------------------------------
# Rig.setup_boolean_image_porosity – unit tests
# ---------------------------------------------------------------------------


class TestSetupBooleanImagePorosity:
    def test_default_no_config_produces_full_boolean_mask(self):
        """Missing config after full-mode setup → all True."""
        rig = _make_rig()
        rig.setup_image_porosity()
        rig.setup_boolean_image_porosity()
        assert np.all(rig.boolean_porosity.img)

    def test_full_mode_always_all_true_regardless_of_tol(self):
        """In full mode the boolean mask must be True everywhere, even with tol=1.0."""
        rig = _make_rig()
        cfg = ImagePorosityConfig(mode="full", tol=1.0)
        rig.setup_image_porosity(config=cfg)
        rig.setup_boolean_image_porosity()
        assert np.all(rig.boolean_porosity.img)

    def test_full_mode_explicit_threshold_still_all_true(self):
        """In full mode, boolean mask is always True even with threshold=1.0 (edge case)."""
        rig = _make_rig()
        cfg = ImagePorosityConfig(mode="full")
        rig.setup_image_porosity(config=cfg)
        # With constant porosity=1.0, threshold=1.0 would yield all-False for the old
        # implementation (1.0 > 1.0 == False).  Full mode must stay all-True.
        rig.setup_boolean_image_porosity(threshold=1.0)
        assert np.all(rig.boolean_porosity.img)

    def test_from_image_uses_tol(self):
        """With from_image mode, tol actually filters the mask."""
        rig = _make_rig(shape=(6, 8))
        cfg = ImagePorosityConfig(mode="from_image", tol=0.5)
        rig.setup_image_porosity(config=cfg)
        # Inject a known porosity map: half < 0.5, half > 0.5
        rig.image_porosity.img[:] = 0.0
        rig.image_porosity.img[:3, :] = 1.0  # top half all porous
        rig.setup_boolean_image_porosity(config=cfg)
        assert np.all(rig.boolean_porosity.img[:3, :])
        assert not np.any(rig.boolean_porosity.img[3:, :])

    def test_from_image_threshold_arg_overrides_config_tol(self):
        """Explicit threshold kwarg has higher precedence than config.tol."""
        rig = _make_rig(shape=(4, 6))
        cfg = ImagePorosityConfig(mode="from_image", tol=0.9)
        rig.setup_image_porosity(config=cfg)
        # All porosity = 0.8 → above 0.5, below 0.9
        rig.image_porosity.img[:] = 0.8
        # With config.tol=0.9 → all False
        rig.setup_boolean_image_porosity(config=cfg)
        assert not np.any(rig.boolean_porosity.img)
        # With explicit threshold=0.5 → all True
        rig.setup_boolean_image_porosity(threshold=0.5, config=cfg)
        assert np.all(rig.boolean_porosity.img)

    def test_fallback_to_stored_config(self):
        """Boolean setup falls back to _image_porosity_config if no config given."""
        rig = _make_rig(shape=(4, 6))
        cfg = ImagePorosityConfig(mode="from_image", tol=0.5)
        rig.setup_image_porosity(config=cfg)
        # Inject known porosity
        rig.image_porosity.img[:] = 0.8  # all > 0.5 → should all be True
        rig.setup_boolean_image_porosity()  # no config passed
        assert np.all(rig.boolean_porosity.img)

    def test_log_saves_jpg(self, tmp_path: Path):
        """Boolean porosity JPG is stored when log is provided."""
        rig = _make_rig()
        rig.setup_image_porosity(config=ImagePorosityConfig(mode="full"))
        rig.setup_boolean_image_porosity(log=tmp_path)
        jpg = tmp_path / "image_porosity" / "boolean_porosity.jpg"
        assert jpg.exists()
        assert jpg.stat().st_size > 0
