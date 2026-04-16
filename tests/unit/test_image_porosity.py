"""Unit tests for ImagePorosityConfig and Rig image-porosity workflow."""

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

    def test_load_full_mode(self):
        cfg = ImagePorosityConfig().load({"mode": "full", "tol": 0.8})
        assert cfg.mode == "full"
        assert cfg.tol == pytest.approx(0.8)

    def test_load_from_image_mode(self):
        cfg = ImagePorosityConfig().load({"mode": "from_image"})
        assert cfg.mode == "from_image"
        assert cfg.tol == pytest.approx(0.9)  # default unchanged

    def test_load_missing_section_uses_defaults(self):
        # Empty dict → all defaults
        cfg = ImagePorosityConfig().load({})
        assert cfg.mode == "full"
        assert cfg.tol == pytest.approx(0.9)

    def test_load_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            ImagePorosityConfig().load({"mode": "bad_mode"})

    def test_load_invalid_tol_raises(self):
        with pytest.raises(ValueError, match="tol must be"):
            ImagePorosityConfig().load({"tol": 0.0})

    def test_load_tol_gt_one_raises(self):
        with pytest.raises(ValueError, match="tol must be"):
            ImagePorosityConfig().load({"tol": 1.5})


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
        """Even when threshold > 1 is forced, full mode stays all-True."""
        rig = _make_rig()
        cfg = ImagePorosityConfig(mode="full")
        rig.setup_image_porosity(config=cfg)
        rig.setup_boolean_image_porosity(threshold=2.0)
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
