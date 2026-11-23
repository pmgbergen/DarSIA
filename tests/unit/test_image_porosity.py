"""Unit tests for ImagePorosityConfig and the image porosity of `processing`."""

from pathlib import Path

import numpy as np
import pytest

import darsia
from darsia.presets.workflows.config.image_porosity import ImagePorosityConfig
from darsia.presets.workflows.processing.image_porosity import (
    boolean_image_porosity,
    compute_image_porosity,
    load_image_porosity,
    save_image_porosity,
)
from darsia.presets.workflows.results_folder import image_porosity_folder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(shape=(10, 20), two_labels=False):
    """Synthetic baseline and labels (optionally two labels split at the middle)."""
    baseline = darsia.Image(np.ones((*shape, 3), dtype=np.float32))
    labels = np.zeros(shape, dtype=np.uint8)
    if two_labels:
        labels[shape[0] // 2 :, :] = 1  # noqa: E203
    return baseline, darsia.Image(labels)


# ---------------------------------------------------------------------------
# ImagePorosityConfig
# ---------------------------------------------------------------------------


class TestImagePorosityConfig:
    def test_defaults(self):
        cfg = ImagePorosityConfig()
        assert cfg.active is False
        assert cfg.method == "kmeans"
        assert cfg.tol == pytest.approx(0.9)
        assert cfg.cutoff == pytest.approx(0.3)
        assert cfg.boundary_band == 5
        assert cfg.kmeans.patches == (1, 1)
        assert cfg.kmeans.num_clusters == 5
        assert cfg.kmeans.tol_color_distance == pytest.approx(0.1)
        assert cfg.kmeans.tol_color_gradient == pytest.approx(0.02)

    def test_load_detection_and_options(self):
        cfg = ImagePorosityConfig()._load_dict(
            {
                "active": True,
                "method": "kmeans",
                "tol": 0.8,
                "cutoff": 0.2,
                "boundary_band": 3,
                "kmeans": {
                    "patches": [2, 3],
                    "num_clusters": 8,
                    "tol_color_distance": 0.05,
                    "tol_color_gradient": 0.01,
                },
            }
        )
        assert cfg.active is True
        assert cfg.tol == pytest.approx(0.8)
        assert cfg.cutoff == pytest.approx(0.2)
        assert cfg.boundary_band == 3
        assert cfg.method_options().patches == (2, 3)
        assert cfg.kmeans.num_clusters == 8
        assert cfg.kmeans.tol_color_distance == pytest.approx(0.05)
        assert cfg.kmeans.tol_color_gradient == pytest.approx(0.01)

    def test_load_empty_section_uses_defaults(self):
        cfg = ImagePorosityConfig()._load_dict({})
        assert cfg.active is False
        assert cfg.boundary_band == 5

    @pytest.mark.parametrize("key", ["mode", "patches", "sample_width"])
    def test_former_keys_raise_with_hint(self, key):
        with pytest.raises(ValueError, match=r"image_porosity\.kmeans"):
            ImagePorosityConfig()._load_dict({key: 1})

    def test_load_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            ImagePorosityConfig()._load_dict({"method": "bad"})

    @pytest.mark.parametrize("tol", [0.0, 1.5])
    def test_load_invalid_tol_raises(self, tol):
        with pytest.raises(ValueError, match="tol must be"):
            ImagePorosityConfig()._load_dict({"tol": tol})

    def test_load_negative_band_raises(self):
        with pytest.raises(ValueError, match="boundary_band"):
            ImagePorosityConfig()._load_dict({"boundary_band": -1})

    def test_load_from_toml_path(self, tmp_path: Path):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            "[image_porosity]\nactive = true\ntol = 0.75\n\n"
            "[image_porosity.kmeans]\nnum_clusters = 7\n"
        )
        cfg = ImagePorosityConfig().load(toml_file)
        assert cfg.active is True
        assert cfg.tol == pytest.approx(0.75)
        assert cfg.kmeans.num_clusters == 7

    def test_load_missing_section_raises(self, tmp_path: Path):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[data]\nfoo = 1\n")
        with pytest.raises(KeyError):
            ImagePorosityConfig().load(toml_file)


# ---------------------------------------------------------------------------
# compute_image_porosity / boolean_image_porosity
# ---------------------------------------------------------------------------


class TestComputeImagePorosity:
    def test_no_detection_one_label_is_ones(self):
        baseline, labels = _make_inputs()
        porosity = compute_image_porosity(baseline, labels, ImagePorosityConfig())
        assert porosity.img.dtype == np.float32
        assert np.all(porosity.img == 1.0)

    def test_boundary_band_zeroes_both_sides(self):
        baseline, labels = _make_inputs(shape=(20, 10), two_labels=True)
        cfg = ImagePorosityConfig(boundary_band=2)
        porosity = compute_image_porosity(baseline, labels, cfg).img
        # Boundary between rows 9 and 10: rows 8..11 zeroed, the rest stays 1.
        assert np.all(porosity[8:12] == 0.0)
        assert np.all(porosity[:8] == 1.0)
        assert np.all(porosity[12:] == 1.0)

    def test_boundary_band_zero_switches_off(self):
        baseline, labels = _make_inputs(shape=(20, 10), two_labels=True)
        cfg = ImagePorosityConfig(boundary_band=0)
        assert np.all(compute_image_porosity(baseline, labels, cfg).img == 1.0)

    def test_kmeans_shape_and_unit_interval(self):
        baseline, labels = _make_inputs(shape=(8, 12))
        cfg = ImagePorosityConfig(active=True, boundary_band=0)
        img = compute_image_porosity(baseline, labels, cfg).img
        assert img.shape == (8, 12)
        assert np.all(img >= 0.0)
        assert np.all(img <= 1.0)

    def test_boolean_without_detection_is_full_domain(self):
        baseline, labels = _make_inputs()
        cfg = ImagePorosityConfig(tol=1.0)
        porosity = compute_image_porosity(baseline, labels, cfg)
        assert np.all(boolean_image_porosity(porosity, cfg).img)

    def test_boolean_with_detection_uses_tol(self):
        baseline, _ = _make_inputs(shape=(6, 8))
        porosity = darsia.zeros_like(baseline, mode="voxels", dtype=float)
        porosity.img[:3, :] = 1.0
        cfg = ImagePorosityConfig(active=True, tol=0.5)
        boolean = boolean_image_porosity(porosity, cfg).img
        assert np.all(boolean[:3, :])
        assert not np.any(boolean[3:, :])


# ---------------------------------------------------------------------------
# Storage and reading
# ---------------------------------------------------------------------------


class TestStoredImagePorosity:
    def test_save_and_load(self, tmp_path: Path):
        baseline, labels = _make_inputs(shape=(20, 10), two_labels=True)
        cfg = ImagePorosityConfig(boundary_band=2)
        porosity = compute_image_porosity(baseline, labels, cfg)
        folder = image_porosity_folder(tmp_path, None)
        save_image_porosity(folder, porosity, cfg)
        assert (folder / "image_porosity.npz").exists()
        assert (folder / "metadata.json").exists()
        assert (folder / "figures" / "image_porosity.jpg").stat().st_size > 0

        stored = load_image_porosity(cfg, tmp_path)
        assert np.array_equal(stored.map_on(baseline).img, porosity.img)
        assert np.all(stored.boolean_on(baseline).img)

    def test_missing_file_gives_hint(self, tmp_path: Path):
        stored = load_image_porosity(ImagePorosityConfig(active=True), tmp_path)
        with pytest.raises(FileNotFoundError, match="Image porosity"):
            stored.map  # noqa: B018

    def test_boolean_without_detection_needs_no_file(self, tmp_path: Path):
        baseline, _ = _make_inputs()
        stored = load_image_porosity(None, tmp_path)
        assert np.all(stored.boolean_on(baseline).img)
