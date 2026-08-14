"""Unit tests for workflow segmentation contour value labels."""

import cv2
import numpy as np

import darsia
from darsia.presets.workflows.config.segmentation import SegmentationConfig
from darsia.presets.workflows.segmentation_contours import SegmentationContours


def _make_test_images() -> tuple[darsia.OpticalImage, darsia.ScalarImage]:
    """Create a small optical image and scalar signal for contour tests."""
    img = darsia.OpticalImage(
        img=np.zeros((120, 120, 3), dtype=np.uint8),
        space_dim=2,
        indexing="ij",
    )
    yy, xx = np.ogrid[:120, :120]
    mask = ((xx - 60) ** 2 + (yy - 60) ** 2) < 35**2
    values = darsia.ScalarImage(img=mask.astype(float), space_dim=2, indexing="ij")
    return img, values


def test_segmentation_config_value_labels_defaults():
    """Default value-label settings stay backward compatible."""
    sec = {
        "label": "phase",
        "mode": "mass",
        "thresholds": [0.5],
        "color": [255, 0, 0],
    }
    cfg = SegmentationConfig().load(sec)
    assert cfg.values.show_values is False
    assert cfg.values.value_color == [255, 0, 0]
    assert cfg.values.value_size == 0.5
    assert cfg.values.value_alpha == 1.0
    assert cfg.values.value_max_per_contour == 3


def test_segmentation_config_value_labels_nested_override():
    """Nested `values` section overrides flat value-label settings."""
    sec = {
        "label": "phase",
        "mode": "mass",
        "thresholds": [0.5],
        "color": [255, 0, 0],
        "show_values": False,
        "value_color": [10, 10, 10],
        "values": {
            "show_values": True,
            "value_color": [1, 2, 3],
            "value_size": 0.9,
            "value_alpha": 0.6,
            "value_density": 1.0,
            "value_min_distance_px": 20.0,
            "value_max_per_contour": 2,
            "value_format": "{:.3f}",
        },
    }
    cfg = SegmentationConfig().load(sec)
    assert cfg.values.show_values is True
    assert cfg.values.value_color == [1, 2, 3]
    assert cfg.values.value_size == 0.9
    assert cfg.values.value_alpha == 0.6
    assert cfg.values.value_density == 1.0
    assert cfg.values.value_min_distance_px == 20.0
    assert cfg.values.value_max_per_contour == 2
    assert cfg.values.value_format == "{:.3f}"


def test_contour_value_labels_toggle_changes_rendered_image():
    """Enabling contour-value labels adds extra rendered pixels."""
    img, values = _make_test_images()

    cfg_no_values = SegmentationConfig().load(
        {
            "label": "phase",
            "mode": "mass",
            "thresholds": [0.5],
            "color": [255, 0, 0],
            "alpha": [1.0],
            "linewidth": 2,
            "show_values": False,
        }
    )
    cfg_with_values = SegmentationConfig().load(
        {
            "label": "phase",
            "mode": "mass",
            "thresholds": [0.5],
            "color": [255, 0, 0],
            "alpha": [1.0],
            "linewidth": 2,
            "show_values": True,
            "value_color": [255, 255, 255],
            "value_size": 0.6,
            "value_density": 1.0,
            "value_min_distance_px": 15.0,
            "value_max_per_contour": 4,
            "value_format": "{:.1f}",
        }
    )

    without_values = SegmentationContours(cfg_no_values)(
        img, saturation_g=None, concentration_aq=None, mass=values
    )
    with_values = SegmentationContours(cfg_with_values)(
        img, saturation_g=None, concentration_aq=None, mass=values
    )

    assert np.any(with_values.img != without_values.img)


def test_segmentation_contours_supports_rescaled_modes():
    """Rescaled segmentation modes are accepted and rendered."""
    img, values = _make_test_images()
    cfg = SegmentationConfig().load(
        {
            "label": "phase",
            "mode": "rescaled_mass",
            "thresholds": [0.5],
            "color": [255, 0, 0],
            "alpha": [1.0],
        }
    )

    rendered = SegmentationContours(cfg)(
        img,
        scalar_products={"rescaled_mass": values},
    )

    assert rendered.img.shape == img.img.shape
    assert np.any(rendered.img != img.img)


def test_segmentation_contours_mass_alias_remains_supported():
    """Legacy mass mode remains supported for backward compatibility."""
    img, values = _make_test_images()
    cfg = SegmentationConfig().load(
        {
            "label": "phase",
            "mode": "mass",
            "thresholds": [0.5],
            "color": [255, 0, 0],
            "alpha": [1.0],
        }
    )

    rendered = SegmentationContours(cfg)(
        img,
        scalar_products={"mass": values},
    )
    assert np.any(rendered.img != img.img)


def test_select_label_positions_respects_spacing_and_cap():
    """Selected positions satisfy min distance and max-per-contour cap."""
    contour = cv2.ellipse2Poly((100, 100), (70, 40), 0, 0, 360, 3).reshape(-1, 1, 2)
    seg = SegmentationContours(
        SegmentationConfig().load(
            {
                "label": "phase",
                "mode": "mass",
                "thresholds": [0.5],
                "color": [255, 0, 0],
            }
        )
    )

    positions, _ = seg._select_label_positions(
        contour=contour,
        min_distance_px=18.0,
        max_per_contour=3,
        density=1.0,
        existing_positions=[],
        existing_boxes=[],
        text="0.5",
        font_scale=0.6,
        thickness=1,
    )

    assert len(positions) <= 3
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.hypot(
                positions[i][0] - positions[j][0], positions[i][1] - positions[j][1]
            )
            assert dist >= 18.0
