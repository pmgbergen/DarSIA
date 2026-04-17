from __future__ import annotations

import numpy as np

import darsia
from darsia import make_coordinate
from darsia.presets.workflows.analysis.expert_knowledge import ExpertKnowledgeAdapter
from darsia.presets.workflows.config.roi import RoiConfig
from darsia.utils.standard_images import roi_to_mask


def _make_scalar_image(rows: int = 10, cols: int = 20) -> darsia.ScalarImage:
    arr = np.ones((rows, cols), dtype=np.float32)
    return darsia.ScalarImage(arr, space_dim=2, dimensions=[1.0, 2.0])


def _make_roi(corner_1: list[float], corner_2: list[float], name: str) -> RoiConfig:
    roi = RoiConfig()
    roi.roi = make_coordinate([corner_1, corner_2])
    roi.name = name
    return roi


def test_adapter_empty_config_is_noop() -> None:
    img = _make_scalar_image()
    adapter = ExpertKnowledgeAdapter()

    constrained_sg = adapter.apply(img, "saturation_g")
    constrained_caq = adapter.apply(img, "concentration_aq")

    assert constrained_sg is img
    assert constrained_caq is img


def test_adapter_constrains_only_target_mode() -> None:
    img = _make_scalar_image()
    roi = _make_roi([0.0, 0.0], [1.0, 1.0], "left_half")
    adapter = ExpertKnowledgeAdapter(
        saturation_g_rois={"left_half": roi},
        concentration_aq_rois={},
    )

    constrained_sg = adapter.apply(img, "saturation_g")
    constrained_caq = adapter.apply(img, "concentration_aq")
    expected_mask = roi_to_mask(roi.roi, img).img

    assert constrained_sg is not None
    assert np.allclose(constrained_sg.img[expected_mask], 1.0)
    assert np.allclose(constrained_sg.img[~expected_mask], 0.0)
    assert constrained_caq is img


def test_adapter_unions_multiple_rois() -> None:
    img = _make_scalar_image()
    roi_left = _make_roi([0.0, 0.0], [0.9, 1.0], "left")
    roi_right = _make_roi([1.1, 0.0], [2.0, 1.0], "right")
    adapter = ExpertKnowledgeAdapter(
        saturation_g_rois={"left": roi_left, "right": roi_right}
    )

    constrained = adapter.apply(img, "saturation_g")
    expected = roi_to_mask([roi_left.roi, roi_right.roi], img).img

    assert constrained is not None
    assert np.allclose(constrained.img[expected], 1.0)
    assert np.allclose(constrained.img[~expected], 0.0)
