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
    return RoiConfig(corner_1=corner_1, corner_2=corner_2, name=name)


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
    gap = roi_to_mask(make_coordinate([[0.9, 0.0], [1.1, 1.0]]), img).img

    assert constrained is not None
    assert np.allclose(constrained.img[expected], 1.0)
    assert np.allclose(constrained.img[~expected], 0.0)
    assert np.allclose(constrained.img[gap], 0.0)


def test_adapter_negative_only_zeroes_inside_keeps_outside() -> None:
    """A no_saturation_g-style exclusion with no positive ROI configured should
    zero the field inside the named region and leave it untouched outside."""
    img = _make_scalar_image()
    exclude_roi = _make_roi([1.1, 0.0], [2.0, 1.0], "exclude")
    adapter = ExpertKnowledgeAdapter(no_saturation_g_rois={"exclude": exclude_roi})

    constrained = adapter.apply(img, "saturation_g")
    excluded = roi_to_mask(exclude_roi.roi, img).img

    assert constrained is not None
    assert np.allclose(constrained.img[excluded], 0.0)
    assert np.allclose(constrained.img[~excluded], 1.0)


def test_adapter_positive_and_negative_apply_independently_without_overlap() -> None:
    """A positive ROI covering the whole domain plus a disjoint negative ROI
    should zero only the negative region, keeping the rest of the positive
    region untouched."""
    img = _make_scalar_image()
    whole_domain = _make_roi([0.0, 0.0], [2.0, 1.0], "whole")
    exclude_roi = _make_roi([1.1, 0.0], [2.0, 1.0], "exclude")
    adapter = ExpertKnowledgeAdapter(
        saturation_g_rois={"whole": whole_domain},
        no_saturation_g_rois={"exclude": exclude_roi},
    )

    constrained = adapter.apply(img, "saturation_g")
    excluded = roi_to_mask(exclude_roi.roi, img).img

    assert constrained is not None
    assert np.allclose(constrained.img[excluded], 0.0)
    assert np.allclose(constrained.img[~excluded], 1.0)


def test_adapter_negative_wins_on_overlap_with_positive() -> None:
    """Where a positive and a negative ROI overlap, the negative rule wins:
    the overlapping sub-region is zeroed even though it's inside the
    positive ROI too."""
    img = _make_scalar_image()
    positive_roi = _make_roi([0.0, 0.0], [0.9, 1.0], "left")
    overlap_roi = _make_roi([0.3, 0.0], [0.6, 1.0], "overlap")
    adapter = ExpertKnowledgeAdapter(
        saturation_g_rois={"left": positive_roi},
        no_saturation_g_rois={"overlap": overlap_roi},
    )

    constrained = adapter.apply(img, "saturation_g")
    positive_mask = roi_to_mask(positive_roi.roi, img).img
    overlap_mask = roi_to_mask(overlap_roi.roi, img).img
    kept = positive_mask & ~overlap_mask

    assert constrained is not None
    assert np.allclose(constrained.img[kept], 1.0)
    assert np.allclose(constrained.img[overlap_mask], 0.0)
    assert np.allclose(constrained.img[~positive_mask], 0.0)
