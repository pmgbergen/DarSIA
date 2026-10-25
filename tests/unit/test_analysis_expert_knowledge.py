from __future__ import annotations

import numpy as np
import pytest

import darsia
from darsia import make_coordinate
from darsia.presets.workflows.analysis.expert_knowledge import ExpertKnowledgeAdapter
from darsia.presets.workflows.config.roi import RoiConfig
from darsia.presets.workflows.config.roi_registry import RoiRegistry
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


def _make_labels_image(rows: int = 10, cols: int = 20) -> darsia.ScalarImage:
    labels = np.zeros((rows, cols), dtype=np.int32)
    labels[:, : cols // 2] = 1  # left half: label 1
    labels[:, cols // 2 :] = 2  # right half: label 2
    return darsia.ScalarImage(labels, space_dim=2, dimensions=[1.0, 2.0])


def _make_facies_image(rows: int = 10, cols: int = 20) -> darsia.ScalarImage:
    """Resolved facies-ID image: both raw labels 1/2 belong to facies 7,
    mirroring what `setup_facies`'s `reassign_labels` would produce."""
    facies = np.full((rows, cols), 7, dtype=np.int32)
    return darsia.ScalarImage(facies, space_dim=2, dimensions=[1.0, 2.0])


def test_adapter_label_shape_excludes_raw_label_region() -> None:
    """shape='label' zeroes exactly the pixels carrying that raw label index."""
    img = _make_scalar_image()
    labels = _make_labels_image()
    label_roi = RoiConfig(shape="label", label_index=1, name="lbl1")
    adapter = ExpertKnowledgeAdapter(
        no_saturation_g_rois={"lbl1": label_roi}, labels_image=labels
    )

    constrained = adapter.apply(img, "saturation_g")

    assert constrained is not None
    assert np.allclose(constrained.img[labels.img == 1], 0.0)
    assert np.allclose(constrained.img[labels.img == 2], 1.0)


def test_adapter_facies_shape_resolves_multiple_raw_labels() -> None:
    """shape='facies' matches every pixel resolved to that facies ID."""
    img = _make_scalar_image()
    facies = _make_facies_image()
    facies_roi = RoiConfig(shape="facies", facies_id=7, name="fac7")
    adapter = ExpertKnowledgeAdapter(
        no_saturation_g_rois={"fac7": facies_roi},
        facies_image=facies,
    )

    constrained = adapter.apply(img, "saturation_g")

    assert constrained is not None
    assert np.allclose(constrained.img, 0.0)  # facies 7 covers both labels


def test_adapter_mixes_box_and_label_shapes_in_same_list() -> None:
    """A geometric and a label-based ROI in the same list union correctly."""
    img = _make_scalar_image()
    labels = _make_labels_image()
    box_roi = _make_roi([0.0, 0.0], [2.0, 0.5], "bottom_box")
    label_roi = RoiConfig(shape="label", label_index=1, name="lbl1")
    adapter = ExpertKnowledgeAdapter(
        no_saturation_g_rois={"bottom_box": box_roi, "lbl1": label_roi},
        labels_image=labels,
    )
    box_mask = roi_to_mask(box_roi.roi, img).img
    label_mask = labels.img == 1
    expected_excluded = box_mask | label_mask

    constrained = adapter.apply(img, "saturation_g")

    assert constrained is not None
    assert np.allclose(constrained.img[expected_excluded], 0.0)
    assert np.allclose(constrained.img[~expected_excluded], 1.0)


def test_adapter_same_roi_usable_in_both_positive_and_negative_lists() -> None:
    """Any named ROI (any shape) can be referenced from both lists at once."""
    img = _make_scalar_image()
    shared_roi = _make_roi([0.0, 0.0], [1.0, 1.0], "shared")

    adapter = ExpertKnowledgeAdapter(
        saturation_g_rois={"shared": shared_roi},
        no_saturation_g_rois={"shared": shared_roi},
    )
    constrained = adapter.apply(img, "saturation_g")
    # Negative wins on overlap -> referencing the same ROI both ways zeroes
    # it entirely (kept nowhere: positive restricts to it, negative excludes it).
    assert constrained is not None
    assert np.allclose(constrained.img, 0.0)


def test_adapter_multiple_positive_and_multiple_negative_rois_combine_correctly() -> (
    None
):
    """Multiple ROIs on both sides at once: positive keeps only their union's
    interior; negative zeroes each of its ROIs' interiors, independent of how
    many ROIs are on either side."""
    img = _make_scalar_image()
    pos1 = _make_roi([0.0, 0.0], [0.8, 1.0], "pos1")
    pos2 = _make_roi([1.2, 0.0], [2.0, 1.0], "pos2")
    neg1 = _make_roi([0.2, 0.0], [0.5, 1.0], "neg1")  # inside pos1
    neg2 = _make_roi([1.5, 0.0], [1.8, 1.0], "neg2")  # inside pos2

    adapter = ExpertKnowledgeAdapter(
        saturation_g_rois={"pos1": pos1, "pos2": pos2},
        no_saturation_g_rois={"neg1": neg1, "neg2": neg2},
    )
    constrained = adapter.apply(img, "saturation_g")

    pos_union = roi_to_mask([pos1.roi, pos2.roi], img).img
    neg_union = roi_to_mask([neg1.roi, neg2.roi], img).img
    expected_kept = pos_union & ~neg_union

    assert constrained is not None
    assert np.array_equal(constrained.img > 0, expected_kept)


def test_adapter_label_shape_without_labels_image_raises() -> None:
    img = _make_scalar_image()
    label_roi = RoiConfig(shape="label", label_index=1, name="lbl1")
    adapter = ExpertKnowledgeAdapter(no_saturation_g_rois={"lbl1": label_roi})

    with pytest.raises(ValueError, match="labels_image"):
        adapter.apply(img, "saturation_g")


def test_adapter_facies_shape_without_facies_image_raises() -> None:
    img = _make_scalar_image()
    facies_roi = RoiConfig(shape="facies", facies_id=7, name="fac7")
    adapter = ExpertKnowledgeAdapter(no_saturation_g_rois={"fac7": facies_roi})

    with pytest.raises(ValueError, match="facies_image"):
        adapter.apply(img, "saturation_g")


def _make_registry(*rois: RoiConfig) -> RoiRegistry:
    registry = RoiRegistry()
    for roi in rois:
        registry.register(roi.name, roi)
    return registry


def test_adapter_intersection_shape_combines_box_and_label() -> None:
    """shape='intersection' replaces the old dead 'label' field's intended
    purpose: AND a geometric ROI with a label-based one."""
    img = _make_scalar_image()
    labels = _make_labels_image()
    box = _make_roi([0.0, 0.0], [2.0, 1.0], "full_box")
    lbl = RoiConfig(shape="label", label_index=1, name="lbl1")
    intersection = RoiConfig(
        shape="intersection", members=["full_box", "lbl1"], name="intersect1"
    )
    registry = _make_registry(box, lbl, intersection)

    adapter = ExpertKnowledgeAdapter(
        saturation_g_rois={"intersect1": intersection},
        labels_image=labels,
        roi_registry=registry,
    )
    constrained = adapter.apply(img, "saturation_g")

    box_mask = roi_to_mask(box.roi, img).img
    label_mask = labels.img == 1
    expected = box_mask & label_mask

    assert constrained is not None
    assert np.array_equal(constrained.img > 0, expected)


def test_adapter_union_shape_combines_box_and_label() -> None:
    img = _make_scalar_image()
    labels = _make_labels_image()
    box = _make_roi([0.0, 0.0], [2.0, 1.0], "full_box")
    lbl = RoiConfig(shape="label", label_index=1, name="lbl1")
    union_roi = RoiConfig(shape="union", members=["full_box", "lbl1"], name="union1")
    registry = _make_registry(box, lbl, union_roi)

    adapter = ExpertKnowledgeAdapter(
        saturation_g_rois={"union1": union_roi},
        labels_image=labels,
        roi_registry=registry,
    )
    constrained = adapter.apply(img, "saturation_g")

    box_mask = roi_to_mask(box.roi, img).img
    label_mask = labels.img == 1
    expected = box_mask | label_mask

    assert constrained is not None
    assert np.array_equal(constrained.img > 0, expected)


def test_adapter_composite_shape_nests_three_levels_deep() -> None:
    """A composite's members may themselves be composites."""
    img = _make_scalar_image()
    labels = _make_labels_image()
    box = _make_roi([0.0, 0.0], [2.0, 1.0], "full_box")
    lbl = RoiConfig(shape="label", label_index=1, name="lbl1")
    union_roi = RoiConfig(shape="union", members=["full_box", "lbl1"], name="union1")
    left_box = _make_roi([0.0, 0.0], [1.0, 1.0], "left_box")
    deep = RoiConfig(shape="intersection", members=["union1", "left_box"], name="deep")
    registry = _make_registry(box, lbl, union_roi, left_box, deep)

    adapter = ExpertKnowledgeAdapter(
        saturation_g_rois={"deep": deep}, labels_image=labels, roi_registry=registry
    )
    constrained = adapter.apply(img, "saturation_g")

    expected = (roi_to_mask(box.roi, img).img | (labels.img == 1)) & roi_to_mask(
        left_box.roi, img
    ).img

    assert constrained is not None
    assert np.array_equal(constrained.img > 0, expected)


def test_adapter_composite_direct_self_reference_raises() -> None:
    img = _make_scalar_image()
    box = _make_roi([0.0, 0.0], [1.0, 1.0], "full_box")
    cyc = RoiConfig(shape="intersection", members=["cyc", "full_box"], name="cyc")
    registry = _make_registry(box, cyc)

    adapter = ExpertKnowledgeAdapter(
        saturation_g_rois={"cyc": cyc}, roi_registry=registry
    )
    with pytest.raises(ValueError, match="circular"):
        adapter.apply(img, "saturation_g")


def test_adapter_composite_indirect_cycle_raises() -> None:
    img = _make_scalar_image()
    box = _make_roi([0.0, 0.0], [1.0, 1.0], "full_box")
    a = RoiConfig(shape="union", members=["b", "full_box"], name="a")
    b = RoiConfig(shape="union", members=["a", "full_box"], name="b")
    registry = _make_registry(box, a, b)

    adapter = ExpertKnowledgeAdapter(saturation_g_rois={"a": a}, roi_registry=registry)
    with pytest.raises(ValueError, match="circular"):
        adapter.apply(img, "saturation_g")


def test_adapter_composite_shape_without_roi_registry_raises() -> None:
    img = _make_scalar_image()
    intersection = RoiConfig(
        shape="intersection", members=["a", "b"], name="intersect1"
    )
    adapter = ExpertKnowledgeAdapter(saturation_g_rois={"intersect1": intersection})

    with pytest.raises(ValueError, match="roi_registry"):
        adapter.apply(img, "saturation_g")
