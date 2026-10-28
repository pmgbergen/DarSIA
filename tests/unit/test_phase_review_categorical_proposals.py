"""Finite saved-label selection contracts; no producer or scientific fixtures."""

from copy import deepcopy
import builtins

import numpy as np
import pytest

from darsia.fluidflower import phase_review_categorical_proposals as proposals


def _arguments(labels, **changes):
    height, width = labels.shape
    values = dict(domain=labels != 255, rule_masks={}, run_id="Run One", frame_index=2,
                  elapsed_h=1.25, model_id="saved-model", candidate_id="saved-candidate",
                  fingerprint="a" * 64, canvas_wh=(width, height),
                  content_box_xywh=(0, 0, width, height), count=8, box_size_px=4)
    values.update(changes)
    return values


def _select(labels, **changes):
    return proposals.propose_saved_category_regions(labels, **_arguments(labels, **changes))


def _split(first=0, second=1, shape=(4, 4)):
    labels = np.full(shape, first, np.uint8)
    labels[:, shape[1] // 2:] = second
    return labels


@pytest.mark.parametrize("first,second", [(0, 1), (0, 2), (1, 2)])
def test_each_actual_adjacent_category_pair_seeds_a_blind_box(first, second):
    labels = _split(first, second)
    rows = _select(labels)
    assert len(rows) == 1
    row = rows[0]
    assert row["bbox_xywh_px"] == [0, 0, 4, 4]
    assert row["selection_basis"] == "saved_final_phase_boundaries"
    assert row["model_suggestion_hidden_until_answer"] is True
    assert not {"suggested_phase", "phase_class", "human_choice", "answered_utc",
                "mean_probabilities", "uncertainty_mean", "uncertainty_p90",
                "confidence", "entropy", "probability"}.intersection(row)


def test_vertical_adjacency_is_equivalent_to_horizontal_without_transposing_geometry():
    labels = _split().T.copy()
    assert _select(labels)[0]["bbox_xywh_px"] == [0, 0, 4, 4]


def test_rule_changes_can_seed_a_uniform_frame_and_rule_names_do_not_preanswer():
    labels = np.zeros((4, 4), np.uint8)
    changed = np.zeros_like(labels, dtype=bool)
    changed[1, 1] = True
    rows = _select(labels, rule_masks={"removed_top": changed})
    assert len(rows) == 1
    assert rows[0]["selection_basis"] == "saved_rule_changes"
    assert "removed_top" not in repr(rows)
    assert _select(labels) == []
    assert _select(labels, rule_masks={"removed_top": np.zeros_like(changed)}) == []


def test_boundary_and_actual_rule_change_are_both_retained_as_selection_basis():
    labels = _split()
    changed = np.zeros_like(labels, dtype=bool)
    changed[0, 0] = True
    assert _select(labels, rule_masks={"post_floor": changed})[0]["selection_basis"] == (
        "saved_final_phase_boundaries_and_rule_changes")


def test_outside_domain_is_neither_a_class_boundary_nor_a_rule_seed():
    labels = np.zeros((4, 4), np.uint8)
    labels[:, 1:3] = 255
    labels[:, 3] = 2
    assert _select(labels, minimum_domain_fraction=0) == []
    assert _select(labels, minimum_domain_fraction=0,
                   rule_masks={"outside_only": labels == 255}) == []
    assert _select(np.full((4, 4), 255, np.uint8), minimum_domain_fraction=0) == []


def test_domain_fraction_is_the_exact_original_box_fraction():
    labels = _split()
    labels[0, 0] = 255
    assert _select(labels) == []
    rows = _select(labels, minimum_domain_fraction=15 / 16)
    assert len(rows) == 1
    assert rows[0]["domain_fraction"] == 15 / 16


def test_padded_canvas_uses_exact_content_pixels_without_resize_or_crop_guess():
    labels = _split()
    row = _select(labels, canvas_wh=(16, 9), content_box_xywh=(6, 2, 4, 4))[0]
    assert row["bbox_xywh_px"] == [0, 0, 4, 4]
    assert row["coordinate_space"] == "normalized_native_review_canvas_16x9"
    assert [row[key] for key in ("x", "y", "width", "height")] == [6 / 16, 2 / 9, 4 / 16, 4 / 9]
    assert row["points"] == [[6 / 16, 2 / 9], [10 / 16, 2 / 9],
                              [10 / 16, 2 / 9 + 4 / 9], [6 / 16, 2 / 9 + 4 / 9]]
    assert row["candidate"] == "A"
    assert row["run_id"] == "run one"
    assert row["frame_index"] == 2 and row["elapsed_h"] == 1.25
    assert row["model_id"] == "saved-model"
    assert row["candidate_id"] == "saved-candidate"
    assert row["params_sha256"] == "a" * 64
    assert row["analysis_domain"] == "registered_porous_medium_2d"


def test_small_exact_native_grid_is_not_enlarged_to_the_default_box_size():
    labels = _split(shape=(2, 3))
    rows = _select(labels, box_size_px=96)
    assert rows
    assert all(row["bbox_xywh_px"][2:] == [2, 2] for row in rows)
    assert all(row["bbox_xywh_px"][0] + 2 <= 3 for row in rows)


def test_last_native_row_and_column_can_seed_a_rule_box():
    labels = np.zeros((11, 13), np.uint8)
    changed = np.zeros_like(labels, dtype=bool)
    changed[-1, -1] = True
    rows = _select(labels, rule_masks={"post_contact": changed})
    assert [row["bbox_xywh_px"] for row in rows] == [[9, 7, 4, 4]]


def test_existing_canvas_and_local_boxes_exclude_the_same_exact_pixels():
    labels = _split()
    geometry = dict(canvas_wh=(16, 9), content_box_xywh=(6, 2, 4, 4))
    first = _select(labels, **geometry)
    retained = deepcopy(first)
    assert _select(labels, existing_regions=first, **geometry) == []
    assert _select(labels, existing_regions=[dict(x=0., y=0., width=1., height=1.)], **geometry) == []
    assert first == retained
    # Exact repeats remain forbidden even if all other overlap is permitted.
    assert _select(labels, existing_regions=first, maximum_iou=1., **geometry) == []


def test_existing_pixel_geometry_must_match_its_canvas_mapping():
    labels = _split()
    existing = _select(labels)
    existing[0]["x"] += .01
    with pytest.raises(ValueError, match="canvas and content"):
        _select(labels, existing_regions=existing)


def test_other_frame_or_run_does_not_hide_this_frames_boxes():
    labels = _split()
    first = _select(labels)
    assert _select(labels, existing_regions=[dict(first[0], frame_index=3)]) == first
    assert _select(labels, existing_regions=[dict(first[0], run_id="other")]) == first


def _iou(first, second):
    ax, ay, aw, ah = first["bbox_xywh_px"]
    bx, by, bw, bh = second["bbox_xywh_px"]
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by))
    return intersection / (aw * ah + bw * bh - intersection)


def test_repeat_generation_is_deterministic_and_excludes_prior_boxes_and_overlap():
    labels = _split(shape=(12, 12))
    first = _select(labels, count=2)
    assert len(first) == 2
    assert _select(labels, count=2) == first
    assert _select(labels, count=1) == first[:1]
    remaining = _select(labels, count=100, existing_regions=first)
    combined = first + remaining
    assert len({row["proposal_id"] for row in combined}) == len(combined)
    assert all(_iou(a, b) <= .18 for i, a in enumerate(combined) for b in combined[i + 1:])
    assert _select(labels, count=100, existing_regions=combined) == []


@pytest.mark.parametrize("key,value", [("run_id", "other"), ("frame_index", 3),
    ("model_id", "other"), ("candidate_id", "other"), ("fingerprint", "b" * 64)])
def test_ids_bind_each_original_selection_identity(key, value):
    labels = _split()
    assert _select(labels)[0]["proposal_id"] != _select(labels, **{key: value})[0]["proposal_id"]


def test_ids_bind_algorithm_and_box_but_not_request_size(monkeypatch):
    labels = _split()
    first = _select(labels, count=1)[0]["proposal_id"]
    assert _select(labels, count=20)[0]["proposal_id"] == first
    monkeypatch.setattr(proposals, "SELECTION_ALGORITHM", "other-test-algorithm")
    assert _select(labels)[0]["proposal_id"] != first


def test_rule_mapping_order_and_facies_storage_layout_do_not_change_results():
    labels = _split(shape=(8, 8))
    a = np.zeros_like(labels, dtype=bool)
    b = np.zeros_like(labels, dtype=bool)
    a[1, 1], b[6, 6] = True, True
    facies = np.indices(labels.shape)[0].astype(np.int16)
    first = _select(labels, rule_masks={"a": a, "b": b}, facies_labels=facies)
    second = _select(labels, rule_masks={"b": b, "a": a}, facies_labels=np.asfortranarray(facies))
    assert first == second
    assert all("facies_id" in row for row in first)
    assert all("facies_id" not in row for row in _select(labels))


def test_readonly_noncontiguous_arrays_are_unchanged_and_selection_performs_no_io(monkeypatch):
    backing = _split(shape=(8, 16))
    labels = backing[:, ::2]
    domain = labels != 255
    changed = np.zeros_like(labels, dtype=bool)
    changed[0, 0] = True
    arrays = (labels, domain, changed)
    original = [value.tobytes() for value in arrays]
    for value in arrays:
        value.flags.writeable = False

    def refuse(*args, **kwargs):
        raise AssertionError("Saved categorical selection must not read files or resize arrays")

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", refuse)
        guard.setattr(np, "load", refuse)
        guard.setattr(np, "resize", refuse)
        rows = _select(labels, domain=domain, rule_masks={"top": changed})
    assert rows
    assert [value.tobytes() for value in arrays] == original


@pytest.mark.parametrize("bad", ["float", "int16", "unknown", "three_dimensions", "list", "too_small"])
def test_labels_refuse_coercion_and_unknown_categories(bad):
    labels = _split()
    arguments = _arguments(labels)
    if bad == "float": labels = labels.astype(np.float32)
    elif bad == "int16": labels = labels.astype(np.int16)
    elif bad == "unknown": labels[0, 0] = 3
    elif bad == "three_dimensions": labels = labels[None, ...]
    elif bad == "list": labels = labels.tolist()
    else: labels = labels[:1]
    with pytest.raises(ValueError, match="exact 2D uint8"):
        proposals.propose_saved_category_regions(labels, **arguments)


@pytest.mark.parametrize("bad", ["domain_dtype", "domain_shape", "domain_membership",
    "rule_dtype", "rule_shape", "rule_name", "facies_dtype", "facies_shape", "content_shape"])
def test_exact_domain_rule_and_optional_facies_contracts(bad):
    labels = _split()
    changes = {}
    if bad == "domain_dtype": changes["domain"] = np.ones(labels.shape, np.uint8)
    elif bad == "domain_shape": changes["domain"] = np.ones((3, 4), bool)
    elif bad == "domain_membership": changes["domain"] = np.zeros_like(labels, dtype=bool)
    elif bad == "rule_dtype": changes["rule_masks"] = {"rule": np.ones_like(labels)}
    elif bad == "rule_shape": changes["rule_masks"] = {"rule": np.ones((3, 4), bool)}
    elif bad == "rule_name": changes["rule_masks"] = {" ": np.ones_like(labels, dtype=bool)}
    elif bad == "facies_dtype": changes["facies_labels"] = np.ones(labels.shape, np.float32)
    elif bad == "facies_shape": changes["facies_labels"] = np.ones((3, 4), np.int32)
    else: changes["content_box_xywh"] = (0, 0, 3, 4)
    with pytest.raises(ValueError):
        _select(labels, **changes)


@pytest.mark.parametrize("changes", [dict(count=True), dict(count=-1), dict(frame_index=1.5),
    dict(box_size_px=0), dict(elapsed_h=float("nan")), dict(fingerprint="not-a-hash"),
    dict(run_id=" "), dict(candidate_id=""), dict(minimum_domain_fraction=1.1),
    dict(maximum_iou=-.1), dict(canvas_wh=(4., 4)), dict(content_box_xywh=(1, 0, 4, 4))])
def test_invalid_selection_or_geometry_metadata_is_refused(changes):
    with pytest.raises(ValueError):
        _select(_split(), **changes)


def test_zero_count_returns_no_proposals_without_preanswer_or_fake_uniform_selection():
    assert _select(_split(), count=0) == []
    assert _select(np.zeros((4, 4), np.uint8)) == []
