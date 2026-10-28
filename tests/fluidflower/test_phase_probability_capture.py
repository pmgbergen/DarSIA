"""Tiny decision-used score copies; no campaign inputs or qualification claims."""
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from darsia.fluidflower.phase import infer, probability_storage as storage, rules
from test_phase_rules import parameters as rule_parameters
from test_phase_domain import parameters as domain_parameters
from test_phase_infer import _full_inputs


def test_capture_is_exact_float32_copy_not_half_reencoding():
    pixel = np.array([[np.float32(.7), np.nextafter(np.float32(.7), np.float32(0)),
                       np.float32(np.float16(.7002))]], np.float32)
    regional = np.array([[.6, .9, .6001]], np.float32)
    support = np.ones(pixel.shape, bool)
    selected = (pixel >= .7) & (regional >= .6)
    capture = {}
    storage.pair(capture, pixel, regional, support, support, selected,
                 branch="centre", pixel_threshold=.7, regional_threshold=.6)
    assert capture["pixel"].tobytes() == pixel.tobytes()
    assert capture["regional"].tobytes() == regional.tobytes()
    assert capture["decision"]["selected_pixels"] == int(selected.sum())
    assert not np.shares_memory(capture["pixel"], pixel)
    expected = capture["pixel"].copy()
    pixel[:] = 0
    assert np.array_equal(capture["pixel"], expected)


def test_centre_on_off_return_exact_and_no_extra_resize(rule_parameters, monkeypatch):
    params = deepcopy(rule_parameters)
    settings = rules.GasDecisionSettings(
        **params["gas_configuration"]["selected_settings"], parameters=params)
    labels = np.ones((5, 7), np.uint8)
    labels[0, 0] = 0
    pixel = np.linspace(.5, .95, labels.size, dtype=np.float32).reshape(labels.shape)
    pixel[0, 1] = np.nan
    regional = np.array([[.59, .61], [.8, .9]], np.float32)
    before = [value.tobytes() for value in (labels, pixel, regional)]
    resize, calls = cv2.resize, []

    def observed(*args, **kwargs):
        calls.append(1)
        return resize(*args, **kwargs)

    monkeypatch.setattr(cv2, "resize", observed)
    old = rules.apply_gas_configuration(labels, pixel, regional, .5, settings, parameters=params)
    capture = {}
    new = rules.apply_gas_configuration(labels, pixel, regional, .5, settings,
                                        parameters=params, output_capture=capture)
    assert len(calls) == 2  # Exactly one original resize for each invocation.
    assert np.array_equal(old[0], new[0]) and old[1] == new[1]
    assert before == [value.tobytes() for value in (labels, pixel, regional)]
    assert capture["regional"].tobytes() == resize(regional, (7, 5), interpolation=cv2.INTER_LINEAR).tobytes()
    assert capture["transitions"][0]["operator"] == "centre-minimum"


def test_seam_captures_original_resize_without_repeating_it(rule_parameters, monkeypatch):
    pixel = np.full((3, 5), .8, np.float32)
    regional = np.array([[.5, .9]], np.float32)
    resize, calls = cv2.resize, []

    def observed(*args, **kwargs):
        calls.append(1)
        return resize(*args, **kwargs)

    monkeypatch.setattr(cv2, "resize", observed)
    capture = {}
    selected = rules.preminimum_gas_selection(pixel, regional, parameters=rule_parameters,
                                               output_capture=capture)
    assert len(calls) == 1
    assert capture["decision"]["selected_pixels"] == int(selected.sum())
    assert np.array_equal(selected, (capture["pixel"] >= .7) & (capture["regional"] >= .6))


def _pair(branch, pixel, regional):
    capture = {}
    support = np.ones(pixel.shape, bool)
    storage.pair(capture, pixel, regional, support, support,
                 (pixel >= .7) & (regional >= .6), branch=branch,
                 pixel_threshold=.7, regional_threshold=.6)
    return capture


def test_composition_preserves_centre_edge_and_seam_origins():
    edge = _pair("edge", np.full((2, 4), .75, np.float32), np.full((2, 4), .65, np.float32))
    centre = _pair("centre", np.full((2, 2), .85, np.float32), np.full((2, 2), .95, np.float32))
    seam = _pair("centre", np.full((2, 2), .9, np.float32), np.full((2, 2), .8, np.float32))
    seam["actual_returned_selection"] = np.ones((2, 2), bool)
    centre["seam_pair"] = seam
    release = np.zeros((2, 4), bool)
    release[:, 2] = True
    commit = np.zeros((2, 4), bool)
    commit[0, 1] = True
    inherited = np.zeros((2, 4), bool)
    inherited[:, 1] = True
    capture = dict(edge=edge, centre=centre, composition=dict(
        central_support=np.ones((2, 2), bool), release=release,
        seam_commit=commit, seam_inherited=inherited), placement={}, postprocessing={})
    classes = dict(modelled=np.ones((2, 4), bool), filled_observed=np.zeros((2, 4), bool),
                   filled_unobserved=np.zeros((2, 4), bool))
    labels = np.full((2, 4), 2, np.uint8)
    storage.finish(capture, labels, classes, (1, 0, 2, 2))
    arrays = capture["product"]["arrays"]
    assert np.array_equal(arrays["branch_support"], np.array([[2, 1, 2, 2]] * 2, np.uint8))
    assert arrays["pixel_probability"][0, 1].tobytes() == seam["pixel"][0, 0].tobytes()
    assert arrays["pixel_probability"][1, 1].tobytes() == centre["pixel"][1, 0].tobytes()
    assert np.array_equal(arrays["pixel_probability"][:, 2], edge["pixel"][:, 2])


def test_placement_exact_pair_overrides_base_but_unscored_nan_wins_outside_zero():
    placement = _pair("placement", np.full((2, 3), .70000005, np.float32),
                      np.full((2, 3), .6000001, np.float32))
    capture = dict(edge={}, composition={}, placement=placement, postprocessing={})
    labels = np.array([[0, 2, 2], [0, 0, 2]], np.uint8)
    classes = dict(modelled=np.ones(labels.shape, bool), filled_observed=np.zeros(labels.shape, bool),
                   filled_unobserved=np.zeros(labels.shape, bool))
    classes["modelled"][1, 0] = False
    classes["filled_observed"][1, 0] = True
    storage.finish(capture, labels, classes, (0, 0, 3, 2))
    arrays = capture["product"]["arrays"]
    for name in ("pixel_probability", "regional_probability"):
        assert arrays[name][0, 0] == arrays[name][1, 1] == 0
        assert np.isnan(arrays[name][1, 0])
        assert arrays[name][0, 1].tobytes() == placement["pixel" if name.startswith("pixel") else "regional"][0, 1].tobytes()
    assert arrays["branch_support"][1, 0] == storage.CODES["unscored"]


def test_ordered_transition_ledger_counts_actual_overlap_not_total_subtraction():
    capture = {}
    before = np.array([[1, 2, 2]], np.uint8)
    middle = np.array([[2, 1, 2]], np.uint8)  # Net gas count is unchanged.
    final = np.array([[1, 1, 2]], np.uint8)
    storage.state(capture, "initial", before)
    storage.state(capture, "Rule-B", middle)
    storage.state(capture, "top", final)
    assert capture["transitions"][0]["transitions"]["1->2"] == 1
    assert capture["transitions"][0]["transitions"]["2->1"] == 1
    assert capture["transitions"][0]["changed_pixels"] == 2
    assert capture["transitions"][1]["changed_pixels"] == 1


def test_full_original_infer_returns_and_scientific_arrays_unchanged(rule_parameters, domain_parameters):
    frame, bundle, params = _full_inputs(rule_parameters, domain_parameters)
    normal = infer.infer_labels(frame, None, bundle, params)
    capture = dict(gas_probability_storage={})
    enabled = infer.infer_labels(frame, None, bundle, params, output_capture=capture)
    assert enabled.keys() == normal.keys()
    for name in ("labels", "pixel_area_mm2"):
        assert enabled[name].dtype == normal[name].dtype
        assert enabled[name].tobytes() == normal[name].tobytes()
    for group in ("classes", "seam_masks", "post_masks"):
        assert enabled[group].keys() == normal[group].keys()
        for name in normal[group]:
            assert enabled[group][name].tobytes() == normal[group][name].tobytes()
    assert set(capture) == {"gas_probability_storage", "g3_pixel", "flatfield_valid",
                            "source_box_xywh", "probability_encoding"}
    assert capture["gas_probability_storage"]["product"]["arrays"]["pixel_probability"].dtype == np.float32


def test_q28_capture_alone_does_not_enable_probability_storage(rule_parameters, domain_parameters, monkeypatch):
    frame, bundle, params = _full_inputs(rule_parameters, domain_parameters)
    capture = {}
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled storage attempted new capture allocation or writing")
    for name in ("pair", "capture_centre", "capture_centre_selection", "capture_placement", "capture_seam_population", "changed_rule", "state", "finish", "write_product"):
        monkeypatch.setattr(storage, name, forbidden)
    infer.infer_labels(frame, None, bundle, params, output_capture=capture)
    assert "gas_probability_storage" not in capture
    assert set(capture) == {"g3_pixel", "flatfield_valid", "source_box_xywh", "probability_encoding"}


def test_unsupported_p2_is_explicit_without_changing_default_mode(rule_parameters):
    mode = rule_parameters["placement"]["placement_modes"]["constrained_neighbourhood"]
    with pytest.raises(ValueError, match="no decision-used regional"):
        storage.require_pair_mode(mode, rule_parameters)


@pytest.mark.parametrize("bad", [None, {}, {"format": storage.REQUEST_FORMAT},
                                 {"format": storage.REQUEST_FORMAT, "encoding": "float16"}])
def test_only_explicit_exact_float32_request_is_admitted(bad):
    if bad is None:
        assert storage.parse_request({}) is None
    else:
        with pytest.raises(ValueError, match="exact-float32"):
            storage.parse_request(dict(gas_probability_storage=bad))


def test_explicit_null_request_is_not_an_enabled_or_disabled_fallback():
    with pytest.raises(ValueError, match="exact-float32"):
        storage.parse_request(dict(gas_probability_storage=None))


def test_capture_refuses_a_selection_not_made_by_the_actual_pair():
    score = np.full((2, 2), .2, np.float32)
    support = np.ones(score.shape, bool)
    with pytest.raises(ValueError, match="does not reproduce"):
        storage.pair({}, score, score, support, support, support,
                     branch="edge", pixel_threshold=.7, regional_threshold=.6)


def _classes(shape):
    return dict(modelled=np.ones(shape, bool), filled_observed=np.zeros(shape, bool),
                filled_unobserved=np.zeros(shape, bool))


def _placement_capture():
    return dict(edge={}, centre={}, composition={}, postprocessing={}, placement=_pair(
        "placement", np.full((1, 2), .8, np.float32), np.full((1, 2), .8, np.float32)))


@pytest.mark.parametrize("wrong_position", [False, True])
def test_same_aggregate_changed_count_requires_actual_spatial_membership(wrong_position):
    capture = _placement_capture()
    before = np.ones((1, 2), bool)
    after = np.array([[wrong_position, not wrong_position]], bool)
    storage.changed_rule(capture["placement"], "placement_minimum", before, after)
    assert capture["placement"]["rules"]["placement_minimum"]["raw_changed_pixels"] == 1
    labels = np.array([[1, 2]], np.uint8)
    if wrong_position:
        with pytest.raises(ValueError, match="unexplained spatial disagreement"):
            storage.finish(capture, labels, _classes(labels.shape), (0, 0, 2, 1))
    else:
        storage.finish(capture, labels, _classes(labels.shape), (0, 0, 2, 1))
        proof = capture["product"]["evidence"]["spatial_proof"]
        assert proof["rules"]["placement_minimum"]["applicable_pixels"] == 1
        assert proof["rules"]["placement_minimum"]["disagreement_pixels"] == 1
        assert proof["rules"]["placement_minimum"]["applicable_mask"] == storage.identity(~after)
        assert proof["unexplained_pixels"] == 0


def test_wrong_centre_branch_cannot_explain_actual_edge_pixel():
    shape = (1, 2)
    edge = _pair("edge", np.full(shape, .8, np.float32), np.full(shape, .8, np.float32))
    centre = _pair("centre", np.full(shape, .8, np.float32), np.full(shape, .8, np.float32))
    storage.changed_rule(centre, "centre_minimum", np.ones(shape, bool), np.array([[False, True]], bool))
    zero = np.zeros(shape, bool)
    capture = dict(edge=edge, centre=centre, composition=dict(central_support=zero.copy(),
        release=zero.copy(), seam_commit=zero.copy(), seam_inherited=np.ones(shape, bool)))
    labels = np.array([[1, 2]], np.uint8)
    with pytest.raises(ValueError, match="unexplained spatial disagreement"):
        storage.finish(capture, labels, _classes(shape), (0, 0, 2, 1))


@pytest.mark.parametrize("base_rule", ["centre_minimum", "edge_B", "seam_B"])
def test_active_placement_cannot_borrow_a_base_rule_mask(base_rule):
    capture = _placement_capture()
    storage.changed_rule(capture["composition"], base_rule,
                         np.array([[2, 2]], np.uint8), np.array([[1, 2]], np.uint8))
    labels = np.array([[1, 2]], np.uint8)
    with pytest.raises(ValueError, match="unexplained spatial disagreement"):
        storage.finish(capture, labels, _classes(labels.shape), (0, 0, 2, 1))


@pytest.mark.parametrize("name", ["base-to-placement-threshold", "centre-edge-composition"])
def test_broad_transition_is_diagnostic_not_a_later_rule_exception(name):
    capture = _placement_capture()
    storage.state(capture["placement"], "initial", np.array([[2, 2]], np.uint8))
    storage.state(capture["placement"], name, np.array([[1, 2]], np.uint8))
    labels = np.array([[1, 2]], np.uint8)
    with pytest.raises(ValueError, match="unexplained spatial disagreement"):
        storage.finish(capture, labels, _classes(labels.shape), (0, 0, 2, 1))


def test_actual_overlap_counts_memberships_and_union_without_double_counting():
    capture = _placement_capture()
    before, after = np.array([[2, 2]], np.uint8), np.array([[1, 2]], np.uint8)
    storage.changed_rule(capture["placement"], "placement_B", before, after)
    storage.changed_rule(capture["postprocessing"], "post_floor", after, before)
    storage.changed_rule(capture["postprocessing"], "post_top", before, after)
    labels = after.copy()
    storage.finish(capture, labels, _classes(labels.shape), (0, 0, 2, 1))
    proof = capture["product"]["evidence"]["spatial_proof"]
    assert proof["applicable_union_pixels"] == proof["overlapping_pixels"] == 1
    assert proof["total_memberships"] == 3 and proof["membership_histogram"]["3"] == 1
    assert proof["rules"]["post_floor"]["disagreement_pixels"] == 1
    assert capture["product"]["evidence"]["raw_later_rules"]["post_floor"]["raw_changed_pixels"] == 1


def test_rethresholded_seam_does_not_borrow_prior_edge_minimum():
    shape = (1, 2)
    edge = _pair("edge", np.full(shape, .8, np.float32), np.full(shape, .8, np.float32))
    storage.changed_rule(edge, "edge_minimum", np.ones(shape, bool), np.array([[False, True]], bool))
    centre = _pair("centre", np.full(shape, .8, np.float32), np.full(shape, .8, np.float32))
    zero = np.zeros(shape, bool)
    capture = dict(edge=edge, centre=centre, composition=dict(central_support=zero.copy(),
        release=zero.copy(), seam_commit=np.array([[True, False]], bool), seam_inherited=zero.copy()))
    labels = np.array([[1, 2]], np.uint8)
    with pytest.raises(ValueError, match="unexplained spatial disagreement"):
        storage.finish(capture, labels, _classes(shape), (0, 0, 2, 1))


def test_released_ring_does_not_borrow_edge_B_before_its_fresh_replay():
    shape = (1, 2)
    edge = _pair("edge", np.full(shape, .8, np.float32), np.full(shape, .8, np.float32))
    centre = _pair("centre", np.full(shape, .8, np.float32), np.full(shape, .8, np.float32))
    zero = np.zeros(shape, bool)
    composition = dict(central_support=zero.copy(), release=np.array([[True, False]], bool),
                       seam_commit=zero.copy(), seam_inherited=zero.copy())
    storage.changed_rule(composition, "edge_B", np.array([[2, 2]], np.uint8), np.array([[1, 2]], np.uint8))
    labels = np.array([[1, 2]], np.uint8)
    with pytest.raises(ValueError, match="unexplained spatial disagreement"):
        storage.finish(dict(edge=edge, centre=centre, composition=composition), labels, _classes(shape), (0, 0, 2, 1))


def test_original_seam_hook_records_exact_minimum_B_and_commit(rule_parameters, domain_parameters):
    from darsia.fluidflower.phase import domain
    shape = (3, 5)
    labels = np.ones(shape, np.uint8)
    labels[1, 1:4] = 2
    ring = np.zeros(shape, bool)
    ring[1, 2] = True
    supported = np.ones(shape, bool)
    parameters = deepcopy(domain_parameters)
    args = (labels, labels.copy(), ring, supported, labels == 2, labels == 2,
            supported, np.ones(shape, np.float64))
    kwargs = dict(open_components=np.ones(shape, np.uint8), source_component_ids=[1],
                  barrier_mask=np.zeros(shape, bool), parameters=parameters, rule_parameters=rule_parameters)
    normal = domain.revise_seam_labels(*args, **kwargs)
    capture = {}
    actual = domain.revise_seam_labels(*args, **kwargs, output_capture=capture)
    assert actual[0].tobytes() == normal[0].tobytes() and actual[1] == normal[1]
    assert set(capture["rules"]) == {"seam_minimum", "seam_B"}
    assert np.array_equal(capture["seam_commit"], supported & np.isin(labels, (1, 2))
                          & (actual[2]["affected_gas_graph"] | ring))
    assert np.array_equal(capture["seam_inherited"], supported & ~ring)


def test_later_removal_mask_cannot_explain_below_threshold_final_gas():
    capture = dict(placement=_pair("placement", np.full((1, 2), .2, np.float32),
                                  np.full((1, 2), .2, np.float32)), postprocessing={})
    storage.changed_rule(capture["postprocessing"], "post_floor",
                         np.array([[2, 2]], np.uint8), np.array([[1, 2]], np.uint8))
    labels = np.array([[2, 1]], np.uint8)
    with pytest.raises(ValueError, match="cannot explain below-threshold final gas"):
        storage.finish(capture, labels, _classes(labels.shape), (0, 0, 2, 1))
