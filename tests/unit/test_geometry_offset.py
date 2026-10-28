"""Read-only geometry-offset contracts on synthetic arrays; no real series."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.geometry_offset import (
    OffsetParameters, SIGN, measure_arrays, propose_registration,
    write_readonly_outputs,
)


def _case(shift=5, spacing_mm=1.0):
    shape = (120, 80)
    labels = np.full(shape, 20, dtype=np.int32)
    labels[:60] = 10
    image = np.zeros((*shape, 3), dtype=np.float32)
    image[60+shift:] = 1
    metadata = dict(space_dim=2, indexing="ij", dimensions=[shape[0]*spacing_mm/1000, shape[1]*spacing_mm/1000])
    regions = [dict(label_id=n, analysis_role="porous_medium", enabled=True) for n in (10, 20)]
    return image, labels, metadata, regions


def _measured(shift=5, run="run1"):
    image, labels, metadata, regions = _case(shift)
    report, drawings = measure_arrays(image, labels, metadata, regions)
    report.update(run_id=run, revision_id="a"*32)
    return report, drawings, image


@pytest.mark.parametrize("shift", [5, -7])
def test_signed_native_normal_shift_and_original_arrays_unchanged(shift):
    image, labels, metadata, regions = _case(shift)
    old_image, old_labels = image.copy(), labels.copy()
    report, drawings = measure_arrays(image, labels, metadata, regions)
    boundary, = report["boundaries"]
    assert boundary["labels"] == [10, 20]
    assert boundary["median_mm"] == pytest.approx(shift, abs=.01)
    assert boundary["p10_mm"] == pytest.approx(shift, abs=.01)
    assert boundary["p90_mm"] == pytest.approx(shift, abs=.01)
    assert report["sign_convention"] == SIGN
    assert report["correction_applied"] is report["registered_revision_written"] is False
    assert np.array_equal(image, old_image) and np.array_equal(labels, old_labels)


def test_metric_scale_comes_from_saved_dimensions_not_ffac_constant():
    image, labels, metadata, regions = _case(5, spacing_mm=2)
    report, _ = measure_arrays(image, labels, metadata, regions)
    assert report["spacing_mm"] == [2, 2]
    assert report["boundaries"][0]["median_mm"] == pytest.approx(10, abs=2)


def test_spatial_percentiles_are_not_replaced_by_the_median():
    image, labels, metadata, regions = _case(-5)
    image[:, 40:] = 0
    image[67:, 40:] = 1
    report, _ = measure_arrays(image, labels, metadata, regions)
    edge, = report["boundaries"]
    assert edge["p10_mm"] < 0 and edge["p90_mm"] > 0
    assert edge["p10_mm"] < edge["median_mm"] < edge["p90_mm"]


def test_existing_chart_or_fixture_role_is_masked_without_new_classification():
    image, labels, metadata, regions = _case(5)
    labels[40:80, 30:50] = 30
    image[40:80, 30:50] = 100
    regions.append(dict(label_id=30, analysis_role="colorchecker", enabled=False))
    report, _ = measure_arrays(image, labels, metadata, regions)
    assert report["declared_excluded_labels"] == [30]
    measured = [b for b in report["boundaries"] if b["measurement_eligible"]]
    assert len(measured) == 1
    assert report["boundary_pair_count"] == 3 and report["not_measurable_boundary_pair_count"] == 2
    assert measured[0]["median_mm"] == pytest.approx(5, abs=.01)
    assert all(b["median_mm"] is None for b in report["boundaries"] if not b["measurement_eligible"])
    assert "no new" in report["exclusions"]


@pytest.mark.parametrize("edge_outside", [False, True])
def test_flat_or_outside_band_never_fabricates_zero_offset(edge_outside):
    image, labels, metadata, regions = _case()
    image[:] = 0
    if edge_outside:
        image[95:] = 1
    report, _ = measure_arrays(image, labels, metadata, regions)
    edge, = report["boundaries"]
    assert edge["median_mm"] is edge["p10_mm"] is edge["p90_mm"] is None
    assert edge["status"] == "uncertain_or_missing_edge"


def test_disconnected_equal_edges_remain_ambiguous():
    image, labels, metadata, regions = _case()
    image[:] = 0
    image[64:70] = 1
    report, _ = measure_arrays(image, labels, metadata, regions)
    edge, = report["boundaries"]
    assert edge["median_mm"] is None
    assert edge["sample_outcomes"].get("ambiguous_equal_edges", 0) > 0


def test_unrecorded_region_role_is_not_assumed_sand():
    image, labels, metadata, regions = _case()
    regions.pop()
    report, _ = measure_arrays(image, labels, metadata, regions)
    assert len(report["boundaries"]) == 1
    assert report["boundaries"][0]["measurement_eligible"] is False
    assert report["boundaries"][0]["roles"]["20"] == "unrecorded"
    assert report["boundaries"][0]["median_mm"] is None
    assert report["labels_without_eligible_role"] == [20]


@pytest.mark.parametrize("change", ["shape", "metric", "role", "band"])
def test_invalid_input_refuses_without_hidden_resize_or_default(change):
    image, labels, metadata, regions = _case()
    parameters = OffsetParameters()
    if change == "shape":
        image = image[:-1]
    elif change == "metric":
        metadata.pop("dimensions")
    elif change == "role":
        regions.append(deepcopy(regions[0]))
    else:
        parameters = OffsetParameters(band_mm=26)
    with pytest.raises(ValueError):
        measure_arrays(image, labels, metadata, regions, parameters=parameters)


def _reports(medians):
    rows = []
    for index, median in enumerate(medians):
        report, _, _ = _measured(run=f"run{index}")
        report["boundaries"][0]["median_mm"] = median
        rows.append(report)
    return rows


def test_original_two_mm_across_three_runs_rule_is_inclusive_and_proposal_only():
    report = propose_registration(_reports([3.0, 4.0, 5.0]), ["run0", "run1", "run2"])
    assert report["suggestion"] == "common_shift_per_layer_candidate"
    assert report["boundaries"][0]["common_candidate_mm"] == 4
    assert report["decision"] == "needs_series_decision_after_overlays_and_table"
    assert report["correction_applied"] is report["registered_revision_written"] is False
    report = propose_registration(_reports([3.0, 4.0, 5.01]), ["run0", "run1", "run2"])
    assert report["suggestion"] == "per_run_registration_candidate"


def test_missing_or_nonfinite_edge_cannot_become_a_shift_proposal():
    report = propose_registration(_reports([3.0, None, 3.0]), ["run0", "run1", "run2"])
    assert report["suggestion"] == "insufficient_edge_evidence"
    report = propose_registration(_reports([3.0, float("nan"), 3.0]), ["run0", "run1", "run2"])
    assert report["suggestion"] == "insufficient_edge_evidence"


@pytest.mark.parametrize("field", ["revision_id", "label_grid", "spacing_mm", "parameters"])
def test_cross_revision_grid_or_estimator_results_are_not_silently_mixed(field):
    reports = _reports([3, 3, 3])
    reports[1][field] = "foreign"
    with pytest.raises(ValueError, match="same saved revision"):
        propose_registration(reports, ["run0", "run1", "run2"])


def test_fresh_readonly_reports_include_english_overlay_and_preserve_old_target(tmp_path):
    measured = [_measured(run=f"run{index}") for index in range(3)]
    proposal = propose_registration([r[0] for r in measured], ["run0", "run1", "run2"])
    target = write_readonly_outputs(tmp_path / "new", measured, proposal)
    assert (target / "overlay_run0.png").is_file()
    assert (target / "offsets.csv").is_file()
    assert json.loads((target / "registration_run0.json").read_bytes())["run_id"] == "run0"
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    with pytest.raises(FileExistsError):
        write_readonly_outputs(target, measured, proposal)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before


def test_top_sand_free_water_boundary_is_measured_with_explicit_sign():
    image, labels, metadata, regions = _case(5)
    regions[0]["analysis_role"] = "free_water"
    report, _ = measure_arrays(image, labels, metadata, regions)
    edge, = report["boundaries"]
    assert edge["measurement_eligible"] is True
    assert edge["roles"] == {"10": "free_water", "20": "porous_medium"}
    assert edge["median_mm"] == pytest.approx(5, abs=.01)
    assert 10 in report["permitted_photographed_labels"] and 10 not in report["declared_excluded_labels"]


def test_frame_water_pair_stays_visible_but_cannot_become_a_photographic_edge():
    image, labels, metadata, regions = _case(5)
    labels[:20] = 0
    image[:20] = 100
    regions[0]["analysis_role"] = "free_water"
    regions.append(dict(label_id=0, analysis_role="frame", enabled=False))
    report, _ = measure_arrays(image, labels, metadata, regions)
    pairs = {tuple(row["labels"]): row for row in report["boundaries"]}
    assert set(pairs) == {(0, 10), (10, 20)}
    assert pairs[(0, 10)]["measurement_eligible"] is False
    assert pairs[(0, 10)]["median_mm"] is None
    assert pairs[(10, 20)]["measurement_eligible"] is True
    assert pairs[(10, 20)]["median_mm"] == pytest.approx(5, abs=.01)
    assert report["measurable_boundary_pair_count"] == report["not_measurable_boundary_pair_count"] == 1


def test_unmeasurable_pairs_remain_in_proposal_coverage_without_fabricating_offset():
    reports = _reports([3, 4, 5])
    for report in reports:
        report["boundaries"].append(dict(boundary_id="labels_0_10", labels=[0, 10],
            measurement_eligible=False, status="not_measurable_from_saved_roles", median_mm=None))
    proposal = propose_registration(reports, ["run0", "run1", "run2"])
    assert proposal["suggestion"] == "common_shift_per_layer_candidate"
    assert proposal["measurable_boundary_pair_count"] == 1
    assert proposal["explicitly_unmeasurable_boundary_pair_count"] == 1
    assert len(proposal["boundaries"]) == 2
    assert next(b for b in proposal["boundaries"] if b["boundary_id"] == "labels_0_10")["suggestion"] == "not_measurable_from_saved_roles"
