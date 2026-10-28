"""Original categorical support and replay-bound descriptive grouping DATA."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import painted_analysis_exclusions as gaps
from darsia.fluidflower import painted_gap_review_groups as groups
from darsia.fluidflower.painted_gap_regions import REGION_CLASSIFICATIONS
from darsia.fluidflower.painted_gap_region_transfer import (
    component_transfer_coverage, transfer_region_masks,
)

from .test_painted_transfer_findings import (  # noqa: F401
    _capture_original_review, _collect, _files,
    imported, registered_setup, synthetic_payload_admission,
)


def _replay():
    """Finite mask/placement DATA; no claim of campaign source authentication."""
    runs, native = [], []
    for run, extent, placement in (
            ("ac18", (1, 4, 1, 5), np.eye(3)),
            ("ac19", (2, 4, 2, 5), np.eye(3)),
            ("ac20", (1, 3, 1, 3), np.array([[1., 0., 20.], [0., 1., 0.], [0., 0., 1.]]))):
        mask = np.zeros((8, 9), bool)
        top, bottom, left, right = extent
        mask[top:bottom, left:right] = True
        components = list(gaps._components(mask))
        metadata = [dict(component, review_row_sha256=str(index+1)*64)
                    for index, (component, _) in enumerate(components)]
        runs.append(dict(run_id=run, gap=dict(shape_hw=list(mask.shape)),
            native_to_painting_h=placement.tolist(), components=metadata))
        native.append(dict(run_id=run, components=components))
    return dict(review=dict(review_binding=dict(path="C:/synthetic/review.json",
        sha256="f"*64, size_bytes=123), runs=runs), runs=native)


def test_support_only_probe_matches_original_class_transfer_and_never_replicates_edge_support():
    source = np.ones((2, 3), bool)
    target = np.ones((5, 8), bool)
    palette = {name: np.zeros_like(source) for name in REGION_CLASSIFICATIONS}
    palette["air"][:] = True
    source_h = [[2., 0., 1.], [0., 2., 1.], [0., 0., 1.]]
    target_h = [[1., 0., -1.], [0., 1., 0.], [0., 0., 1.]]
    actual = transfer_region_masks(palette, source, source_h, target, target_h)
    coverage = component_transfer_coverage(source, source_h, target, target_h)
    assert coverage == {key: actual[key] for key in ("matched_pixels", "remaining_pixels")}
    assert coverage["remaining_pixels"] > 0
    assert not {"labels", "class_masks", "approval", "outside_proof"} & set(coverage)


def test_groups_are_deterministic_full_supported_components_and_inputs_stay_unchanged():
    replay = _replay()
    before = deepcopy(replay["review"])
    masks = [mask.copy() for row in replay["runs"] for _, mask in row["components"]]
    value = groups.deviation_groups_from_replay(replay)
    assert [[member["run_id"] for member in group["members"]] for group in value["groups"]] == [
        ["ac18", "ac19"], ["ac20"]]
    assert value["summary"] == dict(deviation_count=3, classification_groups=2,
        transferred_confirmations=1, failed_pairs=0)
    assert value["review"] == replay["review"]["review_binding"]
    assert groups.deviation_groups_from_replay(replay) == value
    assert replay["review"] == before
    for original, mask in zip(masks, [mask for row in replay["runs"] for _, mask in row["components"]]):
        np.testing.assert_array_equal(original, mask)
    assert not {"actions", "approval", "candidate", "resolved", "index"} & set(value)


def test_partial_supported_destination_remains_a_separate_original_component():
    replay = _replay()
    # Keep the original connected-component writer and binding. The destination
    # crosses one source edge, so ordinary nearest support must not group it.
    mask = np.zeros((8, 9), bool)
    mask[2:4, 4:7] = True
    component, actual = next(gaps._components(mask))
    replay["runs"][1]["components"] = [(component, actual)]
    replay["review"]["runs"][1]["components"] = [dict(component, review_row_sha256="2"*64)]
    value = groups.deviation_groups_from_replay(replay)
    assert value["summary"]["classification_groups"] == 3
    assert value["summary"]["transferred_confirmations"] == 0


@pytest.mark.parametrize("fault", ["review", "row", "duplicate", "missing", "counts"])
def test_group_data_cannot_omit_repeat_rebind_or_recount_an_original_row(fault):
    replay = _replay()
    value = groups.deviation_groups_from_replay(replay)
    if fault == "review": value["review"]["sha256"] = "a"*64
    elif fault == "row": value["groups"][0]["members"][1]["review_row_sha256"] = "a"*64
    elif fault == "duplicate": value["groups"][0]["members"].append(deepcopy(value["groups"][0]["members"][1]))
    elif fault == "missing": value["groups"].pop()
    else: value["summary"]["classification_groups"] = 1
    with pytest.raises(ValueError):
        groups.validate_deviation_groups(value, replay["review"])


def test_changed_replay_component_pixels_are_refused_before_descriptive_publication():
    replay = _replay()
    replay["runs"][0]["components"][0][1][1, 1] = False
    with pytest.raises(ValueError, match="original component binding"):
        groups.deviation_groups_from_replay(replay)


def test_actual_original_collector_adds_only_bound_group_data_without_changing_scientific_findings(
        registered_setup, tmp_path, monkeypatch):
    from darsia.fluidflower import painted_transfer_findings as findings

    case = _capture_original_review(registered_setup, tmp_path, monkeypatch, gaps=True)
    original = findings._assemble_findings
    observed = []
    def observe(*args, **options):
        value = original(*args, **options)
        observed.append(deepcopy(value))
        return value
    monkeypatch.setattr(findings, "_assemble_findings", observe)
    before = _files(tmp_path)
    result = _collect(registered_setup, case)
    value = groups.validate_deviation_groups(result["report"]["deviation_groups"], result["review"])
    actual = sum(len(row["components"]) for row in result["review"]["runs"])
    assert value["summary"]["deviation_count"] == actual > 0
    for key in ("issues", "summary", "unresolved_domains", "run_data", "table_data"):
        assert result["report"][key] == observed[0][key]
    assert not {"approval", "candidate", "actions", "resolved"} & set(value)
    assert _files(tmp_path) == before and not (Path(case["folder"]) / "index.json").exists()
