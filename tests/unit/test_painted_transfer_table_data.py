"""Parent/session table diagnostics from genuine tiny original replay fixtures."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import painted_adoption_table as table
from darsia.fluidflower import painted_analysis_exclusions as gaps
from darsia.fluidflower.painted_transfer_table_data import transfer_table_data

from .test_geometry_registered_export_apply import registered_setup  # noqa: F401
from .test_painted_adoption_table import imported, prepared_painting, synthetic_payload_admission  # noqa: F401
from .test_painted_analysis_exclusions import gap_case, _approve, _fresh  # noqa: F401


def _gap_inputs(case):
    """Load only actual source and native products retained by the tiny fixture."""
    review = case.review
    with np.load(review["source_map"]["path"], allow_pickle=False) as archive:
        source = np.array(archive["array"], copy=True)
    labels = {row["run_id"]: gaps._arrays(row["product"])["array"] for row in review["runs"]}
    current = geometry.GeometryRevisionStore(case.data.project).current()
    return current, dict(painting_binding=review["source"]["painting_facies"],
                         source_analysis=source, effective_labels=labels)


def test_transfer_table_data_keeps_actual_parent_and_unmapped_zero_without_authority(gap_case):
    """Retain parent IDs/properties and report residual zero without a frame row."""
    current, inputs = _gap_inputs(gap_case)
    before = deepcopy(current), deepcopy(inputs)

    result = transfer_table_data(current, **inputs)

    assert [row["label_id"] for row in result["regions"]] == [5, 6]
    assert [row["facies_id"] for row in result["regions"]] == [1, 2]
    assert [row["porosity"] for row in result["regions"]] == [.44, .22]
    assert result["unmapped_label_ids"] == [0]
    assert "0" not in result["spec"]["layer_to_facies"]
    assert 0 not in result["table"].layer_to_facies
    assert result["permeability_unit"] == "mD"
    assert result["parent_source"]["revision"] == inputs["painting_binding"]["frame_basis"]["workspace_revision"]
    assert result["parent_source"]["csv"] in result["guarded_sources"]
    assert result["spec"]["sha256"] == result["table"].sha256
    assert not {"preparation", "configuration", "registration", "painting_geometry"} & set(result)
    assert current == before[0] and inputs["painting_binding"] == before[1]["painting_binding"]
    np.testing.assert_array_equal(inputs["source_analysis"], before[1]["source_analysis"])
    for run, labels in inputs["effective_labels"].items():
        np.testing.assert_array_equal(labels, before[1]["effective_labels"][run])
    assert not (Path(gap_case.review["review_binding"]["path"]).parent / "index.json").exists()


@pytest.mark.parametrize("change", ["current-revision", "parent-csv", "unknown-native-layer"])
def test_transfer_table_data_retains_original_source_and_layer_refusals(gap_case, change):
    """Diagnostic assembly cannot accept another revision, changed CSV or new IDs."""
    current, inputs = _gap_inputs(gap_case)
    if change == "current-revision":
        current = deepcopy(current)
        current["revision_id"] = "changed-immutable-revision"
    elif change == "parent-csv":
        csv_path = Path(current["artifacts"]["facies_properties_path"])
        csv_path.write_bytes(csv_path.read_bytes() + b"\nchanged source\n")
    else:
        run = next(iter(inputs["effective_labels"]))
        inputs["effective_labels"][run] = inputs["effective_labels"][run].copy()
        inputs["effective_labels"][run].flat[0] = 254

    with pytest.raises(ValueError, match="changed|differ|absent"):
        transfer_table_data(current, **inputs)


def test_transfer_table_data_keeps_original_source_and_native_fixture_drawn_ids(
        registered_setup, tmp_path):
    """Use fixture IDs returned by the existing underlay readers and projection."""
    prepared, _, _, _, _ = prepared_painting(registered_setup, tmp_path, fixture=True)
    folder = Path(prepared["folder"])
    from darsia.fluidflower.layer_painting_fixtures import read_underlay
    source_fixture = read_underlay(table._local(folder, prepared["source_fixture_underlay"]))
    source_ids = [int(value) for value in np.unique(source_fixture.drawn_layers) if value]
    labels, drawn_ids = {}, set()
    for row in prepared["runs"]:
        native_labels, _filled, _support, _bound = table._native_flags(folder, row)
        labels[row["run_id"]] = native_labels
        drawn, _ = table._checked_native_underlay(folder, row, source_fixture, native_labels)
        drawn_ids.update(drawn)
    current = geometry.GeometryRevisionStore(registered_setup.project).current()

    result = transfer_table_data(current, painting_binding=prepared["painting_facies"],
        source_analysis=source_fixture.analysis_layers, effective_labels=labels,
        source_fixture_drawn_ids=source_ids, native_fixture_drawn_ids=drawn_ids)

    expected = (set().union(*(set(map(int, np.unique(value))) for value in labels.values())) | drawn_ids) - {0}
    assert {row["label_id"] for row in result["regions"]} == expected
    assert set(source_ids) <= set(result["table"].layer_to_facies)
    assert any(row["analysis_role"] == "colorchecker" for row in result["regions"])
    assert all(row["label_id"] != 0 for row in result["regions"])


def _prepared_inputs(data, prepared):
    """Supply actual original preparation DATA through the materializer interface."""
    folder = Path(prepared["folder"])
    with np.load(table._local(folder, prepared["source_map"])["path"], allow_pickle=False) as archive:
        source = np.array(archive["array"], copy=True)
    from darsia.fluidflower.painted_target_integration import effective_product
    labels = {row["run_id"]: effective_product(prepared, folder, row)[0]["array"]
              for row in prepared["runs"] if row.get("status") == "prepared"}
    return geometry.GeometryRevisionStore(data.project).current(), dict(
        painting_binding=prepared["painting_facies"], source_analysis=source,
        effective_labels=labels, outside_context=dict(preparation=prepared, folder=folder))


def test_materialized_table_data_uses_original_outside_proof_before_frame_row(registered_setup, tmp_path):
    """Only a genuine original outside proof changes zero from unmapped to frame."""
    prepared, _, _, _, _ = prepared_painting(registered_setup, tmp_path, outside=True, continuation=True)
    current, inputs = _prepared_inputs(registered_setup, prepared)
    without_proof = transfer_table_data(current, **dict(inputs, outside_context=None))

    result = transfer_table_data(current, **inputs)
    expected_contract, expected_regions, expected_unit, expected_table = table.candidate_table(
        current, prepared, Path(prepared["folder"]))

    assert without_proof["unmapped_label_ids"] == [0]
    assert 0 not in without_proof["table"].layer_to_facies
    frame = next(row for row in result["regions"] if row["label_id"] == 0)
    assert frame["analysis_role"] == "frame" and frame["enabled"] is False
    assert result["unmapped_label_ids"] == []
    assert result["table"].csv_bytes == expected_table.csv_bytes
    assert result["spec"] == expected_contract["table"]
    assert result["regions"] == expected_regions
    assert result["permeability_unit"] == expected_unit
    assert (0 in result["table"].layer_to_facies) == (0 in expected_table.layer_to_facies)


def test_materialized_table_data_filters_only_original_approved_gap_masks(gap_case):
    """Original gap masks and outside proof determine the same table as adoption."""
    prepared = _fresh(gap_case, _approve(gap_case.review))
    current, inputs = _prepared_inputs(gap_case.data, prepared)

    result = transfer_table_data(current, **inputs)
    expected_contract, expected_regions, expected_unit, expected_table = table.candidate_table(
        current, prepared, Path(prepared["folder"]))
    masks = [gaps.prepared_exclusion(prepared, Path(prepared["folder"]), row["run_id"])
             for row in prepared["runs"] if row.get("status") == "prepared"]

    assert any(np.any(labels == 0) for labels in inputs["effective_labels"].values())
    assert any(mask is not None and mask.any() for mask in masks)
    assert result["unmapped_label_ids"] == []
    assert result["table"].csv_bytes == expected_table.csv_bytes
    assert result["spec"] == expected_contract["table"]
    assert result["regions"] == expected_regions
    assert result["permeability_unit"] == expected_unit
    assert (0 in result["table"].layer_to_facies) == (0 in expected_table.layer_to_facies)


@pytest.mark.parametrize("change", ["effective-array", "outside-proof"])
def test_materialized_table_data_refuses_changed_arrays_or_original_proof(registered_setup, tmp_path, change):
    """Real replay inputs and original proof bytes remain mandatory for frame zero."""
    prepared, _, _, _, _ = prepared_painting(registered_setup, tmp_path, outside=True, continuation=True)
    current, inputs = _prepared_inputs(registered_setup, prepared)
    if change == "effective-array":
        run = next(iter(inputs["effective_labels"]))
        inputs["effective_labels"][run] = inputs["effective_labels"][run].copy()
        inputs["effective_labels"][run].flat[0] = 254
    else:
        bound = next(iter(prepared["outside_frame_proof"]["runs"].values()))
        path = Path(prepared["folder"]) / bound["path"]
        path.write_bytes(path.read_bytes() + b"changed outside proof")

    with pytest.raises(ValueError, match="changed|differ|proof"):
        transfer_table_data(current, **inputs)
