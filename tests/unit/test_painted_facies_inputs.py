"""Absent-id refusal ordering on private metadata, with no scientific build.

Apply/Build cases use their real functions with minimal metadata context. They
qualify refusal before a write, not Store adoption, RAW pixels or native output.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import corrected_setup
from darsia.fluidflower import painted_facies_inputs as inputs


@pytest.fixture
def table():
    """Use a stable noncontiguous table, including water zero and new rows."""
    return {"provenance": [
        {"name": name, "facies_id": number}
        for name, number in (("Free water", 0), ("Fine", 1), ("Medium", 2),
                             ("Coarse", 5), ("Fixture", 9), ("Seal", 10))
    ]}


def _config(table):
    """Select the explicit new painted-table branch without modifying it."""
    return {inputs.KEY: {"table": deepcopy(table)}}


@pytest.mark.parametrize("document,field", [
    ({"facies": {"7": {"labels": [100]}}}, "facies.7"),
    ({"shared": {"porosity_by_facies": {"3": .44}}}, "shared.porosity_by_facies.3"),
    ({"shared": {"swi_by_facies": {"8": .12}}}, "shared.swi_by_facies.8"),
    ({"shared": {"unquantified_facies": [6]}}, "shared.unquantified_facies[0]"),
    ({"calibration": {"color_path": {"ignore_labels": [4]}}}, "calibration.color_path.ignore_labels[0]"),
    ({"profile": {"ignored_label_ids": [7]}}, "profile.ignored_label_ids[0]"),
])
def test_absent_ids_list_exact_source_and_field_without_changing_values(table, document, field):
    """Check category references rather than scalar values or raster labels."""
    before = deepcopy(document)
    with pytest.raises(ValueError) as error:
        inputs.require_current_facies_inputs(_config(table), {"run input": document})
    assert field in str(error.value) and "run input" in str(error.value)
    assert "absent from the adopted table" in str(error.value)
    assert document == before


def test_all_missing_ids_are_listed_instead_of_silently_renumbered(table):
    """All missing ids are listed instead of silently renumbered."""
    document = {"facies": {"7": {"labels": [1]}, "8": {"labels": [2]}}}
    result = inputs.id_keyed_input_problems(table, document, source="run.toml")
    assert len(result) == 2
    assert "facies.7" in result[0] and "id 7" in result[0]
    assert "facies.8" in result[1] and "id 8" in result[1]


def test_current_ids_water_zero_and_native_segment_values_keep_their_meaning(table):
    """Current ids water zero and native segment values keep their meaning."""
    document = {"facies": {"path": "native.npz", "props": "table.csv",
                           "5": {"labels": [3, 7, 8]}},
                "shared": {"porosity_by_facies": {"1": .44, "2": .43, "5": .42},
                           "unquantified_facies": [0, 9, 10]}, "ignore_labels": [0, 9, 10]}
    before = deepcopy(document)
    inputs.require_current_facies_inputs(_config(table), {"current run": document})
    assert document == before


def test_current_id_is_not_refused_by_descriptive_retirement_metadata(table):
    """Current id is not refused by descriptive retirement metadata."""
    table.update(retired_source_ids=[5], changed_source_ids=[5])
    inputs.require_current_facies_inputs(_config(table), {"run": {"ignore_labels": [5]}})


def test_old_default_contract_does_not_add_reads_or_validation():
    """Old default contract does not add reads or validation."""
    assert inputs.require_current_facies_inputs({}, None) is None
    corrected_setup._require_painted_facies_documents({}, [Path("missing-old-default.toml")])


def test_each_check_reads_current_run_bytes_and_never_rewrites_them(table, tmp_path):
    """Each check reads current run bytes and never rewrites them."""
    run = tmp_path / "run.toml"
    run.write_text('[facies.5]\nlabels = [7, 8]\n', encoding="utf-8")
    corrected_setup._require_painted_facies_documents(_config(table), [run])
    run.write_text('[facies.7]\nlabels = [7, 8]\n', encoding="utf-8")
    before = run.read_bytes()
    with pytest.raises(ValueError, match=r"facies\.7.*absent"):
        corrected_setup._require_painted_facies_documents(_config(table), [run])
    assert run.read_bytes() == before


def test_apply_refuses_absent_groups_before_any_apply_write(table, tmp_path, monkeypatch):
    """Stub only the selected series locator; exercise the real Apply entry."""
    series = tmp_path / "series"
    series.mkdir()
    (series / "series.toml").write_text('[runs]\nids = ["one"]\n', encoding="utf-8")
    run = series / "one.toml"
    run.write_text('[facies.7]\nlabels = [1]\n[facies.8]\nlabels = [2]\n', encoding="utf-8")
    current = tmp_path / "current.json"
    current.write_text(json.dumps({"configuration": _config(table)}), encoding="utf-8")
    destination = tmp_path / "export"
    destination.mkdir()
    receipt = {"binding": {"configuration_bundles": {"one": [str(run)]}}}
    (destination / "export.json").write_text(json.dumps(receipt), encoding="utf-8")
    exported = SimpleNamespace(run_id="one", current_json=current, current_bytes=current.read_bytes(),
                               destination=destination, receipt=receipt)
    monkeypatch.setattr(corrected_setup, "native_series_link", lambda project: {"series_root": str(series)})
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError) as error:
        corrected_setup.apply_corrected_selected_run(SimpleNamespace(root=tmp_path), exported)
    assert "facies.7" in str(error.value) and "facies.8" in str(error.value)
    assert {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert not (destination / "apply-claim.json").exists()


def test_build_refuses_absent_ids_before_native_source_admission(table, tmp_path):
    """Use the real Build preflight with a canonical private export location."""
    revision = "a" * 32
    project = tmp_path / "project"
    destination = project / "exports/native_setup" / revision
    destination.mkdir(parents=True)
    current = project / "overrides/geometry_facies/current.json"
    current.parent.mkdir(parents=True)
    current.write_text(json.dumps({"revision_id": revision, "configuration": _config(table)}), encoding="utf-8")
    receipt = {"run_id": "one", "reference_run": "one", "revision_id": revision,
               "binding": {"configuration_bundles": {"one": []}}}
    (destination / "export.json").write_text(json.dumps(receipt), encoding="utf-8")
    run = tmp_path / "run.toml"
    run.write_text('[labeling]\nlabels = "' + (destination / "native_labels.npz").as_posix() +
        '"\n[facies]\npath = "' + (destination / "native_facies.npz").as_posix() +
        '"\nprops = "' + (destination / "facies_properties.csv").as_posix() +
        '"\n[shared.porosity_by_facies]\n7 = 0.44\n', encoding="utf-8")
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match=r"porosity_by_facies\.7.*absent"):
        corrected_setup.require_corrected_export_for_build(run)
    assert {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
