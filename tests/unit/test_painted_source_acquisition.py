"""An explicit facies-source statement follows one verified painting only.

All photographs, decisions and maps are private synthetic fixtures. The target
review photographs deliberately remain the old parent NPZ files; their exposure
does not determine the chemistry of the shared source painting.
"""
from copy import deepcopy
from pathlib import Path

import pytest

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import geometry_raster_sources as sources
from darsia.fluidflower import map_adoption as adoption
from darsia.fluidflower import painted_adoption_table as contract

from .test_geometry_registered_export_apply import registered_setup  # noqa: F401
from .test_painted_adoption_table import imported, prepared_painting, synthetic_payload_admission  # noqa: F401


def _statement(prepared, state="indicator"):
    """Explicit synthetic owner DATA, never an actual acquisition declaration."""
    folder = Path(prepared["folder"])
    binding = prepared["painting_facies"]
    return dict(acquisition_state=state,
        preparation=contract._bound(folder / "index.json"),
        photograph=deepcopy(binding["frame_basis"]["photograph"]),
        version=deepcopy(binding["version"]), approval=deepcopy(binding["approval"]),
        source_map=contract._local(folder, prepared["source_map"]),
        decision=dict(actor="synthetic source owner",
                      statement="Explicit test-only chemistry for this exact painting source."))


def _source_bytes(prepared):
    folder = Path(prepared["folder"])
    binding = prepared["painting_facies"]
    paths = [folder / "index.json", folder / "approval.json", folder / "source_map.npz",
             Path(binding["version"]["path"]),
             *(Path(row["path"]) for row in binding["files"].values())]
    return {path: path.read_bytes() for path in paths}


@pytest.mark.parametrize("state", ["clearwater", "indicator", "unknown"])
def test_explicit_painting_source_state_reaches_every_target_and_original_warnings(
        registered_setup, tmp_path, state):
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path)
    before = _source_bytes(prepared)
    declaration = _statement(prepared, state)
    expected = deepcopy(declaration)
    candidate, _ = adoption.adoption_candidate(data.revision, prepared["folder"],
        facies_source_acquisition=declaration)
    rows = candidate["run_geometry"]["runs"]
    assert {row["run_id"] for row in rows} == {"one", "two"}
    assert all(row["acquisition_state"] == state for row in rows)
    assert all(row["source_image_path"] != expected["photograph"]["path"] for row in rows)
    assert candidate["provenance"]["facies_source_acquisition"] == expected

    # Publication/state provenance is owned by the candidate, not its caller.
    declaration["acquisition_state"] = "clearwater" if state != "clearwater" else "indicator"
    declaration["photograph"]["sha256"] = "0" * 64
    declaration["decision"]["statement"] = "Changed after the call."
    assert candidate["provenance"]["facies_source_acquisition"] == expected
    report = geometry.validate_geometry_config(candidate)
    assert not [row for row in report["issues"] if row["severity"] == "error"]
    codes = [row["code"] for row in report["issues"]]
    assert codes.count("indicator_facies_carryover_risk") == (2 if state == "indicator" else 0)
    assert codes.count("run_geometry_acquisition_unknown") == (2 if state == "unknown" else 0)
    assert sources.canonical_parent_payload(candidate)["painted_table_bytes"]
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_missing_and_none_keep_the_original_unknown_output_and_ignore_manifest_state(
        registered_setup, tmp_path):
    data = registered_setup
    old_config = deepcopy(data.revision["configuration"])
    old_config["provenance"]["facies_source_acquisition"] = {"old_source": "must not be inherited"}
    for row in old_config["run_geometry"]["runs"]:
        row["acquisition_state"] = "indicator"
    data.revision = geometry.GeometryRevisionStore(data.project).save_revision(old_config,
        actor="synthetic prior owner", reason="private preceding source metadata")
    data.config = data.revision["configuration"]
    prepared, _, _, _, _ = prepared_painting(data, tmp_path)
    data.project.manifest["runs"] = [dict(run_id=run, indicator_state="clearwater")
                                     for run in ("one", "two")]
    original, summary = adoption.adoption_candidate(data.revision, prepared["folder"])
    explicit_none, none_summary = adoption.adoption_candidate(data.revision, prepared["folder"],
        facies_source_acquisition=None)
    assert explicit_none == original and none_summary == summary
    assert "facies_source_acquisition" not in original["provenance"]
    assert all(row["acquisition_state"] == "unknown" for row in original["run_geometry"]["runs"])


@pytest.mark.parametrize("anchor", ["photograph", "version", "preparation", "source_map", "approval"])
def test_explicit_statement_refuses_a_different_source_anchor(registered_setup, tmp_path, anchor):
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path)
    before = _source_bytes(prepared)
    declaration = _statement(prepared)
    declaration[anchor]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="another " + anchor):
        adoption.adoption_candidate(data.revision, prepared["folder"],
            facies_source_acquisition=declaration)
    assert all(path.read_bytes() == raw for path, raw in before.items())


@pytest.mark.parametrize("change", ["invalid_state", "empty_decision", "no_decision", "unknown_field"])
def test_incomplete_or_ambiguous_statement_never_defaults_to_clearwater(
        registered_setup, tmp_path, change):
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path)
    declaration = _statement(prepared)
    if change == "invalid_state":
        declaration["acquisition_state"] = "indicator_filled"
    elif change == "empty_decision":
        declaration["decision"] = {}
    elif change == "no_decision":
        declaration.pop("decision")
    else:
        declaration["infer_from_run"] = True
    with pytest.raises(ValueError, match="facies-source acquisition statement"):
        adoption.adoption_candidate(data.revision, prepared["folder"],
            facies_source_acquisition=declaration)


@pytest.mark.parametrize("change", ["photograph", "one_target_state"])
def test_store_source_replay_rechecks_statement_and_all_rows_after_derive_cache_is_warm(
        registered_setup, tmp_path, change):
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path)
    candidate, _ = adoption.adoption_candidate(data.revision, prepared["folder"],
        facies_source_acquisition=_statement(prepared))
    assert sources.canonical_parent_payload(candidate)["painted_table_bytes"]
    if change == "photograph":
        candidate["provenance"]["facies_source_acquisition"]["photograph"]["sha256"] = "0" * 64
        message = "another photograph"
    else:
        candidate["run_geometry"]["runs"][1]["acquisition_state"] = "clearwater"
        message = "states differ from their explicit facies source"
    with pytest.raises(ValueError, match=message):
        sources.canonical_parent_payload(candidate)
