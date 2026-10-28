"""Transfer diagnostics retain original checks and never publish a candidate."""

from copy import deepcopy
from pathlib import Path

import pytest

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import map_adoption as adoption
from darsia.fluidflower import painted_analysis_exclusions as exclusions
from darsia.fluidflower import painted_transfer_findings as subject
from darsia.fluidflower import painted_transfer_replay as replay
from darsia.fluidflower.painted_adoption_table import _source

from .test_painted_adoption_table import (  # noqa: F401
    imported, prepared_painting, registered_setup, synthetic_payload_admission,
)


ACTION = dict(actor="synthetic human", approved_utc="2026-01-01T00:00:00Z",
    reason="Explicit source review in a synthetic fixture", decision_ref="synthetic-source-review")


def _files(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def _capture_original_review(data, tmp_path, monkeypatch, *, gaps):
    captured = {}
    original = exclusions.build_gap_review
    original_prepare = adoption.prepare_adoption

    def observe(preparation, folder, **kwargs):
        # Observe the actual original producer recipe.  No scientific reader,
        # source admission, product proof or original assertion is replaced.
        captured.update(preparation=deepcopy(preparation), folder=Path(folder))
        result = original(preparation, folder, **kwargs)
        captured["review"] = result
        return result

    def prepare_with_review(*args, **kwargs):
        # Ask the original producer for its own bound review before publication;
        # an unbound review appended after index.json is not a valid oracle.
        return original_prepare(*args, **dict(kwargs, review_findings=True))

    with monkeypatch.context() as patch:
        patch.setattr(exclusions, "build_gap_review", observe)
        patch.setattr(adoption, "prepare_adoption", prepare_with_review)
        if gaps:
            with pytest.raises(exclusions.GapReviewRequired):
                prepared_painting(data, tmp_path, outside=True,
                    outside_targets=True, continuation=True)
        else:
            prepared, *_ = prepared_painting(data, tmp_path, continuation=True)
            captured["preparation"] = deepcopy(prepared)
    captured["context"] = replay.source_context_from_preparation(
        captured["preparation"], captured["folder"])
    return captured


def _collect(data, case, **kwargs):
    return subject.collect_transfer_findings(_source(data.revision)["revision"],
        case["review"]["review_binding"], source_context=case["context"], **kwargs)


def test_transfer_data_matches_original_candidate_diagnostics_without_new_authority(
        registered_setup, tmp_path, monkeypatch):
    data = registered_setup
    case = _capture_original_review(data, tmp_path, monkeypatch, gaps=False)
    candidate, _ = adoption.adoption_candidate(data.revision, case["folder"])
    expected = geometry._geometry_input_report(candidate)
    before = _files(tmp_path)

    proof_folder = tmp_path / "verified_findings_inputs"
    result = _collect(data, case, residual_destination=proof_folder)

    after = _files(tmp_path)
    assert {name: value for name, value in after.items()
            if not name.startswith(proof_folder.name + "/")
            and not name.startswith(proof_folder.name + "\\")} == before
    assert not list(proof_folder.rglob("index.json"))
    report = result["report"]
    assert report["format"] == subject.FORMAT
    assert report["issues"] == expected["issues"]
    assert report["summary"] == expected["summary"]
    assert result["review"]["review_binding"] == case["review"]["review_binding"]
    assert not {"candidate", "configuration", "painted_adoption_table", "registration", "approval", "index"} & set(report)
    assert all("registration" not in row for row in report["run_data"])


def test_unresolved_transfer_keeps_actual_gaps_zero_and_original_errors_without_index(
        registered_setup, tmp_path, monkeypatch):
    data = registered_setup
    case = _capture_original_review(data, tmp_path, monkeypatch, gaps=True)
    assert not (case["folder"] / "index.json").exists()
    before = _files(tmp_path)

    result = _collect(data, case)

    assert _files(tmp_path) == before
    report = result["report"]
    assert report["table_data"]["unmapped_label_ids"] == [0]
    assert all(row["label_id"] != 0 for row in report["table_data"]["regions"])
    actual = {(row["run_id"], component["component_id"])
              for row in case["review"]["runs"] for component in row["components"]}
    unresolved = {(row["run_id"], row["component_id"])
                  for row in report["unresolved_domains"] if row["kind"] == "analysis_gap"}
    assert unresolved == actual
    assert any(issue["code"] == "unmapped_labels" and 0 in issue["details"]["labels"]
               for issue in report["issues"])


def test_current_source_choice_changes_original_findings_without_inheriting_target_state(
        registered_setup, tmp_path, monkeypatch):
    data = registered_setup
    case = _capture_original_review(data, tmp_path, monkeypatch, gaps=False)
    before = _files(tmp_path)
    ordinary = _collect(data, case)
    action = dict(ACTION, acquisition_state="indicator")
    current = _collect(data, case, explicit_actions=dict(source_choices=[action]))

    assert _files(tmp_path) == before
    assert all(row["acquisition_state"] == "unknown" for row in ordinary["report"]["run_data"])
    assert all(row["acquisition_state"] == "indicator" for row in current["report"]["run_data"])
    assert current["report"]["input_bindings"]["source_choices"] == [action]
    assert any(issue["code"] == "indicator_facies_carryover_risk"
               for issue in current["report"]["issues"])
    assert not any(issue["code"] == "run_geometry_acquisition_unknown"
                   for issue in current["report"]["issues"])
    assert current["report"]["table_data"] == ordinary["report"]["table_data"]


def test_changed_actual_parent_is_rejected_before_a_new_list_can_replace_it(
        registered_setup, tmp_path, monkeypatch):
    data = registered_setup
    case = _capture_original_review(data, tmp_path, monkeypatch, gaps=False)
    source = _source(data.revision)
    path = Path(source["csv"]["path"])
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError):
        subject.collect_transfer_findings(source["revision"],
            case["review"]["review_binding"], source_context=case["context"])


@pytest.mark.parametrize("source", ["review", "geometry_index"])
def test_consumed_source_change_during_diagnostics_is_rejected_before_result_install(
        registered_setup, tmp_path, monkeypatch, source):
    data = registered_setup
    case = _capture_original_review(data, tmp_path, monkeypatch, gaps=False)
    original = subject._assemble_findings

    def mutate_after_original_diagnostics(*args, **kwargs):
        report = original(*args, **kwargs)
        descriptor = (case["review"]["review_binding"] if source == "review"
                      else case["review"]["geometry_index"])
        path = Path(descriptor["path"])
        path.write_bytes(path.read_bytes() + b"\nchanged after original diagnostics\n")
        return report

    monkeypatch.setattr(subject, "_assemble_findings", mutate_after_original_diagnostics)

    with pytest.raises(ValueError, match="changed|differ"):
        _collect(data, case)
