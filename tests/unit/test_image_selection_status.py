"""Saved Setup choices are independent of later fit and production cohorts."""

from copy import deepcopy
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from darsia.fluidflower.image_selection_status import (
    production_selection_note,
    saved_selection_summary,
    setup_selection_summary,
)
from darsia.fluidflower.series_image_selection import FORMAT
from darsia.fluidflower.workflow_row_status import WorkflowStatusContext, row_status


def selected(run):
    return {
        "selected": [
            {
                "source_path": f"images/{run}/baseline.tif",
                "image_id": "baseline.tif",
                "frame_index": 0,
                "elapsed_h": -0.1,
            },
            {
                "source_path": f"images/{run}/later.tif",
                "image_id": "later.tif",
                "frame_index": 5,
                "elapsed_h": 2.0,
            },
        ],
        "source_metadata": [
            {"path": f"protocols/{run}.csv", "sha256": "a" * 64, "size_bytes": 50}
        ],
        "saved_utc": "2026-10-16T20:00:00+00:00",
    }


@pytest.fixture
def document():
    return {
        "schema_version": 1,
        "series_id": "portable_fixture",
        "runs": {
            "ids": ["run_a", "run_b"],
            "excluded": {"removed_run": "Explicitly removed from this series"},
        },
        "image_selection": {
            "format": FORMAT,
            "schema_version": 1,
            "runs": {run: selected(run) for run in ("run_a", "run_b")},
        },
    }


def test_active_saved_choices_survive_already_removed_exclusions(document, tmp_path, monkeypatch):
    manifest = tmp_path / "series.toml"
    manifest.write_text("schema_version=1\n", encoding="utf-8")
    before = deepcopy(document)
    # The status formatter may load Python's timezone database on Windows.
    ZoneInfo("Europe/Oslo")

    def no_payload(*_args, **_kwargs):
        pytest.fail("Passive status must not open protocols, images or production selections")

    monkeypatch.setattr(Path, "open", no_payload)
    info = setup_selection_summary(document)
    status = row_status(WorkflowStatusContext(tmp_path, manifest=document), "image_selection", "plan")
    assert info["complete"] and info["saved_run_count"] == 2
    assert info["selected_frame_count"] == 4 and info["eligible_count"] == 2
    assert info["inactive_excluded_runs"] == document["runs"]["excluded"]
    assert info["excluded_count"] == 0 and info["missing_runs"] == []
    assert status.state == "done" and "4 images in 2 of 2 active runs" in status.summary
    assert "not a computed result" in status.summary
    assert status.date_source == "saved_utc"
    assert document == before


def test_missing_active_choice_is_not_hidden_by_removed_exclusions(document):
    del document["image_selection"]["runs"]["run_b"]
    info = setup_selection_summary(document)
    assert not info["complete"]
    assert info["saved_run_count"] == 1 and info["missing_runs"] == ["run_b"]
    assert info["selected_frame_count"] == 2


def test_registered_exclusion_removes_only_that_run_from_setup_count(document):
    document["runs"]["excluded"]["RUN_B"] = "Explicit registered exclusion"
    del document["image_selection"]["runs"]["run_b"]
    original = saved_selection_summary(document)
    assert not original["complete"] and original["missing_runs"] == ["run_b"]
    info = setup_selection_summary(document)
    assert info["complete"] and info["eligible_count"] == 1
    assert info["excluded_runs"] == {"run_b": "Explicit registered exclusion"}
    assert info["selected_frame_count"] == 2


@pytest.mark.parametrize("reason", ["", 1, None])
def test_invalid_reason_never_silently_removes_a_missing_active_run(document, reason):
    document["runs"]["excluded"]["run_b"] = reason
    del document["image_selection"]["runs"]["run_b"]
    info = setup_selection_summary(document)
    assert not info["complete"] and info["missing_runs"] == ["run_b"]
    assert info["eligible_count"] == 2 and info["excluded_count"] == 0


def test_production_grid_is_reported_separately_and_cannot_complete_setup(document):
    del document["image_selection"]
    assert "no separate saved grid" in production_selection_note(document)
    document["production_image_selection"] = {
        "selection_manifest": {"path": "jobs/production/selected.json", "sha256": "b" * 64}
    }
    assert not setup_selection_summary(document)["complete"]
    assert "separate saved-manifest reference" in production_selection_note(document)
    assert "not counted above" in production_selection_note(document)


def test_selected_rows_without_source_binding_do_not_claim_saved_ready(document):
    document["image_selection"]["runs"]["run_b"].pop("source_metadata")
    info = setup_selection_summary(document)
    assert not info["complete"] and info["missing_runs"] == ["run_b"]
    document["image_selection"]["runs"]["run_b"]["source_metadata"] = [{}]
    with pytest.raises(ValueError, match="malformed source metadata binding"):
        setup_selection_summary(document)
