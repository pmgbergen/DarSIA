"""Calibration products use one series folder while saved references remain usable."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import calibration, calibration_review, mass_calibration_library, series_paths


@pytest.fixture
def series(tmp_path, monkeypatch):
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "registry.toml"))
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text('schema_version = 1\nseries_id = "TEST"\n', encoding="utf-8")
    return root


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_saved_plan_and_review_products_follow_calibration_root(series, monkeypatch):
    from darsia.fluidflower.indicator_response import IndicatorResponseModelStore

    project = SimpleNamespace(root=series, _audit=lambda *args: None)
    monkeypatch.setattr(calibration, "validate_calibration_plan", lambda *args: {
        "summary": {"error_count": 0}, "objective_contract": {"content_sha256": "a" * 64}})
    monkeypatch.setattr(calibration.PhaseInterpretationRevisionStore, "current", lambda self: {})
    monkeypatch.setattr(calibration_review, "validate_calibration_review_config", lambda *args: {
        "summary": {"error_count": 0, "approved_run_count": 0, "run_count": 0}})
    plans = calibration.CalibrationPlanStore(project)
    reviews = calibration_review.CalibrationResultReviewStore(project)
    plan = plans.save_plan({}, actor="operator", reason="save")
    review = reviews.save_review({"calibration_plan_id": plan["plan_id"]}, actor="operator", reason="review")
    assert (series / "Calibration" / "plans" / plan["plan_id"] / "plan.json").is_file()
    saved = series_paths.calibration_artifact_path(series, "calibration_reviews", review["review_id"], "review.json")
    assert json.loads(saved.read_text())["configuration"]["calibration_plan_id"] == plan["plan_id"]
    assert json.loads(plans.current_path.read_text())["plan_id"] == plan["plan_id"]
    assert json.loads(reviews.current_path.read_text())["review_id"] == review["review_id"]
    assert IndicatorResponseModelStore(project).artifact_root == series / "Calibration" / "indicator_response" / "models"
    assert plans.current_path == series / "overrides" / "calibration" / "current.json"
    assert series_paths.series_folder_path(series, "results") == series / "results"


def test_existing_review_and_explicit_calibration_destinations_remain_usable(series):
    old = _write(series / "results" / "calibration" / "reviews" / "saved" / "review.json", {"review_id": "saved"})
    assert series_paths.calibration_artifact_path(series, "calibration_reviews", "saved", "review.json") == old
    explicit = series / "chosen reviews"
    with (series / "series.toml").open("a", encoding="utf-8") as stream:
        stream.write('\n[paths]\ncalibration = "chosen calibration"\ncalibration_reviews = "chosen reviews"\n')
    project = SimpleNamespace(root=series)
    assert calibration.CalibrationPlanStore(project).artifact_root == series / "chosen calibration" / "plans"
    assert calibration_review.CalibrationResultReviewStore(project).artifact_root == explicit
    current = _write(explicit / "saved" / "review.json", {"review_id": "saved", "new": True})
    assert series_paths.calibration_artifact_path(series, "calibration_reviews", "saved", "review.json") == current
    assert old.is_file()


def test_mass_library_writes_new_references_and_reads_legacy_without_moving_fits(series):
    source = _write(series / "results" / "concentration_mass" / "saved" / "calibration.json", {
        "format": "native-darsia-fixed-phase-mass-v1", "parameters": {"gain": 1.0},
        "native_calibrations": {"r1": {"calibration_files": ["saved.json"]}},
        "fit_performed": True, "training_indices": [0]})
    original = source.read_bytes()
    entry = Path(mass_calibration_library.index_file(source, series)[0])
    assert entry.parent == series / "Calibration" / "mass_calibration"
    legacy = series / "models" / "mass_calibration" / entry.name
    legacy.parent.mkdir(parents=True)
    entry.rename(legacy)
    rows = mass_calibration_library.discover_mass_calibrations(series)
    assert len(rows) == 1 and rows[0]["available"]
    assert Path(rows[0]["calibration_path"]) == source
    assert Path(rows[0]["entry_path"]) == legacy
    # Re-indexing writes the new catalog; the fit and original reference stay intact.
    new_entry = Path(mass_calibration_library.index_file(source, series)[0])
    rows = mass_calibration_library.discover_mass_calibrations(series)
    assert len(rows) == 1 and Path(rows[0]["entry_path"]) == new_entry
    assert legacy.is_file() and source.read_bytes() == original


def test_colour_report_default_uses_series_root_with_standalone_fallback(series):
    from darsia.fluidflower.native_mass_calibration import _default_report_root, REPORT_FOLDER

    config = series / "config" / "runs" / "r1.toml"
    source_folder = series / "existing" / "interpolation" / "mass"
    assert _default_report_root([config], source_folder) == series / "Calibration" / REPORT_FOLDER
    with (series / "series.toml").open("a", encoding="utf-8") as stream:
        stream.write('\n[paths]\ncolour_calibration_reports = "chosen reports"\n')
    assert _default_report_root([config], source_folder) == series / "chosen reports"
    standalone = series.parent / "standalone" / "run.toml"
    assert _default_report_root([standalone], source_folder) == source_folder.parent.parent / REPORT_FOLDER


def test_refit_gui_default_follows_series_but_preserves_chosen_local_output(series, qtbot, monkeypatch):
    from darsia.fluidflower import refit_job_panel

    monkeypatch.setattr(refit_job_panel, "series_refit_roles", lambda root: (None, "No roles"))
    def no_baseline(root):
        raise ValueError("No baseline")
    monkeypatch.setattr(refit_job_panel, "baseline_calibration", no_baseline)
    panel = refit_job_panel.RefitJobPanel()
    qtbot.addWidget(panel)
    panel.set_series(series)
    assert Path(panel.jobs_edit.text()) == series / "Calibration" / "concentration_mass" / "jobs"
    # The series names the folder; it is shown, and only Other folder… changes it.
    assert panel.jobs_edit.isReadOnly() and panel.jobs_button.text() == "Other folder…"
    other = series.parent / "other"
    panel.set_series(other)
    assert Path(panel.jobs_edit.text()) == other / "Calibration" / "concentration_mass" / "jobs"
    panel.jobs_edit.setText(str(series.parent / "chosen local jobs"))
    panel.set_series(series)
    assert Path(panel.jobs_edit.text()) == series.parent / "chosen local jobs"
    panel.jobs_edit.clear()
    panel.set_series(Path(r"\\storage\series\DarSIA"))
    assert not panel.jobs_edit.text()
