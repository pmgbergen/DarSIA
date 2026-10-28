"""Approve run in Calibration review (N141 spec 2.5, Part 2), on the real page.

A temporary local series; the only write is the explicit approval record.
"""

import getpass
from pathlib import Path

import pytest

from darsia.fluidflower import calibration_approvals as approvals
from darsia.fluidflower.reviewer_preferences import save_reviewer_name
from darsia.fluidflower.series_method import MethodSelection, save_series_method
from .calibration_review_fixture import add_saved_results


def _series(tmp_path, *, calibration=True):
    root = tmp_path / "series-a"
    root.mkdir()
    (root / "series.toml").write_text(
        'schema_version = 1\nseries_id = "series-a"\nname = "series-a"\n[runs]\nids = ["ac28", "ac47"]\n',
        encoding="utf-8",
    )
    if calibration:
        (root / "fits").mkdir()
        (root / "fits/frozen.json").write_text('{"frozen": true}', encoding="utf-8")
        save_series_method(root, MethodSelection(active="phase_model", model_version="m",
                                                 mass_calibration="fits/frozen.json"))
    return root


def _page(qtbot, monkeypatch, root):
    from darsia.fluidflower.tools import phase_studio_mass_analysis_workspace as module

    monkeypatch.setattr(getpass, "getuser", lambda: pytest.fail("The OS login is never recorded"))
    monkeypatch.setattr(module, "configured_series", lambda: (("series-a", "series-a", root),))
    page = module.MassAnalysisWorkspace(series="series-a", project_root=root)
    page.saved_results.native_catalog.registry = type(page.saved_results.native_catalog.registry)(root / "review-registrations")
    qtbot.addWidget(page)
    experiments = page.saved_results.view_experiment
    add_saved_results(page, root, ("ac28", "ac47"))
    return page, experiments


def test_approve_run_appends_one_record_for_the_viewed_run(qtbot, tmp_path, monkeypatch):
    root = _series(tmp_path)
    save_reviewer_name(root, "Reviewer A")  # The series' reviewer setting (N175).
    page, experiments = _page(qtbot, monkeypatch, root)
    try:
        page.show_view("results")
        assert page.approval.isHidden()
        page.show_view("review")
        experiments.setCurrentIndex(0)
        assert not page.approval.isHidden()
        assert page.approve_button.text() == "Approve AC28" and page.approve_button.isEnabled()
        assert "AC28: not approved for calibration fits/frozen.json" in page.approval_label.text()
        before = (root / "series.toml").read_bytes()
        page.approve_button.click()
        records = approvals.read_run_approvals(root)
        assert [record["run"] for record in records] == ["ac28"]
        assert records[0]["calibration"] == approvals.calibration_identity(root, "fits/frozen.json")
        assert records[0]["reviewer"] == "Reviewer A"
        assert "AC28: approved by Reviewer A at" in page.approval_label.text()
        assert "1 run(s) approved" in page.approval_label.text()
        assert not page.approve_button.isEnabled()
        experiments.setCurrentIndex(1)
        assert page.approve_button.text() == "Approve AC47" and page.approve_button.isEnabled()
        assert (root / "series.toml").read_bytes() == before
    finally:
        page._stop_lookups()


def test_without_a_reviewer_setting_the_approval_names_nobody(qtbot, tmp_path, monkeypatch):
    root = _series(tmp_path)
    page, experiments = _page(qtbot, monkeypatch, root)
    try:
        page.show_view("review")
        experiments.setCurrentIndex(0)
        page.approve_button.click()
        record = approvals.read_run_approvals(root)[0]
        assert record["reviewer"] is None
        text = page.approval_label.text()
        assert "AC28: approved at " in text and " by " not in text
    finally:
        page._stop_lookups()


def test_approval_is_unavailable_without_a_referenced_calibration(qtbot, tmp_path, monkeypatch):
    root = _series(tmp_path, calibration=False)
    page, experiments = _page(qtbot, monkeypatch, root)
    try:
        page.show_view("review")
        experiments.setCurrentIndex(0)
        assert page.approval_label.text().startswith("Approval unavailable: No mass calibration is referenced")
        assert not page.approve_button.isEnabled()
        page.approve_selected_run()
        assert not (root / approvals.APPROVALS_PATH).exists()
    finally:
        page._stop_lookups()


def test_another_calibration_for_the_same_run_cannot_approve_the_series_reference(qtbot, tmp_path, monkeypatch):
    root = _series(tmp_path)
    (root / "fits/other.json").write_text('{"frozen": true, "variant": 2}', encoding="utf-8")
    page, experiments = _page(qtbot, monkeypatch, root)
    try:
        add_saved_results(page, root, ("ac28",), name="Calibration B", reference="fits/other.json")
        page.show_view("review")
        saved = page.saved_results
        saved.mass_calibration.setCurrentIndex(saved.mass_calibration.findData(str((root / "fits/other.json").resolve())))
        assert experiments.currentData() == "ac28"
        assert not page.approve_button.isEnabled()
        assert "differs from the series' approved calibration" in page.approval_label.text()
        page.approve_selected_run()  # The write path also checks, without trusting the button.
        assert not (root / approvals.APPROVALS_PATH).exists()
        saved.mass_calibration.setCurrentIndex(saved.mass_calibration.findData(str((root / "fits/frozen.json").resolve())))
        assert page.approve_button.isEnabled()
        page.approve_button.click()
        record = approvals.read_run_approvals(root)[0]
        assert record["calibration"] == approvals.calibration_identity(root, "fits/frozen.json")
        assert Path(record["result"]).parts[-3] == "Calibration A"
    finally:
        page._stop_lookups()


@pytest.mark.parametrize("change", ["replace-calibration-bytes", "missing-checksum", "another-series", "conflicting-checksums"])
def test_approval_requires_the_saved_result_binding_not_a_hash_of_todays_path(qtbot, tmp_path, monkeypatch, change):
    root = _series(tmp_path)
    page, _ = _page(qtbot, monkeypatch, root)
    try:
        page.show_view("review")
        result = page.saved_results.result
        if change == "replace-calibration-bytes":
            (root / "fits/frozen.json").write_text('{"frozen": true, "replacement": true}', encoding="utf-8")
        elif change == "missing-checksum":
            result.pop("bindings")
        elif change == "another-series":
            result["selected_inputs"]["candidate"]["dataset"] = "different-series"
        else:
            result["calibration_sha256"] = "0" * 64
        page.refresh_approval()
        assert not page.approve_button.isEnabled()
        page.approve_selected_run()
        assert not (root / approvals.APPROVALS_PATH).exists()
    finally:
        page._stop_lookups()


def test_missing_result_or_changed_calibration_binding_never_writes_approval(qtbot, tmp_path, monkeypatch):
    root = _series(tmp_path)
    page, _experiments = _page(qtbot, monkeypatch, root)
    try:
        page.show_view("review")
        page.saved_results.result.pop("series_calibration")
        page.refresh_approval()
        assert not page.approve_button.isEnabled()
        page.approve_selected_run()
        assert not (root / approvals.APPROVALS_PATH).exists()
        page.saved_results.clear_view()
        page.refresh_approval()
        assert not page.approve_button.isEnabled()
        page.approve_selected_run()
        assert not (root / approvals.APPROVALS_PATH).exists()
    finally:
        page._stop_lookups()
