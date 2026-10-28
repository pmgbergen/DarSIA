"""Five GUI steps and private metadata choices; no scientific action is run."""
import json
from pathlib import Path
import threading

from PySide6.QtWidgets import QFileDialog, QPushButton

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import map_placement as placement
from darsia.fluidflower import painting_saved_transfers_ui as saved
from darsia.fluidflower.map_placement_view import MapPlacementDialog

from .test_painting_corrections import view  # noqa: F401
from .test_painting_manual_step4_route import _saved_manual


def _index(root, leaf, version, when):
    folder = root / leaf
    folder.mkdir(parents=True)
    document = dict(format=placement.PLACEMENT_FORMAT, reference_run="sample",
        map_version=dict(version), created_utc=when, runs=[], layers={})
    (folder / "index.json").write_text(json.dumps(document), encoding="utf8")
    return folder, document


def _files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_actual_step_order_places_facies_before_versions_and_actions_below(view):
    assert [section.toggle.text() for section in
            (view.step_frame, view.step_paint, view.step_facies, view.step_propose, view.step_use_map)] == [
        "Reference points, injection points and analysis boxes", "Mark layers", "Facies", "Versions", "Use the map in the series"]
    assert view.step_frame.content.isAncestorOf(view.analysis_boxes.widget)
    view.step_frame.set_expanded(False)
    assert not view.analysis_boxes.widget.isVisible()
    view.step_frame.toggle.click()
    assert view.analysis_boxes.widget.isVisible()
    view.step_frame.toggle.click()
    assert not view.analysis_boxes.widget.isVisible()
    assert view.step_facies.content.isAncestorOf(view.facies_table)
    assert view.step_propose.content.isAncestorOf(view.versions_table)
    assert not view.step_propose.content.isAncestorOf(view.place_version_button)
    for button in (view.place_version_button, view.chart_review_button,
                   view.prepare_adoption_button, view.adopt_button):
        assert view.step_use_map.content.isAncestorOf(button)
    assert view.step_export.content.isAncestorOf(view.export_button)
    assert view.step_export.isHidden()
    assert not hasattr(view, "setup_pointer")
    assert "Apply to all runs" in view.apply_build_button.text() and not view.apply_build_button.isEnabled()


def test_approval_opens_use_step_and_retains_reachable_versions_without_starting_work(view, qtbot):
    record = _saved_manual(view, qtbot)
    approval = view.approve_version(record)
    assert approval is not None and "Using Selected map (approved" in view.map_status.text()
    assert not view.step_propose.is_expanded() and view.step_use_map.is_expanded()
    assert view.place_version_button.property("seriesMapNext") is True
    assert view.place_version_button.text() == "Transfer map to all runs…"
    assert view.prepare_adoption_button is view.adopt_button
    assert view.adopt_button.text() == "Adopt as the series' sand map…"
    before = _files(lp.painting_folder(view.project))
    view.step_propose.toggle.click()
    assert view.step_propose.is_expanded()
    assert _files(lp.painting_folder(view.project)) == before
    assert view._placement_thread is None and view._adoption_thread is None and view._save_thread is None


def test_disabled_actions_name_the_missing_saved_version_and_approval(view, qtbot):
    view._refresh_use_map()
    for button in (view.place_version_button, view.prepare_adoption_button, view.adopt_button):
        assert not button.isEnabled()
        assert "saved map version under Versions" in view.use_map_reasons[button].text()
    _saved_manual(view, qtbot)
    for button in (view.place_version_button, view.prepare_adoption_button, view.adopt_button):
        assert not button.isEnabled()
        assert "Approve this saved version under Versions" in view.use_map_reasons[button].text()


def test_review_chooser_preselects_latest_transfer_of_selected_approved_version_and_one_click_opens(view, qtbot, monkeypatch):
    record = _saved_manual(view, qtbot)
    approval = view.approve_version(record)
    version = dict(record, approval=approval)
    root = lp.painting_folder(view.project).parent / "map_placement"
    _index(root, "older", version, "2026-01-01T01:00:00Z")
    chosen, _ = _index(root, "chosen", version, "2026-01-01T02:00:00Z")
    other = dict(version, name="Other map", sha256="b" * 64, folder=str(root / "unrelated-version"))
    _index(root, "newest-other-map", other, "2026-01-01T03:00:00Z")
    before = _files(root)
    calls = []
    read = saved.saved_transfers
    def observed(*args):
        calls.append(threading.current_thread() is threading.main_thread())
        return read(*args)
    monkeypatch.setattr(saved, "saved_transfers", observed)
    def no_picker(*_args, **_kwargs):
        raise AssertionError("Saved transfer review must not open a file picker")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", no_picker)
    view.open_placement_button.click()
    chooser = view._saved_transfers_dialog
    qtbot.waitUntil(lambda: not chooser.busy(), timeout=5000)
    assert calls == [False]
    assert chooser.rows[chooser.list.currentRow()]["folder"] == str(chosen)
    chooser.open_button.click()
    assert isinstance(view._placement_dialog, MapPlacementDialog)
    assert view._placement_dialog.folder == chosen
    assert view._placement_dialog.isVisible()
    assert view.adopt_button.isEnabled()
    assert view._adoption_thread is None  # Reviewing metadata never starts adoption.
    assert not view.apply_build_button.isEnabled()
    assert _files(root) == before
    view._placement_dialog.close()


def test_summary_names_map_reference_and_meanings_and_has_no_folder_button(qtbot, tmp_path):
    folder, value = _index(tmp_path, "summary", dict(name="Private version", sha256="a" * 64), "2026-01-01T01:00:00Z")
    dialog = MapPlacementDialog(dict(value, folder=str(folder)))
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.windowTitle() == "Sand map placement summary – Private version in all runs"
    assert "Private version from SAMPLE" in dialog.lead.text() and "Read only" in dialog.lead.text()
    assert "camera displacement" in dialog.table.horizontalHeaderItem(2).toolTip()
    assert "near 0 is good" in dialog.table.horizontalHeaderItem(3).toolTip()
    assert "5 mm" in dialog.table.horizontalHeaderItem(4).toolTip()
    assert "not among the analysed runs" in dialog.table.horizontalHeaderItem(1).toolTip()
    assert dialog.fit_button.text() == "Show whole photo"
    assert all(button.text() not in {"Fit", "Open folder", "Open saved placement folder"}
               for button in dialog.findChildren(QPushButton))
    assert not hasattr(dialog, "open_button")


def test_metadata_list_refuses_damaged_entries_without_writing_anything(qtbot, tmp_path):
    _index(tmp_path, "valid", dict(name="Private map", sha256="a" * 64), "2026-01-01T01:00:00Z")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "index.json").write_bytes(b'{"format":"wrong", "format":"duplicate"}')
    _index(tmp_path, "bad-approval", dict(name="Malformed same-content map", sha256="a" * 64,
        approval=["not an approval object"]), "2026-01-01T02:00:00Z")
    before = _files(tmp_path)
    rows, errors = saved.saved_transfers(tmp_path)
    assert len(rows) == 1 and rows[0]["version"]["name"] == "Private map"
    assert len(errors) == 2 and any("Repeated" in error for error in errors)
    assert any("approval metadata must be an object" in error for error in errors)
    assert _files(tmp_path) == before


def test_closing_during_finite_metadata_read_retains_thread_until_retired(qtbot, tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def finite(_root):
        entered.set()
        assert release.wait(5), "The private test must release its metadata reader"
        return [], []
    monkeypatch.setattr(saved, "saved_transfers", finite)
    chooser = saved.SavedTransferChooser(tmp_path)
    qtbot.addWidget(chooser)
    chooser.show()
    try:
        qtbot.waitUntil(entered.is_set, timeout=2000)
        chooser.close()
        assert chooser.busy() and chooser.isVisible()
        assert "current metadata read" in chooser.status.text()
        release.set()
        qtbot.waitUntil(lambda: not chooser.busy() and not chooser.isVisible(), timeout=5000)
    finally:
        release.set()
        chooser.wait_for_workers()


def test_adopt_button_does_not_fall_back_to_legacy_dialog_for_known_check(view, qtbot, monkeypatch, tmp_path):
    record = _saved_manual(view, qtbot)
    approval = view.approve_version(record)
    view.last_adoption = dict(map_version=dict(record, approval=approval), folder=str(tmp_path / "private-check"))
    opened = []
    monkeypatch.setattr(view, "open_adoption", lambda folder: opened.append(folder))
    def no_picker(*_args, **_kwargs):
        raise AssertionError("The known selected check must open without another path chooser")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", no_picker)
    view._series_adopt_action = None
    view._adopt_clicked()
    assert opened == []
    assert "automatic adoption backend" in view.use_map_progress.text()
    assert view._adoption_thread is None and view._save_thread is None
