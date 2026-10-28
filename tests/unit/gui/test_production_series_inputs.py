"""New analysis prepares from the series' own inputs with one button; files stay optional."""
from PySide6.QtWidgets import QFormLayout

from darsia.fluidflower import analysis_plan_sources, series_geometry
from darsia.fluidflower import native_production_sections as ui
from darsia.fluidflower.analysis_plan_sources import Source
from .test_production_page import page, select  # noqa: F401

FOUND = [Source("Run documents", "runs", found="2 runs", bound=[dict(path="r1.json"), dict(path="r2.json")]),
         Source("Measured frame geometry", "geometry", found="g.json", bound=dict(path="g.json")),
         Source("Running pressure", "pressure", found="p.json", bound=dict(path="p.json"))]


def _editor(page, qtbot, monkeypatch, found):
    """The phase-one editor of the fixture's series, after one read of ``found``."""
    monkeypatch.setattr(analysis_plan_sources, "phase_one_sources", lambda root: list(found))
    page.inline_layout.sync()
    editor = page._preparation_dialog
    assert isinstance(editor, ui._InlineAnalysisPreparation) and editor.series_root == "series"
    editor.check_series_sources()
    qtbot.waitUntil(lambda: editor._series_sources is not None, timeout=5000)
    return editor


def _row_shown(editor, widget):
    form = next(form for form in editor.findChildren(QFormLayout) if form.indexOf(editor.stage) >= 0)
    return form.isRowVisible(ui._form_row(form, widget))


def test_missing_inputs_are_listed_and_prepare_waits_for_them(page, qtbot, monkeypatch):
    missing = [FOUND[0], Source("Running pressure", "inputs/pressure_index.json", "Not saved yet.")]
    editor = _editor(page, qtbot, monkeypatch, missing)
    text = editor.series_sources.text()
    assert "Found: Run documents: 2 runs" in text
    assert "Missing: Running pressure. Not saved yet.\n    Expected: inputs/pressure_index.json" in text
    assert not editor.prepare_from_series_button.isEnabled()
    assert editor.check_sources_button.isEnabled()


def test_file_choices_stay_available_behind_one_checkbox(page, qtbot, monkeypatch):
    editor = _editor(page, qtbot, monkeypatch, FOUND)
    assert not editor.choose_files.isChecked()
    assert not any(_row_shown(editor, field) for field in (editor.inputs, editor.geometry, editor.folder))
    assert _row_shown(editor, editor.stage) and _row_shown(editor, editor.probability)
    assert editor.prepare_button.isHidden() and not editor.adopt_button.isHidden()
    editor.choose_files.setChecked(True)
    assert all(_row_shown(editor, field) for field in (editor.inputs, editor.geometry, editor.pressure, editor.folder))
    assert not editor.prepare_button.isHidden()


def test_one_button_prepares_the_found_inputs_for_review(page, qtbot, monkeypatch):
    editor = _editor(page, qtbot, monkeypatch, FOUND)
    assert editor.approval_actor.text() == series_geometry.default_actor()
    assert editor.prepare_from_series_button.isEnabled()
    queued, calls = [], []
    editor._run = lambda call, **kwargs: queued.append((call, kwargs))
    monkeypatch.setattr(analysis_plan_sources, "prepare_phase_one_plan",
                        lambda *a, **k: calls.append((a, k)) or "prepared")
    editor.probability.setChecked(True)
    editor.prepare_from_series_button.click()
    assert len(queued) == 1 and queued[0][1] == {}
    assert (editor.geometry.text(), editor.pressure.text(), editor.name.text()) == ("g.json", "p.json", "phase-maps")
    assert editor.inputs.text() == editor.folder.text() + "-source-inputs.json"
    assert queued[0][0]() == "prepared"
    (args, kwargs), = calls
    assert args == ("series", FOUND) and kwargs["gas_probability"] is True and kwargs["job_id"] == "phase-maps"
    assert str(kwargs["folder"]) == editor.folder.text() and str(kwargs["manifest"]) == editor.inputs.text()


def test_another_stage_reads_its_own_inputs(page, qtbot, monkeypatch):
    editor = _editor(page, qtbot, monkeypatch, FOUND)
    stages = []
    monkeypatch.setattr(analysis_plan_sources, "phase_two_sources",
                        lambda root, calibration: stages.append(calibration) or [
                            Source("Fitted mass calibration", "jobs", "Choose a completed mass calibration job.")])
    editor.stage.setCurrentIndex(editor.stage.findData("phase2"))
    qtbot.waitUntil(lambda: editor._series_sources is not None, timeout=5000)
    assert stages == [None] and "Missing: Fitted mass calibration" in editor.series_sources.text()
    assert not editor.prepare_from_series_button.isEnabled()
