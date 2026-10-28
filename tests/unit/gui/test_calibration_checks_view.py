"""Checks page: held-out checks 1-3 of one or two check-score files, read-only.

Real offscreen Qt with synthetic files in tmp_path; the file dialog is a
stand-in. Nothing is computed, written or stored as a setting.
"""

import gc
import json
import weakref

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QFileDialog, QScrollArea

from darsia.fluidflower import row_scope as scopes
from darsia.fluidflower.calibration_checks_view import EMPTY_TEXT, CalibrationChecksPage

from .test_calibration_checks import score_document, write_scores
from .test_native_layout_pages import PHASE_CALIBRATION, _visible
from .test_native_series_method import _files, _roots, method_window  # noqa: F401
from .test_row_scope import context, series  # noqa: F401


def _snapshot(folder):
    return {
        str(path.relative_to(folder)): (path.stat().st_mtime_ns, path.read_bytes())
        for path in folder.rglob("*")
        if path.is_file()
    }


def _headers(table):
    return [
        table.horizontalHeaderItem(column).text()
        for column in range(table.columnCount())
    ]


def _rows(table):
    return [
        tuple(table.item(row, column).text() for column in range(table.columnCount()))
        for row in range(table.rowCount())
    ]


def _summary(page):
    return {values[0]: values[1:] for values in _rows(page.summary)}


@pytest.fixture
def page(qtbot):
    widget = CalibrationChecksPage()
    qtbot.addWidget(widget)
    widget.show()
    return widget


@pytest.fixture
def files(tmp_path):
    first = tmp_path / "on" / "variant_on.json"
    first.parent.mkdir()
    write_scores(first)
    document = score_document()
    document["verdict"] = "FAIL"
    document["check1"]["per_time"]["1.0"]["median_abs_error"] = 0.0789
    document["check1"]["status"] = "FAIL"
    second = tmp_path / "off" / "variant_off.json"
    second.parent.mkdir()
    write_scores(second, document)
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"verdict": "PASS", "check1": {}}), encoding="utf-8")
    return first, second, broken


def test_empty_page_explains_which_file_to_open(page):
    assert page.empty_note.isVisible() and page.empty_note.text() == EMPTY_TEXT
    assert "check-score file (JSON)" in EMPTY_TEXT and "writes nothing" in EMPTY_TEXT
    assert not page.tabs.isVisible() and not page.verdict.isVisible()
    assert (
        page.open_button.text() == "Open check results…"
        and page.open_button.isEnabled()
    )
    assert not page.second_button.isEnabled() and not page.path_fields[1].isEnabled()
    assert not page.remove_button.isEnabled() and not page.details_row.isVisible()
    # Only the inline controls scroll; result tables keep their own scrolling.
    assert page.findChildren(QScrollArea) == [page.controls_scroll]
    assert page.controls_scroll.isAncestorOf(page.calculate_button)
    assert not page.controls_scroll.isAncestorOf(page.tabs)


def test_path_field_opens_the_file_and_every_check_is_shown(page, files, tmp_path):
    first, _second, _broken = files
    before = _snapshot(tmp_path)
    page.path_fields[0].setText(f'"{first}"')
    page.path_fields[0].returnPressed.emit()
    assert [scores.name for scores in page.scores()] == ["variant_on.json"]
    assert not page.empty_note.isVisible() and page.tabs.isVisible()
    assert page.verdict.text() == "variant_on.json: verdict Pass"
    assert [page.tabs.tabText(index) for index in range(page.tabs.count())] == [
        "Summary",
        "Check 1",
        "Check 2",
        "Check 3",
    ]
    assert _headers(page.summary) == ["Item", "variant_on.json"]
    summary = _summary(page)
    assert summary["Verdict"] == ("Pass",)
    assert summary["Check 3"] == ("Fail, decided by the fallback rule",)
    assert summary["Check 1 at 1.0: median |M/d − 1|"] == ("0.041",)
    assert page.check1_note.text().startswith("Status in the file: Pass.")
    assert _headers(page.check1_table) == ["Item", "Time 1.0", "Time 2.5"]
    check1 = _rows(page.check1_table)
    assert check1[1] == ("Median |M/d − 1|", "0.041", "0.030")
    assert check1[2] == ("group_a median M/d", "0.951", "0.970")
    assert check1[7] == ("r02 M/d", "0.912", "—")
    note, tree = page.rule_views[1]
    assert note.text().startswith(
        "Decision in the file: Fail, decided by the fallback rule (zero_rule)."
    )
    primary, fallback = tree.topLevelItem(0), tree.topLevelItem(1)
    assert (
        primary.text(0) == "Primary rule: capacity_rule" and not primary.font(0).bold()
    )
    assert (
        fallback.text(0) == "Fallback rule: zero_rule (decides this check)"
        and fallback.font(0).bold()
    )
    assert [tree.headerItem().text(column) for column in range(tree.columnCount())] == [
        "Rule, time or run",
        "Status",
        "Assessable",
        "Runs counted",
        "δ",
        "M/d",
        "W/d",
        "Set",
    ]
    late = primary.child(1)
    assert (late.text(0), late.text(2), late.text(4)) == ("Time 60.0", "No", "—")
    loro = primary.child(0).child(1)
    assert loro.text(7) == "Fit run, LORO-predicted" and loro.font(0).italic()
    assert late.isExpanded() and primary.isExpanded()
    assert _snapshot(tmp_path) == before


def test_open_button_uses_the_dialog_and_a_refused_file_keeps_the_display(
    page, files, monkeypatch
):
    first, _second, broken = files
    chosen = [str(broken)]
    calls = []

    def dialog(parent, title, folder, filters):
        calls.append((parent, title, folder, filters))
        return chosen[0], filters

    monkeypatch.setattr(QFileDialog, "getOpenFileName", dialog)
    page.open_button.click()
    assert (
        calls[0][0] is page
        and calls[0][1] == "Open check results"
        and calls[0][2] == ""
    )
    assert page.message.isVisible()
    assert page.message.text() == (
        "broken.json was not opened. Not a check-score file: check1 has no 'status'."
    )
    assert page.empty_note.isVisible() and not page.tabs.isVisible()
    chosen[0] = str(first)
    page.open_button.click()
    assert calls[1][2] == str(broken.parent)
    assert not page.message.isVisible() and page.tabs.isVisible()
    chosen[0] = str(broken)
    page.open_button.click()
    assert page.message.isVisible() and "check1 has no 'status'" in page.message.text()
    # What was shown before stays shown.
    assert [scores.name for scores in page.scores()] == ["variant_on.json"]
    assert page.verdict.text() == "variant_on.json: verdict Pass"
    chosen[0] = ""
    page.open_button.click()  # Cancelled: nothing changes.
    assert [scores.name for scores in page.scores()] == ["variant_on.json"]


def test_unreadable_path_is_reported_without_a_crash(page, tmp_path):
    assert not page.open_file(tmp_path / "absent.json")
    assert page.message.text().startswith(
        "absent.json was not opened. The file could not be read:"
    )
    assert page.empty_note.isVisible()


def test_second_file_shows_both_side_by_side_and_can_be_removed(
    page, files, monkeypatch
):
    first, second, _broken = files
    assert not page.open_file(second, second=True)
    assert page.message.text() == "Open the check results before a file to compare."
    assert page.open_file(first)
    assert page.second_button.isEnabled() and page.path_fields[1].isEnabled()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(second), ""))
    page.second_button.click()
    assert [scores.name for scores in page.scores()] == [
        "variant_on.json",
        "variant_off.json",
    ]
    assert _headers(page.summary) == ["Item", "variant_on.json", "variant_off.json"]
    summary = _summary(page)
    assert summary["Verdict"] == ("Pass", "Fail")
    assert summary["Check 1"] == ("Pass", "Fail")
    assert summary["Check 1 at 1.0: median |M/d − 1|"] == ("0.041", "0.079")
    assert summary["Check 2, primary rule at 4.0: median δ"] == ("0.026", "0.026")
    assert (
        page.verdict.text()
        == "variant_on.json: verdict Pass\nvariant_off.json: verdict Fail"
    )
    assert page.details_row.isVisible() and page.remove_button.isEnabled()
    choice = page.details_choice
    assert [choice.itemText(index) for index in range(choice.count())] == [
        "variant_on.json",
        "variant_off.json",
    ]
    assert page.tabs.cornerWidget(Qt.Corner.TopRightCorner) is page.details_row
    assert _rows(page.check1_table)[1][1] == "0.041"
    choice.setCurrentIndex(1)
    assert _rows(page.check1_table)[1][1] == "0.079"
    assert page.check1_note.text().startswith("Status in the file: Fail.")
    page.remove_button.click()
    assert [scores.name for scores in page.scores()] == ["variant_on.json"]
    assert _headers(page.summary) == ["Item", "variant_on.json"]
    assert not page.details_row.isVisible() and not page.remove_button.isEnabled()
    assert page.path_fields[1].text() == ""
    assert _rows(page.check1_table)[1][1] == "0.041"


def test_integrity_problems_in_the_file_are_shown_with_the_verdict(page, tmp_path):
    document = score_document()
    document["integrity_problems"] = ["row 2: duplicate", "row 9: mismatch"]
    path = tmp_path / "flagged.json"
    write_scores(path, document)
    assert page.open_file(path)
    assert (
        page.verdict.text()
        == "flagged.json: verdict Pass, 2 integrity problems reported"
    )
    assert _summary(page)["Integrity problems"] == (
        "2: row 2: duplicate; row 9: mismatch",
    )


def test_nothing_is_written_and_no_setting_is_stored(
    page, files, tmp_path, monkeypatch
):
    first, second, broken = files
    monkeypatch.setattr(
        QSettings, "setValue", lambda *args: pytest.fail("No setting is stored")
    )
    before = _snapshot(tmp_path)
    page.open_file(first)
    page.open_file(second, second=True)
    page.open_file(broken)
    page.details_choice.setCurrentIndex(1)
    page.remove_second()
    assert _snapshot(tmp_path) == before


def test_a_parentless_page_is_freed_without_garbage_collection(qapp, files):
    first, second, _broken = files
    gc.collect()
    gc.disable()
    try:
        widget = CalibrationChecksPage()
        widget.open_file(first)
        widget.open_file(second, second=True)
        reference = weakref.ref(widget)
        del widget
        assert reference() is None
    finally:
        gc.enable()


def test_checks_row_follows_calibration_review_and_opens_the_page(
    method_window, tmp_path, qtbot  # noqa: F811
):
    window = method_window
    qtbot.waitUntil(lambda: not window._series_loading, timeout=10000)
    roots = _roots(window)
    before = {key: _files(root) for key, root in roots.items()}
    rows = _visible(window, "calibration")
    index = rows.index(("phase_model:calibration_review", "Calibration review"))
    assert rows[index + 1] == ("phase_model:checks", "Checks")
    assert PHASE_CALIBRATION[-1] == ("phase_model:checks", "Checks")
    assert window.sidebar.display_selection("phase_model", "checks")[0] == "calibration"
    stack = window.content_stack
    minimum = stack.minimumSizeHint()
    window._select_row(("phase_model", "checks"))
    page = stack.currentWidget()
    assert (
        isinstance(page, CalibrationChecksPage)
        and page is window.ensure_calibration_checks()
    )
    assert window.sidebar.get_selected() == ("phase_model", "checks")
    assert page.empty_note.isVisible()
    # A whole-series page: the header names the series and the run selector hides.
    assert window.status_header.line.text().endswith(" · Whole series (0 runs)")
    assert not window.run_combo.isVisible()
    row = window.sidebar._sections["calibration"].get_row("phase_model:checks")
    assert "Applies to: whole series" in row.help_text
    assert "performs no calculation and writes nothing" in row.help_text
    # No saved-output status line is invented for a viewer row.
    assert ("phase_model", "checks") not in window.workflow_status.rows
    assert not window.workflow_status.panel.isVisible()
    assert not window.toolbar_builder.play_action.isEnabled()
    # The page does not make the window larger than before, empty or with two
    # files of long names side by side.
    after = stack.minimumSizeHint()
    assert after.width() <= minimum.width() and after.height() <= minimum.height()
    folder = tmp_path / "scores"
    folder.mkdir()
    long_name = "a_rather_long_check_score_file_name_without_any_break_for_the_column"
    for name in ("first", "second"):
        write_scores(folder / f"{long_name}_{name}.json")
    assert page.open_file(folder / f"{long_name}_first.json")
    assert page.open_file(folder / f"{long_name}_second.json", second=True)
    after = stack.minimumSizeHint()
    assert after.width() <= minimum.width() and after.height() <= minimum.height()
    window._select_row(("phase_model", "overflow"))
    assert stack.currentWidget() is window.mass_calibration_display
    window._select_row(("phase_model", "checks"))
    assert stack.currentWidget() is page
    assert [stack.widget(i) for i in range(stack.count())].count(page) == 1
    assert {key: _files(root) for key, root in roots.items()} == before


def test_checks_row_scope_is_the_whole_series(series):  # noqa: F811
    value = scopes.row_scope(context(series), "phase_model", "checks")
    assert value.scope == "series" and not value.run_selector_enabled
    assert value.line == "Applies to: whole series AC (2 runs)"
    assert value.note == (
        "Check results cover the whole series; opening a file performs no calculation "
        "and writes nothing."
    )


def test_page_text_is_plain_and_selectable(page):
    for label in (page.empty_note, page.message, page.verdict, page.check1_note):
        assert label.textFormat() == Qt.TextFormat.PlainText and label.wordWrap()
