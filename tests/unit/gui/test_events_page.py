"""Comparisons > Events page: questions from menus, answered for every run (synthetic data only).

The mass analysis is a fixture reader; no job folder, map or NAS input is
read. Saved questions go to a temporary series folder.
"""
import threading

from PySide6.QtWidgets import QFileDialog, QMessageBox

from darsia.fluidflower import analysis_events as events
from darsia.fluidflower.analysis_events import Condition, RunSeries
from darsia.fluidflower.events_page import EventsPage

FRAMES = ("c0", "c1", "c2", "c3")


def run(name, aqueous, gas=(0, 0, 0, 0)):
    parts = dict(aqueous=tuple(aqueous), gas=tuple(gas), total=tuple(a + g for a, g in zip(aqueous, gas)))
    return RunSeries(name, FRAMES, (0.0, 1.5, 3.0, 4.5), (0.0, 10.0, 20.0, 30.0),
                     {"full_rig": parts, "box1": parts, "box:Upper": parts})


SERIES = [run("ac01", (0, .2, 1, 2), gas=(0, 0, .5, .5)), run("ac02", (0, 0, 0, .4))]


def page(qtbot, tmp_path, *, reader=None, jobs=None):
    calls = []

    def read(root, cancelled, progress):
        calls.append(threading.current_thread() is threading.main_thread())
        progress(4, 8)
        progress(8, 8)
        return SERIES, "Masses from the mass analysis's frame records", "series S · 2 runs, 2 complete"

    widget = EventsPage(jobs_reader=lambda root, series_id: jobs if jobs is not None else [
        dict(name="Mass job", root=str(tmp_path / "job"))], reader=reader or read)
    qtbot.addWidget(widget)
    widget.calls = calls
    return widget


def test_a_question_is_answered_for_every_run(qtbot, tmp_path):
    widget = page(qtbot, tmp_path)
    assert not widget.answer_button.isEnabled()
    widget.set_series(tmp_path, "S")
    qtbot.waitUntil(lambda: bool(widget.series), timeout=10000)
    assert widget.calls == [False]  # Read off the GUI thread.
    assert widget.status.text().startswith("2 finished runs, 8 frames.")
    row = widget.rows[0]
    assert [row.where.itemText(i) for i in range(row.where.count())] == ["Whole rig", "Box 1", "Upper"]
    row.where.setCurrentIndex(row.where.findData("box:Upper"))
    widget.answer_button.click()
    table = widget.table
    assert [[table.item(r, c).text() for c in range(4)] for r in range(2)] == [
        ["AC01", "1.50", "c1", ""], ["AC02", "4.50", "c3", ""]]
    assert widget.sentence.text() == "Dissolved CO2 is first detected in Upper: found in 2 of 2 runs."
    # A second condition, combined with and: free gas must be detected too.
    second = widget.add_condition(Condition("gas", "detected", "box1"))
    widget.answer_button.click()
    assert [table.item(r, 2).text() for r in range(2)] == ["c2", ""]
    assert table.item(1, 3).text() == "Never"
    widget.combine.setCurrentIndex(widget.combine.findData("any"))
    widget.answer_button.click()
    assert [table.item(r, 2).text() for r in range(2)] == ["c1", "c3"]
    second.remove.click()
    assert len(widget.rows) == 1 and not widget.rows[0].remove.isEnabled()
    # The answers are also drawn: one row per run at its time.
    assert not widget.chart.isHidden() and not widget.chart.pixmap().isNull()


def test_an_answer_is_compared_with_a_saved_question(qtbot, tmp_path):
    widget = page(qtbot, tmp_path)
    events.save_question(tmp_path, events.EventQuestion("Gas", (Condition("gas", "detected", "box1"),)))
    widget.set_series(tmp_path, "S")
    qtbot.waitUntil(lambda: bool(widget.series), timeout=10000)
    widget.compare_with.setCurrentIndex(widget.compare_with.findData("Gas"))
    widget.answer_button.click()
    table = widget.table
    assert table.horizontalHeaderItem(4).text() == "Before Gas"
    # AC01: dissolved CO2 at 1.5 h, gas at 3.0 h; AC02 has no free gas.
    assert [table.item(r, 4).text() for r in range(2)] == ["Yes", "—"]
    widget.relation.setCurrentIndex(widget.relation.findData("after"))
    widget.answer_button.click()
    assert table.horizontalHeaderItem(4).text() == "After Gas" and table.item(0, 4).text() == "No"
    widget.compare_with.setCurrentIndex(0)
    widget.answer_button.click()
    assert table.columnCount() == 4


def test_threshold_and_noise_filter_follow_their_choices(qtbot, tmp_path):
    widget = page(qtbot, tmp_path)
    row = widget.rows[0]
    widget.show()
    assert row.threshold.isHidden() and row.unit.isHidden()
    row.event.setCurrentIndex(row.event.findData("reaches"))
    assert not row.threshold.isHidden() and not row.unit.isHidden()
    row.threshold.setValue(5)
    assert widget.question().conditions[0] == Condition("dissolved", "reaches", "full_rig", 5.0, "percent_injected")
    assert not widget.hold.isEnabled()
    widget.noise.setChecked(True)
    assert widget.hold.isEnabled()
    question = widget.question()
    assert (question.noise_min, question.noise_unit, question.hold_frames) == (0.01, "g", 2)


def test_questions_are_saved_in_the_series_and_loaded_again(qtbot, tmp_path, monkeypatch):
    widget = page(qtbot, tmp_path)
    widget.set_series(tmp_path, "S")
    qtbot.waitUntil(lambda: bool(widget.series), timeout=10000)
    widget.save_button.click()
    assert "Give the question a name" in widget.sentence.text()
    widget.name.setText("First gas in Upper")
    widget.rows[0].what.setCurrentIndex(widget.rows[0].what.findData("gas"))
    widget.rows[0].where.setCurrentIndex(widget.rows[0].where.findData("box:Upper"))
    widget.noise.setChecked(True)
    widget.save_button.click()
    path = tmp_path / "config" / "events" / "First gas in Upper.json"
    assert path.is_file() and widget.saved.currentText() == "First gas in Upper"
    # Saving again under the same name asks first.
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args: asked.append(args) or QMessageBox.StandardButton.No)
    widget.save_button.click()
    assert asked and path.is_file()
    widget.show_question(events.EventQuestion("Other", (Condition("co2", "maximum"),)))
    assert widget.rows[0].event.currentData() == "maximum" and not widget.noise.isChecked()
    widget.load_saved()
    assert widget.name.text() == "First gas in Upper" and widget.noise.isChecked()
    assert widget.rows[0].condition() == Condition("gas", "detected", "box:Upper")


def test_answers_are_exported_and_failures_are_named(qtbot, tmp_path, monkeypatch):
    widget = page(qtbot, tmp_path)
    widget.set_series(tmp_path, "S")
    qtbot.waitUntil(lambda: bool(widget.series), timeout=10000)
    widget.answer_button.click()
    target = tmp_path / "answers.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(target), "CSV (*.csv)"))
    widget.export_button.click()
    assert target.read_text(encoding="utf-8") == events.answers_csv(widget.asked, widget.answers)

    def fail(root, cancelled, progress):
        raise OSError("the job folder is not reachable")

    broken = page(qtbot, tmp_path, reader=fail)
    broken.set_series(tmp_path, "S")
    qtbot.waitUntil(lambda: not broken._read.busy, timeout=10000)
    assert broken.status.text() == "The mass analysis could not be read: OSError: the job folder is not reachable"
    assert not broken.answer_button.isEnabled()
    empty = page(qtbot, tmp_path, jobs=[])
    empty.set_series(tmp_path, "S")
    assert "no registered mass analysis job" in empty.status.text()


def test_the_chart_is_saved_as_a_png(qtbot, tmp_path, monkeypatch):
    from PySide6.QtGui import QImage

    widget = page(qtbot, tmp_path)
    assert not widget.figure_button.isEnabled()
    widget.set_series(tmp_path, "S")
    qtbot.waitUntil(lambda: bool(widget.series), timeout=10000)
    widget.answer_button.click()
    target = tmp_path / "answers.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(target), "PNG image (*.png)"))
    widget.figure_button.click()
    image = QImage(str(target))
    assert not image.isNull() and image.size() == widget.chart.pixmap().size()
