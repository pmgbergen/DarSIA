"""Calibration review's run list and Play toggle: local series, real widgets, no writes.

Roles come from the referenced calibration and series.toml; review status from
the approval records. The only write in these tests is the explicit approval.
"""

import json

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QScrollArea, QSlider

from darsia.fluidflower import calibration_approvals as approvals
from darsia.fluidflower import calibration_review_runs as review
from darsia.fluidflower.series_method import MethodSelection, save_series_method
from .calibration_review_fixture import add_saved_results

RUNS = ("ac17", "ac18", "ac19", "ac20", "ac38", "ac99")


def _files(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _series(tmp_path, *, calibration=True):
    root = tmp_path / "series-r"
    root.mkdir()
    (root / "series.toml").write_text(
        'schema_version = 1\nseries_id = "series-r"\nname = "series-r"\n[runs]\nids = ['
        + ", ".join(f'"{run}"' for run in RUNS) + "]\n"
        '[runs.excluded]\nac20 = "chemical outlier"\n',
        encoding="utf-8",
    )
    if calibration:
        (root / "fits").mkdir()
        (root / "fits" / "frozen.json").write_text(json.dumps({
            "frozen": True,
            "calibration_runs": ["ac17", "AC18"],
            "heldout_runs": ["ac19"],
            "descriptive_runs": ["ac38"],
        }), encoding="utf-8")
        save_series_method(root, MethodSelection(active="phase_model", model_version="m",
                                                 mass_calibration="fits/frozen.json"))
    return root


def _approve(root, run, reviewer=None):
    identity = approvals.calibration_identity(root, "fits/frozen.json")
    approvals.approve_run(root, run, identity, reviewer=reviewer, retry=lambda append: append())


def test_every_run_has_its_role_and_review_status(tmp_path):
    root = _series(tmp_path)
    _approve(root, "ac17", reviewer="Reviewer A")
    before = _files(root)
    runs = review.review_run_rows(root)
    rows = {row.run: row for row in runs.rows}
    assert [row.run for row in runs.rows] == list(RUNS)
    assert {run: row.role for run, row in rows.items()} == {
        "ac17": "fit", "ac18": "fit", "ac19": "held-out", "ac20": "excluded",
        "ac38": "descriptive", "ac99": review.NO_ROLE}
    assert rows["ac17"].review == review.APPROVED and "by Reviewer A" in rows["ac17"].detail
    assert {row.review for run, row in rows.items() if run != "ac17"} == {review.NOT_APPROVED}
    assert "chemical outlier" in rows["ac20"].detail
    assert runs.note == ("2 fit · 1 held-out · 1 descriptive · 1 excluded · 1 of 6 approved. "
                         "Roles from frozen.json and series.toml [runs] excluded.")
    assert _files(root) == before


def test_without_a_calibration_only_excluded_runs_have_a_role(tmp_path):
    root = _series(tmp_path, calibration=False)
    runs = review.review_run_rows(root)
    assert [row.role for row in runs.rows] == ["—", "—", "—", "excluded", "—", "—"]
    assert {row.review for row in runs.rows} == {review.NO_REVIEW}
    assert runs.note.startswith("1 excluded. Roles from series.toml [runs] excluded. "
                                "No mass calibration is referenced")
    (root / "series.toml").write_text('schema_version = 1\n[runs]\nids = ["x1"]\n', encoding="utf-8")
    assert review.review_run_rows(root).note.startswith("No roles recorded. No mass calibration")


def test_new_model_keeps_prior_run_roles_without_activating_previous_approvals(tmp_path):
    root = _series(tmp_path)
    _approve(root, "ac17")
    save_series_method(root, MethodSelection(active="phase_model", model_version="new"), prepared_model=True)
    before = _files(root)
    current = review.review_run_rows(root)
    assert {row.role for row in current.rows if row.run != "ac20"} == {review.NO_ROLE}
    roles = review.review_run_rows(root, roles_only=True)
    assert roles.rows[0].role == "fit" and roles.rows[2].role == "held-out"
    assert {row.review for row in roles.rows} == {review.NO_REVIEW}
    assert "no active mass calibration" in roles.note
    assert _files(root) == before


def test_unreadable_inputs_are_named_not_guessed(tmp_path):
    root = tmp_path / "broken"
    root.mkdir()
    runs = review.review_run_rows(root)
    assert runs.rows == () and runs.note.startswith("The series' runs could not be read:")
    series = _series(tmp_path)
    (series / "fits" / "frozen.json").write_text("[1, 2]", encoding="utf-8")
    runs = review.review_run_rows(series)
    assert {row.role for row in runs.rows if row.run != "ac20"} == {review.NO_ROLE}
    assert {row.review for row in runs.rows} == {review.NO_REVIEW}
    assert "The referenced calibration could not be read" in runs.note
    (series / "fits" / "frozen.json").write_text('{"calibration_runs": ["ac17"]}', encoding="utf-8")
    damaged = series / approvals.APPROVALS_PATH
    damaged.parent.mkdir(parents=True)
    damaged.write_text("not a record\n", encoding="utf-8")
    runs = review.review_run_rows(series)
    assert runs.rows[0].role == "fit" and {row.review for row in runs.rows} == {review.NO_REVIEW}
    assert "The approval records could not be read" in runs.note
    assert review.read_series_runs(series) == (RUNS, {"ac20": "chemical outlier"})
    assert review.calibration_roles({"heldout_runs": ["A1"], "calibration_runs": ["a1"]}) == {"a1": "fit"}


def test_list_shows_every_run_and_emits_only_choosable_user_choices(qtbot, tmp_path):
    rows = review.review_run_rows(_series(tmp_path))
    panel = review.CalibrationReviewRunList()
    qtbot.addWidget(panel)
    chosen = []
    panel.run_selected.connect(chosen.append)
    panel.set_rows(rows)
    assert panel.runs() == list(RUNS) and panel.title.text() == "Runs — whole series (6)"
    table = panel.table
    assert [table.item(i, 1).text() for i in range(table.rowCount())][:3] == ["fit", "fit", "held-out"]
    assert table.editTriggers() == table.EditTrigger.NoEditTriggers
    for _ in range(2):
        panel.set_available(["ac17", "AC19"])
    assert not table.item(1, 0).flags() & Qt.ItemFlag.ItemIsSelectable
    assert table.item(1, 0).toolTip().count(review.NO_RESULT) == 1
    panel.select_run("ac18")  # No saved result: cannot be chosen.
    assert panel.selected_run() is None
    panel.select_run("ac17")
    assert panel.selected_run() == "ac17" and chosen == []
    table.selectRow(2)
    assert chosen == ["ac19"]
    panel.set_rows(rows)  # A re-read keeps the chosen run.
    assert panel.selected_run() == "ac19" and chosen == ["ac19"]


def test_play_steps_the_slider_with_a_timer_and_stops_at_the_end(qtbot):
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 3)
    button = QPushButton()
    qtbot.addWidget(slider)
    qtbot.addWidget(button)
    slider.show()
    player = review.SliderPlayer(slider, button, interval_ms=1)
    assert button.text() == "Play" and button.isCheckable()
    seen = []
    slider.valueChanged.connect(seen.append)
    button.click()
    assert player.playing() and button.text() == "Pause"
    qtbot.waitUntil(lambda: not player.playing(), timeout=5000)
    assert slider.value() == 3 and seen == [1, 2, 3]
    assert not button.isChecked() and button.text() == "Play"
    button.click()  # From the last image Play starts again at the first.
    qtbot.waitUntil(lambda: not player.playing(), timeout=5000)
    assert seen == [1, 2, 3, 0, 1, 2, 3]
    slider.setValue(1)
    slider.setEnabled(False)
    button.click()
    assert not player.playing() and not button.isChecked() and slider.value() == 1


def _page(qtbot, monkeypatch, root):
    from darsia.fluidflower.tools import phase_studio_mass_analysis_workspace as module

    monkeypatch.setattr(module, "configured_series", lambda: (("series-r", "series-r", root),))
    page = module.MassAnalysisWorkspace(series="series-r", project_root=root)
    qtbot.addWidget(page)
    experiments = page.saved_results.view_experiment
    add_saved_results(page, root, ("ac17", "ac19", "ac38"))
    return page, experiments


def test_calibration_review_lists_all_runs_beside_the_one_scroll_area(qtbot, tmp_path, monkeypatch):
    root = _series(tmp_path)
    page, experiments = _page(qtbot, monkeypatch, root)
    try:
        page.show()
        page.show_view("review")
        listed = page.run_list
        assert listed.isVisible() and listed.runs() == list(RUNS)
        assert page.findChildren(QScrollArea) == [page.scroll]
        assert not page.scroll.isAncestorOf(listed)  # No scroll inside a scrolling page.
        assert listed.selected_run() == "ac17"
        # Choosing a run in the list views it through the unchanged Experiment selector.
        listed.table.selectRow(2)
        assert experiments.currentData() == "ac19"
        assert page.approve_button.text() == "Approve AC19"
        experiments.setCurrentIndex(experiments.findData("ac38"))
        assert listed.selected_run() == "ac38"
        assert not listed.table.item(1, 0).flags() & Qt.ItemFlag.ItemIsSelectable  # ac18
        before = _files(root)
        page.approve_button.click()
        assert [record["run"] for record in approvals.read_run_approvals(root)] == ["ac38"]
        assert listed.table.item(4, 2).text() == review.APPROVED
        after = _files(root)
        assert {path for path in after if after[path] != before.get(path)} == {
            str(approvals.APPROVALS_PATH)}
        page.show_view("results")
        assert not listed.isVisible()
    finally:
        page._stop_lookups()


def test_play_sits_beside_previous_and_a_run_change_stops_it(qtbot, tmp_path, monkeypatch):
    root = _series(tmp_path)
    page, experiments = _page(qtbot, monkeypatch, root)
    try:
        saved = page.saved_results
        navigation = review._layout_holding(saved.layout(), saved.previous_image)
        assert navigation.indexOf(page.play_button) == navigation.indexOf(saved.previous_image) + 1
        assert navigation.indexOf(saved.slider) == navigation.indexOf(page.play_button) + 1
        page.show()
        page.show_view("review")
        saved.slider.setRange(0, 5)
        saved.slider.setEnabled(True)
        page.play_button.click()
        assert page.player.playing()
        experiments.setCurrentIndex(experiments.findData("ac19"))
        assert not page.player.playing() and not page.play_button.isChecked()
    finally:
        page._stop_lookups()


def test_a_result_that_switches_runs_without_a_signal_moves_the_list(qtbot, tmp_path, monkeypatch):
    from darsia.fluidflower.tools import phase_studio_series_ui as viewer

    def silent_load(self, path):
        # As the retained viewer does: set the selector with its signals blocked.
        from pathlib import Path
        if str(path).endswith(".json"):
            path = json.loads(Path(path).read_text(encoding="utf-8"))["selected_inputs"]["candidate"]["run"]
        self.view_experiment.blockSignals(True)
        self.view_experiment.setCurrentIndex(self.view_experiment.findData(path))
        self.view_experiment.blockSignals(False)

    monkeypatch.setattr(viewer.SeriesMassWorkspace, "load_result", silent_load)
    root = _series(tmp_path)
    page, experiments = _page(qtbot, monkeypatch, root)
    try:
        page.show()
        page.show_view("review")
        assert page.run_list.selected_run() == "ac17"
        page.saved_results.load_result("ac38")
        assert experiments.currentData() == "ac38"
        assert page.run_list.selected_run() == "ac38"
        assert page.approve_button.text() == "Approve AC38"
    finally:
        page._stop_lookups()


def test_insert_after_needs_a_layout_that_holds_the_anchor(qtbot):
    lonely = QPushButton()
    qtbot.addWidget(lonely)
    assert review.insert_after(lonely, QPushButton()) is False
