"""Comparisons > Events: when does something happen in an area, for every run.

The page reads a finished mass analysis's per-frame masses in the background
(small records only; no map is read). The user builds a question from menus,
one row per condition (what, event, threshold, where), combines conditions
with and/or, can turn on a noise filter, and gets the answer for every
finished run at once. Questions are saved by name in the series folder and
can be loaded again. The rules are in ``analysis_events``.
"""
from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import analysis_events as events

COLUMNS = ("Run", "Time (h)", "Frame", "Note")
NO_SERIES = "Choose a registered series on Start."
CHART_WIDTH_PX = 760


def render_answers(question: events.EventQuestion, answers, *, width_px: int = CHART_WIDTH_PX) -> np.ndarray:
    """The answers as a chart, one row per run at its hours since injection start; RGBA pixels.

    A run without a time shows its note (for example "Never") instead of a point.
    """
    from .production_results_view import _RENDER_LOCK, _figure, _pixels

    rows = list(answers)
    with _RENDER_LOCK:
        figure = _figure((width_px, min(1400, 140 + 24 * max(1, len(rows)))))
        axis = figure.subplots()
        found = [(index, row.elapsed_h) for index, row in enumerate(rows) if row.elapsed_h is not None]
        if found:
            axis.plot([hours for _index, hours in found], [index for index, _hours in found], "o", color="#1f6feb")
        for index, row in enumerate(rows):
            if row.elapsed_h is None:
                axis.text(0.01, index, row.note or "Not found", transform=axis.get_yaxis_transform(),
                          va="center", color="#667085", fontsize=8)
        axis.set_yticks(range(len(rows)), [row.run.upper() for row in rows])
        axis.set_ylim(len(rows) - 0.5, -0.5)
        axis.set_xlabel("Hours since injection start")
        axis.set_title(textwrap.fill(question.text(), 100), fontsize=10)
        axis.grid(axis="x", alpha=0.3)
        return _pixels(figure)


class _Cancelled(Exception):
    """The read was replaced by another one."""


def _registered_jobs(series_root, series_id):
    from .analysis_box_reanalysis_ui import series_mass_jobs
    return series_mass_jobs(series_root, series_id)


def _read_job(root, cancelled, progress):
    """A finished mass analysis's run series, where they came from, and one line about the job."""
    from .production_results_view import ProductionJob

    def check():
        if cancelled():
            raise _Cancelled()

    job = ProductionJob(root)
    series, source = events.read_job_series(job, check_cancelled=check, on_progress=progress)
    return series, source, job.facts()


def _plain(label: QLabel) -> QLabel:
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    return label


class ConditionRow(QWidget):
    """One condition: what, event, its threshold where the event needs one, and where."""

    changed = Signal()
    remove_requested = Signal(object)

    def __init__(self, areas: dict[str, str], condition: events.Condition | None = None, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.what = QComboBox()
        for key, label in events.WHAT.items():
            self.what.addItem(label, key)
        self.event = QComboBox()
        for key, label in events.EVENTS.items():
            self.event.addItem(label, key)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1e9)
        self.threshold.setDecimals(3)
        self.threshold.setValue(5.0)
        self.unit = QComboBox()
        for key, label in events.UNITS.items():
            self.unit.addItem(label, key)
        self.where = QComboBox()
        self.where.setToolTip("Areas with saved masses: the whole rig, the mass analysis's boxes, and "
                              "the named boxes and sand layers of Mass from saved maps.")
        self.remove = QPushButton("Remove")
        self.remove.clicked.connect(lambda: self.remove_requested.emit(self))
        for widget in (self.what, self.event, self.threshold, self.unit):
            row.addWidget(widget)
        row.addWidget(QLabel("in"))
        row.addWidget(self.where, 1)
        row.addWidget(self.remove)
        self.set_areas(areas)
        if condition is not None:
            self.set_condition(condition)
        for combo in (self.what, self.event, self.unit, self.where):
            combo.currentIndexChanged.connect(self._changed)
        self.threshold.valueChanged.connect(self._changed)
        self._sync()

    def _changed(self, *_args) -> None:
        self._sync()
        self.changed.emit()

    def _sync(self) -> None:
        needs = self.event.currentData() in events.THRESHOLD_EVENTS
        self.threshold.setVisible(needs)
        self.unit.setVisible(needs)

    def set_areas(self, areas: dict[str, str]) -> None:
        """Offer ``areas``, keeping the chosen one (marked when this analysis lacks it)."""
        current = self.where.currentData()
        self.where.blockSignals(True)
        try:
            self.where.clear()
            for key, label in (areas or {events.WHOLE_RIG: events.area_label(events.WHOLE_RIG)}).items():
                self.where.addItem(label, key)
            if current is not None and self.where.findData(current) < 0:
                self.where.addItem(events.area_label(current) + " (not in this analysis)", current)
            self.where.setCurrentIndex(max(0, self.where.findData(current)) if current is not None else 0)
        finally:
            self.where.blockSignals(False)

    def condition(self) -> events.Condition:
        """The condition shown."""
        needs = self.event.currentData() in events.THRESHOLD_EVENTS
        return events.Condition(self.what.currentData(), self.event.currentData(), self.where.currentData(),
                                float(self.threshold.value()) if needs else None,
                                self.unit.currentData() if needs else "percent_injected")

    def set_condition(self, condition: events.Condition) -> None:
        """Show ``condition``; an area this analysis lacks is kept and marked."""
        for combo, value in ((self.what, condition.what), (self.event, condition.event),
                             (self.unit, condition.unit)):
            index = combo.findData(value)
            if index >= 0:
                combo.setCurrentIndex(index)
        if condition.threshold is not None:
            self.threshold.setValue(float(condition.threshold))
        if self.where.findData(condition.where) < 0:
            self.where.addItem(events.area_label(condition.where) + " (not in this analysis)", condition.where)
        self.where.setCurrentIndex(self.where.findData(condition.where))
        self._sync()


class EventsPage(QScrollArea):
    """Ask when something happens in an area, for every run of a finished mass analysis."""

    _progress = Signal(int, int)

    def __init__(self, parent=None, *, jobs_reader=None, reader=None):
        super().__init__(parent)
        from .ui_metadata_preview import MetadataPreview

        self.jobs_reader = jobs_reader or _registered_jobs
        self.reader = reader or _read_job
        self.series_root: Path | None = None
        self.series_id: str | None = None
        self.series: list[events.RunSeries] = []
        self.answers: list[events.EventAnswer] = []
        self.asked: events.EventQuestion | None = None
        self.comparison: tuple[str, dict] | None = None
        self.rows: list[ConditionRow] = []
        self._read = MetadataPreview(self)
        self._read.completed.connect(self._read_done)
        self._progress.connect(self._show_progress)
        self.setWidgetResizable(True)
        body = QWidget()
        self.setWidget(body)
        layout = QVBoxLayout(body)
        layout.addWidget(QLabel("Events"))
        layout.addWidget(_plain(QLabel(
            "Find when something happens in an area, for every run of a finished mass analysis: for "
            "example when dissolved CO2 is first detected in a box. The masses the mass analysis saved "
            "for each frame are read; no map is read and nothing is changed.")))
        jobs = QHBoxLayout()
        jobs.addWidget(QLabel("Mass analysis"))
        self.jobs = QComboBox()
        self.jobs.setMinimumContentsLength(16)
        self.jobs.currentIndexChanged.connect(self._job_chosen)
        jobs.addWidget(self.jobs, 1)
        self.choose_job = QPushButton("Choose job…")
        self.choose_job.clicked.connect(self._choose_job)
        jobs.addWidget(self.choose_job)
        self.refresh_jobs = QPushButton("Refresh jobs")
        self.refresh_jobs.clicked.connect(self.load_jobs)
        jobs.addWidget(self.refresh_jobs)
        layout.addLayout(jobs)
        self.status = _plain(QLabel(NO_SERIES))
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)

        question = QGroupBox("Question")
        form = QVBoxLayout(question)
        self.rows_layout = QVBoxLayout()
        form.addLayout(self.rows_layout)
        actions = QHBoxLayout()
        self.add_button = QPushButton("Add condition")
        self.add_button.clicked.connect(lambda: self.add_condition())
        actions.addWidget(self.add_button)
        actions.addWidget(QLabel("Answer when"))
        self.combine = QComboBox()
        for key, label in events.COMBINE.items():
            self.combine.addItem(label, key)
        actions.addWidget(self.combine)
        actions.addStretch(1)
        form.addLayout(actions)
        noise = QHBoxLayout()
        self.noise = QCheckBox("Noise filter:")
        self.noise.setToolTip("Off: any mass counts as detected, and the first frame where the question "
                              "holds is the answer. The phase maps already leave out most noise.")
        noise.addWidget(self.noise)
        noise.addWidget(QLabel("smallest detected mass"))
        self.noise_min = QDoubleSpinBox()
        self.noise_min.setRange(0.0, 1e9)
        self.noise_min.setDecimals(3)
        self.noise_min.setValue(0.01)
        noise.addWidget(self.noise_min)
        self.noise_unit = QComboBox()
        for key, label in events.UNITS.items():
            self.noise_unit.addItem(label, key)
        self.noise_unit.setCurrentIndex(self.noise_unit.findData("g"))
        noise.addWidget(self.noise_unit)
        noise.addWidget(QLabel("and must hold for"))
        self.hold = QSpinBox()
        self.hold.setRange(1, 1000)
        self.hold.setValue(2)
        noise.addWidget(self.hold)
        noise.addWidget(QLabel("frames in a row"))
        noise.addStretch(1)
        form.addLayout(noise)
        self.noise.toggled.connect(self._sync_noise)
        saved = QHBoxLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("Name, for example First dissolved CO2 in Box 1")
        saved.addWidget(self.name, 1)
        self.save_button = QPushButton("Save question")
        self.save_button.clicked.connect(self.save)
        saved.addWidget(self.save_button)
        saved.addWidget(QLabel("Saved questions"))
        self.saved = QComboBox()
        self.saved.setMinimumContentsLength(16)
        saved.addWidget(self.saved, 1)
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load_saved)
        saved.addWidget(self.load_button)
        form.addLayout(saved)
        layout.addWidget(question)

        answer = QHBoxLayout()
        self.answer_button = QPushButton("Answer for all runs")
        self.answer_button.clicked.connect(self.answer)
        answer.addWidget(self.answer_button)
        answer.addWidget(QLabel("Compare with"))
        self.compare_with = QComboBox()
        self.compare_with.setMinimumContentsLength(16)
        self.compare_with.setToolTip("A saved question: per run, did this event happen before (or after) it?")
        answer.addWidget(self.compare_with)
        self.relation = QComboBox()
        for key, label in events.RELATIONS.items():
            self.relation.addItem(label, key)
        answer.addWidget(self.relation)
        self.export_button = QPushButton("Export CSV…")
        self.export_button.clicked.connect(self.export)
        answer.addWidget(self.export_button)
        self.figure_button = QPushButton("Save figure…")
        self.figure_button.setToolTip("Save the chart of the answers as a PNG file.")
        self.figure_button.clicked.connect(self.save_figure)
        answer.addWidget(self.figure_button)
        answer.addStretch(1)
        layout.addLayout(answer)
        self.sentence = _plain(QLabel())
        layout.addWidget(self.sentence)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(240)
        self.table.setMinimumWidth(420)
        # The table and its chart side by side; the chart grows with the runs.
        results = QHBoxLayout()
        results.addWidget(self.table, 2)
        self.chart = QLabel()
        self.chart.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.chart.hide()
        results.addWidget(self.chart, 3, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(results, 1)
        self.add_condition(events.Condition("dissolved", "detected"))
        self._sync_noise()
        self._sync_actions()

    # ------------------------------------------------------------ series and jobs

    def set_series(self, series_root, series_id) -> None:
        """Show the jobs and saved questions of a series; nothing is read twice for the same one."""
        root = Path(series_root) if series_root else None
        if (root, series_id) == (self.series_root, self.series_id):
            return
        self.series_root, self.series_id = root, series_id
        self._read.cancel()
        self.series, self.answers, self.asked = [], [], None
        self._show_answers()
        self.load_jobs()
        self.load_saved_list()

    def load_jobs(self) -> None:
        """List the series' finished mass analyses (its registered jobs) and read the first one."""
        self.jobs.blockSignals(True)
        try:
            self.jobs.clear()
            rows, problem = [], ""
            if self.series_root is not None and self.series_id:
                try:
                    rows = list(self.jobs_reader(self.series_root, self.series_id))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    problem = f"The series' jobs could not be listed: {error}"
            for row in rows:
                self.jobs.addItem(str(row["name"]), str(row["root"]))
        finally:
            self.jobs.blockSignals(False)
        if self.series_root is None or not self.series_id:
            self.status.setText(NO_SERIES)
        elif problem:
            self.status.setText(problem)
        elif not rows:
            self.status.setText("This series has no registered mass analysis job yet. Use Choose job… "
                                "for a finished job folder.")
        self._job_chosen()

    def _choose_job(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Finished mass analysis job folder")
        if folder:
            index = self.jobs.findData(folder)
            if index < 0:
                self.jobs.addItem(Path(folder).name, folder)
                index = self.jobs.count() - 1
            self.jobs.setCurrentIndex(index)
            self._job_chosen()

    def _job_chosen(self, *_args) -> None:
        root = self.jobs.currentData()
        self.series, self.answers, self.asked = [], [], None
        self._show_answers()
        if not root:
            self._read.cancel()
            self._set_areas()
            self._sync_actions()
            return
        self.status.setText("Reading the mass analysis's frame records…")
        reader = self.reader

        def progress(done, total):
            if done == total or done % 25 == 0:
                try:
                    self._progress.emit(done, total)
                except RuntimeError:
                    pass  # The page closed meanwhile.

        self._read.start(lambda cancelled: reader(root, cancelled, progress))
        self._sync_actions()

    def _show_progress(self, done: int, total: int) -> None:
        if self._read.busy:
            self.status.setText(f"Reading the mass analysis's frame records: {done} of {total} frames…")

    def _read_done(self, value, error: str) -> None:
        if error or value is None:
            self.series = []
            self.status.setText("The mass analysis could not be read: " + (error or "no answer was returned"))
        else:
            self.series, source, facts = value
            frames = sum(len(row.frames) for row in self.series)
            self.status.setText(f"{len(self.series)} finished runs, {frames} frames. {source}. {facts}")
        self._set_areas()
        self._sync_actions()

    def _set_areas(self) -> None:
        areas = events.areas(self.series)
        for row in self.rows:
            row.set_areas(areas)

    # ---------------------------------------------------------------- question

    def add_condition(self, condition: events.Condition | None = None) -> ConditionRow:
        """Add a condition row (by default dissolved CO2 is first detected in the whole rig)."""
        row = ConditionRow(events.areas(self.series), condition)
        row.remove_requested.connect(self._remove_condition)
        self.rows.append(row)
        self.rows_layout.addWidget(row)
        self._sync_actions()
        return row

    def _remove_condition(self, row: ConditionRow) -> None:
        if len(self.rows) <= 1:
            return
        self.rows.remove(row)
        self.rows_layout.removeWidget(row)
        row.deleteLater()
        self._sync_actions()

    def _sync_noise(self, *_args) -> None:
        for widget in (self.noise_min, self.noise_unit, self.hold):
            widget.setEnabled(self.noise.isChecked())

    def question(self) -> events.EventQuestion:
        """The question shown, checked; ValueError says what is missing."""
        noise = self.noise.isChecked()
        return events.EventQuestion(
            self.name.text().strip() or "Question",
            tuple(row.condition() for row in self.rows),
            combine=self.combine.currentData(),
            noise_min=float(self.noise_min.value()) if noise else 0.0,
            noise_unit=self.noise_unit.currentData() if noise else "g",
            hold_frames=int(self.hold.value()) if noise else 1,
        ).check()

    def show_question(self, question: events.EventQuestion) -> None:
        """Show a saved question in the menus."""
        for row in list(self.rows):
            self.rows.remove(row)
            self.rows_layout.removeWidget(row)
            row.deleteLater()
        for condition in question.conditions:
            self.add_condition(condition)
        self.combine.setCurrentIndex(max(0, self.combine.findData(question.combine)))
        noise = question.noise_min > 0 or question.hold_frames > 1
        self.noise.setChecked(noise)
        if noise:
            self.noise_min.setValue(question.noise_min)
            self.noise_unit.setCurrentIndex(max(0, self.noise_unit.findData(question.noise_unit)))
            self.hold.setValue(question.hold_frames)
        self.name.setText(question.name)

    def _sync_actions(self) -> None:
        series = self.series_root is not None
        for row in self.rows:
            row.remove.setEnabled(len(self.rows) > 1)
        self.answer_button.setEnabled(bool(self.series))
        self.export_button.setEnabled(bool(self.answers))
        self.figure_button.setEnabled(bool(self.answers))
        self.save_button.setEnabled(series)
        self.load_button.setEnabled(series and self.saved.count() > 0)

    # ------------------------------------------------------------------- answers

    def answer(self) -> None:
        """Answer the question for every finished run of the chosen mass analysis."""
        try:
            question = self.question()
        except ValueError as error:
            self.sentence.setText(str(error))
            return
        self.asked, self.answers = question, events.answer_all(question, self.series)
        other = getattr(self, "_saved", {}).get(self.compare_with.currentData())
        self.comparison = None
        if other is not None:
            relation = self.relation.currentData()
            self.comparison = (f"{relation} {other.name}",
                               events.compare(self.answers, events.answer_all(other, self.series), relation))
        self._show_answers()

    def _show_answers(self) -> None:
        comparison = getattr(self, "comparison", None) if self.answers else None
        columns = COLUMNS + ((comparison[0][:1].upper() + comparison[0][1:],) if comparison else ())
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.horizontalHeaderItem(1).setToolTip("Hours since injection start")
        self.table.setRowCount(len(self.answers))
        for index, row in enumerate(self.answers):
            hours = "" if row.elapsed_h is None else f"{row.elapsed_h:.2f}"
            texts = [row.run.upper(), hours, row.frame or "", row.note]
            if comparison:
                result = comparison[1].get(row.run)
                texts.append("—" if result is None else ("Yes" if result else "No"))
            for column, text in enumerate(texts):
                self.table.setItem(index, column, QTableWidgetItem(text))
        if self.asked is None or not self.answers:
            self.sentence.clear()
            self.chart.clear()
            self.chart.hide()
        else:
            found = sum(row.frame is not None for row in self.answers)
            self.sentence.setText(f"{self.asked.text()}: found in {found} of {len(self.answers)} runs.")
            from .production_results_view import _pixmap
            self.chart.setPixmap(_pixmap(render_answers(self.asked, self.answers)))
            self.chart.show()
        self._sync_actions()

    def export(self) -> None:
        """Save the answers as a CSV file the user chooses."""
        if self.asked is None or not self.answers:
            return
        path, _filter = QFileDialog.getSaveFileName(self, "Export the answers", self.asked.name + ".csv",
                                                    "CSV (*.csv)")
        if path:
            Path(path).write_text(events.answers_csv(self.asked, self.answers,
                                                     comparison=getattr(self, "comparison", None)),
                                  encoding="utf-8")

    def save_figure(self) -> None:
        """Save the chart of the answers as a PNG file the user chooses."""
        pixmap = self.chart.pixmap()
        if self.asked is None or not self.answers or pixmap.isNull():
            return
        path, _filter = QFileDialog.getSaveFileName(self, "Save the figure", self.asked.name + ".png",
                                                    "PNG image (*.png)")
        if path and not pixmap.save(path, "PNG"):
            self.sentence.setText(f"The figure could not be saved to {path}.")

    # ---------------------------------------------------------- saved questions

    def load_saved_list(self) -> None:
        """List the series' saved questions; a file that cannot be read is named in the status."""
        self.saved.clear()
        self.compare_with.clear()
        self.compare_with.addItem("(nothing)", None)
        self._saved: dict[str, events.EventQuestion] = {}
        if self.series_root is not None:
            questions, problems = events.load_questions(self.series_root)
            for question in questions:
                self._saved[question.name] = question
                self.saved.addItem(question.name, question.name)
                self.compare_with.addItem(question.name, question.name)
            if problems:
                self.sentence.setText("Saved questions that could not be read: " + "; ".join(problems))
        self._sync_actions()

    def save(self) -> None:
        """Save the question by its name in the series folder; a name in use is replaced after asking."""
        if self.series_root is None:
            return
        if not self.name.text().strip():
            self.sentence.setText("Give the question a name before saving it.")
            return
        try:
            question = self.question()
            path = events.question_path(self.series_root, question.name)
            replace = path.exists()
            if replace and QMessageBox.question(
                    self, "Save question", f"Replace the saved question {question.name!r}?") \
                    != QMessageBox.StandardButton.Yes:
                return
            path = events.save_question(self.series_root, question, replace=replace)
        except (OSError, ValueError) as error:
            self.sentence.setText(f"The question was not saved: {error}")
            return
        self.load_saved_list()
        self.saved.setCurrentIndex(max(0, self.saved.findData(question.name)))
        self.sentence.setText(f"Saved as {path}.")

    def load_saved(self) -> None:
        """Show the chosen saved question."""
        question = getattr(self, "_saved", {}).get(self.saved.currentData())
        if question is not None:
            self.show_question(question)
