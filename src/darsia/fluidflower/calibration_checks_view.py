"""Checks page of the Phase-map method: held-out checks 1-3 of check-score files.

The page opens one check-score file, or a second one to compare side by side,
and shows what darsia.fluidflower.calibration_checks reads from them. Opening
a file computes nothing, writes nothing and stores no setting.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .calibration_check_calculation_ui import CalculateChecksDialog
from .workflow_section import WorkflowSection

from darsia.fluidflower.calibration_checks import (
    CHECK_TITLES,
    RULE_TREE_HEADERS,
    CheckScoreError,
    CheckScores,
    check1_note,
    check1_table,
    file_titles,
    load_check_scores,
    rule_check_note,
    rule_tree,
    status_text,
    summary_rows,
)

EMPTY_TEXT = (
    "Open a check-score file (JSON) to see the held-out checks 1-3 of a calibration: "
    "the verdict, check 1 per time with M/d per run and group, and checks 2 and 3 per "
    "rule and time with their per-run rows. Open a second file to compare two "
    "calibrations side by side. Opening a file computes nothing, writes nothing and "
    "stores no setting."
)
FILE_FILTER = "Check-score files (*.json);;All files (*)"
WARNING_STYLE = "color: #b77900;"


def _label(text: str = "", *, bold: bool = False) -> QLabel:
    """A plain, wrapped, selectable text line of the page."""
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    # A long file name has no break; it must not set the window's width.
    policy = label.sizePolicy()
    policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
    label.setSizePolicy(policy)
    if bold:
        font = label.font()
        font.setBold(True)
        label.setFont(font)
    return label


def _table() -> QTableWidget:
    """A read-only table without row numbers."""
    table = QTableWidget(0, 0)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.verticalHeader().hide()
    table.horizontalHeader().setStretchLastSection(True)
    table.setWordWrap(False)
    return table


def _fill_table(table: QTableWidget, headers, rows) -> None:
    """Show rows of texts under headers; each cell's text is also its tooltip."""
    table.clear()
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    table.setRowCount(len(rows))
    for row, values in enumerate(rows):
        for column, text in enumerate(values):
            item = QTableWidgetItem(text)
            item.setToolTip(text)
            table.setItem(row, column, item)
    table.resizeColumnsToContents()


def _fill_tree(tree: QTreeWidget, headers, rows) -> None:
    """Show nested rows expanded; deciding rule bold, LORO-predicted runs italic."""
    tree.clear()
    tree.setColumnCount(len(headers))
    tree.setHeaderLabels(list(headers))

    def add(parent, row):
        """Add one row under parent (None for the top level), then its children."""
        item = QTreeWidgetItem(list(row.texts))
        if row.mark:
            font = item.font(0)
            font.setBold(row.mark == "deciding")
            font.setItalic(row.mark == "loro")
            for column in range(len(row.texts)):
                item.setFont(column, font)
        for column, text in enumerate(row.texts):
            item.setToolTip(column, text)
        if parent is None:
            tree.addTopLevelItem(item)
        else:
            parent.addChild(item)
        for child in row.children:
            add(item, child)

    for row in rows:
        add(None, row)
    tree.expandAll()
    for column in range(len(headers)):
        tree.resizeColumnToContents(column)


class CalibrationChecksPage(QWidget):
    """Read-only view of the held-out checks 1-3 in one or two check-score files."""

    def __init__(self, parent=None):
        """Build the empty page; no file is read until one is opened."""
        super().__init__(parent)
        self.setObjectName("calibrationChecksPage")
        self._scores: list[CheckScores | None] = [None, None]
        layout = QVBoxLayout(self)
        layout.addWidget(_label("Checks", bold=True))
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.splitter, 1)
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.controls_scroll.setMinimumWidth(440)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 8, 0)
        controls_layout.setSpacing(8)
        self._calculation_dialog = CalculateChecksDialog(self, embedded=True)
        self._calculation_dialog.calculated.connect(self._calculated)
        self.calculation_section = WorkflowSection(
            1, "Calculate checks…", self._calculation_dialog)
        controls_layout.addWidget(self.calculation_section)
        self.calculate_button = self._calculation_dialog.calculate_button
        file_controls = QWidget()
        file_layout = QVBoxLayout(file_controls)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.results_section = WorkflowSection(2, "Open check results…", file_controls)
        controls_layout.addWidget(self.results_section)
        controls_layout.addStretch(1)
        self.controls_scroll.setWidget(controls)
        self.splitter.addWidget(self.controls_scroll)
        results = QWidget()
        layout = QVBoxLayout(results)
        layout.setContentsMargins(0, 0, 0, 0)
        self.splitter.addWidget(results)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([500, 800])
        self.empty_note = _label(EMPTY_TEXT)
        layout.addWidget(self.empty_note)

        files = QGridLayout()
        self.path_fields = (QLineEdit(), QLineEdit())
        self.path_fields[0].setPlaceholderText("Path of a check-score file (JSON)")
        self.path_fields[1].setPlaceholderText(
            "Path of a second check-score file to compare"
        )
        self.open_button = QPushButton("Open check results…")
        self.open_button.setToolTip("Choose a check-score file; nothing is written.")
        self.second_button = QPushButton("Open second file…")
        self.second_button.setToolTip(
            "Choose a second check-score file to show side by side, for example "
            "the same checks with a variant switched on and off."
        )
        self.remove_button = QPushButton("Remove second file")
        files.addWidget(QLabel("Check results:"), 0, 0, 1, 2)
        files.addWidget(self.path_fields[0], 1, 0)
        files.addWidget(self.open_button, 1, 1)
        files.addWidget(QLabel("Compare with:"), 2, 0, 1, 2)
        files.addWidget(self.path_fields[1], 3, 0)
        files.addWidget(self.second_button, 3, 1)
        files.addWidget(self.remove_button, 4, 1)
        files.setColumnStretch(0, 1)
        file_layout.addLayout(files)
        self.open_button.clicked.connect(self._choose_first)
        self.second_button.clicked.connect(self._choose_second)
        self.remove_button.clicked.connect(self.remove_second)
        self.path_fields[0].returnPressed.connect(self._open_first_field)
        self.path_fields[1].returnPressed.connect(self._open_second_field)

        self.message = _label()
        self.message.setStyleSheet(WARNING_STYLE)
        self.message.hide()
        layout.addWidget(self.message)
        self.verdict = _label(bold=True)
        layout.addWidget(self.verdict)

        self.tabs = QTabWidget()
        # With two files, the tab bar's corner chooses whose details the tabs show.
        self.details_row = QWidget()
        details = QHBoxLayout(self.details_row)
        details.setContentsMargins(0, 0, 0, 0)
        details.addWidget(QLabel("Details for:"))
        self.details_choice = QComboBox()
        self.details_choice.setToolTip(
            "The file whose checks 1-3 the tabs show in detail."
        )
        # Long file names must not widen the window.
        self.details_choice.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.details_choice.setMinimumContentsLength(18)
        details.addWidget(self.details_choice)
        self.details_choice.currentIndexChanged.connect(self._show_details)
        self.tabs.setCornerWidget(self.details_row, Qt.Corner.TopRightCorner)
        self.summary = _table()
        self.tabs.addTab(self.summary, "Summary")
        check1 = QWidget()
        column = QVBoxLayout(check1)
        self.check1_note = _label()
        column.addWidget(self.check1_note)
        self.check1_table = _table()
        column.addWidget(self.check1_table, 1)
        self.tabs.addTab(check1, CHECK_TITLES["check1"])
        self.rule_views = []
        for key in ("check2", "check3"):
            page = QWidget()
            column = QVBoxLayout(page)
            note = _label()
            column.addWidget(note)
            tree = QTreeWidget()
            tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            tree.setUniformRowHeights(True)
            column.addWidget(tree, 1)
            self.rule_views.append((note, tree))
            self.tabs.addTab(page, CHECK_TITLES[key])
        layout.addWidget(self.tabs, 1)
        # Keeps the empty page at the top; the tabs take the space once shown.
        layout.addItem(
            QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )
        self._render()

    # ------------------------------------------------------------- opening

    @Slot(object)
    def _calculated(self, result):
        self.open_file(result["scores"]["path"])

    def scores(self) -> tuple[CheckScores, ...]:
        """The files shown now: the check results, then the comparison file if any."""
        return tuple(scores for scores in self._scores if scores is not None)

    def open_file(self, path, *, second: bool = False) -> bool:
        """Read a check-score file into the first or the comparison place.

        The page shows a refused or unreadable file's reason and keeps what it
        showed before. Returns True when the file was opened.
        """
        index = 1 if second else 0
        text = str(path)
        self.path_fields[index].setText(text)
        if second and self._scores[0] is None:
            return self._refuse("Open the check results before a file to compare.")
        name = Path(text).name or text
        try:
            scores = load_check_scores(Path(text).expanduser())
        except CheckScoreError as error:
            return self._refuse(
                f"{name} was not opened. Not a check-score file: {error}"
            )
        except OSError as error:
            reason = error.strerror or str(error)
            return self._refuse(
                f"{name} was not opened. The file could not be read: {reason}."
            )
        self._scores[index] = scores
        self.message.hide()
        self.message.clear()
        self._render()
        return True

    def remove_second(self) -> None:
        """Stop comparing; the check results stay shown."""
        self._scores[1] = None
        self.path_fields[1].clear()
        self._render()

    def _refuse(self, text: str) -> bool:
        """Show why a file was not opened; what was shown stays. Returns False."""
        self.message.setText(text)
        self.message.show()
        return False

    def _choose(self, second: bool) -> None:
        """Ask for a file, starting in the folder of the path already given."""
        current = self.path_fields[1 if second else 0].text().strip().strip('"')
        folder = str(Path(current).parent) if current else ""
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Open second check-score file" if second else "Open check results",
            folder,
            FILE_FILTER,
        )
        if path:
            self.open_file(path, second=second)

    def _choose_first(self) -> None:
        """Open check results…: choose the first file."""
        self._choose(False)

    def _choose_second(self) -> None:
        """Open second file…: choose the file to compare."""
        self._choose(True)

    def _open_field(self, second: bool) -> None:
        """Open the path typed or pasted into a path field, quotes removed."""
        text = self.path_fields[1 if second else 0].text().strip().strip('"')
        if text:
            self.open_file(text, second=second)

    def _open_first_field(self) -> None:
        """Enter in the first path field."""
        self._open_field(False)

    def _open_second_field(self) -> None:
        """Enter in the second path field."""
        self._open_field(True)

    # ------------------------------------------------------------- display

    def _render(self) -> None:
        """Show the empty state, or the verdicts, the summary and the details."""
        files = self.scores()
        first_open = self._scores[0] is not None
        self.empty_note.setVisible(not files)
        self.tabs.setVisible(bool(files))
        self.verdict.setVisible(bool(files))
        self.path_fields[1].setEnabled(first_open)
        self.second_button.setEnabled(first_open)
        self.remove_button.setEnabled(self._scores[1] is not None)
        self.details_row.setVisible(len(files) > 1)
        if not files:
            self.verdict.clear()
            return
        titles = file_titles(files)
        lines = []
        for title, scores in zip(titles, files):
            line = f"{title}: verdict {status_text(scores.verdict)}"
            if scores.integrity_problems:
                line += (
                    f", {len(scores.integrity_problems)} integrity problems reported"
                )
            lines.append(line)
        self.verdict.setText("\n".join(lines))
        self.verdict.setStyleSheet(
            WARNING_STYLE if any(scores.integrity_problems for scores in files) else ""
        )
        rows = summary_rows(files)
        _fill_table(
            self.summary,
            ("Item", *titles),
            [(label, *values) for label, values in rows],
        )
        with QSignalBlocker(self.details_choice):
            index = self.details_choice.currentIndex()
            self.details_choice.clear()
            self.details_choice.addItems(list(titles))
            self.details_choice.setCurrentIndex(
                index if 0 <= index < len(titles) else 0
            )
        self._show_details()

    def _show_details(self, *_args) -> None:
        """Fill the check tabs with the file chosen under Details for."""
        files = self.scores()
        if not files:
            return
        index = self.details_choice.currentIndex()
        scores = files[index if 0 <= index < len(files) else 0]
        self.check1_note.setText(check1_note(scores.check1))
        _fill_table(self.check1_table, *check1_table(scores.check1))
        for check, (note, tree) in zip(scores.rule_checks, self.rule_views):
            note.setText(rule_check_note(check))
            _fill_tree(tree, RULE_TREE_HEADERS, rule_tree(check))

    def _calculate_checks(self):
        """Reveal the existing calculation controls without opening a window."""
        self.calculation_section.set_expanded(True)
        self.controls_scroll.ensureWidgetVisible(self.calculation_section)
        self._calculation_dialog.fields["series"].setFocus()

    def can_close(self):
        dialog = self._calculation_dialog
        return dialog is None or dialog.can_close()
