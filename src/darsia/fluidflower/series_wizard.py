"""New series wizard of the native application's Start page.

Three pages create a native series folder with
:mod:`darsia.fluidflower.series_creation`:

1. Series: identity, rig class, experiment inputs, and shared project/setup/results
   locations. Setup defaults to ``DarSIA`` and results to ``Results`` beside it.
2. Experiments: one run per experiment with its image folders and baseline
   photograph.
3. Create: the files that will be written, with a preview.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
    QWizard,
    QWizardPage,
)

from darsia.fluidflower import series_creation as sc
from darsia.fluidflower.series_paths import series_key
from darsia.fluidflower.templates import RIG_CLASSES

_ID_FROM_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")

# Physical Medium rigs reference the existing upstream profiles; extend this list.
PHYSICAL_RIGS = (("Albus", "FF AC"), ("Bilbo", "FF Bilbo"))


def _note(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setObjectName("secondaryText")
    return label


def _warning_color(widget: QWidget) -> str:
    foreground = widget.palette().color(QPalette.ColorRole.WindowText)
    base = QColor(201, 138, 20)
    return QColor(
        int(0.7 * base.red() + 0.3 * foreground.red()),
        int(0.7 * base.green() + 0.3 * foreground.green()),
        int(0.7 * base.blue() + 0.3 * foreground.blue()),
    ).name()


def _default_actor() -> str:
    try:
        from darsia.fluidflower.series_geometry import default_actor

        return default_actor()
    except Exception:  # noqa: BLE001 - only a default
        return ""


class SeriesPage(QWizardPage):
    """Name, ID, rig class and experiment folder."""

    def __init__(self, wizard: NewSeriesFolderWizard) -> None:
        super().__init__()
        self.owner = wizard
        self.setTitle("Series")
        self.setSubTitle(
            "Name and ID of the new series, the rig it was run in and the folder "
            "that holds its experiments."
        )
        form = QFormLayout(self)
        self.name = QLineEdit()
        self.name.setPlaceholderText("For example FF_XY (sand layers, CO2 injections)")
        self.name.textChanged.connect(self._name_changed)
        form.addRow("Name", self.name)
        self.series_id = QLineEdit()
        self.series_id.setPlaceholderText("For example FF_XY")
        self.series_id.setToolTip(
            "Starts with a letter; letters, digits, _ and -. Used to register "
            "the series and in its run IDs."
        )
        self.series_id.textEdited.connect(self._id_edited)
        self.series_id.textChanged.connect(self._changed)
        self._id_from_name = True
        form.addRow("Series ID", self.series_id)
        self.rig_class = QComboBox()
        self.rig_class.addItem("Choose the rig class…", "")
        for rig in RIG_CLASSES.values():
            self.rig_class.addItem(
                rig.label if rig.key == "medium" else rig.title(), rig.key
            )
            self.rig_class.setItemData(
                self.rig_class.count() - 1,
                rig.source or rig.label,
                Qt.ItemDataRole.ToolTipRole,
            )
        self.rig_class.currentIndexChanged.connect(self._changed)
        form.addRow("Rig class", self.rig_class)
        self.physical_rig_label = QLabel("Physical rig")
        self.physical_rig = QComboBox()
        self.physical_rig.addItem("Choose the physical rig…", "")
        for name, profile in PHYSICAL_RIGS:
            self.physical_rig.addItem(name, profile)
        self.physical_rig.setToolTip(
            "Use the existing rig profile in Preprocessing; depth and corrections "
            "are configured separately."
        )
        self.physical_rig.currentIndexChanged.connect(self._changed)
        form.addRow(self.physical_rig_label, self.physical_rig)
        folder_row = QHBoxLayout()
        self.folder = QLineEdit()
        self.folder.setPlaceholderText("Folder with the experiments' image folders")
        self.folder.textChanged.connect(self._changed)
        folder_row.addWidget(self.folder, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        folder_row.addWidget(browse)
        form.addRow("Experiment folder", folder_row)
        self.project_folder = QLineEdit()
        self.setup_folder = QLineEdit()
        self.results_folder = QLineEdit()
        self.jobs_folder = QLineEdit()
        for title, widget, placeholder in (
            ("Project folder", self.project_folder, "Shared folder for this series"),
            ("DarSIA setup folder", self.setup_folder, "<Project folder>/DarSIA"),
            ("Results folder", self.results_folder, "<Project folder>/Results"),
            ("Shared Jobs / workers folder", self.jobs_folder, "<Results folder>/jobs"),
        ):
            row = QHBoxLayout()
            widget.setPlaceholderText(placeholder)
            widget.textChanged.connect(self._changed)
            row.addWidget(widget, 1)
            button = QPushButton("Browse…")
            button.clicked.connect(lambda _checked=False, edit=widget, label=title: self._browse_location(edit, label))
            row.addWidget(button)
            form.addRow(title, row)
        self.project_folder.textChanged.connect(self._location_defaults)
        self.folder.textChanged.connect(self._location_defaults)
        self._default_project = self._default_setup = self._default_results = self._default_jobs = ""
        self.results_folder.textChanged.connect(self._job_location_default)
        self.created_by = QLineEdit(_default_actor())
        form.addRow("Created by", self.created_by)
        self.status = _note()
        form.addRow(self.status)

    def darsia_folder(self) -> Path | None:
        text = self.setup_folder.text().strip()
        return Path(text) if text else None

    def _location_defaults(self, *_args):
        source = self.folder.text().strip()
        if not self.project_folder.text().strip() or self.project_folder.text() == self._default_project:
            self._default_project = source
            self.project_folder.setText(source)
        project = self.project_folder.text().strip()
        for edit, attribute, name in ((self.setup_folder, "_default_setup", sc.SERIES_FOLDER_NAME),
                                      (self.results_folder, "_default_results", "Results")):
            if not edit.text().strip() or edit.text() == getattr(self, attribute):
                value = str(Path(project) / name) if project else ""
                setattr(self, attribute, value)
                edit.setText(value)

        self._job_location_default()

    def _job_location_default(self, *_args):
        if not self.jobs_folder.text().strip() or self.jobs_folder.text() == self._default_jobs:
            results = self.results_folder.text().strip()
            self._default_jobs = str(Path(results) / "jobs") if results else ""
            self.jobs_folder.setText(self._default_jobs)

    def _browse_location(self, edit, title):
        chosen = QFileDialog.getExistingDirectory(self, title, edit.text().strip() or self.folder.text().strip())
        if chosen:
            edit.setText(chosen)

    def problems(self) -> list[str]:
        problems: list[str] = []
        if not self.name.text().strip():
            problems.append("Enter a name for the series.")
        try:
            key = series_key(self.series_id.text().strip())
        except ValueError:
            key = ""
            problems.append(
                "The series ID starts with a letter and contains letters, digits, "
                "_ or - only."
            )
        if key and key in self.owner.registered_keys:
            problems.append(f"A series with the ID {self.series_id.text()} exists.")
        if not self.rig_class.currentData():
            problems.append("Choose the rig class.")
        elif (
            self.rig_class.currentData() == "medium"
            and not self.physical_rig.currentData()
        ):
            problems.append("Choose the physical rig for this Medium series.")
        folder = self.darsia_folder()
        if folder is None:
            problems.append("Choose the DarSIA setup folder.")
        elif not folder.parent.is_dir():
            problems.append(f"The folder {folder.parent} does not exist.")
        elif (folder / "series.toml").exists():
            problems.append(
                f"{folder} already holds a series; add it on Start with 'Add "
                "existing series folder'."
            )
        return problems

    def isComplete(self) -> bool:
        return not self.problems()

    def _changed(self, *_args: Any) -> None:
        is_medium = self.rig_class.currentData() == "medium"
        self.physical_rig_label.setVisible(is_medium)
        self.physical_rig.setVisible(is_medium)
        problems = self.problems()
        folder = self.darsia_folder()
        if problems:
            self.status.setText(" ".join(problems))
            self.status.setStyleSheet(f"color: {_warning_color(self)};")
        else:
            self.status.setText(
                f"Setup: {folder}. Results: {self.results_folder.text()}. Images stay in their selected folders."
            )
            self.status.setStyleSheet("")
        self.completeChanged.emit()

    def _name_changed(self, text: str) -> None:
        if self._id_from_name:
            match = _ID_FROM_NAME.search(text)
            self.series_id.setText(match.group(0) if match else "")
        self._changed()

    def _id_edited(self, _text: str) -> None:
        self._id_from_name = False

    def _browse(self) -> None:
        start = self.folder.text().strip() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(
            self, "Folder with the experiments", start
        )
        if chosen:
            self.folder.setText(str(Path(chosen)))


class _ScanWorker(QObject):
    """Find the runs below a folder, or scan one experiment folder."""

    # The runs found, and whether one experiment folder was scanned.
    finished = Signal(object, bool)
    failed = Signal(str)

    def __init__(self, folder: Path, *, single: bool) -> None:
        super().__init__()
        self.folder = Path(folder)
        self.single = single

    @Slot()
    def run(self) -> None:
        try:
            if self.single:
                runs = [sc.scan_run_folder(self.folder)]
            else:
                runs = sc.find_runs(self.folder)
        except (OSError, ValueError) as error:
            self.failed.emit(str(error))
            return
        except Exception as error:  # noqa: BLE001 - reported to the user
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.finished.emit(runs, self.single)


class ExperimentsPage(QWizardPage):
    """One run per experiment with its image folders and baseline."""

    COLUMNS = ("Run ID", "Image folders", "Images", "Baseline", "Comment")

    def __init__(self, wizard: NewSeriesFolderWizard) -> None:
        super().__init__()
        self.owner = wizard
        self.setTitle("Experiments")
        self.setSubTitle(
            "One run per experiment: its image folders (folders named after "
            "injections are ticked) and baseline photograph, the first photograph "
            "before the first injection. Images are not copied."
        )
        self.runs: list[sc.RunDraft] = []
        self._scanned_folder = ""
        self._menu_open = False
        self._thread: QThread | None = None
        self._worker: _ScanWorker | None = None
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self.table.horizontalHeader()
        for column in (0, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._item_changed)
        layout.addWidget(self.table, 1)
        buttons = QHBoxLayout()
        self.find_button = QPushButton("Find experiments")
        self.find_button.setToolTip(
            "One run per subfolder of the experiment folder that contains images, "
            "directly or in its subfolders"
        )
        self.find_button.clicked.connect(lambda: self.find_experiments(confirm=True))
        buttons.addWidget(self.find_button)
        self.add_button = QPushButton("Add experiment…")
        self.add_button.clicked.connect(self._add_experiment)
        buttons.addWidget(self.add_button)
        baseline = QPushButton("Choose baseline…")
        baseline.clicked.connect(self._choose_baseline)
        buttons.addWidget(baseline)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove)
        buttons.addWidget(remove)
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setMaximumWidth(110)
        self.busy.setTextVisible(False)
        self.busy.hide()
        buttons.addWidget(self.busy)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.status = _note()
        layout.addWidget(self.status)

    # ------------------------------------------------------------ scans

    def scanning(self) -> bool:
        try:
            return self._thread is not None and self._thread.isRunning()
        except RuntimeError:
            return False

    def wait_for_scan(self) -> None:
        """Let a running scan finish (before closing, and in tests)."""

        thread = self._thread
        if thread is not None:
            try:
                thread.wait()
            except RuntimeError:
                return
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()
        QApplication.processEvents()

    def _scan(self, folder: Path, *, single: bool, background: bool) -> None:
        if not background:
            worker = _ScanWorker(folder, single=single)
            worker.finished.connect(self._scan_done)
            worker.failed.connect(self._scan_failed)
            worker.run()
            return
        if self.scanning():
            return
        thread = QThread(self)
        worker = _ScanWorker(folder, single=single)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Bound methods of this page run in the GUI thread (queued).
        worker.finished.connect(self._scan_done)
        worker.failed.connect(self._scan_failed)
        # Direct: quit from the worker thread, so that wait() in the GUI
        # thread cannot block a queued quit.
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.failed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._scan_thread_finished)
        self._thread = thread
        self._worker = worker
        for button in (self.find_button, self.add_button):
            button.setEnabled(False)
        self.busy.show()
        self.status.setStyleSheet("")
        self.status.setText(
            f"Looking for experiments in {folder}… On a network drive this can "
            "take a minute."
        )
        thread.start()
        self.completeChanged.emit()

    @Slot(object, bool)
    def _scan_done(self, runs: list[sc.RunDraft], single: bool) -> None:
        if single:
            used = {row.run_id for row in self.runs}
            for run in runs:
                base, number = run.run_id, 2
                while run.run_id in used:
                    run.run_id = f"{base}_{number}"
                    number += 1
                used.add(run.run_id)
                self.runs.append(run)
        else:
            self.runs = list(runs)
        self._refresh()

    @Slot(str)
    def _scan_failed(self, message: str) -> None:
        self.status.setText(message)
        self.status.setStyleSheet(f"color: {_warning_color(self)};")

    def _scan_thread_finished(self) -> None:
        thread = self._thread
        self._thread = None
        self._worker = None
        if thread is not None:
            thread.deleteLater()
        self.busy.hide()
        for button in (self.find_button, self.add_button):
            button.setEnabled(True)
        self.completeChanged.emit()

    # ------------------------------------------------------------- runs

    def initializePage(self) -> None:
        folder = self.owner.series_page.folder.text().strip()
        self._refresh()
        if folder and folder != self._scanned_folder and not self.runs:
            self.find_experiments(confirm=False)

    def find_experiments(self, *, confirm: bool, background: bool = True) -> None:
        """List one run per experiment found in the experiment folder."""

        folder = self.owner.series_page.folder.text().strip()
        if not folder or self.scanning():
            return
        if confirm and self.runs:
            answer = QMessageBox.question(
                self,
                "Find experiments",
                "Replace the experiments listed with those found in the folder?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._scanned_folder = folder
        self._scan(Path(folder), single=False, background=background)

    def add_run_folder(self, folder: Path, *, background: bool = True) -> None:
        """Add the experiment in a folder as a run."""

        self._scan(Path(folder), single=True, background=background)

    def _add_experiment(self) -> None:
        start = self.owner.series_page.folder.text().strip() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(
            self, "Folder of one experiment", start
        )
        if chosen:
            self.add_run_folder(Path(chosen))

    def selected_row(self) -> int:
        rows = {index.row() for index in self.table.selectedIndexes()}
        return min(rows) if rows else -1

    def _choose_baseline(self) -> None:
        row = self.selected_row()
        if not 0 <= row < len(self.runs):
            return
        run = self.runs[row]
        start = str(run.folders[0]) if run.folders else ""
        chosen, _filter = QFileDialog.getOpenFileName(
            self,
            f"Baseline photograph of {run.run_id}",
            start,
            "Images (" + " ".join(f"*{suffix}" for suffix in sc.IMAGE_FORMATS) + ")",
        )
        if chosen:
            self.set_baseline(row, Path(chosen))

    def set_baseline(self, row: int, path: Path) -> None:
        run = self.runs[row]
        path = Path(path)
        if path.parent not in [Path(folder) for folder in run.folders]:
            run.set_folders([*run.folders, path.parent])
        run.baseline = path.name
        self._refresh()

    def _remove(self) -> None:
        row = self.selected_row()
        if 0 <= row < len(self.runs):
            del self.runs[row]
            self._refresh()

    # ---------------------------------------------------------- display

    def _folders_text(self, run: sc.RunDraft) -> str:
        if not run.folders:
            return "No folders: tick…"
        return ", ".join(run.folder_label(folder) for folder in run.folders)

    def _folders_button(self, row: int) -> QToolButton:
        run = self.runs[row]
        button = QToolButton()
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        button.setText(self._folders_text(run))
        button.setToolTip(
            "Tick the image folders of this run (in order of capture time)\n"
            + "\n".join(str(folder) for folder in run.folders)
        )
        menu = QMenu(button)
        menu.aboutToShow.connect(
            lambda target=menu, index=row: self._fill_folder_menu(target, index)
        )
        menu.aboutToHide.connect(self._folder_menu_closed)
        button.setMenu(menu)
        return button

    def _fill_folder_menu(self, menu: QMenu, row: int) -> None:
        self._menu_open = True
        menu.clear()
        run = self.runs[row]
        chosen = {Path(folder) for folder in run.folders}
        for info in run.candidates:
            box = QCheckBox(f"{run.folder_label(info.path)}   ({info.count} images)")
            box.setChecked(info.path in chosen)
            box.toggled.connect(
                lambda checked, index=row, path=info.path: self.set_folder(
                    index, path, checked
                )
            )
            holder = QWidget()
            line = QHBoxLayout(holder)
            line.setContentsMargins(10, 3, 12, 3)
            line.addWidget(box)
            action = QWidgetAction(menu)
            action.setDefaultWidget(holder)
            menu.addAction(action)

    def _folder_menu_closed(self) -> None:
        self._menu_open = False
        QTimer.singleShot(0, self._refresh)

    def set_folder(self, row: int, path: Path, used: bool) -> None:
        """Tick or untick an image folder of a run."""

        run = self.runs[row]
        folders = [Path(folder) for folder in run.folders if Path(folder) != Path(path)]
        if used:
            folders.append(Path(path))
        run.set_folders(folders)
        if self._menu_open:
            button = self.table.cellWidget(row, 1)
            if isinstance(button, QToolButton):
                button.setText(self._folders_text(run))
            self._update_status()
        else:
            self._refresh()

    def _refresh(self) -> None:
        if self._menu_open:
            return
        table = self.table
        table.blockSignals(True)
        try:
            for row in range(table.rowCount()):
                widget = table.cellWidget(row, 1)
                if widget is not None:
                    widget.hide()
                    table.removeCellWidget(row, 1)
            table.setRowCount(len(self.runs))
            for row, run in enumerate(self.runs):
                folders = QTableWidgetItem(self._folders_text(run))
                folders.setToolTip("\n".join(str(folder) for folder in run.folders))
                count = QTableWidgetItem(str(run.image_count or ""))
                for item in (folders, count):
                    item.setFlags(
                        Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                    )
                count.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                values = (
                    QTableWidgetItem(run.run_id),
                    folders,
                    count,
                    QTableWidgetItem(run.baseline),
                    QTableWidgetItem(run.comment),
                )
                for column, item in enumerate(values):
                    table.setItem(row, column, item)
                if run.candidates:
                    table.setCellWidget(row, 1, self._folders_button(row))
        finally:
            table.blockSignals(False)
        self._update_status()

    def _item_changed(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if not 0 <= row < len(self.runs):
            return
        run = self.runs[row]
        text = item.text().strip()
        if item.column() == 0:
            run.run_id = text
        elif item.column() == 3:
            run.baseline = text
        elif item.column() == 4:
            run.comment = text
        self._update_status()

    def problems(self) -> list[str]:
        return sc.check_runs(self.runs)

    def _update_status(self) -> None:
        problems = self.problems()
        if problems:
            self.status.setText(" ".join(problems[:4]))
            self.status.setStyleSheet(f"color: {_warning_color(self)};")
        else:
            images = sum(run.image_count for run in self.runs)
            self.status.setText(
                f"{len(self.runs)} run(s) with {images} images. Run IDs are lower "
                "case; the baseline is the first photograph unless you choose "
                "another."
            )
            self.status.setStyleSheet("")
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return not self.scanning() and bool(self.runs) and not self.problems()


class CreatePage(QWizardPage):
    """The files that will be written, with a preview."""

    def __init__(self, wizard: NewSeriesFolderWizard) -> None:
        super().__init__()
        self.owner = wizard
        self.setTitle("Create")
        self.setSubTitle("These files are written; nothing existing is changed.")
        self.setFinalPage(True)
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.summary)
        row = QHBoxLayout()
        row.addWidget(QLabel("Preview"))
        self.file_choice = QComboBox()
        self.file_choice.currentIndexChanged.connect(self._show_file)
        row.addWidget(self.file_choice, 1)
        layout.addLayout(row)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.preview.setFont(font)
        layout.addWidget(self.preview, 1)
        layout.addWidget(
            _note(
                "Next: Setup > Protocols writes the imaging protocols and templates "
                "for the injection and pressure-temperature protocols; Geometry and "
                "facies marks the rig frame and the sand layers."
            )
        )
        self._files: dict[str, str] = {}

    def initializePage(self) -> None:
        draft = self.owner.draft()
        self._files = sc.series_files(draft)
        rig = RIG_CLASSES.get(draft.rig_class)
        self.summary.setText(
            f"Series folder: {draft.darsia_folder}\n"
            f"Project folder: {sc.project_folder(draft)}\n"
            f"Results folder: {sc.results_folder(draft)}\n"
            f"Shared Jobs / workers folder: {sc.jobs_folder(draft)}\n"
            f"Rig class: {rig.title() if rig else draft.rig_class}\n"
            f"Runs: {', '.join(run.run_id for run in draft.runs)}"
        )
        self.file_choice.blockSignals(True)
        self.file_choice.clear()
        for name in self._files:
            self.file_choice.addItem(name)
        self.file_choice.blockSignals(False)
        self.file_choice.setCurrentIndex(0)
        self._show_file()

    def _show_file(self, *_args: Any) -> None:
        self.preview.setPlainText(self._files.get(self.file_choice.currentText(), ""))


class NewSeriesFolderWizard(QWizard):
    """Create a native series folder; ``created`` holds it after acceptance."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        registered_ids: Sequence[str] = (),
        experiment_folder: Path | None = None,
    ) -> None:
        super().__init__(parent)
        # Aero needs a native Windows handle and can leave the controls unpainted.
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setWindowTitle("New series")
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self.resize(980, 640)
        self.registered_keys = set()
        for value in registered_ids:
            try:
                self.registered_keys.add(series_key(value))
            except ValueError:
                continue
        self.created: Path | None = None
        self.series_page = SeriesPage(self)
        self.experiments_page = ExperimentsPage(self)
        self.create_page = CreatePage(self)
        for page in (self.series_page, self.experiments_page, self.create_page):
            self.addPage(page)
        self.setButtonText(QWizard.WizardButton.FinishButton, "Create series")
        if experiment_folder is not None:
            self.series_page.folder.setText(str(experiment_folder))
        self.series_page._changed()

    def draft(self) -> sc.SeriesDraft:
        page = self.series_page
        return sc.SeriesDraft(
            series_id=page.series_id.text().strip(),
            name=page.name.text().strip(),
            rig_class=str(page.rig_class.currentData() or ""),
            darsia_folder=page.darsia_folder() or Path(),
            runs=self.experiments_page.runs,
            created_by=page.created_by.text().strip(),
            rig_profile=(
                str(page.physical_rig.currentData() or "")
                if page.rig_class.currentData() == "medium"
                else ""
            ),
            project_folder=Path(page.project_folder.text()) if page.project_folder.text().strip() else None,
            results_folder=Path(page.results_folder.text()) if page.results_folder.text().strip() else None,
            jobs_folder=Path(page.jobs_folder.text()) if page.jobs_folder.text().strip() else None,
        )

    def create(self) -> Path | None:
        """Write the series folder; None (after a message) if it failed."""

        try:
            self.created = sc.create_series_folder(
                self.draft(), registered_ids=sorted(self.registered_keys)
            )
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could not create the series", str(error))
            return None
        return self.created

    def accept(self) -> None:
        if self.create() is None:
            return
        super().accept()

    def done(self, result: int) -> None:
        self.experiments_page.wait_for_scan()
        super().done(result)


__all__ = ["CreatePage", "ExperimentsPage", "NewSeriesFolderWizard", "SeriesPage"]
