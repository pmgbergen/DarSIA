"""Native calibration observation choices; phase-history inputs stay intact."""
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QFormLayout, QLabel, QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .calibration_image_selection import (
    DEFAULT_IMAGES_PER_RUN, _targets, calibration_selection_preview,
    save_calibration_image_selection,
)
from .series_method_calls import SeriesMethodCalls
from .ui_metadata_preview import MetadataPreview


class CalibrationImageSelection(QWidget):
    saved = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.root = self.snapshot = None
        self.generation = 0
        self.preview = MetadataPreview(self)
        self.preview.completed.connect(self._read)
        self.calls = SeriesMethodCalls(self, verify_approval=False)
        self.calls.action_finished.connect(self._saved)
        layout = QVBoxLayout(self)
        title = QLabel("Image selection")
        title.setStyleSheet("font-weight: bold")
        layout.addWidget(title)
        calibration_title = QLabel("Calibration images")
        calibration_title.setStyleSheet("font-weight: bold")
        layout.addWidget(calibration_title)
        explanation = QLabel(
            "Choose the number of observations used to fit each run. Comparable nominal times are "
            "spread over the available duration, with more detail early in the experiment. "
            "The fit windows in Model parameters still apply. Clear-water/baseline references and "
            "the injection-end anchor remain available in addition to the observations.")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        history = QLabel(
            "Phase inference keeps the complete chronological image sequence and its previous-map history. "
            "This selection does not change Setup's colour-path images or the production image selection. "
            "Production runs its own full selected sequence; calibration phase maps are not reused when "
            "that sequence changes.")
        history.setWordWrap(True)
        layout.addWidget(history)
        form = QFormLayout()
        self.count = QSpinBox()
        self.count.setRange(3, 1000)
        self.count.setValue(DEFAULT_IMAGES_PER_RUN)
        self.count.valueChanged.connect(self._changed)
        form.addRow("Target images per run", self.count)
        layout.addLayout(form)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(("Run", "Candidate targets", "Source/history images", "Nominal times [h]"))
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)
        self.save_button = QPushButton("Save calibration image selection")
        self.save_button.clicked.connect(self._save)
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)
        self.status = QLabel("Choose a series on Start.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        # The phase model's training images are shown, not chosen: they are the
        # frames with phase labels, plus each run's baseline.
        training_title = QLabel("Phase-model training images")
        training_title.setStyleSheet("font-weight: bold")
        layout.addWidget(training_title)
        training_note = QLabel(
            "The phase model named under Calibration Method is trained on each run's baseline and the images "
            "that have phase labels. This list is not a free choice: to train on other images, label them "
            "under Phase model → Label phases, then train a new model version. Calibration images may "
            "overlap with these. Train model version also prepares each run's baseline and the frames the "
            "labels' history needs, so it prepares more images than are listed here.")
        training_note.setWordWrap(True)
        layout.addWidget(training_note)
        self.training_status = QLabel()
        self.training_status.setWordWrap(True)
        layout.addWidget(self.training_status)
        self.training_table = QTableWidget(0, 4)
        self.training_table.setHorizontalHeaderLabels(("Run", "Labelled images", "Labels", "Image times [h]"))
        self.training_table.verticalHeader().hide()
        self.training_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.training_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.training_table, 1)
        self.training_preview = MetadataPreview(self)
        self.training_preview.completed.connect(self._training_read)

    def set_series(self, root):
        self.root = Path(root) if root is not None else None
        self.generation += 1
        self.snapshot = None
        self.preview.cancel()
        self.training_preview.cancel()
        self.calls.set_context(self.root, None, None, None)
        self.table.setRowCount(0)
        self.training_table.setRowCount(0)
        self.count.setEnabled(False)
        self.save_button.setEnabled(False)
        if self.root is None:
            self.status.setText("Choose a series on Start.")
            self.training_status.setText("")
            return
        root = self.root
        self.status.setText("Reading saved image times…")
        self.preview.start(lambda cancelled: calibration_selection_preview(root, cancelled))
        self.training_status.setText("Reading the phase model's labelled training images…")
        from .phase_training_images import series_training_images
        self.training_preview.start(lambda _cancelled: (root, series_training_images(root)))

    def _training_read(self, value, error):
        """Show, per run, the labelled images the series' phase model is trained on."""
        if error:
            self.training_status.setText("The phase model's training images could not be read: " + error)
            return
        if value is None or value[0] != self.root:
            return
        result = value[1]
        runs = result["runs"]
        images = sum(len(row["times_h"]) for row in runs.values())
        text = f"Model {result['model']}: {images} labelled images in {len(runs)} runs."
        if result["unreadable"]:
            text += f" {len(result['unreadable'])} label files could not be read: " + result["unreadable"][0]
        self.training_status.setText(text)
        self.training_table.setRowCount(len(runs))
        for index, (run, row) in enumerate(runs.items()):
            values = (run, str(len(row["times_h"])), str(row["labels"]),
                      ", ".join(f"{time:g}" for time in row["times_h"]))
            for column, cell in enumerate(values):
                item = QTableWidgetItem(cell)
                item.setToolTip(cell)
                self.training_table.setItem(index, column, item)
        self.training_table.resizeColumnsToContents()
        self.training_table.setColumnWidth(3, max(240, self.training_table.columnWidth(3)))

    def _read(self, value, error):
        if error:
            self.status.setText("Could not read calibration image choices: " + error)
            return
        if value is None or value["root"] != self.root:
            return
        self.snapshot = value
        self.calls.set_context(self.root, value["count"], value["raw"], None)
        with QSignalBlocker(self.count):
            self.count.setValue(value["count"])
        self.count.setEnabled(True)
        self.save_button.setEnabled(True)
        self._show_targets()
        self.status.setText(f"Current choice: up to {value['count']} target images per run. "
            "Preview uses recorded camera targets; the statistics collected with the generated phase maps and fit windows determine "
            "the final eligible observations. Opening this page starts no job.")

    def _show_targets(self):
        if self.snapshot is None:
            return
        rows = self.snapshot["runs"]
        targets = _targets([time for row in rows.values() for time in row["available_targets"]], self.count.value())
        self.table.setRowCount(len(rows))
        for index, (run, row) in enumerate(rows.items()):
            chosen = [time for time in targets if time in row["available_targets"]]
            values = (run, str(len(chosen)), str(row["source_images"]), ", ".join(f"{time:g}" for time in chosen) or "No recorded eligible targets")
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.table.setItem(index, column, item)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(3, max(240, self.table.columnWidth(3)))

    def _changed(self, *_args):
        if self.snapshot is None:
            return
        self._show_targets()
        self.status.setText("Unsaved calibration image count." if self.count.value() != self.snapshot["count"]
                            else "The saved calibration image count is selected.")

    def _save(self):
        if self.snapshot is None or not self.save_button.isEnabled():
            return
        root, generation, raw = self.root, self.generation, self.snapshot["raw"]
        count = self.count.value()
        def save():
            save_calibration_image_selection(root, count, expected_bytes=raw,
                context_current=lambda: self.root == root and self.generation == generation
                and getattr(self.parent(), "_method_series_root", lambda: root)() == root)
            return calibration_selection_preview(root)
        if self.calls.start_action("calibration_images", save, self.calls.context):
            self.count.setEnabled(False)
            self.save_button.setEnabled(False)
            self.status.setText("Saving calibration image selection…")

    def _saved(self, result):
        if not result["current"]:
            return
        if result["error"]:
            self.status.setText("Calibration image selection was not saved: " + result["error"])
            self.count.setEnabled(True)
            self.save_button.setEnabled(True)
            return
        self._read(result["result"], "")
        self.status.setText(f"✓ Calibration image selection saved: up to {self.snapshot['count']} "
            "observations per run. Complete phase history is retained. No job started.")
        self.saved.emit(self.root)

    def can_close(self):
        self.preview.cancel()
        self.training_preview.cancel()
        self.calls.stop_pending()
        return self.calls.can_close()
