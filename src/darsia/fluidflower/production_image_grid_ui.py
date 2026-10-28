"""Production-purpose choices inside the existing image-selection editor."""
from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Slot
from PySide6.QtWidgets import (QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .production_image_grid import propose_production_grid, save_production_grid


class ProductionGridControls(QObject):
    def __init__(self, dialog, form):
        super().__init__(dialog)
        self.dialog, self.proposed, self.metadata, self.quality = dialog, None, None, None
        self._thread = self._worker = None
        self.group = QGroupBox("Production image selection (separate from calibration)")
        self.group.setCheckable(True)
        self.group.setChecked(False)
        outer = QVBoxLayout(self.group)
        body = QWidget()
        layout = QVBoxLayout(body)
        outer.addWidget(body)
        body.hide()
        self.group.toggled.connect(body.setVisible)
        note = QLabel("One logarithmic time grid for all runs. The baseline is additional; "
            "shorter runs stop at their last photograph. Saving preserves the calibration image choices.")
        note.setWordWrap(True)
        layout.addWidget(note)
        fields = QFormLayout()
        self.count, self.late = QSpinBox(), QSpinBox()
        self.count.setRange(2, 10000)
        self.count.setValue(80)
        self.late.setRange(0, 10000)
        self.late.setValue(10)
        self.first, self.through = QDoubleSpinBox(), QDoubleSpinBox()
        self.first.setRange(.000001, 10000.)
        self.first.setDecimals(6)
        self.first.setValue(.25)
        self.through.setRange(.000001, 10000.)
        self.through.setValue(48.)
        saved = dialog.document.get("production_image_selection", {}).get("parameters", {})
        for key, control in (("main_count", self.count), ("late_count", self.late),
                             ("first_h", self.first), ("through_h", self.through)):
            if key in saved:
                control.setValue(saved[key])
        for title, control in (("Images through the main interval", self.count),
                ("First target (hours after injection)", self.first),
                ("Main interval ends (hours)", self.through),
                ("Maximum images after this interval", self.late)):
            fields.addRow(title, control)
            control.valueChanged.connect(self.invalidate)
        layout.addLayout(fields)
        row = QHBoxLayout()
        self.propose_button = QPushButton("Propose production grid")
        self.save_button = QPushButton("Save production selection")
        self.save_button.setEnabled(False)
        row.addWidget(self.propose_button)
        row.addWidget(self.save_button)
        layout.addLayout(row)
        self.status = QLabel("Production selection has not been proposed.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(("Run", "Baseline", "Main images", "Late images", "Total", "Missing targets", "Quality not recorded"))
        self.table.setMaximumHeight(160)
        layout.addWidget(self.table)
        form.addRow(self.group)
        self.propose_button.clicked.connect(lambda: dialog._clicked("Production grid not proposed", self.propose))
        self.save_button.clicked.connect(lambda: dialog._clicked("Production selection not saved", self.save))

    @property
    def busy(self):
        return self._thread is not None

    def parameters(self):
        return dict(main_count=self.count.value(), first_h=self.first.value(),
                    through_h=self.through.value(), late_count=self.late.value())

    def invalidate(self, *_args):
        self.proposed = None
        self.save_button.setEnabled(False)

    def propose(self):
        if self.busy:
            return
        if self.dialog._dirty_runs:
            raise ValueError("Save or reload the current calibration draft before proposing production")
        from .jobs_panel import _CallWorker
        root, reader = self.dialog.root, self.dialog.metadata_reader
        runs = [self.dialog.run_combo.itemData(index) for index in range(self.dialog.run_combo.count())]
        saved = deepcopy(self.dialog.saved_table)
        parameters = self.parameters()
        def prepare():
            from .image_selection_proposal import load_saved_quality
            metadata, quality = {}, {}
            for run in runs:
                metadata[run] = reader(root, run)
                bound = saved.get("runs", {}).get(run, {}).get("quality_source")
                if bound:
                    quality[run] = load_saved_quality(bound["path"], metadata[run], expected=bound)
            return dict(proposal=propose_production_grid(metadata, quality=quality, **parameters),
                        metadata=metadata, quality=quality)
        self.status.setText("Reading protocol metadata for the shared production grid; images are not decoded...")
        self.invalidate()
        self.propose_button.setEnabled(False)
        thread, worker = QThread(self.dialog), _CallWorker(prepare)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._ready)
        worker.failed.connect(self.status.setText)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._closed)
        thread.finished.connect(thread.deleteLater)
        self._thread, self._worker = thread, worker
        thread.start()

    @Slot(object)
    def _ready(self, value):
        if value["proposal"]["parameters"] != self.parameters():
            self.status.setText("Production parameters changed; propose again")
            return
        self.proposed, self.metadata, self.quality = value["proposal"], value["metadata"], value["quality"]
        self.table.setRowCount(len(self.proposed["runs"]))
        for index, (run, row) in enumerate(self.proposed["runs"].items()):
            counts = row["counts"]
            for column, item in enumerate((run.upper(), counts["baselines"], counts["main"],
                                          counts["late"], counts["selected"], counts["missing"], row["unrecorded_quality"])):
                self.table.setItem(index, column, QTableWidgetItem(str(item)))
        self.table.resizeColumnsToContents()
        self.status.setText(f"{len(self.proposed['main_target_hours'])} shared main targets and "
            f"{len(self.proposed['late_target_hours'])} later targets, plus each baseline. "
            "Missing times remain explicit. Inspect the counts, then Save production selection.")
        self.save_button.setEnabled(True)

    @Slot()
    def _closed(self):
        self._thread.wait()
        self._thread = self._worker = None
        self.propose_button.setEnabled(True)

    def save(self):
        if self.busy or self.proposed is None:
            raise ValueError("Propose and inspect the production grid before saving")
        if self.dialog._dirty_runs:
            raise ValueError("Save or reload the calibration draft first")
        result = save_production_grid(self.dialog.root, self.proposed, self.metadata,
            quality=self.quality, expected_bytes=self.dialog.expected_bytes)
        self.dialog.expected_bytes = result["series_bytes"]
        self.save_button.setEnabled(False)
        self.status.setText("Production selection saved. Calibration choices are unchanged. "
            "Production requires phase maps for this saved selection; no calculation was started.")
        self.dialog.saved.emit(str(Path(self.dialog.root) / "series.toml"))
        return result
