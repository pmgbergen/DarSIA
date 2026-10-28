"""Measured-frame adapter for the existing analysis-domain editor and store."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, QRectF, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QGraphicsRectItem, QMessageBox,
    QPushButton, QSplitter, QTableWidgetItem,
)

from . import metric_analysis_domains as data
from .map_placement import read_run_geometry
from .ui.analysis_domains import AnalysisDomainDialog, AnalysisDomainImageView


class _Call(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, function):
        super().__init__()
        self.function = function

    @Slot()
    def run(self):
        try:
            result = self.function()
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
        else:
            self.finished.emit(result)


class MetricBoxCanvas(AnalysisDomainImageView):
    rectangle_edited = Signal(str, object)

    def __init__(self):
        super().__init__()
        self._edit = None

    def set_mode(self, mode):
        if mode != "edit":
            return super().set_mode(mode)
        self.mode = mode
        self.setDragMode(self.DragMode.NoDrag)

    def mousePressEvent(self, event):
        if self.mode == "edit" and event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if isinstance(item, QGraphicsRectItem) and item.data(0):
                point = self.mapToScene(event.position().toPoint())
                rect = QRectF(item.rect())
                corners = (rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft())
                distance = [np.hypot(*(self.mapFromScene(c)-event.position().toPoint()).toTuple()) for c in corners]
                corner = int(np.argmin(distance)) if min(distance) < 12 else None
                self._edit = (item, rect, point, corner)
                self.domain_selected.emit(str(item.data(0)))
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._edit:
            item, original, start, corner = self._edit
            delta = self.mapToScene(event.position().toPoint())-start
            rect = QRectF(original)
            if corner is None:
                rect.translate(delta)
            else:
                setters = (rect.setTopLeft, rect.setTopRight, rect.setBottomRight, rect.setBottomLeft)
                points = (original.topLeft(), original.topRight(), original.bottomRight(), original.bottomLeft())
                setters[corner](points[corner]+delta)
            item.setRect(rect.normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._edit and event.button() == Qt.MouseButton.LeftButton:
            item, original, _, _ = self._edit
            self._edit = None
            rect, bounds = item.rect(), self.pixmap_item.boundingRect()
            if rect == original:
                event.accept()
                return
            if rect.width() > 0 and rect.height() > 0 and bounds.width() > 0:
                self.rectangle_edited.emit(str(item.data(0)), [rect.left()/bounds.width(),
                    rect.top()/bounds.height(), rect.right()/bounds.width(), rect.bottom()/bounds.height()])
            else:
                item.setRect(original)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class MetricAnalysisDomainDialog(AnalysisDomainDialog):
    """Series-wide physical boxes; all file/image work runs outside the GUI thread."""

    def __init__(self, project, configuration, *, models, store=None,
                 photo_loader=None, labels_loader=None, series_runs=(), parent=None):
        self.models = models
        self.photo_loader, self.labels_loader = photo_loader, labels_loader
        self._thread = self._worker = None
        self._completion = None
        self._run = next(iter(models))
        store = store or data.MetricAnalysisDomainStore(project)
        current = store.current().get("configuration", {})
        if current.get("format") == data.FORMAT:
            if current.get("geometry_index") != configuration["geometry_index"]:
                raise ValueError("Saved boxes bind another geometry index; review it before opening")
            configuration = current
        self.metric_config = deepcopy(configuration)
        initial = dict(configuration, domains=[data.ui_domain(row) for row in configuration["domains"]])
        super().__init__(project, parent, store=store, configuration=initial,
                         load_reference=False, canvas=MetricBoxCanvas())
        self.setWindowTitle("Analysis boxes · measured rig frame [mm]")
        self.domain_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        splitters = self.findChildren(QSplitter)
        if splitters:
            splitters[0].setSizes([720, 620])
        self.domain_source.hide()
        for button in self.findChildren(QPushButton):
            if button.text() == "Load corrected image":
                button.setText("Load run photograph + sand map")
        self.run_choice = QComboBox()
        ordered = list(series_runs) or list(models)
        for run in ordered:
            self.run_choice.addItem(str(run).upper() + (" · model unavailable" if run not in models else ""), run)
        self.run_choice.setCurrentIndex(max(0, self.run_choice.findData(self._run)))
        self.run_choice.currentIndexChanged.connect(self._run_changed)
        self.layout().insertWidget(0, self.run_choice)
        self.edit_button = QPushButton("Move / resize box (drag corner)")
        self.edit_button.clicked.connect(lambda: self._set_mode("edit"))
        self.layout().insertWidget(2, self.edit_button)
        self.legacy_button = QPushButton("Review imported coordinates in measured frame")
        self.legacy_button.clicked.connect(self._acknowledge_legacy)
        self.layout().insertWidget(3, self.legacy_button)
        self.domain_reference_status.setText(
            "Millimetres, x right, y up. Width comes from each measured model. "
            "900 × 490 mm is the view extent; the top height has not been independently measured. "
            "Imported legacy coordinates stay unchanged and need review before mass use.")
        self.domain_canvas.rectangle_edited.connect(self._rectangle_edited)
        self._show_empty_view()

    def _refresh_domain_table(self, *, select_id=""):
        super()._refresh_domain_table(select_id=select_id)
        self._updating_table = True
        self.domain_table.setColumnCount(7)
        self.domain_table.setHorizontalHeaderLabels(
            ["Use", "Name", "x min [mm]", "x max [mm]", "y min [mm]", "y max [mm]", "Mass role"])
        for row, value in enumerate(self.domains):
            for column, number in enumerate(value["bounds_mm"], 2):
                self.domain_table.item(row, column).setText(f"{number:.12g}")
            self.domain_table.setItem(row, 6, QTableWidgetItem(value["role"]))
        self._updating_table = False

    def _domain_table_changed(self, item):
        if self._updating_table or item.row() >= len(self.domains):
            return
        old = self.domains[item.row()]
        try:
            # Display rounding must not change unedited physical coordinates.
            values = list(old["bounds_mm"])
            if 2 <= item.column() <= 5:
                values[item.column()-2] = float(item.text().replace(",", "."))
            legacy = deepcopy(old.get("legacy_import"))
            if legacy and values != old["bounds_mm"]:
                legacy["unmodified"] = False
            value = data.box(self.domain_table.item(item.row(), 1).text(), values,
                domain_id=old["domain_id"], role=self.domain_table.item(item.row(), 6).text().strip(),
                enabled=self.domain_table.item(item.row(), 0).checkState() == Qt.CheckState.Checked, legacy=legacy)
            self.domains[item.row()] = data.ui_domain(value)
        except (ValueError, TypeError):
            self.domain_reference_status.setText("Enter finite increasing bounds and a role: box1, box2, or blank.")
        self._refresh_domain_table(select_id=old["domain_id"])

    def _set_mode(self, mode):
        self.domain_pan.setChecked(mode == "pan")
        self.domain_draw.setChecked(mode == "draw")
        self.domain_canvas.set_mode(mode)

    def _rectangle_drawn(self, normalized):
        value = data.from_normalized(normalized, name=f"Analysis box {len(self.domains)+1}")
        self.domains.append(data.ui_domain(value))
        self._refresh_domain_table(select_id=value["domain_id"])

    def _rectangle_edited(self, domain_id, normalized):
        for index, old in enumerate(self.domains):
            if old["domain_id"] == domain_id:
                legacy = deepcopy(old.get("legacy_import"))
                if legacy:
                    legacy["unmodified"] = False
                value = data.from_normalized(normalized, name=old["name"], role=old["role"],
                    domain_id=domain_id, enabled=old["enabled"], legacy=legacy)
                self.domains[index] = data.ui_domain(value)
                break
        self._refresh_domain_table(select_id=domain_id)

    def _show_empty_view(self):
        self.domain_canvas.set_image(np.full((490, 900, 3), .18, np.float32))
        self.domain_canvas.set_domains(self.domains)
        self.reference = {"provenance": {"source_shape_rc": [490, 900], "width_m": .9, "height_m": .49}}

    def _run_changed(self):
        self._run = str(self.run_choice.currentData())
        self._show_empty_view()
        if self._run not in self.models:
            self.domain_reference_status.setText("This run has no measured model; no placement or image is inferred.")
        else:
            width = self.models[self._run].model["metric_frame"]["measured_width_mm"]
            self.domain_reference_status.setText(f"{self._run.upper()}: measured width {width:g} mm. Load its photograph to review the same physical boxes.")

    def _background(self, function, done):
        if self._thread is not None and self._thread.isRunning():
            return
        thread, worker = QThread(self), _Call(function)
        self._thread, self._worker = thread, worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        self._completion = done
        worker.finished.connect(self._background_done, Qt.ConnectionType.QueuedConnection)
        worker.failed.connect(self._background_failed, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(thread.quit); worker.failed.connect(thread.quit)
        thread.finished.connect(self._background_closed, Qt.ConnectionType.QueuedConnection)
        thread.start()

    @Slot(object)
    def _background_done(self, result):
        if self._completion is not None:
            self._completion(result)

    @Slot(str)
    def _background_failed(self, message):
        self.domain_reference_status.setText(message)

    @Slot()
    def _background_closed(self):
        thread = self._thread
        if thread is not None:
            thread.wait()
            self._thread = self._worker = self._completion = None
            thread.deleteLater()

    def _load_reference(self, *, show_errors=True):
        if self._run not in self.models or self.photo_loader is None:
            self.domain_reference_status.setText("No verified photograph loader for this run; the metric boxes remain editable.")
            return
        run, model = self._run, self.models[self._run]
        def load():
            report = data.validate_configuration(self.project, self.metric_config)
            if not report["valid"]:
                raise ValueError("The geometry binding changed; reopen the editor before loading")
            photo = self.photo_loader(run, model)
            labels = self.labels_loader(run, model) if self.labels_loader else None
            return run, data.metric_preview(photo, model, labels=labels)
        def display(result):
            if result[0] == self._run:
                self.domain_canvas.set_image(result[1])
                self.domain_canvas.set_domains(self.domains)
                self.domain_reference_status.setText(f"{run.upper()}: measured-frame photograph; boxes are shared by the entire series.")
        self._background(load, display)

    def _import_domains(self):
        selected, _ = QFileDialog.getOpenFileName(self, "Import original box1 / box2", "", "ROI registry (*.toml)")
        if not selected:
            return
        def imported(values):
            existing = {row["name"].casefold() for row in self.domains}
            if any(row["name"].casefold() in existing for row in values):
                self.domain_reference_status.setText("A box with an imported name already exists; import was not added.")
                return
            self.domains.extend(data.ui_domain(row) for row in values)
            self._refresh_domain_table()
            self.domain_reference_status.setText("Imported exact legacy coordinates; review their frame interpretation before mass use.")
        self._background(lambda: data.load_legacy_registry(selected), imported)

    def _acknowledge_legacy(self):
        # Explicit user action on the selected draft rows, never an automatic conversion.
        selected = self.domain_table.selectionModel().selectedRows()
        if not selected:
            self.domain_reference_status.setText("Select the imported rows whose frame interpretation you have reviewed.")
            return
        for index in selected:
            legacy = self.domains[index.row()].get("legacy_import")
            if legacy:
                legacy["decision"] = "use_coordinates_in_measured_frame"
        self.domain_reference_status.setText("Selected imported coordinates acknowledged without scaling or clipping.")

    def _save_domains(self):
        if self._thread is not None and self._thread.isRunning():
            self.domain_reference_status.setText("Wait for the current image or revision operation to finish.")
            return
        config = deepcopy(self.metric_config)
        config["domains"] = [{k: deepcopy(v) for k, v in row.items()
            if k not in {"physical_bounds_m", "normalized_xyxy_top_left"}} for row in self.domains]
        actor, reason = self.domain_reviewer.text().strip(), self.domain_reason.text().strip()
        def saved(revision):
            self.metric_config = deepcopy(config)
            self.domains_saved.emit(revision)
            try:
                data.mass_boxes(config)
                note = "Both mass roles are ready as existing N1 input coordinates."
            except ValueError as error:
                note = str(error)
            self.domain_reference_status.setText("Saved append-only box revision " + revision["revision_id"] + ". " + note)
        self._background(lambda: self.store.save_revision(config, actor=actor, reason=reason), saved)

    def closeEvent(self, event):
        if self._thread is not None and self._thread.isRunning():
            event.ignore()
            self.domain_reference_status.setText("Wait for the current operation before closing.")
            return
        super().closeEvent(event)

    def reject(self):
        if self._thread is not None and self._thread.isRunning():
            self.domain_reference_status.setText("Wait for the current operation before closing.")
            return
        super().reject()


class NativeMetricBoxesWorkflow(QObject):
    """One additive Geometry/facies entry point; existing page owners are reused."""

    def __init__(self, window, workspace):
        super().__init__(window)
        self.workspace = workspace
        self._thread = self._worker = self.dialog = None
        self._context = None
        window.installEventFilter(self)
        self.button = QPushButton("Analysis boxes [mm]…")
        self.button.clicked.connect(self.open)
        workspace.layout().insertWidget(1, self.button)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Close:
            own_busy = self._thread is not None and self._thread.isRunning()
            dialog_busy = (self.dialog is not None and self.dialog._thread is not None
                           and self.dialog._thread.isRunning())
            if own_busy or dialog_busy:
                event.ignore()
                self.button.setToolTip("Wait for the current box-editor operation before closing.")
                return True
        return super().eventFilter(watched, event)

    def open(self):
        workspace = self.workspace
        project = workspace.project
        if project is None or workspace.simple_view is None:
            QMessageBox.information(workspace, "Analysis boxes", "Open the geometry/facies package and save its geometry revision first.")
            return
        if self._thread is not None and self._thread.isRunning():
            return
        selected, _ = QFileDialog.getOpenFileName(workspace, "Measured geometry index", "", "Geometry index (index.json);;JSON (*.json)")
        if not selected:
            return
        loader = workspace.simple_view._reference_photo_loader()
        runs = [row.run_id for row in workspace.run_inputs]
        self._context = project, loader, runs
        thread = QThread(self)
        worker = _Call(lambda: (data.new_configuration(project, selected), read_run_geometry(selected)))
        self._thread, self._worker = thread, worker
        worker.moveToThread(thread); thread.started.connect(worker.run)
        self.button.setEnabled(False)
        worker.finished.connect(self._ready)
        worker.failed.connect(self._failed)
        worker.finished.connect(thread.quit); worker.failed.connect(thread.quit)
        thread.finished.connect(self._finished)
        thread.start()

    @Slot(object)
    def _ready(self, result):
        project, loader, runs = self._context
        if self.workspace.project is not project:
            return
        try:
            self.dialog = MetricAnalysisDomainDialog(project, result[0], models=result[1],
                photo_loader=loader,
                labels_loader=lambda run, model: data.load_saved_sandmap_camera_grid(project, run, model),
                series_runs=runs, parent=self.workspace)
            self.dialog.show()
        except Exception as error:
            self._failed(str(error))

    @Slot(str)
    def _failed(self, message):
        QMessageBox.warning(self.workspace, "Analysis boxes", message)

    @Slot()
    def _finished(self):
        thread = self._thread
        if thread is not None:
            thread.wait()
            self._thread = self._worker = None
            thread.deleteLater()
        self.button.setEnabled(True)
