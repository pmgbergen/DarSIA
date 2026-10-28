"""Independent panels for the existing saved mass-result reader.

Only display copies are made. No RAW decoding, fitting, field publication or
selection of a production method belongs to this module.
"""
from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget,
)

from .tools.phase_studio_series_ui import _FullscreenPixmap, _photometry_filter_values


CONTENTS = (
    ("Photograph", "photograph"),
    ("Dissolved CO₂ concentration map (g/L)", "concentration"),
    ("Total CO₂ mass map (g per cell)", "mass_total"),
    ("Dissolved CO₂ mass map (g per cell)", "mass_aqueous"),
    ("Free-gas CO₂ mass map (g per cell)", "mass_gas"),
    ("Mass graph", "mass_curve"),
    ("Area and dilution graph", "dilution_curve"),
    ("Overflow / free-gas graph", "overflow_curve"),
)
LAYOUTS = {
    "one": ((0, 0, 1, 2),),
    "side_by_side": ((0, 0, 1, 1), (0, 1, 1, 1)),
    "pyramid": ((0, 0, 1, 2), (1, 0, 1, 1), (1, 1, 1, 1)),
}


def asset_path(result_path, reference):
    if not reference:
        raise ValueError("This saved result does not contain the requested asset.")
    path = Path(reference)
    return path if path.is_absolute() else Path(result_path).resolve().parent / path


def concentration_figure(result_path, row):
    """Display the original saved g/L field and support, at its recorded scale.

    This is deliberately a concentration map, not an inferred per-cell mass
    map. Missing scale or support is reported instead of guessed from pixels.
    """
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    limit = row.get("color_limit_g_l")
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
        raise ValueError("The saved concentration map has no fixed positive g/L display limit.")
    with np.load(asset_path(result_path, row.get("fields")), allow_pickle=False) as archive:
        field = archive["concentration_g_l"]
        support_key = next((key for key in ("aqueous_volume_m3", "av", "aqueous_fraction") if key in archive), None)
        if support_key is None:
            raise ValueError("The saved concentration map has no aqueous support field.")
        support = archive[support_key] > 0
        if field.ndim != 2 or support.shape != field.shape:
            raise ValueError("Saved concentration and aqueous support grids differ.")
        mask = ~support | ~np.isfinite(field)
        if "unquantified_phase_mask" in archive:
            unknown = archive["unquantified_phase_mask"]
            if unknown.shape != field.shape:
                raise ValueError("Saved unquantified mask and concentration grids differ.")
            mask |= unknown.astype(bool)
        displayed = np.ma.array(field, mask=mask, copy=True)
    figure = Figure(figsize=(7, 3.8), layout="constrained")
    try:
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.set_facecolor("#e6edf2")
        image = axis.imshow(displayed, vmin=0, vmax=limit, cmap="viridis", interpolation="nearest")
        axis.set_title("Dissolved CO₂ concentration · saved field")
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, label="g/L · fixed linear scale")
        return figure
    except BaseException:
        from .storage_saved_rendering import release_stored_figure
        release_stored_figure(figure)
        raise


def read_panel(result, result_path, index, content, photo_path=None, entry=None):
    """Reuse saved-result/photo readers, including exact alternate-photo joins."""
    from .tools.phase_studio_photometry_view import _photo_rgb, _frame_for_same_source
    from .tools.phase_studio_saved_results import verify_saved_series_entry
    from . import phase_saved_review_metadata as backend

    if entry is not None and entry.get("receipt_path"):
        verify_saved_series_entry(entry)
    row = result["frames"][index]
    if content == "photograph":
        path, selected, selected_index = result_path, result, index
        if photo_path and Path(photo_path).resolve() != Path(result_path).resolve():
            path = photo_path
            selected = backend.read_result(path)
            if selected.get("complete") is not True:
                raise ValueError("The saved photograph variant is incomplete.")
            selected_index = _frame_for_same_source(result, index, selected)
        rgb = _photo_rgb(path, selected["frames"][selected_index])
        height, width = rgb.shape[:2]
        return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
    if content == "concentration":
        from .storage_saved_rendering import release_stored_figure
        figure = concentration_figure(result_path, row)
        try:
            figure.canvas.draw()
            rgba = figure.canvas.buffer_rgba()
            width, height = figure.canvas.get_width_height()
            return QImage(rgba, width, height, 4 * width, QImage.Format.Format_RGBA8888).copy()
        finally:
            release_stored_figure(figure)
    if content in {"mass_total", "mass_aqueous", "mass_gas"}:
        raise ValueError("This result has no verified saved per-cell mass binding. Its concentration field cannot supply a mass map.")
    if content not in {"mass_curve", "dilution_curve", "overflow_curve"}:
        raise ValueError("Unknown calibration-review panel content.")
    image = QImage(str(asset_path(result_path, result.get(content))))
    if image.isNull():
        raise ValueError("The saved graph is unavailable or unreadable.")
    return image


class _PanelRead(QThread):
    ready = Signal(int, object)

    def __init__(self, generation, result, path, index, requests, parent):
        super().__init__(parent)
        self.generation, self.result, self.path = generation, result, path
        self.index, self.requests = index, requests
        self.native_curve = None
        self.gas_notice = ""

    def run(self):
        loaded = []
        native_frame = None
        local_gas = None
        for request in self.requests:
            if self.isInterruptionRequested():
                return
            try:
                if request.get("error"):
                    raise ValueError(request["error"])
                from .calibration_review_native import NativeSavedReview
                from .calibration_review_gas import LocalGasReview, gas_figure, gas_summary_text
                if isinstance(self.result, LocalGasReview):
                    if local_gas is None:
                        local_gas = self.result.gas(self.index)
                        self.gas_notice = self.result.run.upper() + " · " + gas_summary_text(local_gas)
                    if request["content"] == "photograph":
                        image = self.result.photograph(self.index)
                    elif request["content"] == "mass_gas":
                        from .calibration_review_native_panels import _figure_image
                        image = _figure_image(gas_figure(local_gas, request.get("gas_view", "total"),
                            request.get("gas_palette", "blue_orange")))
                    elif request["content"] in {"mass_curve", "overflow_curve"}:
                        from .calibration_review_native_panels import _figure_image
                        if self.native_curve is None:
                            self.native_curve = _figure_image(self.result.curve())
                        image = self.native_curve
                    else:
                        raise ValueError("This local calibration has no saved field for this display. Choose its free-gas map; other saved results retain their original displays.")
                elif isinstance(self.result, NativeSavedReview):
                    if request["content"] == "mass_gas" and request.get("gas_view", "total") != "total":
                        raise ValueError("This saved result has no recorded model/overflow gas decomposition")
                    from .calibration_review_native_panels import read_native_panel
                    if request["content"] == "mass_curve":
                        if self.native_curve is None:
                            self.native_curve = read_native_panel(self.result, None, "mass_curve",
                                cancelled=self.isInterruptionRequested)
                        image = self.native_curve
                    else:
                        if native_frame is None:
                            native_frame = self.result.read_frame(self.index)
                        image = read_native_panel(self.result, native_frame, request["content"],
                            limit_g_per_cell=request.get("mass_limit"), cancelled=self.isInterruptionRequested)
                else:
                    if request["content"] == "mass_gas" and request.get("gas_view", "total") != "total":
                        raise ValueError("This saved result has no recorded model/overflow gas decomposition")
                    image = read_panel(self.result, self.path, self.index, request["content"],
                                       request.get("photo_path"), request.get("entry"))
                loaded.append((image, ""))
            except Exception as error:
                loaded.append((QImage(), str(error)))
        if not self.isInterruptionRequested():
            self.ready.emit(self.generation, loaded)


class ReviewPanel(QWidget):
    changed = Signal()

    def __init__(self, number, default, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        layout.addLayout(bar)
        bar.addWidget(QLabel(f"Panel {number}"))
        self.content = QComboBox()
        for label, key in CONTENTS:
            self.content.addItem(label, key)
        self.content.setCurrentIndex(self.content.findData(default))
        bar.addWidget(self.content, 1)
        self.photo_controls = QWidget()
        controls = QGridLayout(self.photo_controls)
        controls.setContentsMargins(0, 0, 0, 0)
        self.light, self.colour, self.photograph = QComboBox(), QComboBox(), QComboBox()
        for label, key in (("All saved light settings", "all"), ("No spatial correction", "none"),
                           ("Historical correction", "historical"), ("Facies correction", "facies"),
                           ("Not recorded", "unknown")):
            self.light.addItem(label, key)
        for label, key in (("All saved colour settings", "all"), ("ColorChecker on", "on"),
                           ("ColorChecker off", "off"), ("Not recorded", "unknown")):
            self.colour.addItem(label, key)
        for row, (label, widget) in enumerate((("Light", self.light), ("Colour", self.colour),
                                               ("Saved photograph", self.photograph))):
            controls.addWidget(QLabel(label), row, 0)
            controls.addWidget(widget, row, 1)
        self.photo_controls.setToolTip("Choose an already saved photograph correction. Mass maps and graphs keep the selected calibration.")
        layout.addWidget(self.photo_controls)
        self.mass_controls = QWidget()
        mass_controls = QHBoxLayout(self.mass_controls)
        mass_controls.setContentsMargins(0, 0, 0, 0)
        mass_controls.addWidget(QLabel("Fixed maximum · g per stored cell"))
        self.mass_limit = QLineEdit()
        self.mass_limit.setPlaceholderText("Enter a positive display limit")
        self.mass_limit.setToolTip("This fixed display limit stays the same across images. It does not change saved values.")
        self.mass_limit.editingFinished.connect(self.changed)
        mass_controls.addWidget(self.mass_limit, 1)
        layout.addWidget(self.mass_controls)
        from .calibration_review_gas import GAS_VIEWS
        self.gas_controls = QWidget()
        gas_controls = QHBoxLayout(self.gas_controls)
        gas_controls.setContentsMargins(0, 0, 0, 0)
        gas_controls.addWidget(QLabel("Gas view"))
        self.gas_view, self.gas_palette = QComboBox(), QComboBox()
        for label, key in GAS_VIEWS:
            self.gas_view.addItem(label, key)
        self.gas_palette.addItem("Blue / orange / magenta", "blue_orange")
        self.gas_palette.addItem("Green / violet / red", "green_violet")
        self.gas_palette.setToolTip("Display colours only. The saved masses and fixed scales stay unchanged.")
        gas_controls.addWidget(self.gas_view, 1)
        gas_controls.addWidget(self.gas_palette)
        self.gas_view.currentIndexChanged.connect(self.changed)
        self.gas_palette.currentIndexChanged.connect(self.changed)
        layout.addWidget(self.gas_controls)
        self.image = _FullscreenPixmap(QPixmap())
        self.image.setMinimumSize(220, 160)
        layout.addWidget(self.image, 1)
        self.notice = QLabel()
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        self._choices = []
        self._has_gas_provenance = False
        self.content.currentIndexChanged.connect(self._content_changed)
        self.light.currentIndexChanged.connect(self._filter_photographs)
        self.colour.currentIndexChanged.connect(self._filter_photographs)
        self.photograph.currentIndexChanged.connect(self.changed)
        self._content_changed()

    def _content_changed(self, *_):
        self.photo_controls.setVisible(self.content.currentData() == "photograph")
        self.mass_controls.setVisible(self.content.currentData() in {"mass_total", "mass_aqueous", "mass_gas"})
        self.set_gas_provenance(self._has_gas_provenance)
        self.changed.emit()

    def set_gas_provenance(self, available):
        self._has_gas_provenance = bool(available)
        self.gas_controls.setVisible(self._has_gas_provenance and self.content.currentData() == "mass_gas")
        self.gas_view.setEnabled(self._has_gas_provenance)
        self.gas_palette.setEnabled(self._has_gas_provenance)

    def set_photographs(self, choices):
        self._choices = choices
        self._filter_photographs(emit=False)

    def _filter_photographs(self, *_, emit=True):
        previous = self.photograph.currentData()
        self.photograph.blockSignals(True)
        self.photograph.clear()
        for path, label, entry in self._choices:
            light, colour = _photometry_filter_values(entry)
            if (self.light.currentData() not in ("all", light)
                    or self.colour.currentData() not in ("all", colour)):
                continue
            self.photograph.addItem(label, path)
        position = self.photograph.findData(previous)
        if position >= 0:
            self.photograph.setCurrentIndex(position)
        self.photograph.blockSignals(False)
        self.photograph.setEnabled(bool(self.photograph.count()))
        if emit:
            self.changed.emit()


class CalibrationReviewPanels(QWidget):
    """One active saved-asset reader; navigation coalesces to the latest frame."""

    display_requested = Signal(int, int)
    display_presented = Signal(int, int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = self.path = None
        self.index, self.entries = 0, []
        self._generation, self._worker, self._pending = 0, None, False
        self._active = False
        self._overflow_enabled = True
        self._native_curve = None
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        layout.addLayout(controls)
        controls.addWidget(QLabel("Layout"))
        self.layout_choice = QComboBox()
        for label, key in (("One panel", "one"), ("Side by side", "side_by_side"),
                           ("Three panels · pyramid", "pyramid")):
            self.layout_choice.addItem(label, key)
        controls.addWidget(self.layout_choice)
        controls.addStretch()
        self.grid = QGridLayout()
        self.gas_summary = QLabel()
        self.gas_summary.setWordWrap(True)
        self.gas_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.gas_summary.hide()
        layout.addWidget(self.gas_summary)
        layout.addLayout(self.grid, 1)
        self.panels = [ReviewPanel(index + 1, content, self) for index, content in enumerate(
            ("photograph", "concentration", "mass_curve"))]
        for panel in self.panels:
            panel.changed.connect(self.refresh)
        self.layout_choice.currentIndexChanged.connect(self.arrange)
        self.layout_choice.setCurrentIndex(2)
        self.arrange()
        QApplication.instance().aboutToQuit.connect(self.stop)

    def arrange(self, *_):
        for panel in self.panels:
            self.grid.removeWidget(panel)
            panel.hide()
        positions = LAYOUTS[self.layout_choice.currentData()]
        for panel, position in zip(self.panels, positions):
            self.grid.addWidget(panel, *position)
            panel.show()
        self.grid.setRowStretch(0, 1)
        self.grid.setRowStretch(1, int(len(positions) == 3))
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)
        self.refresh()

    def set_active(self, active):
        self._active = bool(active)
        self.setVisible(active)
        self.refresh()

    def set_overflow_enabled(self, enabled):
        self._overflow_enabled = bool(enabled)
        for panel in self.panels:
            panel.content.model().item(panel.content.findData("overflow_curve")).setEnabled(enabled)
        self.refresh()

    def set_result(self, result, path, index, entries):
        self.gas_summary.hide()
        self._native_curve = None
        self.result, self.path, self.index, self.entries = result, path, index, list(entries)
        choices = []
        if result and path:
            from .tools.phase_studio_photometry_view import variant_choices
            selected = {"method": result.get("method", {}).get("settings", {})}
            choices.append((str(path), "As saved for this calibration", selected))
            by_path = {str(entry["path"]): entry for entry in entries}
            for _key, label, photo_path in variant_choices(entries, result, path):
                if Path(photo_path).resolve() != Path(path).resolve():
                    choices.append((str(photo_path), label, by_path[str(photo_path)]))
        for panel in self.panels:
            panel.set_gas_provenance(False)
            panel.set_photographs(choices)
        self.refresh()

    def set_native_result(self, reader, index):
        """Keep the native reader separate from the older result dictionaries."""
        from .calibration_review_native import NativeSavedReview
        self.gas_summary.hide()
        changing_kind = not isinstance(self.result, NativeSavedReview)
        if self.result is not reader:
            self._native_curve = None
        self.result, self.path, self.index, self.entries = reader, reader.index["path"], index, []
        for panel in self.panels:
            panel.set_gas_provenance(False)
            panel.set_photographs([(str(self.path), "Saved original photograph", {"method": {}})])
            if changing_kind and panel.content.currentData() == "concentration":
                panel.content.blockSignals(True)
                panel.content.setCurrentIndex(panel.content.findData("mass_total"))
                panel.content.blockSignals(False)
                panel.photo_controls.setVisible(False)
                panel.mass_controls.setVisible(True)
        self.refresh()

    def set_local_result(self, reader, index):
        from .calibration_review_gas import LocalGasReview
        changing_kind = not isinstance(self.result, LocalGasReview)
        if self.result is not reader:
            self._native_curve = None
        self.result, self.path, self.index, self.entries = reader, reader.frozen["path"], index, []
        self.gas_summary.show()
        for position, panel in enumerate(self.panels):
            panel.set_gas_provenance(True)
            panel.set_photographs([(str(self.path), "Saved photograph", {"method": {}})])
            if changing_kind and panel.content.currentData() in {"concentration", "photograph"}:
                was_photo = panel.content.currentData() == "photograph"
                panel.content.blockSignals(True)
                panel.content.setCurrentIndex(panel.content.findData("mass_gas"))
                panel.content.blockSignals(False)
                if position == 0 and was_photo:
                    panel.gas_view.blockSignals(True)
                    panel.gas_view.setCurrentIndex(panel.gas_view.findData("origin"))
                    panel.gas_view.blockSignals(False)
            panel.photo_controls.setVisible(panel.content.currentData() == "photograph")
            panel.mass_controls.setVisible(False)
            panel.gas_controls.setVisible(panel.content.currentData() == "mass_gas")
        self.refresh()

    def refresh(self, *_):
        self._generation += 1
        self._pending = self._active and self.result is not None
        from .calibration_review_gas import LocalGasReview
        if isinstance(self.result, LocalGasReview):
            self.gas_summary.setText("Loading saved gas amounts for this image…" if self._pending else "Choose a saved image.")
        self.display_requested.emit(self._generation, self.index)
        for panel in self.panels:
            panel.image.set_source_pixmap(QPixmap())
            panel.image.clear()
            panel.notice.setText("Loading saved display…" if self._pending else "Choose a saved result.")
        if self._worker is not None:
            self._worker.requestInterruption()
            return
        self._start_pending()

    def _start_pending(self):
        if not self._pending:
            return
        self._pending = False
        requests = []
        for panel in self.panels[:len(LAYOUTS[self.layout_choice.currentData()])]:
            content, photo_path = panel.content.currentData(), panel.photograph.currentData()
            path = photo_path if content == "photograph" else str(self.path)
            entry = next((entry for entry in self.entries if str(entry["path"]) == path), None)
            error = ("No saved photograph matches these light and colour settings."
                     if content == "photograph" and not photo_path else "")
            if content == "overflow_curve" and not self._overflow_enabled:
                error = "Overflow comparison is off. Choose Compare only on the Mass calibration page to display it."
            mass_limit = None
            from .calibration_review_gas import LocalGasReview
            local = isinstance(self.result, LocalGasReview)
            panel.mass_controls.setVisible(not local and content in {"mass_total", "mass_aqueous", "mass_gas"})
            if not local and content in {"mass_total", "mass_aqueous", "mass_gas"}:
                try:
                    mass_limit = float(panel.mass_limit.text())
                    if not math.isfinite(mass_limit) or mass_limit <= 0:
                        raise ValueError
                except ValueError:
                    error = "Enter a fixed positive display limit in grams per stored cell."
            requests.append(dict(content=content, photo_path=photo_path, entry=entry,
                                 mass_limit=mass_limit, error=error, gas_view=panel.gas_view.currentData() if local else "total",
                                 gas_palette=panel.gas_palette.currentData()))
        worker = _PanelRead(self._generation, self.result, self.path, self.index, requests, self)
        if self._native_curve is not None and self._native_curve[0] is self.result:
            worker.native_curve = self._native_curve[1]
        self._worker = worker
        worker.ready.connect(self._ready)
        worker.finished.connect(lambda: self._finished(worker))
        worker.start()

    def _ready(self, generation, loaded):
        if generation != self._generation or not self._active:
            return
        for panel, (image, error) in zip(self.panels, loaded):
            panel.image.set_source_pixmap(QPixmap.fromImage(image))
            panel.notice.setText("Unavailable: " + error if error else "")
        if getattr(self._worker, "gas_notice", ""):
            self.gas_summary.setText(self._worker.gas_notice)
        else:
            from .calibration_review_gas import LocalGasReview
            if isinstance(self.result, LocalGasReview):
                self.gas_summary.setText("Saved gas amounts are unavailable for this image.")
        self.display_presented.emit(generation, self.index,
            bool(loaded) and all(not error and not image.isNull() for image, error in loaded))

    def _finished(self, worker):
        curve = getattr(worker, "native_curve", None)
        if worker.result is self.result and curve is not None:
            self._native_curve = worker.result, curve
        if self._worker is worker:
            self._worker = None
        worker.deleteLater()
        self._start_pending()

    def stop(self):
        self._active = self._pending = False
        self._generation += 1
        if self._worker is not None:
            self._worker.requestInterruption()
            self._worker.wait()

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)
