"""Named lighting choice in the native GUI; numerical admission stays separate."""

from PySide6.QtCore import QCoreApplication, QSignalBlocker, QThread, Qt, Signal, Slot
import tomllib
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from . import image_normalisation as method


class ImageNormalisationChoice(QWidget):
    choice_saved = Signal()
    busy_changed = Signal(bool)
    context_updated = Signal()

    def __init__(self, parent=None, *, allow_save=True, owns_context=True, reader=None, writer=None):
        super().__init__(parent)
        self.allow_save = allow_save
        self.owns_context = owns_context
        self._reader = reader or self._read_context
        self._writer = writer or method.save_choice
        self.thread = self.worker = None
        self._epoch, self._pending, self._closing = 0, None, False
        self._owner_busy = False
        QCoreApplication.instance().aboutToQuit.connect(self.wait_for_workers)
        self.series_root, self.series_id, self.runs, self.original = None, "", (), None
        self._source_family = "unknown"
        self._saved_form = None
        layout = QVBoxLayout(self)
        line = QHBoxLayout()
        line.addWidget(QLabel("Lighting / intensity normalisation"))
        self.method = QComboBox()
        for key, label in method.METHODS:
            self.method.addItem(label, key)
        self.method.setItemData(self.method.findData(method.EQUALISE_SAND),
            "One light field per run from same-facies sand on its start photo, outside the band along "
            "the frame. Changing the method requires preparing images and retraining the phase model.",
            Qt.ItemDataRole.ToolTipRole)
        self.method.setItemData(self.method.findData(method.MATCH_REFERENCE),
            "Old method: fits light in the band along the frame on every photo. CO2 entering that "
            "band is taken as light.", Qt.ItemDataRole.ToolTipRole)
        self.method.setCurrentIndex(-1)
        self.method.currentIndexChanged.connect(self._changed)
        line.addWidget(self.method, 1)
        line.addWidget(QLabel("ColorChecker colour correction"))
        self.color_method = QComboBox()
        for key, label in method.COLOR_METHODS:
            self.color_method.addItem(label, key)
        self.color_method.model().item(self.color_method.findData(method.COLOR_SAVED_NATIVE)).setEnabled(False)
        self.color_method.currentIndexChanged.connect(self._changed)
        line.addWidget(self.color_method)
        self.reference = QComboBox()
        self.reference.currentIndexChanged.connect(self._changed)
        line.addWidget(self.reference)
        for combo in (self.method, self.color_method, self.reference):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(18)
        self.save_button = QPushButton("Save image settings")
        self.save_button.clicked.connect(self._save)
        line.addWidget(self.save_button)
        layout.addLayout(line)
        band = QHBoxLayout()
        self.band_label = QLabel("Band along the frame")
        band.addWidget(self.band_label)
        self.stable_band = QDoubleSpinBox()
        self.stable_band.setRange(0.0, 1_000_000.0)
        self.stable_band.setDecimals(2)
        self.stable_band.setSingleStep(1.0)
        self.stable_band.setSuffix(" mm")
        self.stable_band.setSpecialValueText("Not set — retain saved method")
        self.stable_band.setToolTip(
            "Width measured inwards from the sand map's outer boundary beside the rig frame, "
            "not from the metal frame or internal layer boundaries. "
            "Use Preview stable area to see exactly which pixels are sampled.")
        self.stable_band.valueChanged.connect(self._changed)
        band.addWidget(self.stable_band)
        layout.addLayout(band)
        self.source_status = QLabel("Image source is read from the saved Setup selection.")
        self.source_status.setWordWrap(True)
        layout.addWidget(self.source_status)
        self.explanation = QLabel()
        self.explanation.setWordWrap(True)
        layout.addWidget(self.explanation)
        self.status = QLabel("Choose an experimental series first.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._controls()

    @staticmethod
    def _read_context(root, series):
        from .normalisation_step import read_series_context
        return read_series_context(root, series)

    def _context(self):
        return self._epoch, str(self.series_root or ""), self.series_id

    def set_context(self, series_root, series_id):
        """Capture selection only; all manifest I/O belongs to a finite worker."""
        if self._closing:
            return
        self._epoch += 1
        self.series_root, self.series_id = series_root, str(series_id or "")
        self.original, self.runs = None, ()
        self._saved_form = None
        self._pending = None
        with QSignalBlocker(self.method), QSignalBlocker(self.color_method), QSignalBlocker(self.reference), QSignalBlocker(self.stable_band):
            self.method.setCurrentIndex(-1)
            self.color_method.setCurrentIndex(0)
            self.color_method.model().item(3).setEnabled(False)
            self.reference.clear()
            self.stable_band.setValue(0.0)
        self.status.setText("Reading the selected series method…" if series_root and series_id else "Choose an experimental series first.")
        self._controls()
        self.context_updated.emit()
        if self.owns_context and series_root and series_id:
            root, series, reader = str(series_root), self.series_id, self._reader
            self._schedule("read", self._context(), lambda: reader(root, series))

    def install_snapshot(self, series_root, series_id, packet):
        """Use the owning normalisation pane's already-read immutable snapshot."""
        if self._closing or (str(series_root or ""), str(series_id or "")) != (str(self.series_root or ""), self.series_id):
            return
        self._install(packet)

    def _install(self, packet):
        raw, document, runs = packet
        self.original, self.runs = raw, tuple(runs)
        self._saved_form = None
        self._source_family = method.selected_source_family(document)
        descriptions = {
            "raw": "Source from Setup: camera RAW. RAW development, lighting correction and crop use these saved source images.",
            "processed": "Source from Setup: processed images. RAW development and saved RAW-result reproduction are unavailable; geometric corrections remain available.",
            "mixed": "Source from Setup: mixed RAW and processed images. Use one compatible prepared image basis before training.",
            "unknown": "Image source is not recorded yet. Save the source-image selection in Setup; no second RAW choice is required here.",
        }
        self.source_status.setText(descriptions[self._source_family])
        with QSignalBlocker(self.method), QSignalBlocker(self.color_method), QSignalBlocker(self.reference), QSignalBlocker(self.stable_band):
            self.method.setCurrentIndex(-1)
            self.color_method.setCurrentIndex(0)
            self.color_method.model().item(3).setEnabled(False)
            self.reference.clear()
            self.stable_band.setValue(0.0)
            try:
                saved = method.saved_choice(document, self.series_id, self.runs)
                self.reference.addItem("Choose reference…", None)
                for run in self.runs:
                    self.reference.addItem(run.upper(), run)
                if saved:
                    self.method.setCurrentIndex(self.method.findData(saved["method"]))
                    color = method.selected_corrections(saved)["color_method"]
                    self.color_method.model().item(3).setEnabled(color == "saved_native")
                    self.color_method.setCurrentIndex(self.color_method.findData(color))
                    self.reference.setCurrentIndex(self.reference.findData(saved.get("reference_run")))
                    band = saved.get("exclusion_band") or saved.get("stable_area") or {}
                    self.stable_band.setValue(band.get("width_mm", 0.0))
                    self._saved_form = self._form()
                    self._saved_feedback()
                else:
                    self._saved_form = None
                    self.status.setText("Choose a method explicitly. Automatic reference proposal needs current light-only evidence; chemistry ranking is not used.")
            except (ValueError, TypeError) as error:
                self.original = None
                self.status.setText(str(error))
        self._controls()
        self.context_updated.emit()

    def _schedule(self, operation, context, function):
        if self._closing:
            return
        if self.thread is not None:
            self._pending = operation, context, function
            return
        from .painting_gap_review_ui import _Read
        self.thread, self.worker = QThread(self), _Read(function)
        self.worker.key = operation, context
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit, Qt.ConnectionType.DirectConnection)
        self.thread.finished.connect(self._retired)
        self._controls()
        self.busy_changed.emit(True)
        self.thread.start()

    @Slot()
    def _retired(self, expected=None):
        expected = self.sender() if expected is None else expected
        if self.thread is None or self.thread is not expected:
            return
        thread, worker = self.thread, self.worker
        thread.wait()
        self.thread = self.worker = None
        operation, context = worker.key
        if not self._closing and context == self._context():
            if worker.error:
                self.original = None
                self.status.setText("The method metadata operation could not finish — " + worker.error)
            else:
                self._install(worker.result)
                if operation == "save":
                    self.choice_saved.emit()
        del worker
        thread.deleteLater()
        pending, self._pending = self._pending, None
        if pending is not None and not self._closing:
            self._schedule(*pending)
        self._controls()
        self.busy_changed.emit(self.thread is not None)
        if self._closing:
            self.close()

    def can_close(self):
        return self.thread is None

    def wait_for_workers(self):
        self._closing, self._pending = True, None
        self._epoch += 1
        self._controls()
        if self.thread is not None:
            thread = self.thread
            thread.wait()
            self._retired(thread)

    def closeEvent(self, event):
        self._closing, self._pending = True, None
        self._epoch += 1
        self._controls()
        if self.thread is not None:
            self.status.setText("The method metadata operation is finishing; the window will close afterwards.")
            event.ignore()
            return
        super().closeEvent(event)

    def _controls(self):
        usable = self.original is not None and self.thread is None and not self._closing and not self._owner_busy
        selected = self.method.currentData()
        color = self.color_method.currentData()
        equalise = selected == method.EQUALISE_SAND
        lighting = selected in (method.MATCH_REFERENCE, method.EQUALISE_SAND)
        needs_reference = lighting or color != "off"
        self.method.setEnabled(usable)
        self.color_method.setEnabled(usable)
        self.stable_band.setEnabled(usable and lighting)
        self.reference.setEnabled(usable and needs_reference)
        # One field, two meanings: equalisation leaves the band out, the old fit samples it.
        self.band_label.setText("Band along the frame left out of the light fit" if equalise else
                                "Stable sand band inwards (old per-photo fit)" if lighting else
                                "Band along the frame")
        self.stable_band.setSpecialValueText("None — all mapped sand enters the light fit" if equalise else
                                             "Not set — retain saved method")
        self.stable_band.setToolTip(
            "Width measured inwards from the sand map's outer boundary beside the rig frame. "
            "Indicator there can have changed colour already at the start, so this band is left out "
            "of the light fit. Use Preview to see the band." if equalise else
            "Width measured inwards from the sand map's outer boundary beside the rig frame, "
            "not from the metal frame or internal layer boundaries. "
            "Use Preview stable area to see exactly which pixels are sampled.")
        chosen = selected in (method.OFF, method.MATCH_REFERENCE, method.EQUALISE_SAND) and color is not None and (
            not needs_reference or self.reference.currentData() in self.runs)
        legacy_color_only = selected == method.OFF and color == method.COLOR_SAVED_NATIVE
        legacy_band_change = color == method.COLOR_SAVED_NATIVE and self.stable_band.value() > 0
        equalise_native = equalise and color == method.COLOR_SAVED_NATIVE
        missing_band = (selected == method.MATCH_REFERENCE and color != method.COLOR_SAVED_NATIVE
                        and self.stable_band.value() <= 0)
        chosen = chosen and not (legacy_color_only or legacy_band_change or missing_band or equalise_native)
        self.save_button.setEnabled(usable and chosen and self.allow_save)
        saving = self.worker is not None and self.worker.key[0] == "save"
        self.save_button.setText("Saving…" if saving else
            "✓ Image settings saved" if self._saved_form is not None and self._form() == self._saved_form else
            "Save image settings")
        if selected == method.OFF:
            explanation = "Lighting is off. Geometric corrections and crop are retained."
        elif equalise:
            explanation = (
                "Each run's start photo gives one smooth light field from sand of the same facies, outside "
                "the band along the frame. The field is applied unchanged to every photo of the run, so "
                "colour changes from CO2 remain. The ColorChecker in each photo corrects drift in light "
                "level and colour and links the runs to the reference run."
            )
        elif selected == method.MATCH_REFERENCE:
            explanation = (
                "Old method: a smooth brightness field is fitted on the band along the frame in every photo "
                "to match the reference run. CO2 entering that band is taken as light."
            )
        else:
            explanation = (
                "Choose Off for geometric corrections only, or equalise sand types to correct the light "
                "across the rig from each run's start photo."
            )
        colour_explanation = {
            "off": "ColorChecker colour correction is off.",
            "darsia_original": "ColorChecker colour correction uses DarSIA's original method independently of lighting normalisation.",
            "uniform24": "ColorChecker colour correction uses Uniform24 independently of lighting normalisation.",
            "saved_native": "The existing saved native colour correction is retained; it has not been renamed or replaced by another method.",
        }.get(color, "Choose a ColorChecker colour-correction method.")
        self.explanation.setText(explanation + " " + colour_explanation +
            " Saving these choices does not recompute images or train a phase model.")
        if legacy_band_change or legacy_color_only or equalise_native:
            self.explanation.setText("Choose Off, DarSIA original or Uniform24 to prepare a new independent "
                "correction. The saved native method retains its original combined light/colour fit.")
        elif missing_band:
            self.explanation.setText("Choose the stable sand-band width in millimetres before saving lighting normalisation.")
        unavailable = ("Wait for the current metadata operation to finish." if self.original is not None
                       else "Choose an experimental series and wait for its method to load.")
        self.method.setToolTip(
            "Choose the lighting correction for this series." if usable else unavailable)
        self.reference.setToolTip(
            "Both corrections are off; no reference run is used." if not needs_reference else
            "Choose the reference for the enabled lighting and/or colour correction. No chemistry ranking is used." if usable else unavailable)
        self.save_button.setToolTip(
            "Read-only preview; no method choice can be saved." if not self.allow_save else
            unavailable if not usable else
            "Choose Off, DarSIA original or Uniform24 for independent color correction." if legacy_color_only else
            "Choose the independent corrections and their reference when needed." if not chosen else
            "Save these corrections and reference for the series. Existing images and phase models are not recomputed.")

    def set_owner_busy(self, busy):
        self._owner_busy = bool(busy)
        self._controls()

    def _changed(self, *_args):
        self._controls()
        if self.original is not None:
            legacy_color_only = (self.method.currentData() == method.OFF
                                 and self.color_method.currentData() == method.COLOR_SAVED_NATIVE)
            if self._saved_form is not None and self._form() == self._saved_form:
                self._saved_feedback()
                return
            self.status.setText(
                "Choose Off, DarSIA original or Uniform24 for independent color correction. "
                "The saved native correction combines colour and lighting." if legacy_color_only else
                "Unsaved image settings — save explicitly. No RAW fit or correction has run.")

    def _form(self):
        return (self.method.currentData(), self.color_method.currentData(),
                self.reference.currentData(), self.stable_band.value())

    def _saved_feedback(self):
        selected = self.method.currentData()
        lighting = {method.OFF: "Off", method.EQUALISE_SAND: "Equalise sand types"}.get(
            selected, "Per-photo band fit (old)")
        reference = self.reference.currentData()
        details = f"Lighting: {lighting}; colour: {self.color_method.currentText()}"
        if reference and (selected != method.OFF or self.color_method.currentData() != "off"):
            details += "; reference: " + str(reference).upper()
        if selected == method.EQUALISE_SAND:
            details += f"; band left out: {self.stable_band.value():g} mm"
        elif selected == method.MATCH_REFERENCE and self.stable_band.value() > 0:
            details += f"; stable band: {self.stable_band.value():g} mm"
        self.status.setText("✓ Image settings saved. " + details +
            ". No images were processed; the Setup job reports processing completion.")

    def _save(self):
        if not self.save_button.isEnabled():
            return
        root, series, runs = str(self.series_root), self.series_id, tuple(self.runs)
        selected = self.method.currentData()
        color = self.color_method.currentData()
        # The same field is the band left out (equalisation) or the band fitted on (old).
        band = (dict(exclusion_band_mm=self.stable_band.value()) if selected == method.EQUALISE_SAND
                else dict(stable_band_mm=self.stable_band.value() or None))
        reference = (self.reference.currentData()
                     if selected in (method.MATCH_REFERENCE, method.EQUALISE_SAND) or color != "off" else None)
        original, writer = self.original, self._writer
        def save():
            raw = writer(root, series, runs, selected, reference,
                         color_method=color, expected_bytes=original, **band)
            return raw, tomllib.loads(raw.decode("utf-8-sig")), runs
        self.status.setText("Saving the explicit method choice…")
        self._schedule("save", self._context(), save)
