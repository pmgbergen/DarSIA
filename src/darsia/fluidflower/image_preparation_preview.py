"""Read-only image preparation preview and explicit reference proposals."""
from copy import deepcopy
import json
import math
from pathlib import Path
from threading import Lock

import numpy as np
from PySide6.QtCore import QCoreApplication, QSignalBlocker, QStandardPaths, QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGraphicsPixmapItem,
    QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .image_normalisation import EQUALISE_SAND
from .pages.image_correction.page import ZoomableImageView
from .ui_metadata_preview import MetadataPreview


def percent_correction(log_correction) -> str:
    """A log correction as the change it makes, exp(x) - 1, in percent (e.g. "39 %"); "—" when missing."""
    if isinstance(log_correction, bool) or not isinstance(log_correction, (int, float)):
        return "—"
    if not math.isfinite(log_correction):
        return "—"
    percent = 100.0 * math.expm1(float(log_correction))
    return f"{percent:.0f} %" if abs(percent) >= 10 else f"{percent:.1f} %"


# Find reference's columns and what each measures. Distances are optical
# corrections between runs, shown as exp(x) - 1 in percent.
RANKING_COLUMNS = (
    ("Rank", "1 is the run with the least mean distance to the other runs."),
    ("Run", "(saved) marks the saved reference and (proposed) the proposal."),
    ("Mean distance (%)", "Mean optical correction from this run to each of the other runs: "
                          "the mean of the sand and ColorChecker distances that are compared."),
    ("Largest distance (%)", "Largest optical correction from this run to any one other run."),
    ("Sand distance (%)", "Mean lighting correction between this run's sand cells and the other runs' "
                          "(median over the shared cells). — when lighting is not compared."),
    ("ColorChecker distance (%)", "Mean colour correction between this run's ColorChecker patches and the "
                                  "other runs' (median over the usable patches). — when colour is not compared."),
)


class PreparationSource:
    """Use the same explicit local job registrations as the Jobs workspace."""

    def __init__(self, registry_root=None):
        from .native_jobs_registry import JobsUIRegistry
        self.registry = JobsUIRegistry(registry_root or (
            Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation))
            / "native_jobs"))
        self._context_cache = {}
        self._operations_cache = {}
        self._preview_lock = Lock()

    def __call__(self, operation, root, series_id, **options):
        from .pipeline import current_image_preparation as preparation
        from .normalisation_step import read_series_context
        # Studio uses a canonical filter ID, whereas Jobs records the literal
        # manifest identity. Resolve that declared owner, never search another
        # registry or silently use a different series' results.
        _raw, document, _runs = read_series_context(root, series_id)
        actual = document["series_id"]
        registrations = self.registry.jobs(root, actual)
        loader = {"preview": preparation.get_preparation_preview,
                  "colour_fit": preparation.get_colour_fit_preview,
                  "proposal": preparation.get_reference_proposal}[operation]
        if operation in ("preview", "colour_fit"):
            if operation == "preview":
                options["context_cache"] = self._context_cache
            options["operations_cache"] = self._operations_cache
            # A cancelled reader may still be finishing its archive read. Do
            # not decode the same baseline in a second concurrent reader.
            with self._preview_lock:
                if options.get("check_cancelled"):
                    options["check_cancelled"]()
                return loader(root, actual, registrations=registrations, **options)
        return loader(root, actual, registrations=registrations, **options)


class ImagePreparationPreview(QWidget):
    """One cancellable reader; accepting its proposal changes only the form."""
    progress = Signal(object, object)
    preview_loaded = Signal(object)

    def __init__(self, choice, parent=None, *, loader=None):
        super().__init__(parent)
        self.choice, self.loader = choice, loader or PreparationSource()
        self.reader = MetadataPreview(self)
        self.reader.completed.connect(self._completed)
        self.progress.connect(self._progress, Qt.ConnectionType.QueuedConnection)
        self._requested = None
        self._series = None
        self._proposal = None
        self._mask = None
        self._displayed = None
        self._photo_key = None
        self._photo_binding = None
        self._auto_preview = QTimer(self)
        self._auto_preview.setSingleShot(True)
        self._auto_preview.setInterval(120)
        self._auto_preview.timeout.connect(lambda: self._start("preview"))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Baseline"))
        self.run = QComboBox()
        self.run.currentIndexChanged.connect(self._changed)
        controls.addWidget(self.run)
        self.preview = QPushButton("Refresh preview")
        self.preview.clicked.connect(lambda: self._start("preview"))
        controls.addWidget(self.preview)
        self.colour_fit = QPushButton("Check ColorChecker fit")
        self.colour_fit.setToolTip(
            "Check the selected baseline against the saved reference with the same full-photograph "
            "colour producer used by Finish setup. Reads at most two RAW baselines; saves no results.")
        self.colour_fit.clicked.connect(lambda: self._start("colour_fit"))
        controls.addWidget(self.colour_fit)
        self.show_mask = QCheckBox("Show stable area")
        self.show_mask.setChecked(True)
        self.show_mask.toggled.connect(lambda checked: self.mask_item.setVisible(checked and self._mask is not None))
        view_controls = QHBoxLayout()
        view_controls.addWidget(self.show_mask)
        self.fit = QPushButton("Fit")
        self.fit.clicked.connect(lambda: self.photo.fitInView(self.photo.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio))
        view_controls.addWidget(self.fit)
        controls.addStretch(1)
        view_controls.addStretch(1)
        layout.addLayout(controls)
        layout.addLayout(view_controls)
        self.photo = ZoomableImageView()
        self.mask_item = QGraphicsPixmapItem()
        self.mask_item.setZValue(5)
        self.photo.scene().addItem(self.mask_item)
        layout.addWidget(self.photo, 1)
        proposals = QHBoxLayout()
        self.propose = QPushButton("Find reference")
        self.propose.setToolTip("Compare candidate reference runs and propose the least overall optical correction.")
        self.propose.clicked.connect(lambda: self._start("proposal"))
        proposals.addWidget(self.propose)
        self.use_proposal = QPushButton("Use proposal")
        self.use_proposal.clicked.connect(self._use_proposal)
        proposals.addWidget(self.use_proposal)
        self.cancel = QPushButton("Cancel preview")
        self.cancel.clicked.connect(self.cancel_preview)
        proposals.addWidget(self.cancel)
        layout.addLayout(proposals)
        self.ranking = QTableWidget(0, len(RANKING_COLUMNS))
        self.ranking.setHorizontalHeaderLabels([name for name, _tip in RANKING_COLUMNS])
        for column, (_name, tip) in enumerate(RANKING_COLUMNS):
            self.ranking.horizontalHeaderItem(column).setToolTip(tip)
        self.ranking.verticalHeader().hide()
        self.ranking.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.ranking.setMaximumHeight(180)
        self.ranking.horizontalHeader().setStretchLastSection(True)
        self.ranking.hide()
        layout.addWidget(self.ranking)
        self.status = QLabel("Choose a series and a stable-band width to preview the sampled area.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.colour_status = QLabel()
        self.colour_status.setWordWrap(True)
        self.colour_status.hide()
        layout.addWidget(self.colour_status)
        self.fit_details = QPushButton("Show ColorChecker fit details")
        self.fit_details.setCheckable(True)
        self.fit_details.hide()
        layout.addWidget(self.fit_details)
        self.colour_details = QPlainTextEdit()
        self.colour_details.setReadOnly(True)
        self.colour_details.setMaximumHeight(180)
        self.colour_details.hide()
        self.fit_details.toggled.connect(self.colour_details.setVisible)
        layout.addWidget(self.colour_details)
        for signal in (choice.method.currentIndexChanged, choice.color_method.currentIndexChanged,
                       choice.reference.currentIndexChanged, choice.stable_band.valueChanged):
            signal.connect(self._changed)
        choice.context_updated.connect(self._context_updated)
        QCoreApplication.instance().aboutToQuit.connect(self.cancel_preview)
        self._context_updated()

    def _progress(self, request, value):
        if request != self._requested or not self.reader.busy:
            return
        run, position, total = value.get("run", ""), value.get("position"), value.get("total")
        suffix = f" ({position + 1}/{total})" if isinstance(position, int) and isinstance(total, int) else ""
        self.status.setText("Reading baseline " + str(run).upper() + suffix + "…")

    def _context(self):
        choice = self.choice
        return (str(choice.series_root or ""), choice.series_id, choice.original,
                tuple(choice.runs), self.run.currentData(), choice.stable_band.value(),
                choice.method.currentData(), choice.color_method.currentData(),
                choice.reference.currentData())

    def _context_updated(self):
        self.cancel_preview()
        choice = self.choice
        context = (str(choice.series_root or ""), choice.series_id, tuple(choice.runs))
        if context != self._series:
            self._series = context
            with QSignalBlocker(self.run):
                self.run.clear()
                for run in choice.runs:
                    self.run.addItem(run.upper(), run)
                selected = self.run.findData(choice.reference.currentData())
                if selected >= 0:
                    self.run.setCurrentIndex(selected)
            self._clear_image()
        self._changed()
        if self.isVisible():
            # Revalidate source pins on a refreshed context, including a new
            # Build with unchanged lighting choices. The same baseline stays
            # visible and its cached pixels are reused when pins still match.
            self._auto_preview.start()

    def _clear_image(self):
        self.photo.pixmap_item.setPixmap(QPixmap())
        self.photo._has_image = False
        self.mask_item.setPixmap(QPixmap())
        self._mask = None
        self._displayed = None
        self._photo_key = None
        self._photo_binding = None

    def _image_key(self):
        return (str(self.choice.series_root or ""), self.choice.series_id,
                self.run.currentData(), self.choice.stable_band.value())

    def _changed(self, *_args):
        self.cancel_preview()
        self._proposal = None
        self.ranking.hide()
        self.colour_status.hide()
        self.fit_details.setChecked(False)
        self.fit_details.hide()
        self.colour_details.clear()
        # Saving lighting/colour choices changes the manifest bytes, but does
        # not change the already displayed geometry baseline or stable band.
        if self._displayed != self._image_key():
            if self._photo_key != self._image_key()[:3]:
                self._clear_image()
            else:
                self.mask_item.setVisible(False)
                self._mask = None
            self.status.setText("Loading the selected baseline and its saved geometry…")
            if self.isVisible():
                self._auto_preview.start()
        self._controls()

    def _controls(self):
        ready = self.choice.original is not None and bool(self.run.currentData())
        method = self.choice.method.currentData()
        # Sand-type equalisation may leave no band out; the old fit needs its band.
        band = self.choice.stable_band.value() > 0 or method == EQUALISE_SAND
        lighting = method in ("match_reference_lighting", EQUALISE_SAND)
        colour = self.choice.color_method.currentData() not in (None, "off")
        self.preview.setEnabled(ready and not self.reader.busy)
        saved = self.choice._saved_form is not None and self.choice._saved_form == self.choice._form()
        self.colour_fit.setEnabled(ready and saved and colour and not self.reader.busy)
        self.propose.setEnabled(ready and band and (lighting or colour) and not self.reader.busy)
        self.use_proposal.setEnabled(self._proposal is not None and not self.reader.busy)
        self.cancel.setEnabled(self.reader.busy)
        self.fit.setEnabled(not self.photo.pixmap_item.pixmap().isNull())

    def _start(self, operation):
        self._auto_preview.stop()
        button = {"preview": self.preview, "colour_fit": self.colour_fit, "proposal": self.propose}[operation]
        if not button.isEnabled():
            return
        context = self._context()
        root, series, _raw, _runs, run, band, method, colour, _ref = context
        options = dict(band_width_mm=band)
        if operation in ("preview", "colour_fit"):
            options["run"] = run
            if operation == "colour_fit":
                options.pop("band_width_mm")
            elif method == EQUALISE_SAND:
                options["excluded"] = True
        else:
            options.update(use_lighting=method in ("match_reference_lighting", EQUALISE_SAND),
                           use_color=colour != "off")
            if method == EQUALISE_SAND:
                from .image_normalisation import LIGHT_FIELD
                options["light_field"] = dict(LIGHT_FIELD)
        self._requested = (operation, context)
        requested = self._requested
        self._proposal = None
        loader = self.loader
        def read(cancelled):
            def check_cancelled():
                if cancelled():
                    raise InterruptedError("Image correction preview cancelled")
            def on_progress(value):
                check_cancelled()
                try:
                    self.progress.emit(requested, value)
                except RuntimeError:
                    pass
            check_cancelled()
            result = loader(operation, root, series, check_cancelled=check_cancelled,
                            on_progress=on_progress, **options)
            check_cancelled()
            return result
        self.reader.start(read)
        self.status.setText({"preview": "Reading baseline and stable area…",
            "colour_fit": "Checking selected baseline and saved reference with the Finish setup colour producer…",
            "proposal": "Comparing baseline optical corrections…"}[operation])
        self._controls()

    def _completed(self, result, error):
        requested, self._requested = self._requested, None
        if requested is None or requested[1] != self._context():
            return
        if error:
            self.status.setText("Image correction preview unavailable: " + error)
        else:
            try:
                if requested[0] == "preview":
                    self._show_preview(result)
                elif requested[0] == "colour_fit":
                    self._show_colour_fit(result)
                else:
                    self._show_proposal(result)
            except (ValueError, TypeError, KeyError) as failure:
                self.status.setText("Image correction preview unavailable: " + str(failure))
        self._controls()

    def _show_colour_fit(self, result):
        if result["run"] != self.run.currentData() or result.get("complete") is not False:
            raise ValueError("The colour-fit preview must be read-only and belong to the selected baseline")
        label = f"{result['run'].upper()} → {result['reference_run'].upper()} · "
        if result["accepted"]:
            label += "ColorChecker fit passed. Preview only; Setup completion is unchanged."
        else:
            label += f"{result.get('failed_run', result['run']).upper()}: {result['error']}. No correction was accepted."
        self.colour_status.setText(label)
        self.colour_status.show()
        self.colour_details.setPlainText(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        self.fit_details.show()
        self.status.setText(result["note"])

    def _show_preview(self, result):
        if result["run"] != self.run.currentData():
            raise ValueError("The returned baseline belongs to another run.")
        rgb = np.asarray(result["rgb"])
        mask = None if result["stable_mask"] is None else np.asarray(result["stable_mask"])
        if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                or (mask is not None and (mask.dtype != np.bool_ or mask.shape != rgb.shape[:2]))):
            raise ValueError("The baseline and stable mask need the same declared image grid.")
        rgb = np.ascontiguousarray(rgb)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, int(rgb.strides[0]), QImage.Format.Format_RGB888).copy()
        # Band width is not part of the image source. Its support descriptor
        # changes without changing the baseline's coordinate system.
        binding = (result.get("baseline"), result.get("grid"))
        if self._photo_key != self._image_key()[:3] or self._photo_binding != binding:
            self.photo.set_image(image)
        else:
            # A band-only update retains the user's pan/zoom.
            self.photo.pixmap_item.setPixmap(QPixmap.fromImage(image))
        self._photo_key = self._image_key()[:3]
        self._photo_binding = binding
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        if mask is not None:
            rgba[mask] = (255, 210, 0, 120)
        overlay = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        self.mask_item.setPixmap(QPixmap.fromImage(overlay))
        self._mask = None if mask is None else mask.copy()
        self._displayed = self._image_key()
        self.mask_item.setVisible(self.show_mask.isChecked() and self._mask is not None)
        self.status.setText(str(result.get("note") or "Yellow: stable area sampled from the sand map's outer boundary."))
        self.preview_loaded.emit(result)

    def _show_proposal(self, result):
        proposed = result["reference_run"]
        rows = result["ranking"]
        if proposed not in self.choice.runs or not rows:
            raise ValueError("The reference proposal does not belong to this series.")
        saved = self.choice._saved_form[2] if self.choice._saved_form is not None else None
        self.ranking.setRowCount(len(rows))
        for index, row in enumerate(rows):
            run = str(row.get("run"))
            marks = [mark for mark, here in (("proposed", run == proposed), ("saved", run == saved)) if here]
            texts = (str(index + 1), run.upper() + (f" ({', '.join(marks)})" if marks else ""),
                     *(percent_correction(row.get(key)) for key in (
                         "score", "maximum_log_correction", "mean_light_correction", "mean_colour_correction")))
            for column, text in enumerate(texts):
                self.ranking.setItem(index, column, QTableWidgetItem(text))
        self.ranking.resizeColumnsToContents()
        self.ranking.show()
        self._proposal = deepcopy(result)
        compared = (" Runs are compared with their own light divided out, outside the band along the frame."
                    if self.choice.method.currentData() == EQUALISE_SAND else "")
        self.status.setText(f"Proposed reference: {proposed.upper()}. Use the proposal to change the form, then save explicitly. "
                            + str(result.get("note") or "") + compared)

    def _use_proposal(self):
        if self._proposal is None:
            return
        proposed = self._proposal["reference_run"]
        self.choice.reference.setCurrentIndex(self.choice.reference.findData(proposed))
        self.status.setText(f"{proposed.upper()} selected in the form. The saved reference is unchanged until you save.")

    def cancel_preview(self):
        self._auto_preview.stop()
        self.reader.cancel()
        self._requested = None
        self._controls()

    def showEvent(self, event):
        super().showEvent(event)
        self._auto_preview.start()

    def hideEvent(self, event):
        self.cancel_preview()
        super().hideEvent(event)

    def closeEvent(self, event):
        self.cancel_preview()
        super().closeEvent(event)
