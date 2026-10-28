"""Drop-in Qt panel for measured indicator-response fitting."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QWidget,
)

from darsia.fluidflower.calibration import apply_indicator_response_constraints
from darsia.fluidflower.indicator_response import (
    IndicatorResponseModelStore,
    default_indicator_response_config,
)
from darsia.fluidflower.project import FluidFlowerProject


class IndicatorResponseFitWorker(QObject):
    """Execute full-resolution buffer fitting away from the Qt event loop."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        project: FluidFlowerProject,
        config: Mapping[str, Any],
        *,
        actor: str,
        reason: str,
    ) -> None:
        super().__init__()
        self.project = project
        self.config = deepcopy(dict(config))
        self.actor = str(actor)
        self.reason = str(reason)

    @Slot()
    def run(self) -> None:
        try:
            revision = IndicatorResponseModelStore(self.project).fit_and_save(
                self.config,
                actor=self.actor,
                reason=self.reason,
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(revision)


class IndicatorResponsePanel(QGroupBox):
    """Self-contained editor, fitter, and current-model selector."""

    model_changed = Signal(object)
    use_changed = Signal(bool)

    def __init__(
        self, project: FluidFlowerProject, parent: QWidget | None = None
    ) -> None:
        super().__init__("Measured indicator response", parent)
        self.project = project
        self.store = IndicatorResponseModelStore(project)
        self._thread: QThread | None = None
        self._worker: IndicatorResponseFitWorker | None = None
        self._build_ui()
        self.load_current()

    @staticmethod
    def _concentration_field(*, maximum: float = 100.0) -> QDoubleSpinBox:
        field = QDoubleSpinBox()
        field.setRange(0.0, maximum)
        field.setDecimals(6)
        field.setSingleStep(0.01)
        field.setSuffix(" mM")
        field.setKeyboardTracking(False)
        return field

    @staticmethod
    def _ph_field() -> QDoubleSpinBox:
        field = QDoubleSpinBox()
        field.setRange(0.501, 13.499)
        field.setDecimals(3)
        field.setSingleStep(0.05)
        field.setKeyboardTracking(False)
        return field

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        explanation = QLabel(
            "Known-pH buffer images are registered and corrected with the current "
            "image pipeline. A monotone full-color path is fitted per material class, "
            "then linked to DarSIA's carbonate model. The fitted color path is fixed "
            "during Optuna optimization."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("secondaryText")
        layout.addRow(explanation)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setObjectName("integrationNotice")
        layout.addRow("Current model", self.status)

        self.use_model = QCheckBox(
            "Attach the current deployable response automatically downstream"
        )
        self.use_model.setToolTip(
            "A deployable measured response is fixed in later mass-calibration plans. "
            "Select another immutable response revision in this module to change it."
        )
        self.use_model.toggled.connect(self.use_changed)
        layout.addRow("Downstream contract", self.use_model)

        self.per_facies = QCheckBox("Fit robust response separately per facies")
        self.per_facies.setChecked(True)
        self.per_facies.setToolTip(
            "Separates material reflectance from indicator response while retaining "
            "a global fallback model."
        )
        layout.addRow("Material handling", self.per_facies)
        self.maximum_samples = QSpinBox()
        self.maximum_samples.setRange(100, 2_000_000)
        self.maximum_samples.setValue(30_000)
        self.maximum_samples.setSingleStep(5_000)
        self.maximum_samples.setKeyboardTracking(False)
        self.maximum_samples.setToolTip(
            "Maximum deterministic pixel sample per image and facies. Source images "
            "remain full resolution; this only bounds robust statistics."
        )
        layout.addRow("Pixels per image/facies", self.maximum_samples)

        bounds_widget = QWidget()
        bounds = QGridLayout(bounds_widget)
        bounds.setContentsMargins(0, 0, 0, 0)
        for column, title in enumerate(("Value", "Lower", "Upper"), start=1):
            bounds.addWidget(QLabel(title), 0, column)
        self.ph_value = self._ph_field()
        self.ph_low = self._ph_field()
        self.ph_high = self._ph_field()
        self.alkalinity_value = self._concentration_field(maximum=20.0)
        self.alkalinity_low = self._concentration_field(maximum=20.0)
        self.alkalinity_high = self._concentration_field(maximum=20.0)
        self.indicator_value = self._concentration_field(maximum=20.0)
        self.indicator_low = self._concentration_field(maximum=20.0)
        self.indicator_high = self._concentration_field(maximum=20.0)
        self.saturation_value = self._concentration_field(maximum=200.0)
        self.saturation_low = self._concentration_field(maximum=200.0)
        self.saturation_high = self._concentration_field(maximum=200.0)
        rows = (
            ("Initial pH", self.ph_value, self.ph_low, self.ph_high),
            (
                "Effective alkalinity",
                self.alkalinity_value,
                self.alkalinity_low,
                self.alkalinity_high,
            ),
            (
                "BTB concentration",
                self.indicator_value,
                self.indicator_low,
                self.indicator_high,
            ),
            (
                "CO2 saturation",
                self.saturation_value,
                self.saturation_low,
                self.saturation_high,
            ),
        )
        for row_index, (label, value, low, high) in enumerate(rows, start=1):
            bounds.addWidget(QLabel(label), row_index, 0)
            bounds.addWidget(value, row_index, 1)
            bounds.addWidget(low, row_index, 2)
            bounds.addWidget(high, row_index, 3)
        layout.addRow("Chemistry and bounds", bounds_widget)

        self.bounds_reviewed = QCheckBox(
            "Values and uncertainty bounds are measured or protocol-derived"
        )
        self.bounds_reviewed.setToolTip(
            "Leave unchecked for exploratory optical fitting. A provisional model "
            "cannot constrain an Optuna production plan."
        )
        layout.addRow("Scientific review", self.bounds_reviewed)
        self.author = QLineEdit(str(self.project.manifest.get("created_by") or ""))
        self.reason = QLineEdit()
        self.reason.setPlaceholderText("Source and purpose of this response fit")
        layout.addRow("Model author", self.author)
        layout.addRow("Fit note", self.reason)

        actions = QHBoxLayout()
        self.fit_button = QPushButton("Fit buffer response")
        self.fit_button.clicked.connect(self.start_fit)
        actions.addWidget(self.fit_button)
        self.refresh_button = QPushButton("Refresh current model")
        self.refresh_button.clicked.connect(self.load_current)
        actions.addWidget(self.refresh_button)
        actions.addStretch(1)
        layout.addRow(actions)

    def _set_concentration_triplet(
        self,
        value_field: QDoubleSpinBox,
        low_field: QDoubleSpinBox,
        high_field: QDoubleSpinBox,
        value_molar: float,
        bounds_molar: list[float],
    ) -> None:
        value_field.setValue(1000.0 * value_molar)
        low_field.setValue(1000.0 * float(bounds_molar[0]))
        high_field.setValue(1000.0 * float(bounds_molar[1]))

    def _load_configuration(self, config: Mapping[str, Any]) -> None:
        sampling = dict(config.get("sampling") or {})
        self.per_facies.setChecked(bool(sampling.get("per_facies", True)))
        self.maximum_samples.setValue(
            int(sampling.get("max_pixels_per_image_and_facies") or 30_000)
        )
        chemistry = dict(config.get("chemistry") or {})
        ph_bounds = list(chemistry.get("initial_ph_bounds") or [6.0, 10.0])
        self.ph_low.setValue(float(ph_bounds[0]))
        self.ph_high.setValue(float(ph_bounds[1]))
        self.ph_value.setValue(0.5 * (float(ph_bounds[0]) + float(ph_bounds[1])))
        measured = dict(chemistry.get("measured_bounds") or {})
        self._set_concentration_triplet(
            self.alkalinity_value,
            self.alkalinity_low,
            self.alkalinity_high,
            float(chemistry.get("alkalinity_M") or 0.0),
            list(measured.get("alkalinity_M") or [0.0, 0.0]),
        )
        self._set_concentration_triplet(
            self.indicator_value,
            self.indicator_low,
            self.indicator_high,
            float(chemistry.get("btb_M") or 0.0),
            list(measured.get("btb_M") or [0.0, 0.0]),
        )
        self._set_concentration_triplet(
            self.saturation_value,
            self.saturation_low,
            self.saturation_high,
            float(chemistry.get("co2_saturation_M") or 0.0),
            list(measured.get("co2_saturation_M") or [0.0, 0.0]),
        )
        self.bounds_reviewed.setChecked(bool(chemistry.get("bounds_reviewed")))

    def configuration(self) -> dict[str, Any]:
        config = default_indicator_response_config(self.project)
        chemistry = dict(config["chemistry"])
        chemistry.update(
            {
                "alkalinity_M": self.alkalinity_value.value() / 1000.0,
                "btb_M": self.indicator_value.value() / 1000.0,
                "co2_saturation_M": self.saturation_value.value() / 1000.0,
                "initial_ph_bounds": [self.ph_low.value(), self.ph_high.value()],
                "measured_bounds": {
                    "initial_ph": [self.ph_low.value(), self.ph_high.value()],
                    "alkalinity_M": [
                        self.alkalinity_low.value() / 1000.0,
                        self.alkalinity_high.value() / 1000.0,
                    ],
                    "btb_M": [
                        self.indicator_low.value() / 1000.0,
                        self.indicator_high.value() / 1000.0,
                    ],
                    "co2_saturation_M": [
                        self.saturation_low.value() / 1000.0,
                        self.saturation_high.value() / 1000.0,
                    ],
                },
                "bounds_reviewed": self.bounds_reviewed.isChecked(),
                "parameter_source": (
                    "reviewed_in_fluidflower_gui"
                    if self.bounds_reviewed.isChecked()
                    else "unreviewed_gui_values"
                ),
            }
        )
        config["chemistry"] = chemistry
        config["sampling"].update(
            {
                "per_facies": self.per_facies.isChecked(),
                "max_pixels_per_image_and_facies": self.maximum_samples.value(),
            }
        )
        return config

    def load_current(self) -> None:
        revision = self.store.current()
        if not revision:
            self._load_configuration(default_indicator_response_config(self.project))
            reference = dict(
                (self.project.manifest.get("chemistry") or {}).get(
                    "indicator_response_reference"
                )
                or {}
            )
            self.status.setText(
                f"No fitted model. {int(reference.get('known_ph_buffer_count') or 0)} "
                "known-pH buffer image(s) are registered."
            )
            self.use_model.setChecked(False)
            self.use_model.setEnabled(False)
            return
        config = dict(revision.get("configuration") or {})
        if config:
            self._load_configuration(config)
        summary = dict(revision.get("summary") or {})
        validation = dict(revision.get("validation") or {})
        status = str(revision.get("status") or "unknown")
        self.status.setText(
            f"{status.replace('_', ' ').upper()} · model "
            f"{str(revision.get('model_id') or '')[:8]} · "
            f"{summary.get('reference_image_count', 0)} images / "
            f"{summary.get('distinct_ph_count', 0)} pH levels / "
            f"{summary.get('facies_group_count', 0)} response groups · "
            f"holdout records {validation.get('holdout_record_count', 0)}"
        )
        deployable = status in {"ready", "ready_limited_holdout"}
        self.use_model.setChecked(deployable)
        self.use_model.setEnabled(False)

    def current_reference(self) -> dict[str, Any]:
        return self.store.current_reference()

    def constrain_plan(self, config: Mapping[str, Any]) -> dict[str, Any]:
        if not self.use_model.isChecked():
            return deepcopy(dict(config))
        return apply_indicator_response_constraints(config, self.current_reference())

    @Slot()
    def start_fit(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        actor = self.author.text().strip()
        if not actor:
            QMessageBox.warning(
                self,
                "Model author required",
                "Enter the person responsible for the chemical-response fit.",
            )
            return
        self.fit_button.setEnabled(False)
        self.status.setText(
            "Registering and correcting full-resolution buffers, fitting robust "
            "facies paths, and running leave-one-image-out validation..."
        )
        thread = QThread(self)
        worker = IndicatorResponseFitWorker(
            self.project,
            self.configuration(),
            actor=actor,
            reason=self.reason.text().strip(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._fit_ready)
        worker.failed.connect(self._fit_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    @Slot(object)
    def _fit_ready(self, revision: object) -> None:
        self.reason.clear()
        self.load_current()
        current = dict(revision) if isinstance(revision, Mapping) else {}
        if str(current.get("status") or "") in {"ready", "ready_limited_holdout"}:
            self.use_model.setChecked(True)
        self.model_changed.emit(current)

    @Slot(str)
    def _fit_failed(self, message: str) -> None:
        self.status.setText("Buffer-response fitting failed; no current model changed.")
        QMessageBox.critical(self, "Could not fit indicator response", message)

    @Slot()
    def _thread_finished(self) -> None:
        if self._thread is not None:
            # After wait() the worker is freed here, in the GUI thread (Q185).
            self._thread.wait()
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self.fit_button.setEnabled(True)


__all__ = ["IndicatorResponseFitWorker", "IndicatorResponsePanel"]
