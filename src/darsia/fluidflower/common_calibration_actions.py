"""Inline options for the existing colour-signal calibration process."""

from pathlib import Path

from PySide6.QtCore import QProcess, QSettings, Signal
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)


class PhaseColourPathPage(QWidget):
    """Colour-path choices; the separate Finish setup page owns execution."""

    images_requested = Signal()
    preparation_requested = Signal()
    calibration_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Colour paths"))
        text = QLabel(
            "Choose representative experiment images for optical colour paths from every included run. "
            "Finish setup calculates separate colour paths for each run and facies after the saved crop, lighting and colour corrections. "
            "This does not require a phase model. Model training and phase maps belong to Calibration.")
        text.setWordWrap(True)
        layout.addWidget(text)
        self.source = QLabel()
        self.source.setWordWrap(True)
        layout.addWidget(self.source)
        self.images_button = QPushButton("Inspect Setup image selection")
        self.images_button.clicked.connect(self.images_requested)
        layout.addWidget(self.images_button)
        self.prepare_button = QPushButton("Open preparation controls")
        self.prepare_button.clicked.connect(self.preparation_requested)
        layout.addWidget(self.prepare_button)
        self.prepare_button.hide()
        note = QLabel(
            "Opening this page starts no job. Colour paths use their own smaller selection "
            "from each run's own saved images; paths are not pooled across runs. A fixed shared scale per facies "
            "keeps the signal coordinates comparable between runs and is retained for production. Mass calibration and production "
            "retain their separate image selections.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.preparation_host = QWidget()
        QVBoxLayout(self.preparation_host).setContentsMargins(0, 0, 0, 0)
        from .native_finish_setup import SetupColourPathChoices
        self.choices = SetupColourPathChoices(self)
        self.preparation_host.layout().addWidget(self.choices)
        layout.addWidget(self.preparation_host)
        self.calibration_button = QPushButton("Continue to calibration")
        self.calibration_button.clicked.connect(self.calibration_requested)
        layout.addWidget(self.calibration_button)
        self.calibration_button.hide()
        layout.addStretch(1)
        self.set_series(None)

    def set_series(self, series_root):
        self.source.setText("Series: " + str(series_root) if series_root is not None
                            else "Choose an experimental series on Start.")
        self.images_button.setEnabled(series_root is not None)
        self.prepare_button.setEnabled(series_root is not None)
        self.choices.set_series(series_root)
        self.set_setup_complete(False)

    def set_setup_complete(self, complete):
        self.calibration_button.setEnabled(bool(complete))
        self.calibration_button.setToolTip(
            "Open mass calibration to review its choices. Opening starts no job."
            if complete else "Complete colour paths and phase inputs to finish Setup.")

    def mount_preparation(self, preparation):
        """Compatibility entry: Setup never mounts model/P1 preparation."""
        self.prepare_button.hide()


class OriginalCalibrationControls(QWidget):
    """Opening or editing options never starts or changes a calibration."""

    start_requested = Signal(dict)
    inputs_requested = Signal()
    interactive_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.explanation = QLabel(
            "Fit the colour-path signal values to injected mass using the existing "
            "automatic colour calibration. The saved simple-flash split is retained. "
            "The previous calibration and trial report are kept by that calculation.")
        self.explanation.setWordWrap(True)
        layout.addWidget(self.explanation)
        self.parameters = QWidget()
        form = QFormLayout(self.parameters)
        form.setContentsMargins(0, 0, 0, 0)
        settings = QSettings()
        self.trials = QSpinBox()
        self.trials.setRange(10, 5000)
        self.trials.setSingleStep(50)
        self.trials.setValue(settings.value("native_app/optuna/trials", 200, type=int))
        form.addRow("Trials", self.trials)
        self.values = QComboBox()
        self.values.addItem("Shared by the sand facies", False)
        self.values.addItem("Separate per sand facies", True)
        self.values.setCurrentIndex(
            int(settings.value("native_app/optuna/per_facies", False, type=bool)))
        self.values.setToolTip(
            "This chooses shared or separate signal functions across facies. "
            "It does not select a shared calibration across experimental runs.")
        form.addRow("Signal functions", self.values)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(settings.value("native_app/optuna/seed", 0, type=int))
        form.addRow("Seed", self.seed)
        layout.addWidget(self.parameters)
        self.inputs_button = QPushButton("Inspect calibration inputs")
        self.inputs_button.clicked.connect(self.inputs_requested)
        layout.addWidget(self.inputs_button)
        self.start_button = QPushButton("Start automatic calibration")
        self.start_button.clicked.connect(lambda: self.start_requested.emit(self.options()))
        layout.addWidget(self.start_button)
        self.interactive_button = QPushButton("Open interactive calibration controls")
        self.interactive_button.clicked.connect(self.interactive_requested)
        layout.addWidget(self.interactive_button)
        self.status = QLabel(
            "Uses the selected run's saved calibration.mass image selection and regions. "
            "Start is an explicit action; changing these options starts nothing.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def options(self):
        return dict(trials=self.trials.value(), per_facies=bool(self.values.currentData()),
                    seed=self.seed.value())

    def remember(self):
        settings = QSettings()
        for name, value in self.options().items():
            settings.setValue("native_app/optuna/" + name, value)


def start_original_automatic(window, controls, options):
    """Use the original process runner, only for an explicit compatible GUI action."""
    from .native_darsia_app import _same_file, optuna_mass_calibration_argv

    try:
        if window._saved_method.active != "colour_path":
            raise ValueError("Select the colour-signal model to use this calibration.")
        if getattr(window, "_series_loading", False):
            raise ValueError("Wait for the selected series to finish loading.")
        process = getattr(window.calibration_tab, "process", None)
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            raise ValueError("A calibration process is already running.")
        window._ensure_run_loaded()
        series = window.phase_model_workspace.series_id
        selected = window.run_combo.currentData()
        config_file = window.config_file
        if (not series or window._config_series != series or not selected
                or not config_file or not _same_file(config_file, selected)):
            raise ValueError("Open the selected run's configuration before starting calibration.")
        config_file = Path(config_file)
        if not config_file.is_file():
            raise ValueError("The selected run's saved configuration is unavailable.")
        controls.remember()
        toolbar = window.toolbar_builder
        process = window.process_runner.start_workflow_process(
            optuna_mass_calibration_argv(config_file, options),
            toolbar.play_action, toolbar.stop_action, cwd=Path.cwd(),
            workflow="calibration", actions=["mass"], config_path=config_file)
        window.calibration_tab.process = process
        controls.status.setText(
            "Calibration launch requested using " + str(config_file)
            + ". Follow the original process log for progress and completion.")
        return process
    except (OSError, ValueError, TypeError) as error:
        controls.status.setText(str(error))
        window.print_log(str(error))
        return None
