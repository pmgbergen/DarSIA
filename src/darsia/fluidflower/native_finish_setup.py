"""Metadata-only final Setup page using the existing Setup job controller."""
import json
from pathlib import Path
import tomllib

from PySide6.QtCore import QObject, QSignalBlocker, Signal
from PySide6.QtWidgets import QFormLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

from .series_method_calls import SeriesMethodCalls
from .ui_metadata_preview import MetadataPreview


def setup_choices(root):
    from .pipeline.current_optical_preparation import colour_path_image_count
    path = Path(root) / "series.toml"
    with path.open("rb") as stream:
        raw = stream.read(4 * 1024**2 + 1)
    if len(raw) > 4 * 1024**2:
        raise ValueError("Series metadata exceeds the Setup preview limit.")
    value = tomllib.loads(raw.decode("utf-8-sig"))
    return dict(root=Path(root), raw=raw, document=value, count=colour_path_image_count(value),
                runs=list(value.get("runs", {}).get("ids", [])))


def finish_setup_status(root, registrations, *, snapshot_reader=None):
    """Inspect one registered Setup job, never RAW images or model directories."""
    from .jobs_metadata import JobSnapshotReader
    from .mass.n1_pipeline import asset
    from .pipeline.setup_image_preparation import completed_setup_colour_paths
    from .series_paths import series_folder_path
    choice = setup_choices(root)
    candidates = list(reversed([row for row in registrations if not row.get("archived")]))
    candidates.sort(key=lambda row: bool(row.get("active")), reverse=True)
    snapshot = None
    for row in candidates:
        try:
            with (Path(row["root"]) / "job.json").open("rb") as stream:
                raw = stream.read(4 * 1024**2 + 1)
        except FileNotFoundError:
            continue
        if len(raw) > 4 * 1024**2:
            raise ValueError("Job metadata exceeds the Setup preview limit.")
        if json.loads(raw).get("pipeline") == "series-setup":
            snapshot = (snapshot_reader or JobSnapshotReader())(row)
            break
    if snapshot is None:
        return dict(choice, complete=False, message="No Finish setup job is prepared. The action below prepares a fresh job for all Setup steps.")
    if not snapshot.get("complete") or snapshot.get("errors"):
        return dict(choice, complete=False, message="Finish setup is not complete. Follow the selected Setup job in Jobs; Build alone does not finish Setup.", snapshot=snapshot)
    try:
        receipt = completed_setup_colour_paths(snapshot, asset(Path(root) / "series.toml"))
        from .pipeline.setup_image_preparation import completed_image_preparation
        basis = completed_image_preparation(snapshot, asset(Path(root) / "series.toml"))
        current = series_folder_path(root, "geometry_facies") / "overrides/geometry_facies/current.json"
        if not current.is_file() or asset(current)["sha256"] != basis["adopted_revision"]["sha256"]:
            raise ValueError("The saved geometry changed after this Setup job.")
        return dict(choice, complete=True, receipt=receipt, snapshot=snapshot,
            message=f"✓ Setup complete: geometry, crop, lighting/colour corrections and separate colour paths per run and facies for all {len(receipt['runs'])} runs. Model training and mass calibration remain under Calibration.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(choice, complete=False, snapshot=snapshot, message="Finish setup is not complete for the current choices: " + str(error))


class SetupColourPathChoices(QWidget):
    saved = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.root = self.value = None
        self.preview = MetadataPreview(self)
        self.preview.completed.connect(self._loaded)
        self.calls = SeriesMethodCalls(self, verify_approval=False)
        self.calls.action_finished.connect(self._saved)
        form = QFormLayout(self)
        self.count = QSpinBox()
        self.count.setRange(1, 1000)
        self.count.setValue(15)
        self.count.valueChanged.connect(lambda: self.status.setText("Unsaved colour-path image count.") if self.value else None)
        form.addRow("Colour-path images per run", self.count)
        self.save_button = QPushButton("Save colour-path selection")
        self.save_button.clicked.connect(self._save)
        form.addRow(self.save_button)
        self.status = QLabel()
        self.status.setWordWrap(True)
        form.addRow(self.status)
        self.set_series(None)

    def set_series(self, root):
        self.root = Path(root) if root else None
        self.value = None
        self.preview.cancel()
        self.count.setEnabled(False)
        self.save_button.setEnabled(False)
        if self.root is not None:
            captured = self.root
            self.preview.start(lambda _cancelled: setup_choices(captured))

    def _loaded(self, value, error):
        if error:
            self.status.setText(error)
            return
        if value is None or value["root"] != self.root:
            return
        self.value = value
        with QSignalBlocker(self.count):
            self.count.setValue(value["count"])
        self.calls.set_context(self.root, value["count"], value["raw"], None)
        self.count.setEnabled(True)
        self.save_button.setEnabled(True)
        self.status.setText(f"Up to {value['count']} experiment images from each of {len(value['runs'])} runs, plus its saved references. Finish setup fits separate colour paths per run and facies.")

    def _save(self):
        if self.value is None:
            return
        from .colour_path_selection import save_colour_path_selection
        root, raw, count = self.root, self.value["raw"], self.count.value()
        def save():
            save_colour_path_selection(root, count, expected_bytes=raw, context_current=lambda: self.root == root)
            return setup_choices(root)
        if self.calls.start_action("setup_colour_path_choice", save, self.calls.context):
            self.save_button.setEnabled(False)
            self.status.setText("Saving colour-path selection…")

    def _saved(self, result):
        if not result["current"]:
            return
        if result["error"]:
            self.status.setText("Not saved: " + result["error"])
            self.save_button.setEnabled(True)
            return
        self._loaded(result["result"], "")
        self.status.setText("✓ Colour-path selection saved. Run Finish setup to apply it. No model or calibration starts.")
        self.saved.emit(self.root)

    def can_close(self):
        self.preview.cancel()
        return self.calls.can_close()


class _SetupContext(QObject):
    reference_changed = Signal()

    def __init__(self, window, parent):
        super().__init__(parent)
        self.owner = window
        self.project = self._loaded_root = None
        self._loaded_series = self.series_id = ""

    def window(self):
        return self.owner


class FinishSetupPage(QWidget):
    busy_changed = Signal(bool)

    def __init__(self, window):
        super().__init__(window)
        self.root = None
        self.setup_complete = False
        self.preview = MetadataPreview(self)
        self.preview.completed.connect(self._loaded)
        self.workspace = _SetupContext(window, self)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Finish setup"))
        note = QLabel("Run all Setup steps from the saved choices: apply geometry, build each run, prepare the crop and selected lighting/colour corrections, then generate colour paths. Model training, phase maps and mass calibration belong to Calibration.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.apply_build_button = QPushButton("Finish setup")
        self.apply_build_reason = QLabel("Choose a series and save the Setup choices first.")
        self.apply_build_reason.setWordWrap(True)
        self.apply_build_job_label, self.apply_build_job_edit = QLabel(), QLabel()
        layout.addWidget(self.apply_build_button)
        layout.addWidget(self.apply_build_reason)
        self.refresh_button = QPushButton("Refresh Setup status")
        self.refresh_button.clicked.connect(lambda: self.set_series(self.root))
        layout.addWidget(self.refresh_button)
        self.calibration_button = QPushButton("Continue to calibration")
        self.calibration_button.clicked.connect(lambda: window._select_row(("calibration", "mass")))
        layout.addWidget(self.calibration_button)
        layout.addStretch(1)
        from .series_setup_jobs_ui import SeriesSetupJobsController
        self.controller = SeriesSetupJobsController(self.workspace, self)
        self.controller.colour_paths_button.hide()
        self.apply_build_button.setText("Finish setup")
        self.apply_build_button.setToolTip("Prepare a fresh job for the entire Setup workflow, then start its workers in Jobs.")
        self.calibration_button.setEnabled(False)

    def busy(self):
        return False

    def set_series(self, root):
        self.root = Path(root) if root else None
        self.setup_complete = False
        self.calibration_button.setEnabled(False)
        self.apply_build_button.setEnabled(False)
        self.preview.cancel()
        if self.root is None:
            self.status.setText("Choose an experimental series on Start.")
            return
        window, captured = self.workspace.owner, self.root
        jobs = window.ensure_jobs()
        series = jobs.series_id
        registrations = jobs.registry.jobs(captured, series)
        self.status.setText("Reading saved Setup status; no images are loaded…")
        def read(_cancelled):
            from .project import FluidFlowerProject
            from .series_paths import series_folder_path
            result = finish_setup_status(captured, registrations)
            result["project"] = FluidFlowerProject.load(series_folder_path(captured, "geometry_facies"))
            result["series_id"] = series
            return result
        self.preview.start(read)

    def _loaded(self, value, error):
        if error:
            self.status.setText("Setup status unavailable: " + error)
            return
        if value is None or value["root"] != self.root:
            return
        self.workspace.project, self.workspace._loaded_root = value["project"], self.root
        self.workspace._loaded_series = self.workspace.series_id = value["series_id"]
        self.setup_complete = value["complete"]
        self.status.setText(value["message"])
        self.calibration_button.setEnabled(self.setup_complete)
        self.controller.refresh()

    def can_close(self):
        self.preview.cancel()
        return not self.controller.busy()
