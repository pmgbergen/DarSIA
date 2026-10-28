"""Display actual Setup image-preparation completion separately from saved choices."""
import hashlib
from pathlib import Path
import tomllib

from PySide6.QtCore import QStandardPaths, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from .ui_metadata_preview import MetadataPreview


class PreparationStatusSource:
    def __init__(self):
        from .native_jobs_registry import JobsUIRegistry
        from .jobs_metadata import JobSnapshotReader
        self.registry = JobsUIRegistry(Path(QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppLocalDataLocation)) / "native_jobs")
        self.jobs = JobSnapshotReader()

    def __call__(self, root, series, original):
        from .pipeline.setup_image_preparation import completed_image_preparation, OPERATION
        from .image_normalisation import saved_choice, selected_corrections
        from .series_paths import series_key
        root = Path(root)
        document = tomllib.loads(original.decode("utf-8-sig"))
        actual = document.get("series_id")
        if not isinstance(actual, str) or series_key(actual) != series_key(series):
            raise ValueError("The selected folder's series manifest identifies another series.")
        if document.get("method", {}).get("active") != "phase_model":
            return dict(complete=False, text="The original colour method uses its saved Rig correction settings.")
        # Existing choices may retain Studio's canonical spelling. Keep their
        # original record/approval bytes while verifying the same series owner.
        owner = (document.get("image_normalisation") or {}).get("series_id", actual)
        if series_key(owner) != series_key(actual):
            raise ValueError("The saved image choice belongs to another series.")
        choice = saved_choice(document, owner, document.get("runs", {}).get("ids", []))
        if not choice or "color_method" not in choice or selected_corrections(choice)["color_method"] == "saved_native":
            return dict(complete=False, text="The saved native correction route is selected. "
                        "Save independent image settings to prepare a new image basis.")
        authority = dict(path=str(root / "series.toml"), size_bytes=len(original),
                         sha256=hashlib.sha256(original).hexdigest())
        rows = self.registry.jobs(root, actual)
        rows.sort(key=lambda row: bool(row.get("active")), reverse=True)
        pending = "No completed image correction for these saved settings. Update and start the Setup job."
        for row in rows:
            job = self.jobs.read(Path(row["root"]) / "job.json", row["job"])
            if job.get("pipeline") != "series-setup":
                continue
            snapshot = self.jobs(row)
            operations = [item for item in snapshot.get("runs", []) if item.get("operation") == OPERATION]
            if not operations:
                continue
            try:
                prepared = completed_image_preparation(snapshot, authority)
            except (ValueError, OSError, KeyError) as error:
                state = str(operations[0].get("state") or "Pending")
                pending = f"Image correction: {state}. {error}"
                continue
            return dict(complete=True, text=f"Image correction completed for {len(prepared['runs'])} runs. "
                        "These saved settings and this geometric basis match the prepared output.",
                        receipt=prepared["receipt"])
        return dict(complete=False, text=pending)


class ImagePreparationStatus(QWidget):
    # True only while a completed preparation matches the saved settings.
    completion_changed = Signal(bool)

    def __init__(self, choice, parent=None, *, loader=None):
        super().__init__(parent)
        self.choice = choice
        self.loader = loader or PreparationStatusSource()
        self.reader = MetadataPreview(self)
        self.reader.completed.connect(self._completed)
        self._requested = None
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.status = QLabel("Save image settings before checking the prepared output.")
        self.status.setWordWrap(True)
        row.addWidget(self.status, 1)
        self.refresh_button = QPushButton("Refresh preparation status")
        self.refresh_button.clicked.connect(self.refresh)
        row.addWidget(self.refresh_button)
        choice.context_updated.connect(self.refresh)
        for name in ("method", "color_method", "reference", "stable_band"):
            control = getattr(choice, name, None)
            if control is not None:
                signal = control.valueChanged if name == "stable_band" else control.currentIndexChanged
                signal.connect(self._draft_changed)
        self.refresh()

    def _context(self):
        return (str(self.choice.series_root or ""), self.choice.series_id, self.choice.original)

    def refresh(self):
        self.reader.cancel()
        context = self._context()
        self._requested = context
        root, series, original = context
        ready = bool(root and series and original is not None)
        self.refresh_button.setEnabled(ready)
        self.status.setStyleSheet("")
        if ready and not self._matches_saved():
            self.status.setText("Unsaved image settings. Save them before checking or running their preparation.")
            self.completion_changed.emit(False)
            return
        if not ready:
            self.status.setText("Save image settings before checking the prepared output.")
            self.completion_changed.emit(False)
            return
        self.status.setText("Reading the saved Setup image-correction status…")
        self.refresh_button.setEnabled(False)
        self.reader.start(lambda cancelled: self.loader(root, series, original))

    def _matches_saved(self):
        if not hasattr(self.choice, "method"):
            return True
        from .image_normalisation import saved_choice, selected_corrections
        try:
            saved = saved_choice(tomllib.loads(self.choice.original.decode("utf-8-sig")),
                                 self.choice.series_id, self.choice.runs)
            if not saved:
                return False
            band = saved.get("exclusion_band") or saved.get("stable_area") or {}
            return (self.choice.method.currentData() == saved["method"]
                    and self.choice.color_method.currentData() == selected_corrections(saved)["color_method"]
                    and self.choice.reference.currentData() == saved.get("reference_run")
                    and self.choice.stable_band.value() == band.get("width_mm", 0.0))
        except (ValueError, TypeError, KeyError, AttributeError):
            return False

    def _draft_changed(self, *_args):
        self.refresh()

    def _completed(self, value, error):
        if self._requested != self._context() or not self._matches_saved():
            return
        self.refresh_button.setEnabled(True)
        if error:
            self.status.setText("Preparation status unavailable: " + error)
            self.status.setStyleSheet("")
            self.completion_changed.emit(False)
            return
        self.status.setText(value["text"])
        self.completion_changed.emit(bool(value.get("complete")))
        self.status.setStyleSheet("color: #25a56a;" if value.get("complete") else "")

    def closeEvent(self, event):
        self.reader.cancel()
        super().closeEvent(event)
