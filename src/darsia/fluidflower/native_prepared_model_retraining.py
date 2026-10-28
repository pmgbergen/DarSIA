"""Retrain an existing architecture from its saved labels on current Setup images."""
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from .series_method_calls import SeriesMethodCalls


class PreparedModelRetraining(QObject):
    progress = Signal(object)

    def __init__(self, steps):
        super().__init__(steps)
        self.steps, self.studio = steps, steps.studio
        self.calls = SeriesMethodCalls(self, verify_approval=False)
        self.calls.action_finished.connect(self._prepared)
        self.progress.connect(self._progress)
        self._context = None

    @property
    def busy(self):
        return self.calls.thread is not None

    def window_owner(self):
        getter = getattr(self.steps, "window", None)
        window = getter() if callable(getter) else None
        return window if callable(getattr(window, "ensure_jobs", None)) else None

    def available(self):
        return bool(self.steps.series_root is not None and self.window_owner() is not None
                    and self.steps._selected_calibration_model())

    def start(self):
        """Only the explicit Train action prepares files and launches fitting."""
        if self.busy or self.studio.training_process is not None:
            return
        if not self.available():
            raise ValueError("Choose an existing model and complete its current Setup Apply/Build first.")
        from .calibration_preparation import prepare_calibration_training, training_worker_count
        window = self.window_owner()
        jobs = window.ensure_jobs()
        root, series = Path(self.steps.series_root), self.steps.series_id
        execution_series = str(getattr(jobs, "series_id", "") or series)
        template = self.steps._selected_calibration_model()
        name = self.steps.model_name.text().strip()
        actor = self.studio.reviewer.text().strip()
        if not name or not actor:
            raise ValueError("Name the new model version before retraining.")
        registrations = jobs.registry.jobs(root, execution_series)
        workers = training_worker_count(jobs)
        context = (root, series, template, name)
        self._context = context
        self.calls.set_context(root, context, None, None)
        def prepare():
            return prepare_calibration_training(root, execution_series, registrations, actor=actor,
                workers=workers, template_reference=template, on_progress=self.progress.emit)
        if self.calls.start_action("prepared_model_retraining", prepare, self.calls.context):
            self.studio.status.setText("Preparing existing saved labels on the current Setup crop, lighting and colour basis. No new boxes are needed…")
            self.steps.refresh()

    def _progress(self, value):
        if not self.busy:
            return
        self.studio.status.setText("Preparing saved labels: " + " · ".join(str(value[key])
            for key in ("state", "run", "position", "total") if key in value))

    def _current(self):
        return (Path(self.steps.series_root) if self.steps.series_root else None,
                self.steps.series_id, self.steps._selected_calibration_model(), self.steps.model_name.text().strip())

    def _prepared(self, result):
        if result["error"]:
            self.studio.status.setText("Retraining did not start: " + result["error"])
            self.steps.refresh()
            return
        if not result["current"] or self._current() != self._context:
            self.studio.status.setText("Prepared label inputs were retained; the selected series, model or version name changed. No training started.")
            self.steps.refresh()
            return
        value = result["result"]
        if value.get("model_ready"):
            self.studio.status.setText("A completed model already exists on this Setup basis: "
                + str(value["request"]["output_root"]) + ". Select that version for calibration.")
            self.steps.refresh()
            return
        try:
            from .calibration_model_training import launch_prepared_training
            launch_prepared_training(self.steps, value, name=self._context[3])
            self.studio.status.setText("Training a new model version on current Setup images, using only the selected model's saved labels. No new boxes or calibration job are created.")
        except Exception as error:
            self.studio.status.setText("Retraining did not start: " + str(error))
        self.steps.refresh()

    def can_close(self):
        return self.calls.can_close()
