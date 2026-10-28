"""Attach explicit saved-model continuation to one native Phase model page."""
from __future__ import annotations

import json
from contextvars import copy_context
from copy import deepcopy
from pathlib import Path
from threading import Thread

from PySide6.QtCore import QObject, Signal

from .phase_training import continuation
from .phase_training.scope import requested_reviewers, series_id


class _Readiness(QObject):
    finished = Signal(object)


class NativeContinuationTraining:
    """One page's continuation readiness, request launch and honest completion."""

    def __init__(self, steps):
        """Attach readiness, launch and completion hooks to the training page."""
        self.steps, self.studio = steps, steps.studio
        self.ready, self.reason = False, "Choose an indexed saved model as the training basis."
        self._cache_key = None
        self._parent_reviewers = None
        self._requested_key = None
        self._reading_key = None
        self._result = None
        self._reader = _Readiness(steps) if isinstance(steps, QObject) else None
        if self._reader is not None:
            self._reader.finished.connect(self._inspected)
        self._original_finished = getattr(self.studio, "_training_finished", None)
        steps.continuation_training = self
        steps.training_routes["existing"] = self.start
        self.studio._training_finished = self.finished
        for name, signal_name in (("annotators", "itemChanged"), ("bundles", "textChanged"),
                                  ("validation_model", "currentIndexChanged")):
            signal = getattr(getattr(self.studio, name, None), signal_name, None)
            if signal is not None:
                signal.connect(lambda *_args: steps.refresh())

    def _inspect(self, entry, selected_series):
        """The original admission, including source hashes, stays off the GUI thread."""
        root = Path(entry["root"])
        paths = [root / "training_request.json", *[root / "models/generic" / f"{head}_signal.json"
                                                   for head in continuation.HEADS]]
        labels = {str(paths[0]): "saved training request", **{
            str(path): "saved " + head.replace("_", " ").replace("co2", "CO2") + " model file"
            for head, path in zip(continuation.HEADS, paths[1:])}}
        try:
            stamps = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths)
            _bound, parent, _previous, _heads = continuation.inspect_parent(entry)
            selected = {series_id(value) for value in parent["settings"].get("series") or ()}
            if selected != {series_id(selected_series)}:
                raise ValueError("The selected model belongs to another series.")
            return (stamps, requested_reviewers(parent["settings"])), ""
        except (OSError, KeyError, TypeError, ValueError) as error:
            detail = str(error)
            label = labels.get(str(error.filename)) if isinstance(error, OSError) else None
            if label:
                detail = ("The selected model's " + label + " is missing. Choose a saved model that includes it."
                    if isinstance(error, FileNotFoundError) else
                    "The selected model's " + label + " cannot be read. Check access to the saved model files.")
            elif isinstance(error, KeyError) and error.args == ("settings",):
                detail = ("This saved revision has no training settings. Choose its original trained model "
                          "to continue training, or Start from scratch for a new fit.")
            return None, "Continuation unavailable: " + detail

    def _inspected(self, result):
        key, value = result
        self._reading_key = None
        if key == self._requested_key:
            self._result = value
        self.steps.refresh()

    def _parent_readiness(self, entry):
        if self._reader is None:
            return self._inspect(entry, self.steps.series_id)
        key = (str(entry.get("root") or ""), json.dumps(entry.get("library_source_identity"), sort_keys=True),
               self.steps.series_id, getattr(self.studio, "_model_catalog_generation", 0))
        if key != self._requested_key:
            self._requested_key, self._result = key, None
        if self._result is None and self._reading_key is None:
            self._reading_key = key
            captured, selected_series, context = deepcopy(entry), self.steps.series_id, copy_context()
            def inspect():
                result = self._inspect(captured, selected_series)
                try:
                    self._reader.finished.emit((key, result))
                except RuntimeError:
                    pass  # The page was closed while its selected source was being read.
            Thread(target=lambda: context.run(inspect), name="phase-continuation-readiness", daemon=True).start()
        return self._result or (None, "Checking the selected model's continuation settings in the background…")

    def refresh(self):
        """Return the model-specific reason; opening the page writes nothing."""
        try:
            getter = getattr(self.studio, "_selected_validation_model", None)
            entry = getter() if callable(getter) else None
            if not entry:
                self._requested_key, self._result = None, None
                raise ValueError("Choose an indexed saved model in Basis before continuing training.")
            result, reason = self._parent_readiness(entry)
            if reason:
                self.ready, self.reason, self._cache_key = False, reason, None
                return self.reason
            self._cache_key, self._parent_reviewers = result
            selector = getattr(self.studio, "_selected_annotators", None)
            selected_reviewers = selector() if callable(selector) else None
            if selected_reviewers == []:
                raise ValueError("Tick the parent's annotators before continuing training.")
            if requested_reviewers({"reviewers": selected_reviewers}) != self._parent_reviewers:
                raise ValueError("The annotator selection differs from the parent; create a fresh training version to change reviewers.")
            if not callable(self._original_finished):
                raise ValueError("The training completion handler is unavailable.")
            actor = getattr(self.studio, "reviewer", None)
            evidence = getattr(self.studio, "bundles", None)
            if actor is None or not actor.text().strip():
                raise ValueError("Set the reviewer identity before continuing training.")
            if evidence is None or not evidence.toPlainText().strip():
                raise ValueError("Save and select new reviewed evidence before continuing training.")
            self.ready = True
            self.reason = ("Continue the selected model on CPU from its saved weights and scaling. "
                           "Earlier evidence and evaluation exclusions are retained; the new version requires independent validation.")
        except (OSError, KeyError, TypeError, ValueError) as error:
            self._cache_key = None
            detail = str(error)
            self.ready, self.reason = False, "Continuation unavailable: " + detail
        return self.reason

    def start(self):
        """Freeze compatible evidence and launch a continuation job from saved weights."""
        from .phase_workflow import parse_training_bundles
        self.refresh()
        if not self.ready:
            raise ValueError(self.reason)
        actor = self.studio.reviewer.text().strip()
        if not actor:
            raise ValueError("Set the reviewer identity before continuing training.")
        bundles = parse_training_bundles(self.studio.bundles.toPlainText())
        if not bundles:
            raise ValueError("Save and select new reviewed evidence before continuing training.")
        request = continuation.prepare_request(parent_entry=self.studio._selected_validation_model(),
            output_parent=self.studio._models_folder_for_series(self.steps.series_id),
            cache_parent=Path(self.studio.cache_parent.text()), bundles=bundles, actor=actor)
        self.studio._launch_training_job(request, continuation.build_command(request),
            Path(request["request_path"]).parent,
            "Continuing the saved model as a new CPU candidate; independent validation remains required.")

    def finished(self, exit_code, exit_status):
        """Report completion and index successful continuation candidates."""
        request = dict(self.studio.training_request or {})
        if (int(exit_code) != 0
                or (request.get("continuation") or {}).get("contract") != continuation.CONTRACT):
            return self._original_finished(exit_code, exit_status) if callable(self._original_finished) else None
        self.studio._read_output()
        process, self.studio.training_process = self.studio.training_process, None
        self.studio.stop_button.setEnabled(False)
        if process is not None:
            process.deleteLater()
        try:
            root = Path(request["output_root"])
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            if summary.get("artifact_kind") != "phase-continuation-candidate":
                raise ValueError("The completed job has no continuation candidate summary")
            self.studio._record_completed_model(request["model_id"], root)
            self.studio.training_completed.emit(str(root))
            incomplete = summary.get("nonconverged_heads") or []
            message = "New candidate saved. Prepare phase maps for review and complete independent validation before using it."
            if incomplete:
                message += " Optimizer convergence was not established for: " + ", ".join(incomplete) + "."
            self.studio.status.setText(message)
        except (OSError, ValueError, KeyError) as error:
            self.studio.status.setText("Continuation files were retained; candidate inspection failed: " + str(error))
        self.studio._update_action_states()


__all__ = ["NativeContinuationTraining"]
