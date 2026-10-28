"""UI-only version choice and existing disagreement views for phase review.

Model switches reuse the original binding verifier, workspace builder and
save machinery. No existing window's model, store or session is repurposed.
"""

from __future__ import annotations

from copy import deepcopy
from contextvars import copy_context
from dataclasses import dataclass
import json
from pathlib import Path
from threading import Thread
import uuid

from PySide6.QtCore import QObject, QSignalBlocker, QTimer, Qt, Signal, Slot
from shiboken6 import isValid
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QWidget,
    QToolBar,
)


@dataclass(frozen=True)
class ModelChoice:
    root: Path
    label: str


@dataclass(frozen=True)
class ReviewTargetSnapshot:
    """Only immutable scope DATA crosses the metadata-reading thread boundary."""
    datasets: tuple[str, ...]
    purpose: tuple[str, str, bool, bool, bool]


def target_snapshot(window):
    """Capture the original inventory and review purpose on the GUI thread."""
    workspace = window.workspace
    return ReviewTargetSnapshot(
        tuple(str(pair.candidate_a.dataset) for pair in window.app_model.inventory.pairs),
        (workspace.key, workspace.review_surface, workspace.single_model,
         workspace.approval_only, workspace.read_only),
    )


def model_choice_arguments(entries, current_root):
    """Transfer already listed UI choices as arguments, never generated code."""
    result = ["--review-current-model-root", str(current_root)]
    seen = set()
    for entry in entries:
        root = str(entry.get("root") or "")
        key = str(Path(root).absolute()).casefold() if root else ""
        if not key or key in seen:
            continue
        seen.add(key)
        result += [
            "--review-model-choice",
            json.dumps(
                {
                    "root": root,
                    "label": str(
                        entry.get("display_name")
                        or entry.get("model_id")
                        or Path(root).name
                    ),
                },
                ensure_ascii=False,
            ),
        ]
    return result


def parse_model_choices(values):
    if len(values) > 256:
        raise ValueError("Too many supplied review model choices.")
    result, seen = [], set()
    for raw in values:
        value = json.loads(raw)
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("root"), str)
            or not value["root"].strip()
        ):
            raise ValueError("A review model choice needs its saved model folder.")
        root = Path(value["root"])
        key = str(root.absolute()).casefold()
        if key not in seen:
            seen.add(key)
            result.append(ModelChoice(root, str(value.get("label") or root.name)))
    return tuple(result)


def verified_target(choice, window, args, api):
    """Reuse the same saved model/map admission as the existing Studio Open."""
    snapshot = window if isinstance(window, ReviewTargetSnapshot) else target_snapshot(window)
    from darsia.fluidflower.phase_evidence_audit import verified_review_binding
    from darsia.fluidflower.ui.model_training import _validate_saved_phase_binding

    binding = verified_review_binding(choice.root, saved_comparison=True)
    # Keep the existing series scope when changing only the model version.
    from darsia.fluidflower.series_paths import series_key

    original_datasets = {
        series_key(dataset) for dataset in snapshot.datasets
    }
    manifest = json.loads(
        (Path(binding["combined_root"]) / "postflight_manifest.json").read_bytes()
    )
    _validate_saved_phase_binding(choice.root, binding, manifest)
    target_datasets = {
        series_key(str(job["dataset"])) for job in manifest.get("jobs") or []
    }
    if not original_datasets or target_datasets != original_datasets:
        raise ValueError(
            "The selected model's saved maps belong to another experimental series."
        )
    model_id = str(binding.get("model_id") or "")
    if not model_id:
        raise ValueError("The selected saved review has no model identity.")
    selected = binding.get("selected_candidates")
    if not isinstance(selected, dict) or not selected:
        raise ValueError("The selected saved model has no bound review candidates.")
    updated = deepcopy(args)
    updated.ffac_root = Path(binding["ffac_root"])
    updated.ffum_root = Path(binding["ffum_root"])
    updated.combined_root = Path(binding["combined_root"])
    updated.target_model_id = model_id
    updated.review_current_model_root = choice.root
    updated.review_session_id = "model-change-" + uuid.uuid4().hex
    updated.single_candidate = [
        f"{run}={candidate}" for run, candidate in sorted(selected.items())
    ]
    updated.evidence_audit = None
    # A new model is a new session. No inherited decision_dir or evidence queue.
    workspace = api.review_workspaces(updated)[snapshot.purpose[0]]
    if (
        workspace.key,
        workspace.review_surface,
        workspace.single_model,
        workspace.approval_only,
        workspace.read_only,
    ) != snapshot.purpose:
        raise ValueError("Changing the model must not change this review's purpose.")
    return updated, workspace


def original_window_builder(prepared, old_window, api, *, use_opengl):
    args, workspace = prepared
    actor = old_window.reviewer.text().strip()
    workspace = api.local_reviewer_workspace(
        workspace, Path(args.local_review_root), actor
    )
    if workspace.output_dir.resolve() == old_window.workspace.output_dir.resolve():
        raise ValueError(
            "A different model must not reuse the previous model's review store."
        )
    app_model, store = api.build_workspace_model(args, workspace, reviewer_name=actor)
    window_class = api.workspace_window_class(workspace)
    return window_class(
        app_model,
        store,
        args.cache_root,
        workspace,
        reviewer_name=actor,
        use_opengl=use_opengl,
    )


def save_state(window):
    """Inspect original save state after its original explicit Save was called."""
    if getattr(window, "_comparison_local_error", False):
        return "error"
    if (
        getattr(window, "_comparison_save_task", None) is not None
        or getattr(window, "_comparison_pending_saves", None)
        or getattr(window, "_comparison_unprotected_requests", None)
        or getattr(window, "save_inflight", False)
        or getattr(window, "pending_save_requests", None)
        or getattr(window, "save_dirty", False)
    ):
        return "pending"
    return "saved"


class ReviewVersionControls(QObject):
    """Attach to the existing single-model correction surface only."""
    _verification_ready = Signal(object)

    def __init__(
        self,
        window,
        args,
        api,
        *,
        use_opengl=False,
        target_reader=verified_target,
        window_builder=original_window_builder,
    ):
        super().__init__(window)
        if not (
            window.workspace.key == "current_comparison"
            and window.workspace.review_surface == "open_correction"
            and window.workspace.single_model
        ):
            raise ValueError(
                "Version controls require the existing open-correction review surface."
            )
        self.api, self.use_opengl = api, use_opengl
        self.target_reader, self.window_builder = target_reader, window_builder
        self.choices = parse_model_choices(
            getattr(args, "review_model_choice", None) or []
        )
        self.windows = []
        self.pending = None
        self._read_request = self._verification = None
        self._verification_serial = 0
        self._verification_ready.connect(self._verified)
        self.timer = QTimer(window)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.finish_switch)
        self.attach(window, args)

    def attach(self, window, args):
        panel = QWidget(window)
        line = QHBoxLayout(panel)
        line.setContentsMargins(0, 0, 0, 0)
        line.addWidget(QLabel("Model version"))
        combo = QComboBox()
        active = str(Path(args.review_current_model_root).absolute()).casefold()
        index = -1
        for i, choice in enumerate(self.choices):
            combo.addItem(choice.label, choice)
            combo.setItemData(i, str(choice.root), 3)
            if str(choice.root.absolute()).casefold() == active:
                index = i
        if index < 0:
            choice = ModelChoice(
                Path(args.review_current_model_root), window.workspace.target_model_id
            )
            combo.insertItem(0, choice.label, choice)
            index = 0
        combo.setCurrentIndex(index)
        line.addWidget(combo, 1)
        disagreements = QCheckBox("Show the boxes that disagree most with the model")
        existing = getattr(window, "evidence_audit", None)
        disagreements.setChecked(existing is not None and not existing.isHidden())
        line.addWidget(disagreements)
        cancel = QPushButton("Cancel model change")
        cancel.hide()
        line.addWidget(cancel)
        toolbar = QToolBar("Review model", window)
        toolbar.setObjectName("reviewModelToolbar")
        toolbar.setMovable(False)
        toolbar.addWidget(panel)
        window.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        window.review_model_toolbar = toolbar
        combo.activated.connect(lambda wanted, w=window: self.choose(w, wanted))
        disagreements.toggled.connect(
            lambda visible, w=window: self.toggle_disagreements(w, visible)
        )
        cancel.clicked.connect(self.cancel_switch)
        window.review_version_controls = self
        window.review_version_panel = panel
        window.review_model_choice = combo
        window.review_disagreements = disagreements
        window.review_model_cancel = cancel
        window._review_choice_index = index
        window._review_choice_args = args
        self.windows.append(window)

    def choose(self, window, index):
        if self.pending is not None or self._read_request is not None or index == window._review_choice_index:
            return
        choice = window.review_model_choice.itemData(index)
        try:
            snapshot = target_snapshot(window)
            args = deepcopy(window._review_choice_args)
            with QSignalBlocker(window.review_model_choice):
                window.review_model_choice.setCurrentIndex(window._review_choice_index)
        except Exception as error:
            self._refuse(window, error)
            return
        self._verification_serial += 1
        serial = self._verification_serial
        self._read_request = self._verification = (serial, window)
        window.review_model_cancel.show()
        window.review_model_choice.setEnabled(False)
        window.statusBar().showMessage("Checking the selected saved model version…")
        context, reader, api = copy_context(), self.target_reader, self.api
        emit = self._verification_ready.emit

        def read():
            # No window, store, widget or mutable inventory is read by this worker.
            try:
                result = (serial, context.run(reader, choice, snapshot, args, api), None)
            except Exception as error:
                result = (serial, None, str(error))
            try:
                emit(result)
            except RuntimeError:
                pass  # The QObject owner may have been deleted during metadata I/O.

        try:
            Thread(target=read, daemon=True).start()
        except Exception as error:
            self._read_request = self._verification = None
            self._refuse(window, error)

    @Slot(object)
    def _verified(self, result):
        serial, prepared, error = result
        request = self._read_request
        if request is None or serial != request[0]:
            return
        self._read_request = None
        wanted = self._verification is request
        self._verification = None
        window = request[1]
        if not isValid(window) or getattr(window, "_comparison_closing", False):
            return
        if not wanted:
            window.review_model_choice.setEnabled(True)
            window.statusBar().showMessage("Model change cancelled. This review remains available.")
            return
        if error is not None:
            self._refuse(window, error)
            return
        try:
            window._save_now()  # Original recovery + canonical save stays on the GUI thread.
        except Exception as error:
            self._refuse(window, error)
            return
        if not isValid(window) or getattr(window, "_comparison_closing", False):
            return
        self.pending = window, prepared
        window.statusBar().showMessage("Saved model verified; finishing current labels before opening its review…")
        self.timer.start()

    def finish_switch(self):
        if self.pending is None:
            return
        window, prepared = self.pending
        if not isValid(window) or getattr(window, "_comparison_closing", False):
            self._clear_pending(window)
            return
        state = save_state(window)
        if state == "pending":
            window.statusBar().showMessage(
                "Finishing the current model's saved labels before changing model…"
            )
            return
        if state == "error":
            self._refuse(
                window,
                RuntimeError(
                    "The current labels could not be protected. Keep this review open and resolve its save error."
                ),
            )
            return
        try:
            window.review_model_cancel.hide()
            window.statusBar().showMessage("Opening the verified review window…")
            new_window = self.window_builder(
                prepared, window, self.api, use_opengl=self.use_opengl
            )
            self.attach(new_window, prepared[0])
            new_window.restoreGeometry(window.saveGeometry())
        except Exception as error:
            self._refuse(window, error)
            return
        self._clear_pending(window)
        new_window.show()
        window.hide()  # The previous window, original store and session remain intact.

    def _clear_pending(self, window):
        if isValid(self.timer):
            self.timer.stop()
        self.pending = None
        if isValid(window):
            window.review_model_cancel.hide()
            window.review_model_choice.setEnabled(True)

    def _refuse(self, window, error):
        self._clear_pending(window)
        if not isValid(window) or getattr(window, "_comparison_closing", False):
            return
        with QSignalBlocker(window.review_model_choice):
            window.review_model_choice.setCurrentIndex(window._review_choice_index)
        window.statusBar().showMessage("Model version was not changed. This review remains available.")
        QMessageBox.warning(window, "Review model was not changed", str(error))

    def cancel_switch(self):
        if self._verification is not None:
            _serial, window = self._verification
            self._verification = None
            if not isValid(window) or getattr(window, "_comparison_closing", False):
                return
            window.review_model_cancel.hide()
            window.statusBar().showMessage(
                "Model change cancelled. The metadata read is finishing; this review remains available."
            )
        elif self.pending is not None:
            window, _prepared = self.pending
            self._clear_pending(window)
            with QSignalBlocker(window.review_model_choice):
                window.review_model_choice.setCurrentIndex(window._review_choice_index)
            window.statusBar().showMessage("Model change cancelled. This review remains available.")

    def toggle_disagreements(self, window, visible):
        dock = getattr(window, "evidence_audit", None)
        if not visible:
            if dock is not None:
                dock.hide()
            return
        try:
            if dock is None:
                path = getattr(window._review_choice_args, "evidence_audit", None)
                if not path:
                    path, _ = QFileDialog.getOpenFileName(
                        window,
                        "Saved disagreement queue for this model",
                        "",
                        "JSON (*.json)",
                    )
                if not path:
                    with QSignalBlocker(window.review_disagreements):
                        window.review_disagreements.setChecked(False)
                    return
                from darsia.fluidflower.ui.evidence_audit import attach_evidence_audit

                dock = attach_evidence_audit(window, Path(path), self.api)
                window._review_choice_args.evidence_audit = Path(path)
            dock.reason.setCurrentIndex(dock.reason.findData(""))
            dock.order.setCurrentIndex(dock.order.findData("score"))
            dock.refresh()
            dock.show()
            dock.raise_()
        except Exception as error:
            with QSignalBlocker(window.review_disagreements):
                window.review_disagreements.setChecked(False)
            QMessageBox.warning(
                window, "Disagreement boxes were not opened", str(error)
            )


def attach_review_version_controls(window, args, api, *, use_opengl=False):
    if (
        getattr(args, "review_current_model_root", None)
        and window.workspace.key == "current_comparison"
        and window.workspace.review_surface == "open_correction"
        and window.workspace.single_model
    ):
        return ReviewVersionControls(window, args, api, use_opengl=use_opengl)
    return None
