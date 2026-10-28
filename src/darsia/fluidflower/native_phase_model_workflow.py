"""Phase-model presentation around the retained review and training routes.

This module does not generate predictions, choose evidence, or replace a worker.
Saved evidence remains in the original Studio objects even when its old controls
are hidden. The native review owner retains one model-bound correction process.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)
from PySide6.QtCore import QSignalBlocker, Qt


PIXEL_LIMIT_TEXT = (
    "Caps the training pixels taken from each drawn region. This weights small "
    "and large boxes more evenly, so a large box cannot dominate simply because "
    "it covers more pixels. Higher limits use more memory and fitting time; "
    "they do not increase image resolution or replace independent labels."
)

# The opened training log stays a modest panel; the full text scrolls inside it.
LOG_MAX_HEIGHT = 160

PRECISION_LIMIT_TEXT = (
    "Later steps cannot recover precision lost during training or phase-map "
    "calculation. The current phase predictor and probability-map format use "
    "float32, including after CPU float64 fitting. Float64 production requires "
    "a supported float64 predictor and newly computed float64 phase maps; "
    "changing an output setting or casting the existing maps is not sufficient."
)


def _path_key(value):
    # UI comparison only: do not stat NAS paths during an ordinary refresh.
    return os.path.normcase(os.path.normpath(str(value))) if value else ""


def training_summary(bundle_text, sessions):
    """Summarise only the explicit inputs from the original evidence field.

    Counts are cached saved-review counts, not scientific admission or a promise
    that each label is eligible. Never silently count a selected-but-unadded
    session, duplicate a decision file, or treat unknown evidence as zero.
    """
    paths = set()
    for raw in str(bundle_text or "").splitlines():
        if not raw.strip():
            continue
        parts = [part.strip() for part in raw.replace("\t", "|").split("|")]
        if len(parts) != 2 or not all(parts):
            return "Training evidence summary unavailable: an input row is incomplete."
        paths.add(_path_key(parts[0]))
    if not paths:
        return "No saved review evidence has been selected for training."
    by_path = {}
    for row in sessions:
        key = _path_key(row.get("decision_path"))
        if key and key not in by_path:
            by_path[key] = row
    if paths - by_path.keys():
        return "Training evidence selected; refresh saved reviews to read its label counts."
    try:
        counts = [(int(by_path[path]["manual_regions"]), int(by_path[path]["blind_region_labels"]))
                  for path in paths]
        if any(min(pair) < 0 for pair in counts):
            raise ValueError("Negative saved count")
        labels, blind = (sum(pair[index] for pair in counts) for index in (0, 1))
    except (KeyError, TypeError, ValueError):
        return "Training evidence selected; its saved label counts are unavailable."
    return (
        f"Selected for training: {labels:,} phase labels and {blind:,} blind validations. "
        "These are saved counts before the original eligibility, reviewer, "
        "withdrawal and held-out-evidence checks."
    )


def recorded_precision_text(entry):
    """Use the existing catalogue's per-head final-fit provenance verbatim."""
    heads = entry.get("heads") if isinstance(entry, Mapping) else None
    if not isinstance(heads, (tuple, list)) or not heads:
        return "Selected model fitting precision: not recorded in the available metadata."
    parts = []
    for row in heads:
        if not isinstance(row, Mapping):
            continue
        name = " / ".join(str(row.get(key) or "unspecified") for key in ("adapter", "head"))
        precision = str(row.get("precision") or "not recorded")
        backend = str(row.get("backend") or "device not recorded")
        parts.append(f"{name}: {precision} ({backend})")
    if not parts:
        return "Selected model fitting precision: not recorded in the available metadata."
    return "Selected model fitting precision — " + "; ".join(parts) + "."


def _label(text):
    label = QLabel(text)
    label.setWordWrap(True)
    return label


def _move(widget, layout):
    widget.setParent(layout.parentWidget())
    layout.addWidget(widget)
    widget.show()


class NativePhaseModelWorkflow:
    """Reparent actual controls; keep numerical values and original slots intact."""

    def __init__(self, steps):
        self.steps, self.studio = steps, steps.studio
        self.archive_dialog = None
        self.dependency_dialog = None
        self._archive_dialog_context = None
        from .native_phase_management_sections import NativePhaseManagementSections
        self.management_sections = NativePhaseManagementSections(steps)
        self._archive_context = None
        self._archive_snapshot = None
        self._archive_read_error = ""
        self._saved_selection = None
        self._saved_context = None
        self._build_labels()
        self._build_blind_validation()
        self._build_training()
        from .native_phase_review_owner import NativePhaseReviewOwner
        self.review_owner = NativePhaseReviewOwner(self)
        from .native_prepared_model_retraining import PreparedModelRetraining
        self.prepared_retraining = PreparedModelRetraining(steps)
        steps.prepared_retraining = self.prepared_retraining
        self.discard_session.clicked.connect(self.review_owner.discard_dialog)
        self._install_archive_filter()
        for name, signal in (("bundles", "textChanged"), ("saved_sessions", "currentIndexChanged"),
                             ("backend", "currentIndexChanged")):
            control = getattr(self.studio, name, None)
            if control is not None:
                getattr(control, signal).connect(steps.refresh)
        if callable(getattr(self.studio, "_install_review_sessions", None)):
            original = self.studio._install_review_sessions

            def install_sessions(*args, **kwargs):
                result = original(*args, **kwargs)
                self.refresh()
                return result

            self.studio._install_review_sessions = install_sessions
        self.refresh()

    def _build_labels(self):
        steps, studio = self.steps, self.studio
        steps.layouts[2].insertWidget(0, _label(
            "Use the model chosen in Basis to draw boxes and label phases here. "
            "Phase map shows or hides the saved prediction. Blind validation uses the same "
            "image and saved session with predictions hidden."
        ))
        _move(studio.open_model_correction, steps.layouts[2])
        # Native panel selection opens the retained review route. Keep the
        # original controls and their admission state, without duplicate buttons.
        studio.open_model_correction.hide()
        steps.phase_review_button.hide()
        steps.organise_button.show()
        self.archive_button = QPushButton("Archive unused models…")
        self.archive_button.clicked.connect(self.open_archive)
        steps.layouts[0].addWidget(self.archive_button)
        self.archive_notice = _label("")
        steps.layouts[0].addWidget(self.archive_notice)

    def _build_blind_validation(self):
        steps, studio = self.steps, self.studio
        self.blind_notice = _label(
            "Answer the offered boxes without seeing the model's prediction. "
            "The correction view uses the same drawing tools for labels and blind validation."
        )
        steps.layouts[3].insertWidget(0, self.blind_notice)
        blind = getattr(studio, "open_blind_labeling", None)
        if blind is not None:
            _move(blind, steps.layouts[3])
            blind.hide()
        audit = getattr(studio, "evidence_audit_tab", None)
        if audit is not None:
            # The shared correction window owns contradiction work. Retain the
            # legacy object for historical callers but expose no detached route.
            audit.hide()
        self.session_notice = _label(
            "One active review session is kept for this series and reviewer. "
            "Close the review before training. Submitting all its evidence to training "
            "or explicitly discarding it permits a new session; saved evidence is preserved."
        )
        steps.layouts[3].addWidget(self.session_notice)
        self.discard_session = QPushButton("Discard active review session…")
        steps.layouts[3].addWidget(self.discard_session)
        studio.resume_session_button.hide()

    def _build_training(self):
        steps, studio = self.steps, self.studio
        # Keep the original objects and values alive for their original request
        # builder; none of the removed settings are reset or silently replaced.
        steps.advanced.hide()
        self.summary = _label("")
        steps.layouts[4].insertWidget(0, self.summary)
        self.selected_evidence = QListWidget()
        self.selected_evidence.setMaximumHeight(120)
        self.selected_evidence.setToolTip(
            "Saved reviews explicitly selected for the original trainer. "
            "Removing an input here preserves its saved labels and validation boxes."
        )
        steps.layouts[4].addWidget(_label("Selected saved reviews"))
        steps.layouts[4].addWidget(self.selected_evidence)
        self.remove_evidence = QPushButton("Remove selected review from training inputs")
        self.remove_evidence.clicked.connect(self._remove_evidence)
        steps.layouts[4].addWidget(self.remove_evidence)
        self.selected_evidence.currentRowChanged.connect(self._evidence_selection_changed)
        self.parameters = QGroupBox("Training controls")
        form = QFormLayout(self.parameters)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.parameter_notice = _label("")
        form.addRow(self.parameter_notice)
        controls = (
            ("backend", "Fitting device and precision", "Choose the precision before fresh training. "
             "The existing CPU route fits in float64; CUDA fits in float32. Automatic is "
             "kept as the saved value until you explicitly choose a device."),
            ("stage_workers", "Evidence preparation workers", "Copies and verifies independent run "
             "assets in parallel. More workers increase storage traffic and memory use."),
            ("extraction_workers", "Feature extraction workers", "Builds the existing image features "
             "for different runs in parallel, within the original worker limits."),
            ("pixels", "Maximum pixels sampled per region", PIXEL_LIMIT_TEXT),
            ("mil_iterations", "Mixed-region fitting iterations", "The original upper limit for "
             "fitting mixed regions; increasing it does not repair incorrect labels."),
            ("change_ablation", "Scientific control", "Retains the original optional control fit "
             "without image-change evidence and its existing numerical settings."),
        )
        for name, title, explanation in controls:
            control = getattr(studio, name, None)
            if not isinstance(control, QWidget):
                continue
            field = QWidget()
            layout = QVBoxLayout(field)
            layout.setContentsMargins(0, 0, 0, 0)
            _move(control, layout)
            control.setToolTip(explanation)
            layout.addWidget(_label(explanation))
            form.addRow(title, field)
        steps.layouts[4].insertWidget(1, self.parameters)
        self.precision = _label("")
        steps.layouts[4].insertWidget(2, self.precision)
        self.precision_limit = _label(PRECISION_LIMIT_TEXT)
        steps.layouts[4].insertWidget(3, self.precision_limit)
        # The log is retained without restoring the obsolete evidence controls.
        # Folded, it shows only its latest line; opened, it keeps a modest height.
        self.log_group = QGroupBox("Training log")
        self.log_group.setCheckable(True)
        self.log_group.setChecked(False)
        log_layout = QVBoxLayout(self.log_group)
        self.log_latest = _label("No training output yet.")
        log_layout.addWidget(self.log_latest)
        _move(studio.log, log_layout)
        studio.log.setMinimumHeight(0)
        studio.log.setMaximumHeight(LOG_MAX_HEIGHT)
        studio.log.hide()
        studio.log.textChanged.connect(self._show_latest_log_line)
        self.log_group.toggled.connect(studio.log.setVisible)
        steps.workspace_layout.addWidget(self.log_group)

    def _show_latest_log_line(self):
        """Keep the last non-empty training log line visible while the log is folded."""
        block = self.studio.log.document().lastBlock()
        while block.isValid() and not block.text().strip():
            block = block.previous()
        self.log_latest.setText("Latest: " + block.text().strip() if block.isValid() else "No training output yet.")

    def training_block_reason(self):
        owner = getattr(self, "review_owner", None)
        if owner is not None and (reason := owner.close_block_reason()):
            return reason
        backend = getattr(self.studio, "backend", None)
        raw_refit = bool(getattr(getattr(self.steps, "training_photometry", None), "uses_prepared_raw_basis", False))
        if (self.steps.scratch_basis.isChecked() or raw_refit) and backend is not None:
            if backend.currentData() not in ({"cpu", "cuda", "auto"} if raw_refit else {"cpu", "cuda"}):
                return "Choose the fitting device and precision before training the new model."
        return ""

    def run_training(self, callback, *, saved_labels_only=False):
        return self.review_owner.run_training(callback, saved_labels_only=saved_labels_only)

    def before_launch(self, request):
        self.review_owner.before_launch(request)

    def close_block_reason(self):
        if not self.prepared_retraining.can_close():
            return "Saved labels are being prepared on the current Setup images; wait for preparation to finish."
        dialog = self.dependency_dialog
        if dialog is not None and not dialog.can_close():
            return "Close the model dependency inspector and let its metadata read finish before continuing."
        return self.review_owner.close_block_reason()

    def refresh(self, *, saved_selection=None):
        steps, studio = self.steps, self.studio
        self.management_sections.sync_context()
        owner = getattr(self, "review_owner", None)
        self.discard_session.setEnabled(owner is None or owner.process is None)
        if saved_selection is not None:
            self._saved_selection = saved_selection
            self._saved_context = str(steps.series_root or "")
        studio.open_model_correction.setText("Open model correction")
        studio.open_model_correction.hide()
        studio.resume_session_button.hide()
        studio.session_summary.setText(studio.session_summary.text().replace(
            "Resume reopens this exact editable decision store.",
            "Saved review data remains available for explicit training selection.").replace(
            "Open it to audit every saved decision without changing the evidence.",
            "The original evidence remains saved."))
        steps.phase_review_button.hide()
        steps.organise_button.show()
        steps.advanced.hide()
        blind = getattr(studio, "open_blind_labeling", None)
        if blind is not None:
            blind.setText("Open blind validation")
            blind.hide()
            # Reuse the retained review-admission state; an enabled control
            # never replaces the original model/source guards.
            blind.setEnabled(studio.open_model_correction.isEnabled())
            blind.setToolTip(steps.review_selection_text())
        self.archive_button.setEnabled(steps.series_root is not None and bool(steps.series_id))
        self.archive_button.setToolTip(
            "Hide unused models from this page's lists; restore them from the archive. "
            "Model files and all validation boxes remain in place."
        )
        text = getattr(studio, "bundles", None)
        summary = training_summary(
            text.toPlainText() if text is not None else "",
            getattr(studio, "review_sessions", ()),
        )
        raw_refit = bool(getattr(getattr(steps, "training_photometry", None), "uses_prepared_raw_basis", False))
        choice = steps.existing_choice() if hasattr(steps, "existing_choice") else ""
        prepared_existing = (steps.existing_basis.isChecked() and choice != "continue"
                             and self.prepared_retraining.available())
        if prepared_existing:
            summary = ("Use only the selected model's existing saved labels. They are projected onto the current "
                "Setup crop, lighting and colour basis before a new model version is fitted. No new boxes or review session are required.")
        elif choice == "continue" or (steps.existing_basis.isChecked() and not raw_refit and choice != "retrain"):
            summary = summary.replace("Selected for training:", "Selected additional evidence:")
            summary += " Continuation also retains the parent's evidence; its counts are not included here."
        self.summary.setText(summary)
        self._refresh_selected_evidence()
        for control in (self.selected_evidence, self.remove_evidence, *(
                getattr(studio, name, None) for name in ("saved_sessions", "refresh_review_sessions_button",
                "saved_sessions_status", "session_summary", "add_session_evidence", "pending_training_sessions"))):
            if control is None:
                continue
            control.setVisible(not prepared_existing)
        getter = getattr(studio, "_selected_validation_model", None)
        entry = getter() if callable(getter) else None
        precision = recorded_precision_text(entry)
        backend = getattr(studio, "backend", None)
        continuation = steps.existing_basis.isChecked() and not raw_refit and not prepared_existing
        idle = getattr(studio, "training_process", None) is None and not getattr(studio, "_series_batch_active", False)
        for name in ("backend", "stage_workers", "extraction_workers", "pixels", "mil_iterations", "change_ablation"):
            control = getattr(studio, name, None)
            if control is not None:
                control.setEnabled(not continuation and not prepared_existing and idle)
        self.parameter_notice.setText(
            "Retraining retains the selected model's architecture and training settings; weights are fitted anew. "
            "Current Setup crop, lighting and colour choices replace the old image basis. Preparation workers follow the Jobs capacity choice."
            if prepared_existing else
            "Continuation retains the parent's scientific and worker settings. "
            "The controls below apply when starting a fresh model."
            if continuation else
            "Fit new heads on the prepared RAW basis. Worker counts are bounded by CPU affinity and available RAM; BLAS uses one thread per worker."
            if raw_refit else "These existing controls configure the new model's original training route."
        )
        if backend is not None:
            if continuation:
                precision += " Continuation uses the original CPU float64 fitting route and inherited scientific settings."
            elif raw_refit:
                precision += " New RAW fitting uses CPU float64 or CUDA float32; Automatic selects the available fitting backend."
        self.precision.setText(precision)
        blocker = self.training_block_reason()
        if blocker:
            studio.start_button.setEnabled(False)
            studio.start_button.setToolTip(blocker + " " + studio.start_button.toolTip())
        self._apply_archive_visibility()

    def _refresh_selected_evidence(self):
        bundles = getattr(self.studio, "bundles", None)
        rows = bundles.toPlainText().splitlines() if bundles is not None else []
        selected = self.selected_evidence.currentItem()
        chosen = selected.data(Qt.ItemDataRole.UserRole) if selected else None
        self.selected_evidence.clear()
        for index, row in enumerate(rows):
            if not row.strip():
                continue
            parts = row.replace("\t", "|").split("|")
            name = Path(parts[0].strip()).name
            item = QListWidgetItem(name or "Incomplete saved review input")
            item.setToolTip(row)
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.selected_evidence.addItem(item)
            if index == chosen:
                self.selected_evidence.setCurrentItem(item)
        self._evidence_selection_changed()

    def _evidence_selection_changed(self, *_args):
        idle = getattr(self.studio, "training_process", None) is None and not getattr(self.studio, "_series_batch_active", False)
        self.remove_evidence.setEnabled(idle and self.selected_evidence.currentItem() is not None)

    def _remove_evidence(self):
        item = self.selected_evidence.currentItem()
        if (item is None or getattr(self.studio, "training_process", None) is not None
                or getattr(self.studio, "_series_batch_active", False)):
            return
        rows = self.studio.bundles.toPlainText().splitlines()
        index = item.data(Qt.ItemDataRole.UserRole)
        if 0 <= index < len(rows):
            rows.pop(index)
            self.studio.bundles.setPlainText("\n".join(rows))

    def protected_roots(self):
        """Protect current selection and any loaded editable-review/training basis."""
        studio = self.studio
        roots = {str(studio.validation_model.currentData() or "")}
        owner = getattr(self, "review_owner", None)
        if owner is not None and owner.active_model_root:
            roots.add(owner.active_model_root)
        active_models = {str(row.get("target_model_id") or "")
                         for row in getattr(studio, "review_sessions", ())
                         if not row.get("editing_locked")}
        roots.update(str(row.get("root") or "") for row in getattr(studio, "model_history", ())
                     if str(row.get("model_id") or "") in active_models)
        request = getattr(studio, "training_request", None) or {}
        if getattr(studio, "training_process", None) is not None:
            roots.add(str(request.get("output_root") or ""))
            parent = ((request.get("continuation") or {}).get("parent_request") or {}).get("path")
            if parent:
                roots.add(str(Path(parent).parent))
        return {root for root in roots if root}

    def _install_archive_filter(self):
        # Hide rendered rows after their original callbacks return. Never swap
        # model_history: a binding reader can snapshot it inside those callbacks.
        for name in ("_populate_validation_models", "_rebuild_model_history_view"):
            original = getattr(self.studio, name, None)
            if not callable(original):
                continue

            def filtered(*args, _original=original, **kwargs):
                result = _original(*args, **kwargs)
                self._apply_archive_visibility()
                return result

            setattr(self.studio, name, filtered)

    def _apply_archive_visibility(self):
        from .native_phase_model_archive import read_archive_snapshot, filter_snapshot
        entries = getattr(self.studio, "model_history", ())
        hidden = set()
        try:
            context = str(self.steps.series_root or "")
            if context and context != self._archive_context:
                self._archive_context = context
                self._archive_snapshot = None
                self._archive_read_error = ""
                try:
                    self._archive_snapshot = read_archive_snapshot(self.steps.series_root)
                except (OSError, ValueError, TypeError, RuntimeError) as error:
                    self._archive_read_error = str(error)
            if context and self._archive_read_error:
                raise ValueError(self._archive_read_error)
            if context and self._archive_snapshot is not None and self._saved_context == context:
                visible = filter_snapshot(self._archive_snapshot, entries, self._saved_selection,
                                          protected_roots=self.protected_roots())
                visible_roots = {_path_key(row.get("root")) for row in visible}
                hidden = {_path_key(row.get("root")) for row in entries} - visible_roots
            self.archive_notice.clear()
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            self.archive_notice.setText("Model archive unavailable; all model entries remain visible: " + str(error))
        combo = self.studio.validation_model
        view = combo.view()
        if hasattr(view, "setRowHidden"):
            for row in range(combo.count()):
                hide = _path_key(combo.itemData(row)) in hidden
                view.setRowHidden(row, hide)
                # QComboBox keyboard navigation also consults item flags.
                # Preserve and restore the exact previous flags on this item.
                model = combo.model()
                item = model.item(row) if hasattr(model, "item") else None
                role = int(Qt.ItemDataRole.UserRole) + 975
                if item is not None:
                    previous = item.data(role)
                    if hide:
                        if previous is None:
                            item.setData(item.flags(), role)
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                    elif previous is not None:
                        item.setFlags(previous)
                        item.setData(None, role)
        history = getattr(self.studio, "history_table", None)
        if history is not None:
            for index, row in enumerate(getattr(self.studio, "visible_model_history", ())):
                history.setRowHidden(index, _path_key(row.get("root")) in hidden)

    def _inspect_model_dependencies(self, entry):
        if not isinstance(entry, dict) or not entry.get("root") or self.steps.series_root is None:
            return
        previous = self.dependency_dialog
        if previous is not None:
            if not previous.can_close():
                self.management_sections.show("dependencies")
                return
            previous.close()
            previous.deleteLater()
        from .ui.phase_model_dependency_review import known_dependency_scope
        from .native_phase_management_sections import InlineDependencyReview
        from .ui.model_training import _models_for_series
        scope = known_dependency_scope(self.steps.series_root,
            _models_for_series(getattr(self.studio, "model_history", ()), self.steps.series_id),
            getattr(self.studio, "review_sessions", ()))
        self.dependency_dialog = InlineDependencyReview(self.steps, entry=entry,
            series_root=self.steps.series_root, scope_records=scope,
            protected_roots=self.protected_roots())
        self.management_sections.present("dependencies", self.dependency_dialog.windowTitle(),
                                         self.dependency_dialog, 0)

    def open_archive(self):
        if self.steps.series_root is None:
            return
        self._archive_context = None
        self._archive_snapshot = None
        self._apply_archive_visibility()
        if (self.archive_dialog is not None
                and self._archive_dialog_context == self.management_sections.context()):
            self.archive_dialog.refresh_choices()
            self.management_sections.present("archive", "Archive unused phase models", self.archive_dialog, 0)
            return
        dialog = QDialog(self.steps)
        dialog.setWindowTitle("Archive unused phase models")
        layout = QVBoxLayout(dialog)
        layout.addWidget(_label(
            "Archiving hides an unused model from this page's lists. Restore it here "
            "at any time. Files and validation boxes are preserved. Deletion is unavailable."
        ))
        choices = QComboBox()
        from .ui.model_training import _models_for_series
        layout.addWidget(choices)
        reason = QLineEdit()
        reason.setPlaceholderText("Reason for archiving or restoring this model")
        layout.addWidget(reason)
        status = _label("")
        layout.addWidget(status)
        actions = QHBoxLayout()
        archive, restore = QPushButton("Archive"), QPushButton("Restore")
        actions.addWidget(archive)
        actions.addWidget(restore)
        layout.addLayout(actions)
        inspect_dependencies = QPushButton("Inspect model dependencies…")
        inspect_dependencies.setEnabled(choices.count() > 0)
        layout.addWidget(inspect_dependencies)
        original_root = self.steps.series_root

        def inspect_selected():
            if original_root != self.steps.series_root:
                status.setText("The series changed. Reopen this archive for the current series before inspecting dependencies.")
                return
            self._inspect_model_dependencies(choices.currentData())

        inspect_dependencies.clicked.connect(inspect_selected)

        def apply(restore_model=False):
            if original_root != self.steps.series_root:
                status.setText("The series changed. Close this archive and open it for the current series.")
                return
            from .native_phase_model_archive import archive_model, restore_model as restore_saved_model
            try:
                function = restore_saved_model if restore_model else archive_model
                function(original_root, choices.currentData(), actor=self.steps.actor_provider(),
                         reason=reason.text(), protected_roots=self.protected_roots())
                self._archive_context = None
                self._archive_snapshot = None
                refresh = getattr(self.studio, "_rebuild_model_history_view", None)
                if callable(refresh):
                    refresh()
                self._apply_archive_visibility()
                entry = choices.currentData()
                label = self.studio._validation_model_label(entry)
                choices.setItemText(choices.currentIndex(), label if restore_model else label + " · Archived")
                status.setText("Model restored to the lists." if restore_model else "Unused model archived. All files and validation boxes are preserved.")
            except (OSError, ValueError, TypeError, RuntimeError) as error:
                status.setText(str(error))

        archive.clicked.connect(lambda: apply(False))
        restore.clicked.connect(lambda: apply(True))
        def update_actions(*_args):
            selected = choices.currentIndex() >= 0
            inspect_dependencies.setEnabled(selected)
            for button in (archive, restore):
                button.setEnabled(selected and bool(reason.text().strip()))

        def refresh_choices(preserve=True):
            previous = choices.currentData() or {}
            identity = lambda entry: (_path_key(entry.get("root")), str(entry.get("model_id") or ""))
            archived = {_path_key(row["root"]) for row in (self._archive_snapshot or {}).get("models", ())
                        if row.get("archived")}
            retained = -1
            with QSignalBlocker(choices):
                choices.clear()
                for entry in _models_for_series(getattr(self.studio, "model_history", ()), self.steps.series_id):
                    label = self.studio._validation_model_label(entry)
                    if _path_key(entry.get("root")) in archived:
                        label += " · Archived"
                    choices.addItem(label, dict(entry))
                    if previous and identity(entry) == identity(previous):
                        retained = choices.count() - 1
                choices.setCurrentIndex(retained if preserve else (0 if choices.count() else -1))
            if preserve and retained < 0:
                reason.clear()
                status.setText("The previous selection is no longer listed. Select a model and enter a reason.")
            update_actions()

        reason.textChanged.connect(update_actions)
        choices.currentIndexChanged.connect(update_actions)
        dialog.refresh_choices = refresh_choices
        refresh_choices(preserve=False)
        self.archive_dialog = dialog
        self._archive_dialog_context = self.management_sections.context()
        self.management_sections.present("archive", dialog.windowTitle(), dialog, 0)
