"""One native phase session using the original review canvas and save workers.

Both roles retain the same candidate, canvas and store. Label phases can show
the saved prediction; blind validation always hides it.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import QComboBox, QLabel, QToolBar


def run_shared_main(desktop, args):
    """Use the original inventory, manifest, canvas, readers and save directory."""
    from ..native_phase_review_session import (
        attach_review_manifest, read_current_review_session, review_editing_lease)
    if args.workspace != "training" or not args.phase_workflow_series or args.evidence_audit:
        raise ValueError("The native phase session requires its training workspace and series")
    if not getattr(args, "review_current_model_root", None):
        raise ValueError("The native review requires an exact saved model folder binding")
    reviewer, save_root = desktop.effective_reviewer_settings(args)
    desktop.QApplication.setAttribute(desktop.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    surface = desktop.QSurfaceFormat()
    surface.setRenderableType(desktop.QSurfaceFormat.RenderableType.OpenGL)
    surface.setVersion(3, 3)
    surface.setProfile(desktop.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface.setSamples(4)
    desktop.QSurfaceFormat.setDefaultFormat(surface)
    app = desktop.QApplication(desktop.sys.argv[:1])
    app.setApplicationName("FluidFlower Phase Model Correction")
    app.setStyle("Fusion")
    with review_editing_lease(save_root, series=args.phase_workflow_series, reviewer=reviewer):
        workspace = desktop.local_reviewer_workspace(
            desktop.review_workspaces(args)["training"], save_root, reviewer)
        reservation = read_current_review_session(save_root, series=args.phase_workflow_series, reviewer=reviewer)
        if reservation is None:
            raise ValueError("The native review session has not been reserved")
        binding = validate_workspace(workspace, reviewer=reviewer, reservation=reservation,
                                     model_root=args.review_current_model_root)
        desktop.ensure_writable_directory(workspace.output_dir)
        binding_path = Path(workspace.output_dir) / "review_source_binding.json"
        if not binding_path.exists():
            # The process-wide editing lease still owns this initial publication.
            # Original manifest refreshes never touch this immutable sidecar.
            desktop.atomic_json(binding_path, binding)
        app_model, store = desktop.build_workspace_model(args, workspace, reviewer_name=reviewer)
        attach_review_manifest(save_root, series=args.phase_workflow_series, reviewer=reviewer,
            review_session_id=workspace.review_session_id,
            session_path=workspace.output_dir / "review_session.json")
        window = create_window(desktop, app_model, store, args.cache_root, workspace,
            reviewer_name=reviewer, role=args.phase_workflow_role,
            use_opengl=not args.no_gpu and desktop.os.environ.get("QT_QPA_PLATFORM", "").casefold() != "offscreen")
        window._workflow_model_root = args.review_current_model_root
        # Plain launch metadata is copied for finite workers. They never inspect
        # widgets or acquire the editing lease already held by this GUI thread.
        window._workflow_request_context = dict(save_root=Path(save_root),
            series=args.phase_workflow_series, reviewer=reviewer,
            model_root=args.review_current_model_root, binding=deepcopy(binding),
            workspace={key: deepcopy(getattr(workspace, key)) for key in (
                "key", "single_model", "review_surface", "review_session_id", "target_model_id", "output_dir",
                "combined_root", "ffac_root", "ffum_root", "selected_candidates", "ffac_runs", "ffum_runs")})
        from .phase_review_proposal_generation import install as install_proposal_generation
        window._workflow_adapters.append(install_proposal_generation(window, desktop))
        from .phase_contradiction_review import install as install_contradiction_review
        window._workflow_adapters.append(install_contradiction_review(window, desktop))
        from .phase_prediction_disclosure import install as install_prediction_disclosure
        window._workflow_prediction_disclosure = install_prediction_disclosure(window, desktop)
        window._workflow_adapters.append(window._workflow_prediction_disclosure)
        window.show()
        return int(app.exec())


def _bound_path(value, *, context):
    from ..review_storage import resolve_review_path
    from ..storage_paths import resolve_storage_path
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("The saved review source binding requires an absolute folder")
    mapped = resolve_storage_path(value)
    if not mapped.is_absolute():
        raise ValueError("The saved review source binding requires an absolute folder on this host")
    return os.path.normcase(str(resolve_review_path(mapped, context=context).resolve()))


def _candidate_binding(value):
    if not isinstance(value, Mapping) or not value:
        raise ValueError("The saved review has no explicit selected-candidate binding")
    result = {}
    for run, candidate in value.items():
        if not isinstance(run, str) or not run.strip() or not isinstance(candidate, str) or not candidate.strip():
            raise ValueError("The saved review has an invalid selected-candidate binding")
        key = run.strip().casefold()
        if key in result:
            raise ValueError("The saved review repeats a selected experiment")
        result[key] = candidate.strip()
    return result


def _run_filter(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or any(not isinstance(run, str) or not run.strip() for run in value):
        raise ValueError("The saved review has an invalid experiment filter")
    runs = [run.strip().casefold() for run in value]
    if len(set(runs)) != len(runs):
        raise ValueError("The saved review repeats an experiment filter")
    return sorted(runs)


def _source_binding(value, *, context):
    return {**{key: _bound_path(value.get(key), context=context)
               for key in ("combined_root", "ffac_root", "ffum_root")},
            "selected_candidates": _candidate_binding(value.get("selected_candidates")),
            **{key: _run_filter(value.get(key)) for key in ("ffac_runs", "ffum_runs")}}


def validate_workspace(workspace, *, reviewer=None, reservation=None, model_root=None):
    """Reject incompatible or retired stores before the original writer opens."""
    if (workspace.key != "training" or not workspace.single_model
            or workspace.review_surface != "blind_regions"
            or not workspace.review_session_id or not workspace.target_model_id):
        raise ValueError("The shared phase window requires one model-bound image-only session")
    if reservation is not None and (
            reservation.get("status") != "active"
            or reservation.get("review_session_id") != workspace.review_session_id
            or reservation.get("target_model_id") != workspace.target_model_id):
        raise ValueError("The native review reservation is retired or belongs to another model/session")
    path = Path(workspace.output_dir) / "review_session.json"
    sources = _source_binding({key: getattr(workspace, key, None) for key in (
        "combined_root", "ffac_root", "ffum_root", "selected_candidates", "ffac_runs", "ffum_runs")}, context=path)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict) or payload.get("format") != "fluidflower-local-review-session-v1":
            raise ValueError("The original review session has an unreadable format")
        if (payload.get("editing_locked") or payload.get("include_in_training") is False
                or payload.get("retired_workflow") or payload.get("independent_holdout")
                or payload.get("evidence_role") in {"blind_transfer_validation", "independent_blind_gas_validation"}):
            raise ValueError("This review session is retired or locked for training")
        if reviewer is not None:
            from ..reviewer_preferences import reviewer_identifier
            owner = reviewer_identifier(reviewer)
            if (reviewer_identifier(payload.get("reviewer") or "") != owner
                    or payload.get("reviewer_id", owner) != owner):
                raise ValueError("The original review session has another human owner")
        from ..review_storage import resolve_review_path
        if resolve_review_path(payload.get("decision_dir") or path.parent, context=path).resolve() != path.parent.resolve():
            raise ValueError("The original review session points to another decision store")
        if payload.get("workspace") != "training" or payload.get("review_surface") != "blind_regions":
            raise ValueError("The original review session uses a different review surface")
        if (payload.get("review_session_id") != workspace.review_session_id
                or payload.get("target_model_id") != workspace.target_model_id):
            raise ValueError("The saved review session belongs to a different model or session")
        if _source_binding(payload, context=path) != sources:
            raise ValueError("The saved review source folders or selected experiments changed; submit or discard this session first")
        if ((path.parent / "blind_reveal.json").exists()
                or payload.get("blindness_status") == "revealed" or payload.get("frozen_blind_snapshot")):
            from ..phase_review_contradictions import read_reveal_marker
            read_reveal_marker(path.parent)
    binding_path = path.parent / "review_source_binding.json"
    if model_root is None:
        if binding_path.exists():
            raise ValueError("The exact saved model folder is required to reopen this review")
        return None
    from ..reviewer_preferences import reviewer_identifier
    binding = dict(format="native-phase-review-source-binding-v1",
        review_session_id=workspace.review_session_id, target_model_id=workspace.target_model_id,
        reviewer_id=reviewer_identifier(reviewer or ""), model_root=_bound_path(model_root, context=path), **sources)
    if binding_path.exists():
        retained = json.loads(binding_path.read_text(encoding="utf-8-sig"))
        if not isinstance(retained, dict):
            raise ValueError("The saved review source binding is unreadable")
        normalized = dict(retained, model_root=_bound_path(retained.get("model_root"), context=path),
                          **_source_binding(retained, context=path))
        if normalized != binding:
            raise ValueError("The saved review source binding identifies another model folder or experiment selection")
    elif path.exists():
        raise ValueError("This existing review has no retained exact model folder binding; explicitly discard it before starting a new session")
    return binding


def create_window(desktop, *args, role="label", **kwargs):
    """Extend the loaded desktop class without importing a second script module."""
    if role not in {"label", "blind"}:
        raise ValueError("Unknown phase review role")

    class PhaseModelCorrectionWindow(desktop.CalibrationReviewWindow):
        def __init__(self, *window_args, **window_kwargs):
            self._workflow_selection_epoch = 0
            self._workflow_adapters = []
            self._workflow_closing = False
            self._workflow_close_saved = False
            self._workflow_revealed = False
            self._workflow_sealing = False
            self._workflow_prediction_ready = False
            super().__init__(*window_args, **window_kwargs)
            self.setWindowTitle("Phase model correction")
            # A session has one human owner. The original reviewer provenance
            # stays authoritative for every answer and drawn annotation.
            self.reviewer.setReadOnly(True)
            bar = QToolBar("Review task", self)
            bar.setMovable(False)
            self.addToolBar(bar)
            self.workflow_role = QComboBox()
            self.workflow_role.addItem("Label phases", "label")
            self.workflow_role.addItem("Blindly validate phase model", "blind")
            bar.addWidget(self.workflow_role)
            self.workflow_notice = QLabel(
                "One saved session · Phase map shows or hides the prediction in Label phases. "
                "Blind validation hides predictions.")
            self.workflow_notice.setWordWrap(True)
            bar.addWidget(self.workflow_notice)
            self.workflow_role.currentIndexChanged.connect(self._focus_workflow_role)
            self.workflow_role.setCurrentIndex(0 if role == "label" else 1)
            self._focus_workflow_role()
            self._workflow_close_timer = QTimer(self)
            self._workflow_close_timer.setInterval(100)
            self._workflow_close_timer.timeout.connect(self._finish_workflow_close)

        def _workflow_invalidate_proposals(self):
            for adapter in self._workflow_adapters:
                invalidate = getattr(adapter, "invalidate_request", None)
                if invalidate is not None:
                    invalidate()

        @property
        def _workflow_sealing(self):
            return getattr(self, "_workflow_sealing_value", False)

        @_workflow_sealing.setter
        def _workflow_sealing(self, value):
            if value:
                self._workflow_invalidate_proposals()
            self._workflow_sealing_value = bool(value)

        @property
        def _workflow_revealed(self):
            return getattr(self, "_workflow_revealed_value", False)

        @_workflow_revealed.setter
        def _workflow_revealed(self, value):
            if value:
                self._workflow_invalidate_proposals()
            self._workflow_revealed_value = bool(value)

        def _load_run(self, run):
            if run:
                # Invalidate before the original changes any live pool/answers.
                # A same-run reload is a new generation too, so A -> B -> A
                # cannot revive a request captured against an earlier visit.
                self._workflow_invalidate_proposals()
                self._workflow_selection_epoch += 1
            return super()._load_run(run)

        def _focus_workflow_role(self, *_args):
            blind = self.workflow_role.currentData() == "blind"
            if hasattr(self, "contours"):
                has_model = bool(self.workspace.target_model_id)
                self.contours.setVisible(not blind and has_model)
                self.contours.setToolTip(
                    "Show or hide the selected model's saved prediction. "
                    "Showing it preserves the existing blind answers; later edits are corrections.")
                self.contour_filter.setVisible(not blind and has_model)
                self._request_overlay()
            self.quiz_title.setText("Blind validation · answers sealed" if self._workflow_revealed
                                    else "Blindly validate phase model")
            target = self.quiz_tabs if blind else self.manual_evidence_panel
            # Keep all evidence visible and in the same original widgets; only
            # scroll the task's controls into view.
            parent = target.parentWidget()
            while parent is not None:
                if hasattr(parent, "ensureWidgetVisible"):
                    parent.ensureWidgetVisible(target)
                    break
                parent = parent.parentWidget()
            target.setFocus()

        def _phase_prediction_hidden(self):
            role_control = getattr(self, "workflow_role", None)
            return (role_control is None or role_control.currentData() != "label"
                    or not self._workflow_prediction_ready)

        def _request_overlay(self):
            role_control = getattr(self, "workflow_role", None)
            if (role_control is not None and role_control.currentData() == "label"
                    and self.contours.isChecked() and not self._workflow_prediction_ready):
                disclosure = getattr(self, "_workflow_prediction_disclosure", None)
                if disclosure is not None:
                    disclosure.request(self._workflow_prediction_unlocked)
            super()._request_overlay()
            self._refresh_prediction_notice()

        def _workflow_prediction_unlocked(self):
            self._workflow_prediction_ready = True
            self._request_overlay()

        def _refresh_prediction_notice(self):
            if not hasattr(self, "workflow_role") or not self.workspace.target_model_id:
                return
            blind = self.workflow_role.currentData() == "blind"
            shown = not self._phase_prediction_hidden() and self.contours.isChecked() and not self.reference_mode
            title = self.findChild(QLabel, "pageTitle")
            badge = self.findChild(QLabel, "workspaceBadge")
            if title is not None:
                title.setText("Blindly validate phase model" if blind else "Label phases")
            if badge is not None:
                badge.setText("PREDICTION ON" if shown else "PREDICTION OFF")
            self.workspace_notice.setText(
                ("Prediction hidden. Earlier blind answers are sealed; use Label phases for corrections."
                 if self._workflow_revealed else "Model predictions are hidden for blind validation.")
                if blind else
                "Use Phase map to show or hide the selected model's prediction. "
                "Draw boxes or areas on the same image to add or correct labels.")

        def _refresh_blind_reveal_notice(self):
            super()._refresh_blind_reveal_notice()
            self._refresh_prediction_notice()

        @Slot(int, str, object, str)
        def _save_finished(self, *save_args):
            super()._save_finished(*save_args)
            for adapter in self._workflow_adapters:
                callback = getattr(adapter, "save_finished", None)
                if callback is not None:
                    callback(*save_args)
            if self._workflow_closing and self.sync_error_pending:
                self._cancel_workflow_close()
                self.statusBar().showMessage(
                    "Review remains open because its final save failed. Retry saving before closing.", 15000)

        def _render_vectors(self):
            super()._render_vectors()
            for adapter in self._workflow_adapters:
                render = getattr(adapter, "render", None)
                if render is not None:
                    render()

        def _workflow_blind_sealed(self, _marker):
            self._workflow_revealed = True
            self.workflow_role.setItemText(1, "Blind validation · answers sealed")
            self.workflow_notice.setText(
                "Blind answers are sealed before contradiction details are shown. "
                "Their saved snapshot is preserved; further work in this session is labelled correction.")
            for adapter in self._workflow_adapters:
                callback = getattr(adapter, "blind_sealed", None)
                if callback is not None:
                    callback()
            self._refresh_quiz()

        def _refresh_quiz(self):
            super()._refresh_quiz()
            if self._workflow_revealed or self._workflow_sealing:
                for control in (*self.phase_buttons.values(), self.ignore_region, self.clear_answer,
                                self.confidence, self.answer_note, self.new_round, self.new_round_count):
                    control.setEnabled(False)
                self.quiz_entry_help.setText(
                    "Blind answers are sealed. Resolve contradictions with the label drawing tools."
                    if self._workflow_revealed else "Finishing the saved blind answers before evaluating contradictions.")

        def _answer(self, *answer_args):
            if not (self._workflow_revealed or self._workflow_sealing):
                return super()._answer(*answer_args)

        def _update_existing_answer(self):
            if not (self._workflow_revealed or self._workflow_sealing):
                return super()._update_existing_answer()

        def _clear_answer(self):
            if not (self._workflow_revealed or self._workflow_sealing):
                return super()._clear_answer()

        def _start_new_round(self):
            if not (self._workflow_revealed or self._workflow_sealing):
                return super()._start_new_round()

        def _cancel_workflow_close(self):
            self._workflow_closing = False
            self._workflow_close_saved = False
            self._workflow_close_timer.stop()
            self.centralWidget().setEnabled(True)
            self.workflow_role.setEnabled(True)
            self.quiz_preparation_closed = False
            for adapter in self._workflow_adapters:
                adapter.cancel_close()

        def _workflow_busy(self):
            pools = [self.save_pool, self.quiz_preparation_pool]
            pools.extend(getattr(self.image_cache, name) for name in
                         ("pool", "reference_pool", "display_pool", "persistence_pool"))
            return (self.save_inflight or bool(self.pending_save_requests)
                    or bool(self.save_tasks) or self.quiz_preparation_task is not None
                    or any(pool.activeThreadCount() for pool in pools)
                    or any(not adapter.can_close() for adapter in self._workflow_adapters)
                    or bool(self.image_cache.running) or bool(self.image_cache.reference_running)
                    or bool(self.image_cache.display_running)
                    or self.image_cache.persistence_running is not None
                    or bool(self.image_cache.queued) or bool(self.image_cache.display_queued)
                    or bool(self.image_cache.persistence_queued))

        def _finish_workflow_close(self):
            if self._workflow_closing and not self._workflow_busy():
                self._workflow_close_timer.stop()
                self.close()

        def closeEvent(self, event):
            # Do not call the legacy close handler: it can issue a synchronous
            # second save while the canonical save worker still owns the store.
            if not self._workflow_close_saved:
                self._workflow_invalidate_proposals()
                self._workflow_closing = True
                self.quiz_preparation_closed = True
                if self.quiz_preparation_task is not None:
                    self.quiz_preparation_task.cancelled.set()
                for adapter in self._workflow_adapters:
                    adapter.begin_close()
                self.image_cache.cancel_queued_reference_preparations()
                self.image_cache.focus(())
                self.image_cache.prefetch_window(())
                self.centralWidget().setEnabled(False)
                self.workflow_role.setEnabled(False)
                if not self.store.is_read_only and self.current_run:
                    self._save_now()
                self._workflow_close_saved = True
            if self.sync_error_pending:
                self._cancel_workflow_close()
                event.ignore()
                return
            if self._workflow_busy():
                self.statusBar().showMessage("Finishing saved review and image work before closing…")
                self._workflow_close_timer.start()
                event.ignore()
                return
            self._workflow_close_timer.stop()
            event.accept()

    return PhaseModelCorrectionWindow(*args, **kwargs)
