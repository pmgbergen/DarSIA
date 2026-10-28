"""Original evidence queues on the single CalibrationReviewWindow canvas.

The owning subclass calls ``save_finished`` after its original _save_finished,
and ``render`` after its original _render_vectors. It retains this adapter in
_workflow_adapters and honors begin_close/can_close. No Comparison window fields
are emulated. _workflow_model_root is the explicitly selected saved model.
"""
from __future__ import annotations

from contextvars import copy_context
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import uuid

from PySide6.QtCore import QObject, QPointF, QRunnable, QRectF, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QComboBox, QDockWidget, QFileDialog, QLabel,
    QLineEdit, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QVBoxLayout, QWidget)

from ..phase_evidence_audit import load_queue, read
from ..phase_review_contradictions import (build_contradiction_queue, correction_role,
    preserve_correction_role, read_reveal_marker, require_correction_session_role,
    require_disclosure, seal_blind_evidence)
from ..phase_labels import PHASE_CHOICE_TO_MANUAL_CATEGORY
from ..project import sha256_file, utc_now


@dataclass
class PendingCorrection:
    """Only a matching durable original save result completes a correction."""
    run: str
    minimum_generation: int
    annotation: dict

    def matches(self, generation, run, record, error):
        if error or generation < self.minimum_generation or run != self.run:
            return False
        if record.get("run") != run:
            return False
        keys = ("id", "category", "frame_index", "shape", "points", "x", "y",
                "width", "height", "evidence_audit")
        return any(all(item.get(key) == self.annotation.get(key) for key in keys)
                   for item in record.get("annotations") or [])


class _Signals(QObject):
    finished = Signal(object, str)
    sealed = Signal(object)


class _EvaluationTask(QRunnable):
    def __init__(self, *, model_root, review_root, decisions, session, reviewer, model_id, output,
                 write_session, workspace):
        super().__init__()
        self.signals = _Signals()
        self.context = copy_context()
        self.options = dict(model_root=model_root, review_root=review_root,
            decisions=decisions, session=session, reviewer=reviewer, model_id=model_id, output=output,
            write_session=write_session, workspace=workspace)

    def run(self):
        self.context.run(self._run)

    def _run(self):
        value = {}
        try:
            p = self.options
            value["marker"] = seal_blind_evidence(p["decisions"], p["session"],
                model_id=p["model_id"], reviewer=p["reviewer"])
            self.signals.sealed.emit(value["marker"])
            p["write_session"](p["workspace"], p["reviewer"])
            # The snapshot records first disclosure. A later explicit evaluation
            # measures the current durable labels, preserving the first snapshot.
            value["result"] = build_contradiction_queue(p["model_root"], p["review_root"],
                [dict(decisions=str(p["decisions"]),
                      manifest=str(Path(p["review_root"]) / "postflight_manifest.json"),
                      session=str(p["session"]))], p["output"])
            value["queue"] = load_queue(value["result"]["queue_path"], verify_sources=True)
            require_disclosure(Path(p["session"]).parent, p["model_root"],
                               [row["run"] for row in value["queue"]["entries"]])
            self.signals.finished.emit(value, "")
        except Exception as error:
            marker_path = Path(self.options["session"]).parent / "blind_reveal.json"
            if marker_path.exists() and "marker" not in value:
                # A partial/invalid disclosure marker must not reopen blind
                # editing in the current process, either.
                value["seal_invalid"] = True
            self.signals.finished.emit(value, f"{type(error).__name__}: {error}")


class CalibrationContradictionDock(QDockWidget):
    def __init__(self, window, desktop):
        super().__init__("Review contradictions", window)
        self.window, self.desktop = window, desktop
        self.queue = None
        self.path = None
        self.digest = None
        self.rows = []
        self.current = None
        self.pending = None
        self.completed = {}
        self.closing = False
        self.seal_invalid = False
        self.awaiting_save = False
        self.task = None
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.focus_pending = False
        self._central_was_enabled = True
        body = QWidget(self)
        layout = QVBoxLayout(body)
        self.evaluate = QPushButton("Finish blind answers and evaluate contradictions")
        self.evaluate.clicked.connect(self.request_evaluation)
        layout.addWidget(self.evaluate)
        self.open_button = QPushButton("Open saved contradictions")
        self.open_button.clicked.connect(self.choose_queue)
        layout.addWidget(self.open_button)
        self.status = QLabel("Contradictions reveal model information. Finish the blind answers first.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(("Experiment", "Time [h]", "Original label", "State"))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.select)
        layout.addWidget(self.table)
        self.details = QLabel()
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.category = QComboBox()
        for index in range(window.annotation_category.count()):
            self.category.addItem(window.annotation_category.itemText(index), window.annotation_category.itemData(index))
        layout.addWidget(self.category)
        self.note = QLineEdit()
        self.note.setPlaceholderText("Optional review note")
        layout.addWidget(self.note)
        self.confirm = QPushButton("Confirm original label")
        self.confirm.clicked.connect(lambda: self.save("confirmed"))
        self.change = QPushButton("Save selected label")
        self.change.clicked.connect(lambda: self.save("reclassified"))
        layout.addWidget(self.confirm)
        layout.addWidget(self.change)
        self.setWidget(body)
        window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self)
        self._controls()

    def _idle(self):
        w = self.window
        return not (self.closing or self.seal_invalid or self.task or self.pool.activeThreadCount() or self.awaiting_save or self.pending
                    or w.save_inflight or w.pending_save_requests or w.save_dirty or w.sync_error_pending)

    def _controls(self):
        idle = self._idle()
        w, row = self.window, self.current
        ready = bool(idle and row and not w.store.is_read_only and w.reviewer.text().strip()
                     and row["run"] == w.current_run and row["frame_index"] == w.frame_index
                     and not w.reference_mode and w.base_key and w.base_key == w.displayed_base_key)
        self.confirm.setEnabled(ready)
        self.change.setEnabled(ready)
        self.open_button.setEnabled(idle)
        self.evaluate.setEnabled(not (self.closing or self.seal_invalid or self.awaiting_save or self.task or self.pool.activeThreadCount() or self.pending)
                                 and not w.store.is_read_only)
        if self.task is None and self.pool.activeThreadCount():
            QTimer.singleShot(100, self._controls)

    def _error(self, text):
        self.status.setText(text)
        if not self.closing:
            QMessageBox.warning(self, "Contradiction review incomplete", text)

    def _refresh_peer_controls(self):
        for adapter in self.window._workflow_adapters:
            if adapter is not self and (render := getattr(adapter, "render", None)) is not None:
                render()

    def request_evaluation(self):
        w = self.window
        if self.closing or self.seal_invalid or self.awaiting_save or self.task or self.pool.activeThreadCount() or self.pending or w.store.is_read_only:
            return
        if not w.reviewer.text().strip() or not getattr(w, "_workflow_model_root", None):
            self._error("An explicit saved model and reviewer are required")
            return
        if (w.quiz_preparation_task is not None
                or any(not adapter.can_close() for adapter in w._workflow_adapters if adapter is not self)):
            self._error("Wait for blind-box preparation and the other review actions to finish before sealing")
            return
        self.awaiting_save = True
        w._workflow_sealing = True
        self._refresh_peer_controls()
        self._central_was_enabled = w.centralWidget().isEnabled()
        w.centralWidget().setEnabled(False)
        self.status.setText("Saving the original review before sealing blind answers…")
        try:
            w._save_now()
            if w.sync_error_pending:
                raise RuntimeError("The canonical review save failed; blind answers remain unsealed")
            self.saved_complete()
        except Exception as error:
            self.awaiting_save = False
            self._unlock()
            self._error(str(error))
        self._controls()

    def saved_complete(self):
        """Launch only after the original asynchronous save queue is durable."""
        w = self.window
        if (not self.awaiting_save or self.closing or w.save_inflight
                or w.pending_save_requests or w.save_dirty or w.sync_error_pending):
            return
        self.awaiting_save = False
        self.task = _EvaluationTask(model_root=Path(w._workflow_model_root),
            review_root=Path(w.workspace.combined_root), decisions=Path(w.store.json_path),
            session=Path(w.workspace.output_dir) / "review_session.json",
            reviewer=w.reviewer.text().strip(), model_id=w.workspace.target_model_id,
            output=Path(w.workspace.output_dir) / "evidence_audits" / uuid.uuid4().hex,
            write_session=self.desktop.write_review_session_manifest, workspace=w.workspace)
        self.task.signals.finished.connect(self._evaluated, Qt.ConnectionType.QueuedConnection)
        self.task.signals.sealed.connect(self._sealed, Qt.ConnectionType.QueuedConnection)
        self.status.setText("Sealing blind evidence and measuring the original saved predictions…")
        self.pool.start(self.task)

    def _unlock(self):
        self.window._workflow_sealing = False
        self._refresh_peer_controls()
        if not self.closing and not getattr(self.window, "_workflow_closing", False):
            self.window.centralWidget().setEnabled(self._central_was_enabled)

    @Slot(object)
    def _sealed(self, marker):
        callback = getattr(self.window, "_workflow_blind_sealed", None)
        if callback:
            callback(marker)
        self.evaluate.setText("Evaluate current contradictions")

    @Slot(object, str)
    def _evaluated(self, value, error):
        self.task = None
        self.seal_invalid = bool(value.get("seal_invalid"))
        marker = value.get("marker")
        if marker:
            self._sealed(marker)
        if not self.seal_invalid:
            self._unlock()
        if not self.closing:
            if error:
                self._error(error)
            else:
                try:
                    self._bind_queue(value["result"]["queue_path"], value["queue"])
                except Exception as failure:
                    self._error(str(failure))
        self._controls()

    def choose_queue(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open saved contradiction queue",
            str(self.window.workspace.output_dir), "JSON (*.json)")
        if path:
            try:
                self.load_queue(path)
            except Exception as error:
                self._error(str(error))

    def load_queue(self, path):
        if not self._idle():
            raise ValueError("Wait for the original review save and evaluation to finish")
        queue = load_queue(path, verify_sources=True)
        self._bind_queue(path, queue)

    def _bind_queue(self, path, queue):
        w = self.window
        if queue["model_id"] != w.workspace.target_model_id:
            raise ValueError("The queue belongs to another model")
        if Path(queue["model_root"]).resolve() != Path(w._workflow_model_root).resolve():
            raise ValueError("The queue belongs to another saved model root")
        if not queue.get("automatic_saved_map_review"):
            raise ValueError("Choose a fresh original saved-map contradiction queue")
        marker = require_disclosure(w.workspace.output_dir, queue["model_root"],
                                    [row["run"] for row in queue["entries"]])
        for row in queue["entries"]:
            correction_role(queue, row)  # Old queues without roles cannot be corrected here.
        self.queue, self.path, self.digest = queue, Path(path), sha256_file(Path(path))
        self.rows = list(queue["entries"])
        self.current = None
        self.completed = {}
        callback = getattr(w, "_workflow_blind_sealed", None)
        if callback:
            callback(marker)
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.rows))
        for index, row in enumerate(self.rows):
            values = (row["run"].upper(), f"{row['elapsed_h']:.3f}", row["original_class"], "Awaiting review")
            for column, value in enumerate(values):
                self.table.setItem(index, column, QTableWidgetItem(value))
        self.table.blockSignals(False)
        coverage = queue.get("coverage") or {}
        reasons = "; ".join(f"{reason} ({count})" for reason, count in
                            sorted((coverage.get("skipped_reason_counts") or {}).items()))
        scored = coverage.get("scored_regions", 0)
        if not scored:
            message = "Evaluation unavailable: no supplied regions could be scored. " + reasons
        elif not coverage.get("complete"):
            message = (f"Partial evaluation: {scored} measured regions, {len(self.rows)} contradictions. "
                       + (reasons or "Original evidence conflicts or excluded candidate-specific labels remain."))
        else:
            message = f"{len(self.rows)} contradictions across {scored} measured regions. All eligible supplied regions were measured."
        self.status.setText(message)
        self.status.setToolTip(message + "\nEvaluation: " + str(queue["evaluation_path"]))
        self._controls()

    def select(self):
        if not self._idle():
            return
        index = self.table.currentRow()
        if index < 0:
            return
        row = self.rows[index]
        try:
            self._verify_current_sources(row)
            w = self.window
            matching = [i for i in range(w.run_list.count())
                        if w.run_list.item(i).data(Qt.ItemDataRole.UserRole) == row["run"]]
            if len(matching) != 1:
                raise ValueError("The original canvas has no unique matching experiment")
            w.run_list.setCurrentRow(matching[0])
            w._set_frame(row["frame_index"])
            self._verify_canvas(row)
            self.current = row
            self.focus_pending = True
            self.details.setText(row["reason_text"] +
                f"\nSaved model: gas {row['old_gas_fraction']:.1%}, aqueous {row.get('aqueous_fraction', 0):.1%}.")
            w._render_vectors()
        except Exception as error:
            self.current = None
            self._error(str(error))
        self._controls()

    def _verify_current_sources(self, row):
        if sha256_file(self.path) != self.digest:
            raise ValueError("The displayed contradiction queue changed")
        load_queue(self.path, verify_sources=True)
        require_disclosure(self.window.workspace.output_dir, self.queue["model_root"], [row["run"]])

    def _verify_canvas(self, row):
        w = self.window
        if w.current_run != row["run"] or w.frame_index != row["frame_index"] or w.reference_mode:
            raise ValueError("The canvas is not on the annotation's original experiment and image")
        path = Path(w.current_review_root) / "native_frames" / row["run"] / "manifest.json"
        if sha256_file(path) != self.queue["natives"][row["run"]]["sha256"]:
            raise ValueError("The canvas has a different native geometry or acquisition timeline")

    def render(self):
        self._controls()
        w, row = self.window, self.current
        if not row or w.current_run != row["run"] or w.frame_index != row["frame_index"] or w.reference_mode:
            return
        p = row["annotation"]
        rect = QRectF(*(float(p[key]) for key in ("x", "y", "width", "height")))
        points = [QPointF(float(x), float(y)) for x, y in p.get("points") or []]
        if points:
            w.viewer.add_polygon(points, QColor("#ef3e36"), width=3.)
        else:
            w.viewer.add_rect(rect, QColor("#ef3e36"), width=3.)
        w.viewer.add_label(QPointF(rect.x(), max(.01, rect.y() - .015)), "Original evidence region", QColor("#ef3e36"))
        if self.focus_pending and w.base_key == w.displayed_base_key:
            w.viewer.focus_normalized_rect(rect)
            self.focus_pending = False

    def save(self, verdict):
        w, row = self.window, self.current
        if not row or not self.confirm.isEnabled():
            return
        try:
            self._verify_current_sources(row)
            self._verify_canvas(row)
            role = correction_role(self.queue, row)
            session = read(Path(w.workspace.output_dir) / "review_session.json")
            require_correction_session_role(role, session)
            p = row["annotation"]
            category = ((p.get("category") or PHASE_CHOICE_TO_MANUAL_CATEGORY.get(p.get("human_choice")))
                        if verdict == "confirmed" else self.category.currentData())
            category_index, side_index = w.annotation_category.findData(category), w.candidate_side.findData("A")
            if category_index < 0 or side_index < 0:
                raise ValueError("Choose an original supported phase category")
            previous_category, previous_side = w.annotation_category.currentIndex(), w.candidate_side.currentIndex()
            old_category_block = w.annotation_category.blockSignals(True)
            old_side_block = w.candidate_side.blockSignals(True)
            try:
                w.annotation_category.setCurrentIndex(category_index)
                w.candidate_side.setCurrentIndex(side_index)
                rect = QRectF(*(float(p[key]) for key in ("x", "y", "width", "height")))
                points = [QPointF(float(x), float(y)) for x, y in p.get("points") or []]
                annotation = w._annotation_payload("freehand" if points else "rectangle", rect, points)
            finally:
                w.annotation_category.setCurrentIndex(previous_category)
                w.candidate_side.setCurrentIndex(previous_side)
                w.annotation_category.blockSignals(old_category_block)
                w.candidate_side.blockSignals(old_side_block)
            if verdict == "confirmed":
                annotation["confidence"] = p.get("confidence", "low")
            annotation.update(comment=self.note.text(), updated_utc=utc_now(),
                evidence_audit=dict(original_region_id=row["id"],
                    original_annotation_sha256=row["original_annotation_sha256"], queue_sha256=self.digest,
                    verdict=verdict, source_model_identity=self.queue["model_identity"]))
            annotation = preserve_correction_role(annotation, role)
            # Predict the retained writer's representation through its own
            # geometry cleaner, including duplicate polygon vertices/bounds.
            shape, coordinates, clean_points = self.desktop.review._clean_annotation_geometry(annotation)
            annotation.update(shape=shape, points=clean_points, **coordinates)
            existing = next((i for i, item in enumerate(w.annotations)
                if item.get("evidence_audit", {}).get("original_region_id") == row["id"]
                and item["evidence_audit"].get("queue_sha256") == self.digest), None)
            if existing is None:
                existing = len(w.annotations)
                w.annotations.append(annotation)
            else:
                annotation["id"] = w.annotations[existing]["id"]
                w.annotations[existing] = annotation
            self.pending = PendingCorrection(row["run"], w.save_generation + 1, deepcopy(annotation))
            self._central_was_enabled = w.centralWidget().isEnabled()
            w.centralWidget().setEnabled(False)
            w._refresh_annotations(selected_index=existing)
            w._save_now()
            self.status.setText("Correction queued; waiting for the original durable save…")
        except Exception as error:
            self._error(str(error))
        self._controls()

    def save_finished(self, generation, run, record, error):
        if self.pending and self.pending.matches(generation, run, record, error):
            audit = self.pending.annotation["evidence_audit"]
            self.completed[audit["original_region_id"]] = audit["verdict"]
            self.pending = None
            self._unlock()
            self.status.setText("Correction saved durably. Original evidence remains unchanged.")
            for index, row in enumerate(self.rows):
                if row["id"] in self.completed:
                    self.table.item(index, 3).setText(self.completed[row["id"]])
        elif error and self.pending:
            self.status.setText("Correction not complete: original save failed; its recovery/retry remains active.")
        if error and self.awaiting_save:
            self.awaiting_save = False
            self._unlock()
            self.status.setText("Blind answers remain unsealed because the original save failed.")
        self.saved_complete()
        self._controls()

    def begin_close(self):
        self.closing = True
        self.awaiting_save = False
        if self.task is None and not self.seal_invalid:
            self.window._workflow_sealing = False
        self._controls()

    def cancel_close(self):
        self.closing = False
        if self.task is None and not self.seal_invalid:
            self.window._workflow_sealing = False
            if self.pending is None:
                self._unlock()
        if self.pending or self.task or self.seal_invalid:
            self.window.centralWidget().setEnabled(False)
        self._controls()

    def can_close(self):
        return self.task is None and self.pending is None and self.pool.activeThreadCount() == 0


def install(window, desktop):
    """The owner retains the returned dock and connects the documented hooks."""
    if not isinstance(window, desktop.CalibrationReviewWindow):
        raise ValueError("Contradictions require the original CalibrationReviewWindow canvas")
    root = Path(window.workspace.output_dir)
    session = read(root / "review_session.json")
    revealed = ((root / "blind_reveal.json").exists() or session.get("blindness_status") == "revealed"
                or session.get("frozen_blind_snapshot"))
    marker = read_reveal_marker(root) if revealed else None
    dock = CalibrationContradictionDock(window, desktop)
    if marker:
        dock._sealed(marker)
    return dock
