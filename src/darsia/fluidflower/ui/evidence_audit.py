"""Disagreement triage attached to the existing native phase-review viewer."""
from __future__ import annotations

from pathlib import Path
import uuid

from PySide6.QtCore import QEvent, QPointF, QProcess, QRectF, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QComboBox, QDockWidget, QLabel, QLineEdit,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    QAbstractItemView, QApplication, QFormLayout, QHeaderView, QScrollArea, QSpinBox)

from darsia.fluidflower.phase_evidence_audit import (load_queue, read,
    restore_selected_regions, select_additional_regions, validate_pool)
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.phase_workflow import phase_python_executable
from .interaction_safety import install_interaction_safety
from .evidence_audit_library import EvidenceAuditLibrary


class EvidenceAuditDock(QDockWidget):
    def __init__(self, window, queue_path, api):
        super().__init__("Review disagreements", window)
        install_interaction_safety(QApplication.instance())
        self.window, self.api = window, api
        self.path = Path(queue_path)
        self.queue = load_queue(self.path, verify_sources=True)
        if self.queue.get("automatic_review") and self.queue.get("phase_review_basis") not in {
                "studio-optical-without-context-v1", "baseline-relative-optical-regions-v1"}:
            raise ValueError("Run Check disagreements again: this queue used older context-supported maps")
        if window.workspace.target_model_id != self.queue["model_id"]:
            raise ValueError("Review window and audit queue belong to different models")
        self.digest = sha256_file(self.path)
        audit_root = window.workspace.output_dir
        if self.queue.get("automatic_review"):
            audit_root = audit_root / "evidence_audits" / self.digest
            audit_root.mkdir(parents=True, exist_ok=True)
        self.pool_path = audit_root / "evidence_audit_candidates.json"
        self.selection_path = audit_root / "evidence_audit_selection.json"
        self.pool, self.pool_digest, self.pool_process = None, None, None
        self.selected_entries = []
        if self.pool_path.exists():
            self.pool = load_queue(self.pool_path)
            validate_pool(self.queue, self.pool)
            self.pool_digest = sha256_file(self.pool_path)
        if self.selection_path.exists():
            if self.pool is None:
                raise ValueError("Saved selected regions are missing their frozen candidate pool")
            self.selected_entries = restore_selected_regions(self.queue, self.pool,
                read(self.selection_path), queue_sha256=self.digest, pool_sha256=self.pool_digest)
        self.current = None
        self.focus_pending = False
        self.progress_path = audit_root / "evidence_audit_progress.json"
        self.deferred = read(self.progress_path) if self.progress_path.exists() else {}
        if any(value.get("queue_sha256") != self.digest for value in self.deferred.values()):
            raise ValueError("Saved review progress belongs to a different queue")
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        body = QWidget()
        layout = QVBoxLayout(body)
        self.model_details = QLabel(f"Evaluated model: {self.queue['model_id']}")
        self.model_details.setWordWrap(True)
        self.model_details.setToolTip(str(self.queue["model_root"]))
        layout.addWidget(self.model_details)
        self.run = QComboBox()
        self.run.addItem("All experiments", "")
        for run in sorted({r["run"] for r in self.queue["entries"]}):
            self.run.addItem(run.upper(), run)
        self.reason = QComboBox()
        self.reason.addItem("Temporal-rule conflicts", "temporal_rule_conflict")
        self.reason.addItem("All flagged disagreements", "")
        self.reason.addItem("Low dissolved CO2 coverage — inspect", "low_aqueous_coverage")
        self.reason.addItem("Model / label disagreements", "model_label_disagreement")
        self.reason.addItem("Required phase missing", "missing_required")
        self.reason.addItem("Conflicting annotations", "annotation_conflict")
        self.state = QComboBox()
        for label, value in (("Not yet reviewed", "new"), ("Reviewed", "reviewed"),
                             ("Needs more context", "deferred"), ("All review states", "")):
            self.state.addItem(label, value)
        self.order = QComboBox()
        self.order.addItem("Highest disagreement first", "score")
        self.order.addItem("Earliest image first", "time")
        for control in (self.run, self.reason, self.state, self.order):
            layout.addWidget(control)
            control.currentIndexChanged.connect(self.refresh)
        self.count = QLabel()
        self.count.setWordWrap(True)
        layout.addWidget(self.count)
        batch_form = QFormLayout()
        self.number_to_add = QSpinBox()
        self.number_to_add.setRange(1, 2147483647)
        self.number_to_add.setValue(20)
        self.number_to_add.setKeyboardTracking(False)
        self.number_to_add.setToolTip("Type how many additional regions to request. Already offered regions are not repeated.")
        batch_form.addRow("Number to add", self.number_to_add)
        layout.addLayout(batch_form)
        self.load_more = QPushButton("Load more disagreements")
        self.load_more.setToolTip("Load the next highest-ranked disagreements for the selected experiment, using the existing evaluated model. No retraining is performed.")
        self.load_more.clicked.connect(self.request_more)
        layout.addWidget(self.load_more)
        self.batch_status = QLabel()
        self.batch_status.setWordWrap(True)
        layout.addWidget(self.batch_status)
        self.table = QTableWidget(0, 5)
        self.table.setMinimumHeight(150)
        self.table.setHorizontalHeaderLabels(("Run", "Time [h]", "Original label", "Priority", "State"))
        self.table.horizontalHeaderItem(3).setToolTip("Larger values are reviewed first. This is a ranking score, not the probability that the label is wrong.")
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.select)
        layout.addWidget(self.table, 1)
        self.details = QLabel()
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.back_to_image = QPushButton("Return to labeled image")
        self.back_to_image.clicked.connect(self.open_current)
        layout.addWidget(self.back_to_image)
        self.category = QComboBox()
        for key, label in window.ISSUE_CATEGORY_LABELS:
            self.category.addItem(label, key)
        layout.addWidget(self.category)
        self.note = QLineEdit()
        self.note.setPlaceholderText("Optional review note")
        layout.addWidget(self.note)
        self.confirm = QPushButton("Confirm original label")
        self.change = QPushButton("Save selected label")
        self.defer = QPushButton("Needs more context")
        self.confirm.clicked.connect(lambda: self.save("confirmed"))
        self.change.clicked.connect(lambda: self.save("reclassified"))
        self.defer.clicked.connect(lambda: self.save("deferred"))
        for button in (self.confirm, self.change, self.defer):
            layout.addWidget(button)
        self.next = QPushButton("Next region")
        self.next.clicked.connect(self.next_region)
        layout.addWidget(self.next)
        extra = QPushButton("Additional drawing and review tools")
        extra.setCheckable(True)
        extra.toggled.connect(window.decision_scroll.setVisible)
        layout.addWidget(extra)
        self.saved = QLabel("No new decision saved")
        self.saved.setWordWrap(True)
        layout.addWidget(self.saved)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        self.setWidget(scroll)
        self.setMinimumWidth(420)
        self.setMaximumWidth(560)
        window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self)
        window.evidence_audit = self
        window.installEventFilter(self)
        window.reviewer.textChanged.connect(self._update_batch_controls)
        window.run_panel.setVisible(False)
        window.decision_scroll.setVisible(False)
        # The unused B viewport must not recenter A when a new image arrives.
        window.sync_views.setChecked(False)
        # The existing viewer supplies zoom, playback, layers and manual drawing.
        if not self.queue.get("automatic_review"):
            window.contours.setChecked(False)
        self.reason.setCurrentIndex(self.reason.findData(""))
        self._update_run_filter()
        self.refresh()

    def _entries(self):
        return [*self.queue["entries"], *self.selected_entries]

    def _update_run_filter(self):
        selected = self.run.currentData()
        rows = self.pool["entries"] if self.pool is not None else self._entries()
        self.run.blockSignals(True)
        self.run.clear()
        self.run.addItem("All experiments", "")
        for run in sorted({r["run"] for r in rows}):
            self.run.addItem(run.upper(), run)
        self.run.setCurrentIndex(max(0, self.run.findData(selected)))
        self.run.blockSignals(False)

    def _update_batch_controls(self, *_):
        locked = self.window.store.is_read_only
        busy = self.pool_process is not None
        ready = bool(self.window.reviewer.text().strip()) and not locked and not busy
        self.number_to_add.setEnabled(ready)
        self.load_more.setEnabled(ready)
        if busy:
            return
        if locked:
            self.batch_status.setText("Frozen session: existing selections are read-only.")
        elif self.pool is None:
            self.batch_status.setText("Additional ranked regions have not been loaded yet.")
        else:
            remaining = len(select_additional_regions(self.queue, self.pool,
                [r["id"] for r in self.selected_entries], 2147483647, run=self.run.currentData()))
            self.load_more.setEnabled(ready and remaining > 0)
            self.batch_status.setText(f"{remaining} additional evaluated disagreements available for this experiment filter."
                if remaining else "All evaluated disagreements in this experiment filter have been offered. More require newly evaluated evidence or a new model.")

    def request_more(self):
        w = self.window
        if w.store.is_read_only or not w.reviewer.text().strip() or self.pool_process is not None:
            return
        w._flush_save()
        if w.save_dirty:
            self.saved.setText("Save failed; finish saving before requesting more regions")
            return
        self.pending_count = self.number_to_add.value()
        self.pending_run = self.run.currentData()
        if self.pool is not None:
            self._append_batch()
            return
        self.pending_pool_path = self.pool_path.with_name(f".evidence-candidates-{uuid.uuid4().hex}.json")
        process = QProcess(self)
        self.pool_process = process
        process.setProgram(phase_python_executable())
        process.setArguments(["-m", "darsia.fluidflower.phase_evidence_audit",
            "--expand-queue", str(self.path), "--output", str(self.pending_pool_path)])
        process.finished.connect(self._pool_finished)
        process.errorOccurred.connect(self._pool_error)
        self.batch_status.setText("Ranking previously evaluated regions in the background...")
        self._update_batch_controls()
        process.start()

    def _pool_error(self, error):
        if self.pool_process is not None and error == QProcess.ProcessError.FailedToStart:
            message = self.pool_process.errorString()
            self.pool_process.deleteLater()
            self.pool_process = None
            self._update_batch_controls()
            self.batch_status.setText(f"Could not load more regions: {message}")

    def _pool_finished(self, code, _status):
        process, self.pool_process = self.pool_process, None
        if process is None:
            return
        error = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
        process.deleteLater()
        try:
            if code != 0:
                raise RuntimeError(error[-1000:] or "Candidate ranking failed")
            pool = load_queue(self.pending_pool_path)
            validate_pool(self.queue, pool)
            if self.pool_path.exists():
                raise FileExistsError("Another window already prepared this session's candidate pool; reopen the session")
            self.pending_pool_path.rename(self.pool_path)
            self.pool, self.pool_digest = pool, sha256_file(self.pool_path)
            self._update_run_filter()
            self._append_batch()
        except Exception as exc:
            self._update_batch_controls()
            self.batch_status.setText(f"Could not load more regions: {exc}")

    def _append_batch(self):
        try:
            if self.window.store.is_read_only:
                raise ValueError("This session is locked")
            self.window._flush_save()
            if self.window.save_dirty:
                raise RuntimeError("Finish saving the current decision before adding regions")
            added = select_additional_regions(self.queue, self.pool,
                [r["id"] for r in self.selected_entries], self.pending_count, run=self.pending_run)
            if added:
                entries = [*self.selected_entries, *added]
                state = {"queue_sha256": self.digest, "pool_sha256": self.pool_digest,
                         "selected_ids": [r["id"] for r in entries], "updated_utc": utc_now()}
                self.api.atomic_json(self.selection_path, state)
                if read(self.selection_path) != state:
                    raise RuntimeError("Selected regions could not be verified after saving")
                self.selected_entries = entries
                for control, data in ((self.run, self.pending_run), (self.reason, ""), (self.state, "new"), (self.order, "score")):
                    control.blockSignals(True)
                    control.setCurrentIndex(control.findData(data))
                    control.blockSignals(False)
                self.current = added[0]
                self.saved.setText(f"Added {len(added)} new regions; selection saved locally.")
            self.refresh()
        except Exception as exc:
            self._update_batch_controls()
            self.saved.setText(f"Could not add regions: {exc}. Existing decisions are unchanged.")

    def eventFilter(self, watched, event):
        if watched is self.window and event.type() == QEvent.Type.Close and self.pool_process is not None:
            process, self.pool_process = self.pool_process, None
            process.kill()
            process.waitForFinished(2000)
            process.deleteLater()
            self._update_batch_controls()
        return super().eventFilter(watched, event)

    def decisions(self):
        result = {key: "deferred" for key in self.deferred}
        for decision in self.window.store.decisions().values():
            for annotation in decision.get("annotations", []):
                audit = annotation.get("evidence_audit", {})
                if audit.get("queue_sha256") == self.digest:
                    result[audit["original_region_id"]] = audit["verdict"]
        return result

    def original_label(self, row):
        payload = row["annotation"]
        category = payload.get("category") or self.api.PHASE_CHOICE_TO_ANNOTATION.get(payload.get("human_choice"))
        return dict(self.window.ISSUE_CATEGORY_LABELS).get(category, row["original_class"])

    def refresh(self, *_):
        states = self.decisions()
        scope = [r for r in self._entries()
                 if (not self.run.currentData() or r["run"] == self.run.currentData())
                 and (not self.reason.currentData() or self.reason.currentData() in r["reasons"]
                      or (self.reason.currentData() == "missing_required" and any(reason.startswith("missing_required_") for reason in r["reasons"])))]
        done = sum(states.get(r["id"]) in {"confirmed", "reclassified"} for r in scope)
        coverage = EvidenceAuditLibrary.coverage_summary(self.queue)
        self.count.setText(f"{done} / {len(scope)} reviewed in this filter"
                           + ("\n" + coverage if coverage else ""))
        wanted = self.state.currentData()
        self.rows = [r for r in scope if not wanted
                     or (wanted == "new" and r["id"] not in states)
                     or (wanted == "deferred" and states.get(r["id"]) == "deferred")
                     or (wanted == "reviewed" and states.get(r["id"]) in {"confirmed", "reclassified"})]
        self.rows.sort(key=lambda r: (("annotation_conflict" not in r["reasons"],
                                      "temporal_rule_conflict" not in r["reasons"],
                                      -r["score"], r["run"], r["elapsed_h"])
                                    if self.order.currentData() == "score" else (r["elapsed_h"], r["run"], -r["score"])))
        old_id = self.current["id"] if self.current else ""
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.rows))
        index = 0
        for i, row in enumerate(self.rows):
            if row["id"] == old_id:
                index = i
            for j, value in enumerate((row["run"].upper(), f"{row['elapsed_h']:.3f}",
                                       self.original_label(row), f"{row['score']:.3f}", states.get(row["id"], "New"))):
                item = QTableWidgetItem(str(value))
                item.setToolTip(row["reason_text"])
                self.table.setItem(i, j, item)
        self.table.blockSignals(False)
        if self.rows:
            self.table.selectRow(index)
            self.select()
        else:
            self.current = None
            self.details.setText("No regions remain in this filter.")
            self.window._render_injection_points()
        self._update_batch_controls()

    def select(self):
        index = self.table.currentRow()
        if not 0 <= index < len(self.rows):
            return
        self.window._flush_save()
        if self.window.save_dirty:
            self.saved.setText("Save failed; finish saving before changing region")
            return
        self.current = self.rows[index]
        row = self.current
        reason_text = row["reason_text"].replace(
            "This region requires at least one ", "One of the overlapping annotations requires some ").replace(
            " witness, but the negative annotation union excludes ", ", but the overlapping exclusion annotations forbid ")
        aqueous = row.get("aqueous_fraction", row["co2_fraction"]-row["old_gas_fraction"])
        self.details.setText(
            f"{row['source_name']} | image {row['frame_index']+1} | {row['elapsed_h']:.3f} h\n"
            f"Original: {self.original_label(row)} ({row['original_reviewer']})\n"
            f"{reason_text}\nEvaluated model gas area: {row['old_gas_fraction']:.1%}; "
            f"aqueous area: {aqueous:.1%}; "
            f"total CO2 area: {row['co2_fraction']:.1%}.\n"
            "Disagreement is not proof that the human label is wrong.")
        category = row["annotation"].get("category") or self.api.PHASE_CHOICE_TO_ANNOTATION.get(
            row["annotation"].get("human_choice"))
        existing = next((a for a in self.window.store.decisions().get(row["run"], {}).get("annotations", [])
                         if a.get("evidence_audit", {}).get("original_region_id") == row["id"]
                         and a.get("evidence_audit", {}).get("queue_sha256") == self.digest), None)
        if existing is not None:
            category = existing["category"]
        self.category.setCurrentIndex(self.category.findData(category))
        self.note.setText(str(existing.get("comment", "")) if existing else "")
        self.open_current()

    def open_current(self):
        if self.current is None:
            return
        w, row = self.window, self.current
        if w.current_run != row["run"]:
            w._flush_save()
            if w.save_dirty:
                return
            w._load_run(row["run"])
        self.focus_pending = True
        w._set_frame(row["frame_index"])
        w._render_injection_points()

    def render(self):
        w, row = self.window, self.current
        ready = bool(row and row["run"] == w.current_run and not w.reference_mode
                     and row["frame_index"] == w.frame_index and w.displayed_base_key == w.base_key)
        for button in (self.confirm, self.change, self.defer):
            button.setEnabled(ready and not w.store.is_read_only and bool(w.reviewer.text().strip()))
        if row is None or row["run"] != w.current_run or w.reference_mode:
            w.frame_slider.set_marker(None)
            return
        self.defer.setEnabled(self.defer.isEnabled() and self.decisions().get(row["id"]) not in {"confirmed", "reclassified"})
        p = row["annotation"]
        rect = QRectF(*(float(p[k]) for k in ("x", "y", "width", "height")))
        color = QColor("#ef3e36" if row["frame_index"] == w.frame_index else "#66d9ef")
        w.frame_slider.set_marker(row["frame_index"], "Original label image")
        for viewer in w.viewers.values():
            for peer in row.get("conflict_peers", []):
                region = peer["annotation"]
                peer_rect = QRectF(*(float(region[k]) for k in ("x", "y", "width", "height")))
                peer_points = [QPointF(float(x), float(y)) for x,y in region.get("points", [])]
                if peer_points:
                    viewer.add_polygon(peer_points, QColor("#ffd166"), width=2.)
                else:
                    viewer.add_rect(peer_rect, QColor("#ffd166"), width=2.)
                peer_name = self.original_label(dict(annotation=region,
                    original_class=peer.get("class") or peer.get("original_class") or "Conflicting annotation"))
                viewer.add_label(QPointF(peer_rect.x(), max(.01, peer_rect.y()-.015)),
                    f"Conflicting: {peer_name}", QColor("#ffd166"))
            points = [QPointF(float(x), float(y)) for x,y in p.get("points", [])]
            if points:
                viewer.add_polygon(points, color, width=3.)
            else:
                viewer.add_rect(rect, color, width=3.)
            viewer.add_label(QPointF(rect.x(), max(.01, rect.y()-.015)),
                             "Label image" if ready else "Same region, different image", color)
            if ready and self.focus_pending:
                viewer.focus_normalized_rect(self.focus_bounds(row))
        if ready:
            self.focus_pending = False

    @staticmethod
    def focus_bounds(row):
        """Show the selected annotation and its actual conflict peers together."""
        def bounds(annotation):
            return QRectF(*(float(annotation[key]) for key in ("x", "y", "width", "height")))
        result = bounds(row["annotation"])
        for peer in row.get("conflict_peers", []):
            result = result.united(bounds(peer["annotation"]))
        return result.adjusted(-.015, -.015, .015, .015).intersected(QRectF(0., 0., 1., 1.))

    def save(self, verdict):
        w, row = self.window, self.current
        if not row or w.store.is_read_only or not self.confirm.isEnabled():
            return
        try:
            if verdict == "deferred":
                pending = {**self.deferred, row["id"]: {"reviewer": w.reviewer.text().strip(),
                    "updated_utc": utc_now(), "note": self.note.text(), "queue_sha256": self.digest}}
                self.api.atomic_json(self.progress_path, pending)
                self.deferred = pending
            else:
                p = row["annotation"]
                category = (p.get("category") or self.api.PHASE_CHOICE_TO_ANNOTATION.get(p.get("human_choice"))) if verdict == "confirmed" else self.category.currentData()
                if not category or not w._set_selected_issue_category("A", category):
                    raise ValueError("Select a supported phase or issue label")
                rect = QRectF(*(float(p[k]) for k in ("x", "y", "width", "height")))
                points = [QPointF(float(x), float(y)) for x,y in p.get("points", [])]
                annotation = w._issue_annotation_payload("A", "freehand" if points else "rectangle", rect, points)
                if verdict == "confirmed":
                    annotation["confidence"] = p.get("confidence", "low")
                annotation.update(comment=self.note.text(), updated_utc=utc_now(),
                    evidence_audit={"original_region_id": row["id"],
                        "original_annotation_sha256": row["original_annotation_sha256"],
                        "queue_sha256": self.digest, "verdict": verdict,
                        "source_model_identity": self.queue["model_identity"]})
                if row["id"] in {r["id"] for r in self.selected_entries}:
                    annotation["evidence_audit"].update(candidate_pool_path=str(self.pool_path),
                                                        candidate_pool_sha256=self.pool_digest)
                existing = next((i for i,a in enumerate(w.annotations)
                    if a.get("evidence_audit", {}).get("original_region_id") == row["id"]), None)
                if existing is None:
                    w.annotations.append(annotation)
                else:
                    annotation["id"] = w.annotations[existing]["id"]
                    w.annotations[existing] = annotation
                w._refresh_issue_annotations(selected_id=annotation["id"])
                w._save_now()
                if w.save_dirty:
                    raise RuntimeError("Review save failed; recovery retained. This region is not complete.")
                stored = w.store.decisions()[row["run"]]["annotations"]
                if not any(a.get("evidence_audit") == annotation["evidence_audit"] for a in stored):
                    raise RuntimeError("Saved annotation is missing its audit provenance")
            self.saved.setText(f"Saved locally: {verdict}. Original evidence unchanged.")
            self.refresh()
        except Exception as exc:
            self.saved.setText(f"Save failed: {exc}")
            QMessageBox.critical(self, "Audit decision not completed", str(exc))

    def next_region(self):
        if self.rows:
            self.table.selectRow((self.table.currentRow()+1) % len(self.rows))


def attach_evidence_audit(window, path, api):
    return EvidenceAuditDock(window, path, api)
