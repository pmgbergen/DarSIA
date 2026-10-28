"""Folder participation on this computer, using the existing job workers."""

from pathlib import Path

from PySide6.QtCore import QSettings, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .workflow_section import WorkflowSection


def _label(text):
    widget = QLabel(text)
    widget.setWordWrap(True)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    return widget


class SharedJobsControls(QWidget):
    role_changed = Signal(str)
    show_status_requested = Signal()
    folder_saved = Signal(str)

    def __init__(self, owner, *, service_factory=None, folder_factory=None, tuner_factory=None):
        super().__init__(owner)
        self.owner = owner
        self._service_factory = service_factory
        self._folder_factory = folder_factory
        self._tuner_factory = tuner_factory
        self._service = self._tuner = None
        self._thread = self._worker = None
        self._leaving = False
        self._closing = False
        self._reprepare_feedback = None
        self._series_default = None
        self._series_root = None
        self._settings = QSettings("DarSIA", "SharedJobs")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(_label(
            "Choose the same shared folder on each computer. Start a job here to be its "
            "master, or join as a worker to help with published jobs."))
        self.folder = QLineEdit(str(self._settings.value("folder", "")))
        self.folder.setPlaceholderText("Shared job folder")
        self.browse = QPushButton("Choose shared folder…")
        self.browse.clicked.connect(self._browse)
        layout.addWidget(self.folder)
        layout.addWidget(self.browse)
        self.save_folder_button = QPushButton("Save shared folder for this series")
        self.save_folder_button.clicked.connect(self.save_folder)
        layout.addWidget(self.save_folder_button)
        layout.addWidget(_label("Setup and Jobs share this saved folder. Each computer can map its jobs root separately in Machine stores."))
        self.execution = QWidget(self)
        execution_layout = QVBoxLayout(self.execution)
        execution_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.execution)
        self.automatic = QCheckBox("Automatic worker capacity")
        self.automatic.setChecked(True)
        self.automatic.setToolTip(
            "Measure completed work and adapt CPU/GPU capacity between batches. "
            "Running calculations finish before their capacity changes.")
        execution_layout.addWidget(self.automatic)
        self.capacity_note = _label("CPU and supported GPUs are used automatically. Counts adapt between completed batches.")
        execution_layout.addWidget(self.capacity_note)
        manual = QWidget()
        manual_layout = QVBoxLayout(manual)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        self.cpu, self.gpu = QSpinBox(), QSpinBox()
        capacity_row = QHBoxLayout()
        for widget, prefix, value in ((self.cpu, "CPU workers: ", 1), (self.gpu, "GPU workers (CUDA): ", 0)):
            widget.setRange(0, 256)
            widget.setPrefix(prefix)
            widget.setValue(value)
            capacity_row.addWidget(widget)
        manual_layout.addLayout(capacity_row)
        manual_layout.addWidget(_label("0 disables that backend. Set both to 0 for a master that only coordinates jobs."))
        self.manual_section = WorkflowSection(None, "Manual worker capacity", manual, expanded=False)
        execution_layout.addWidget(self.manual_section)
        self.manual_section.hide()
        self.automatic.toggled.connect(self._automatic_changed)
        self.publish = QPushButton("Start selected job in shared folder")
        self.join = QPushButton("Join folder as worker")
        self.leave = QPushButton("Leave shared folder")
        self.publish.clicked.connect(self.start_selected_job)
        self.join.clicked.connect(self.join_folder)
        self.leave.clicked.connect(self.leave_folder)
        participation_row = QHBoxLayout()
        for widget in (self.publish, self.join, self.leave):
            participation_row.addWidget(widget)
        execution_layout.addLayout(participation_row)
        self.setup_version_note = _label("")
        self.reprepare_setup = QPushButton("Update job")
        self.reprepare_setup.setToolTip("Use the installed application version and saved settings. Keep completed work and old metadata in history. Press Start to continue.")
        self.reprepare_setup.clicked.connect(self._reprepare_selected_setup)
        execution_layout.addWidget(self.setup_version_note)
        execution_layout.addWidget(self.reprepare_setup)
        self.setup_version_note.hide()
        self.reprepare_setup.hide()
        self.recover_admission = QPushButton('Recover old admission lock…')
        self.recover_admission.clicked.connect(self._show_admission_recovery)
        execution_layout.addWidget(self.recover_admission)
        self.admission_recovery = QWidget()
        recovery_layout = QVBoxLayout(self.admission_recovery)
        self.admission_confirmation = QCheckBox(
            'All other DarSIA/Python sessions on this machine are closed')
        self.admission_confirmation.toggled.connect(self.sync)
        self.admission_recover_now = QPushButton('Verify and recover admission')
        self.admission_recover_now.clicked.connect(self._recover_admission)
        recovery_layout.addWidget(self.admission_confirmation)
        recovery_layout.addWidget(self.admission_recover_now)
        execution_layout.addWidget(self.admission_recovery)
        self.admission_recovery.hide()
        management = getattr(owner, "job_management", None)
        if management is not None:
            management.reprepare_message.connect(self._reprepare_message)
            management.reprepare_state_changed.connect(self.sync)
        self.status = _label("Not connected. No shared-folder worker is started.")
        execution_layout.addWidget(self.status)
        self.show_status = QPushButton("Connected workers and progress")
        self.show_status.clicked.connect(self.show_status_requested)
        participation_row.addWidget(self.show_status)
        self.results = QWidget()
        result_layout = QVBoxLayout(self.results)
        result_layout.addWidget(_label("Connected workers"))
        self.workers = QTableWidget(0, 6)
        self.workers.setHorizontalHeaderLabels(["Machine", "Role", "CPU / GPU capacity", "Connection", "Last report", "Progress"])
        result_layout.addWidget(self.workers)
        result_layout.addWidget(_label("Jobs on this computer"))
        self.jobs = QTableWidget(0, 5)
        self.jobs.setHorizontalHeaderLabels(["Series", "Job", "State", "Completed frames", "Workers / details"])
        result_layout.addWidget(self.jobs)
        self.details = _label("")
        result_layout.addWidget(self.details)
        for table in (self.workers, self.jobs):
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            table.horizontalHeader().setStretchLastSection(True)
        self.timer = QTimer(self)
        self.timer.setInterval(5000)
        self.timer.timeout.connect(self._tick)
        self.folder.textChanged.connect(self.sync)
        self.sync()

    def _automatic_changed(self, automatic):
        self.manual_section.setVisible(not automatic)
        self.manual_section.set_expanded(not automatic)
        self.capacity_note.setText(
            "CPU and supported GPUs are used automatically. Counts adapt between completed batches."
            if automatic else "Use the selected CPU and GPU worker counts together when the job supports them.")

    def set_series(self, root, jobs_folder=None):
        """Offer the selected series' jobs folder without joining or starting it."""
        if self._service is not None or self._thread is not None:
            return
        changed_series = self._series_root != (Path(root) if root else None)
        self._series_root = Path(root) if root else None
        if changed_series or not self.folder.text().strip() or self.folder.text() == self._series_default:
            from .series_paths import series_folder_path
            self._series_default = str(jobs_folder or series_folder_path(root, "jobs")) if root else ""
            self.folder.setText(self._series_default)
        self.sync()

    def save_folder(self):
        """Save the shared series choice through the normal manifest writer."""
        if self._series_root is None or self._service is not None or self._thread is not None:
            return
        from .series_locations import save_series_jobs_folder
        root, folder = self._series_root, self.folder.text().strip()
        self.status.setText("Saving shared Jobs folder…")
        def done(value):
            self._series_default = str(value["jobs_folder"])
            self._settings.setValue("folder", self._series_default)
            self.status.setText("Shared folder saved for Setup and Jobs. No files moved; no worker started.")
            self.folder_saved.emit(str(root))
        self._run(lambda: save_series_jobs_folder(root, folder), done)

    def _browse(self):
        chosen = QFileDialog.getExistingDirectory(self, "Choose shared job folder", self.folder.text())
        if chosen:
            self.folder.setText(chosen)

    def sync(self, *_):
        busy, active = self._thread is not None, self._service is not None
        worker_only = active and self._service.role == "worker"
        source = self._setup_source_status()
        management = getattr(self.owner, "job_management", None)
        preparing = bool(management is not None and management._repreparing)
        for widget in (self.folder, self.browse, self.automatic, self.manual_section):
            widget.setEnabled(not active and not busy)
        self.save_folder_button.setEnabled(self._series_root is not None and bool(self.folder.text().strip()) and not busy and not active)
        self.publish.setEnabled(bool(self.folder.text().strip()) and not busy and not worker_only
                                and not preparing and not source.get("blocked", source["mismatch"])
                                and bool((self.owner.panel.current or {}).get("job_facts")))
        self.join.setEnabled(bool(self.folder.text().strip()) and not active and not busy)
        self.leave.setEnabled(active and not busy)
        self.recover_admission.setEnabled(not busy)
        self.admission_recover_now.setEnabled(not busy and bool(self.folder.text().strip())
                                             and self.admission_confirmation.isChecked())
        self.publish.setVisible(not worker_only and bool((self.owner.panel.current or {}).get("job_facts")))
        self.join.setVisible(not active)
        self.leave.setVisible(active)
        feedback = self._reprepare_feedback
        if feedback and feedback[0] != (self.owner.panel.current or {}).get("root"):
            self._reprepare_feedback = feedback = None
        message = "\n".join(filter(None, (source["message"], feedback[1] if feedback else "")))
        self.setup_version_note.setText(message)
        self.setup_version_note.setVisible(bool(message) and not worker_only)
        self.reprepare_setup.setVisible(source["can_reprepare"] and not worker_only)
        self.reprepare_setup.setEnabled(not busy and not preparing and bool(self.owner.series_root))

    def _setup_source_status(self):
        management = getattr(self.owner, "job_management", None)
        return (management.setup_source_status() if management is not None else
                dict(mismatch=False, can_reprepare=False, message="", commit=""))

    def _reprepare_selected_setup(self):
        if self._thread is not None or (self._service is not None and self._service.role == "worker"):
            return
        self.owner.job_management._reprepare_setup()

    def _reprepare_message(self, message):
        management = self.owner.job_management
        if management._reprepare_root in {None, (self.owner.panel.current or {}).get("root")}:
            self.status.setText(message)
            self._reprepare_feedback = ((self.owner.panel.current or {}).get("root"), message)
            self.sync()

    def _block_old_setup_start(self):
        source = self._setup_source_status()
        if not source.get("blocked", source["mismatch"]):
            return False
        self.status.setText(source["message"])
        self.sync()
        return True

    def _show_admission_recovery(self):
        self.admission_recovery.show()
        self.status.setText('Recovery checks for an active OS lock and other local Python/DarSIA sessions. '
                            'It preserves jobs and does not stop any process.')

    def _recover_admission(self):
        if self._thread is not None:
            return
        confirmed = self.admission_confirmation.isChecked()
        machine = self.owner.panel.local_machine
        folder = self._service.folder if self._service is not None else self.folder.text().strip()
        def recover():
            from .shared_admission_lock import recover_legacy
            from .shared_job_folder import SharedJobFolder
            target = SharedJobFolder(folder) if isinstance(folder, (str, Path)) else folder
            return recover_legacy(target, machine, confirmed=confirmed)
        def done(value):
            self.status.setText('Admission recovered. Click Start to continue the selected job.'
                                if value['recovered'] else value['reason'])
            self.admission_recovery.hide()
        self._run(recover, done)

    def _factories(self):
        from .shared_job_folder import SharedJobFolder
        from .shared_job_service import SharedJobService
        from .shared_worker_tuning import AutomaticWorkerCapacity
        return (self._folder_factory or SharedJobFolder,
                self._service_factory or SharedJobService,
                self._tuner_factory or AutomaticWorkerCapacity)

    def _connect(self, role, job_root=None, *, displayed_control=None):
        if self._thread is not None:
            return
        if role == "master" and job_root is not None and self._block_old_setup_start():
            return
        if self._service is not None and self._service.role != role:
            self.status.setText("Leave the shared folder before changing this computer's role.")
            return
        text = self.folder.text().strip()
        if not text:
            self.status.setText("Choose a shared job folder first.")
            return
        if role == "worker" and not self.automatic.isChecked() and not (self.cpu.value() or self.gpu.value()):
            self.status.setText("Enable automatic capacity or at least one worker to join.")
            return
        machine = self.owner.panel.local_machine
        stores = self.owner.stores.path
        automatic = self.automatic.isChecked()
        counts = self.cpu.value(), self.gpu.value()
        current_service, current_tuner = self._service, self._tuner
        current = dict(self.owner.panel.current or {})
        source_commit = self._setup_source_status()["commit"]
        self._settings.setValue("folder", text)
        self.status.setText("Connecting to shared job folder…")

        def connect():
            Folder, Service, Tuner = self._factories()
            local = Path(text)
            if self._series_root is not None:
                from .series_locations import machine_jobs_folder
                local = machine_jobs_folder(self._series_root, text, stores)
            folder = Folder(str(local))
            located_job = job_root
            if job_root is not None and local != Path(text):
                try:
                    located_job = str(local / Path(job_root).relative_to(Path(text)))
                except ValueError:
                    pass
            if job_root is not None:
                if source_commit and (current.get("job_facts") or {}).get("pipeline") == "series-setup":
                    from .pipeline.job_queue import read_json
                    actual = read_json(Path(located_job) / "job.json")
                    if actual.get("commit") != source_commit:
                        raise ValueError("This Setup job uses another code version. Refresh, reprepare it with current code, then click Start.")
            tuner = (current_tuner or Tuner(machine=machine)) if automatic else None
            cpu, gpu = tuner.counts() if tuner is not None else counts
            if job_root is not None:
                folder.publish(located_job, master_machine=machine, cpu_workers=cpu, cuda_workers=gpu)
                from .job_control import read_control
                state = read_control(located_job)
                if state["action"] != "run":
                    if (displayed_control is not None
                            and state.get("command") != displayed_control.get("command")):
                        raise ValueError("Job control changed; refresh before starting this job")
                    from .job_lifecycle import recover_job
                    recover_job(located_job, machine=machine,
                        expected_previous=state.get("command"), stores_path=stores)
            service = current_service or Service(folder, machine, role=role, cpu_workers=cpu,
                cuda_workers=gpu, stores=stores, launcher=self.owner.panel.local_launcher)
            service.set_tuner(tuner)
            if current_service is not None:
                service.update_capacity(cpu, gpu)
            snapshot = service.tick()
            return dict(service=service, tuner=tuner, snapshot=snapshot)

        self._run(connect, self._connected)

    def start_selected_job(self):
        if self._service is not None and self._service.role == "worker":
            return
        current = self.owner.panel.current or {}
        if not current.get("job_facts"):
            self.status.setText("Select a prepared job first.")
            return
        if self._block_old_setup_start():
            return
        if current.get("archived") or current.get("complete") or current.get("errors"):
            self.status.setText("Choose an unfinished, unarchived job with readable metadata.")
            return
        self._reprepare_feedback = None
        self.sync()
        self._connect("master", current["root"],
            displayed_control=(current.get("control") or {}).get("state"))

    def join_folder(self):
        self._connect("worker")

    @Slot(object)
    def _connected(self, value):
        self._service, self._tuner = value["service"], value["tuner"]
        self.role_changed.emit(self._service.role)
        self._display(value["snapshot"])
        if not self._closing:
            self.timer.start()
        self.show_status_requested.emit()

    def _tick(self):
        if self._closing or self._service is None or self._thread is not None:
            return
        self._run(self._service.tick, self._display)

    def leave_folder(self):
        if self._thread is not None or self._service is None:
            return
        self.timer.stop()
        self._leaving = True
        self.status.setText("Leaving folder; running calculations are allowed to finish…")
        self._run(self._service.leave, self._left)

    @Slot(object)
    def _left(self, value):
        self._service = self._tuner = None
        self._leaving = False
        self.role_changed.emit("")
        self.status.setText("Left folder. This computer will take no new jobs; calculations already started may still be finishing.")

    def _run(self, call, ready):
        self._thread, self._worker = self.owner.panel._start_call(call, ready, self._closed, failed=self._failed)
        self.sync()

    @Slot(str)
    def _failed(self, error):
        self.status.setText("Shared folder: " + (error.strip().splitlines()[-1] if error.strip() else "Operation failed."))
        self.details.setText(error)
        if self._leaving:
            self._leaving = False
            if not self._closing:
                self.timer.start()

    @Slot()
    def _closed(self):
        if self._thread is not None:
            self._thread.wait()
        self._thread = self._worker = None
        self.sync()

    @staticmethod
    def _rows(table, rows):
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column, value in enumerate(row):
                table.setItem(row_index, column, QTableWidgetItem(str(value)))

    @Slot(object)
    def _display(self, snapshot):
        from .shared_job_folder import current_worker_reports
        rows = []
        current_reports = current_worker_reports(snapshot.get("workers", []))
        for worker in current_reports:
            progress = []
            for job in worker.get("jobs", []):
                if isinstance(job, dict):
                    p = job.get("progress") or {}
                    progress.append(f"{job.get('job_id', '')[:12]}: {p.get('committed', 0)}/{p.get('total_frames', '?')}")
            rows.append([worker.get("machine", ""), worker.get("role", ""),
                f"{worker.get('cpu_workers', 0)} / {worker.get('cuda_workers', 0)}",
                "Offline" if worker.get("state") == "offline" else
                "Stale — connection unknown" if worker.get("stale") else worker.get("state", "online"),
                worker.get("reported_utc", worker.get("updated_utc", "")), "; ".join(progress)])
        self._rows(self.workers, rows)
        for index, worker in enumerate(current_reports):
            count = worker.get("previous_session_count", 0)
            if count:
                self.workers.item(index, 3).setToolTip(
                    f"Latest connection report for this machine. {count} earlier GUI session report(s) are retained. "
                    "Job progress is reported separately; connection status does not establish worker liveness.")
        rows = []
        for job in snapshot.get("jobs", []):
            p = job.get("progress") or {}
            backends = "; ".join(f"{name}: {info.get('workers', 0)} — {info.get('message') or info.get('state', '')}"
                for name, info in (job.get("backends") or {}).items())
            rows.append([job.get("series_id") or "—", Path(job.get("job_root", job.get("job_id", ""))).name, job.get("state", ""),
                f"{p.get('committed', 0)} / {p.get('total_frames', '?')}", backends])
        self._rows(self.jobs, rows)
        for index, job in enumerate(snapshot.get("jobs", [])):
            capabilities = job.get("capabilities") or {}
            hints = [f"{stage['name']}: {', '.join(stage.get('supported_backends', []))}. {stage.get('reason', '')}"
                     for stage in capabilities.get("stages", [])]
            if hints:
                self.jobs.item(index, 4).setToolTip("\n".join(hints) +
                    "\nWorkers take complete runs; internal steps follow that run's implementation.")
        errors = snapshot.get("errors", [])
        self.details.setText("\n".join(str(error) for error in errors))
        role = "Master" if snapshot.get("role") == "master" else "Worker"
        self.status.setText(f"{role} connected — reporting every 5 seconds." +
            (" Some jobs require attention; see details." if errors else ""))
        if snapshot.get("tuning_note"):
            self.capacity_note.setText(snapshot["tuning_note"])

    def can_close(self):
        return self._thread is None

    def prepare_close(self):
        # A tick can admit a worker or write presence: let that call finish,
        # but never start the next tick while the window is closing.
        self._closing = True
        self.timer.stop()

    def cancel_close(self):
        self._closing = False
        if self._service is not None and self._thread is None:
            self.timer.start()
