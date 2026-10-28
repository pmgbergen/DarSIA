"""Explicit native close choices for registered detached Jobs services."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Slot
from PySide6.QtWidgets import QDockWidget, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from .job_control import _bindings, publish_control, read_control
from .job_launch_state import read_launch_requests
from .jobs_metadata import JobSnapshotReader
from .pipeline.job_queue import descriptor, read_json


def _ordinary_shared_launch(request):
    launch = request.get("launcher_receipt") or {}
    return (request.get("kind") == "local" and launch.get("format") == "darsia-job-local-launch-v2"
            and launch.get("entrypoint") == "python -B -m darsia_service.prestart -- python -m darsia.fluidflower.worker"
            and launch.get("machine") == request.get("machine")
            and launch.get("backend") == request.get("backend") and launch.get("started") is True)


def _shared_service_retired(request, *, local_machine=None, stores_path=None):
    """Use the original claim verifier, including supervisor PID/birth absence.

    A terminal log only locates the service session. It cannot establish
    retirement, and local process inspection cannot establish remote absence.
    """
    from .pipeline.job_queue import name, retired_claim_evidence
    from .job_management import _retirement_binding
    if request.get("local_stop_confirmation") and local_machine:
        from .job_local_stop import verified_local_stop
        verified_local_stop(request["job_root"], machine=local_machine, stores_path=stores_path)
        return True
    if (not local_machine or name(local_machine) != name(request["machine"])
            or not request.get("terminal_report")):
        return False
    root = Path(request["job_root"])
    control = read_control(root)
    if control["action"] not in {"pause", "stop"}:
        return False
    service = name(request["machine"] + "-" + request["backend"])
    claims = sorted((root / "claimed" / service).glob("*.json"))
    if not claims:
        return False
    for claimed in claims:
        try:
            retired_claim_evidence(root, claimed,
                local_output=_retirement_binding(root, claimed, stores_path),
                expected_stop=control.get("command"), expected_action=control["action"])
        except (OSError, ValueError, KeyError, TypeError):
            return False
    return True


def _service_retired(request, *, reader=None, local_machine=None, stores_path=None):
    """Read original whole-family evidence through an explicit UI launch link."""
    if (request.get("preparation_failure")
            or request.get("state") == "Launcher did not create a process"):
        return True
    root = Path(request["job_root"])
    path = Path(request["launch_directory"]) / "service-request.json"
    if not path.is_file():
        if _ordinary_shared_launch(request):
            return _shared_service_retired(request, local_machine=local_machine, stores_path=stores_path)
        return False
    reader = reader if reader is not None else JobSnapshotReader()
    link = reader.read(path, descriptor(path))
    reference, _, _ = _bindings(root)
    if (link.get("format") != "darsia-job-prepared-service-v1"
            or link.get("job") != reference(root / "job.json")
            or link.get("request") != reference(Path(request["source"]["path"]))):
        raise ValueError("The service status link belongs to another launch")
    pin = link["service_request"]
    service = reader.read(Path(pin["path"]), pin)
    provider_pin = service["provider"]
    provider = reader.read(Path(provider_pin["path"]), provider_pin)
    if (service.get("format") != "darsia-setup-job-service-request-v1"
            or service.get("job") != descriptor(root / "job.json")
            or provider.get("format") != "darsia-setup-job-service-provider-v1"
            or provider.get("scope", {}).get("job") != service["job"]):
        raise ValueError("The original service is not bound to this job")
    from darsia_cache.first_qualification import _slot_read
    try:
        native, _ = _slot_read(provider, "native_completion")
    except FileNotFoundError:
        return False
    eligibility = native.get("eligibility") or {}
    if (native.get("request") != pin or native.get("provider") != provider_pin
            or native.get("scope") != provider["scope"]
            or native.get("stores_source") != service.get("stores_source")
            or native.get("complete") is not True
            or native.get("observer_native_handles_closed") is not True
            or eligibility.get("original_retirement_verifier_accepted") is not True
            or eligibility.get("retirement") != native.get("retirement")):
        return False
    retired_pin = native["retirement"]
    retired = reader.read(Path(retired_pin["path"]), retired_pin)
    receipt_pin = eligibility["receipt"]
    receipt = reader.read(Path(receipt_pin["path"]), receipt_pin)
    binding = receipt.get("binding") or {}
    manifest = binding.get("manifest") or {}
    if (receipt.get("format") != "darsia-cache-service-lifecycle-v1"
            or receipt.get("kind") != "eligibility"
            or receipt.get("actor") != eligibility.get("observer")
            or receipt.get("value") != {key: value for key, value in eligibility.items() if key != "receipt"}
            or binding.get("policy") != service.get("stores_source")
            or binding.get("profile") != provider_pin
            or manifest.get("native_launch_context") != retired.get("native_launch")
            or manifest.get("owner") != retired.get("primary")
            or retired.get("format") != "darsia-native-owned-worker-retirement-v1"
            or retired.get("whole_family_kernel_observation") is not True
            or retired.get("terminal", {}).get("accounting", {}).get("active_processes") != 0):
        return False
    return True


def _has_service_report(root):
    """A recorded launch needs a close choice; old claims alone do not."""
    for path in (Path(root) / "logs").glob("*.log"):
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 128 * 1024))
            tail = stream.read(128 * 1024).decode("utf-8", errors="replace")
        for line in tail.splitlines():
            try:
                value = json.loads(line.partition(" ")[2])
            except ValueError:
                continue
            if isinstance(value, dict) and value.get("kind") in ("started", "stopped", "startup_failure"):
                return True
    return False


def review_jobs(registrations, *, retirement_reader=_service_retired):
    """Inspect only explicit UI associations and their original control metadata."""
    jobs, errors = [], []
    for registration in registrations:
        row = dict(registration)
        root = Path(row["root"])
        try:
            if descriptor(root / "job.json") != row["job"]:
                raise ValueError("Registered job metadata changed")
            requests = read_launch_requests(root)
            unresolved = [request for request in requests if not retirement_reader(request)]
            if not unresolved and (requests or not _has_service_report(root)):
                continue
            if not requests or any(not _ordinary_shared_launch(request)
                                   and not (Path(request["launch_directory"]) / "service-request.json").is_file()
                                   for request in unresolved):
                row["retirement_note"] = "This older launch has no service status link. Automatic retirement confirmation is unavailable; you can close without waiting."
            elif any(_ordinary_shared_launch(request) for request in unresolved):
                row["retirement_note"] = "Waiting for the shared worker's original retirement confirmation. You can close without waiting."
            reference, master_format, _ = _bindings(root)
            master = read_json(root / "ui" / "master.json")
            if (master.get("format") != master_format
                    or master.get("job") != reference(root / "job.json")):
                raise ValueError("The recorded control master belongs to another job")
            row.update(master=reference(root / "ui" / "master.json"),
                master_machine=master["machine"], control=read_control(root))
            jobs.append(row)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(str(row.get("name") or root.name) + ": " + str(error))
    return dict(jobs=jobs, errors=errors)


def stop_reviewed_jobs(review, *, machine):
    """Use the unchanged master Stop path, retaining its displayed previous command."""
    if review["errors"]:
        raise ValueError("Job status could not be read; choose Keep jobs running or Cancel")
    for row in review["jobs"]:
        root = Path(row["root"])
        reference, _, _ = _bindings(root)
        if (descriptor(root / "job.json") != row["job"]
                or reference(root / "ui" / "master.json") != row["master"]
                or read_control(root) != row["control"]):
            raise ValueError("Job control changed. Review the close choice again")
        if row["control"]["action"] != "stop" and row["master_machine"] != machine:
            raise ValueError("Request Stop on the recorded master: " + row["master_machine"])
    for row in review["jobs"]:
        if row["control"]["action"] != "stop":
            publish_control(row["root"], "stop", machine=machine,
                expected_previous=row["control"].get("command"))


class NativeJobsCloseController(QObject):
    """Keep close review and cooperative retirement waits inside the native window."""

    def __init__(self, window, jobs):
        super().__init__(window)
        self.window, self.jobs = window, jobs
        self.thread = self.worker = self.review = None
        self._approved = self._cancelled = self._stopping = False
        self._registrations = []
        self._stores_path = None
        self._metadata_reader = JobSnapshotReader()
        self._ready_callback = None
        self._reading = False
        self._central_review_state = None
        self.timer = QTimer(self)
        self.timer.setInterval(3000)
        self.timer.timeout.connect(self._check_retirement)
        self.close_timer = QTimer(self)
        self.close_timer.setInterval(250)
        self.close_timer.timeout.connect(self._finish_close)
        self.dock = QDockWidget("Close DarSIA", window)
        self.dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        body = QWidget(self.dock)
        layout = QVBoxLayout(body)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.stop_button = QPushButton("Stop jobs and close")
        self.keep_button = QPushButton("Keep jobs running and close")
        self.cancel_button = QPushButton("Return to DarSIA")
        for button in (self.stop_button, self.keep_button, self.cancel_button):
            button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            buttons.addWidget(button)
        self.cancel_button.setDefault(True)
        self.stop_button.setAutoDefault(False)
        self.keep_button.setAutoDefault(False)
        # Keep every choice beside the explanation, even on a very wide screen.
        # Stretching all three across the window can put the two available
        # choices far outside the part of the dock containing its status text.
        buttons.addStretch()
        layout.addLayout(buttons)
        self.stop_button.clicked.connect(self.stop_and_close)
        self.keep_button.clicked.connect(self.keep_and_close)
        self.cancel_button.clicked.connect(self.cancel)
        self.dock.setWidget(body)
        window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock)
        self.dock.hide()

    def _show_review(self):
        """Let the close dock fit without recreating or clearing the active page."""
        if self._central_review_state is None:
            central = self.window.centralWidget()
            self._central_review_state = (central, central.isHidden()) if central is not None else (None, True)
            if central is not None:
                central.hide()
        self.dock.show()
        self.dock.raise_()
        if self.window.layout() is not None:
            self.window.layout().activate()
        maximum = self.window.maximumSize()
        if self.window.width() > maximum.width() or self.window.height() > maximum.height():
            self.window.resize(min(self.window.width(), maximum.width()),
                               min(self.window.height(), maximum.height()))
        self.cancel_button.setFocus(Qt.FocusReason.OtherFocusReason)

    def _restore_page(self):
        state, self._central_review_state = self._central_review_state, None
        self.dock.hide()
        if state is not None:
            central, was_hidden = state
            if central is not None and self.window.centralWidget() is central:
                central.setVisible(not was_hidden)

    def allow_close(self):
        if self._approved and self._reading:
            self._cancel_read()
        if self.thread is not None:
            self._show_review()
            return False
        if self._approved:
            self.timer.stop()
            return True
        if self.dock.isVisible():
            return False
        self._cancelled = self._stopping = False
        self.keep_button.setText("Keep jobs running and close")
        try:
            self._registrations = self.jobs.registry.known_jobs()
            provider = getattr(self.jobs.panel, "local_stores_path_provider", None)
            self._stores_path = provider() if provider is not None else None
        except Exception as error:
            self._show_review()
            self.status.setText("Job status unavailable: " + str(error))
            self.stop_button.setEnabled(False)
            self.keep_button.setText("Close without waiting")
            return False
        if not self._registrations:
            return True
        self._show_review()
        self.status.setText("Checking jobs registered with DarSIA…")
        self._run(self._review_jobs, self._review_ready)
        return False

    def _review_jobs(self):
        return review_jobs(self._registrations, retirement_reader=lambda request:
            _service_retired(request, reader=self._metadata_reader, local_machine=self.jobs.panel.local_machine,
                             stores_path=self._stores_path))

    def _run(self, call, ready, *, mutation=False):
        if self.thread is not None:
            return
        self.stop_button.setEnabled(False)
        self._ready_callback = ready
        self._reading = not mutation
        start = self.jobs.panel._start_call if mutation else self.jobs.panel._start_metadata_call
        self.thread, self.worker = start(call, self._ready, self._closed, failed=self._failed)

    @Slot(object)
    def _ready(self, value):
        if not self._cancelled and not self._approved:
            self._ready_callback(value)

    @Slot(str)
    def _failed(self, text):
        if not self._cancelled:
            message = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), text)
            self.status.setText(message[:400])
            self.keep_button.setText("Close without waiting")
            self.timer.stop()

    @Slot()
    def _closed(self):
        if self.thread is not None and not self._reading:
            self.thread.wait()
        self.thread = self.worker = None
        self._reading = False
        if self._approved:
            QTimer.singleShot(0, self._finish_close)

    def _review_ready(self, review):
        self.review = review
        if not review["jobs"] and not review["errors"]:
            self._approved = True
            return
        names = ", ".join(str(row.get("name") or Path(row["root"]).name) for row in review["jobs"])
        self.status.setText("Jobs registered with DarSIA: " + (names or "status unavailable")
            + ". Choose what happens when this window closes."
            + ("\n" + "\n".join(dict.fromkeys(row["retirement_note"] for row in review["jobs"]
                if row.get("retirement_note"))) if any(row.get("retirement_note") for row in review["jobs"]) else "")
            + ("\n" + "\n".join(review["errors"]) if review["errors"] else ""))
        allowed = not review["errors"] and all(row["control"]["action"] == "stop"
            or row["master_machine"] == self.jobs.panel.local_machine for row in review["jobs"])
        self.stop_button.setEnabled(bool(review["jobs"]) and allowed)
        if review["errors"] or any(row.get("retirement_note") for row in review["jobs"]):
            self.keep_button.setText("Close without waiting")
        if not allowed and not review["errors"]:
            self.status.setText(self.status.text() + "\nStop must be requested on each job's recorded master.")

    @Slot()
    def stop_and_close(self):
        if self.thread is not None or self.review is None:
            return
        self._stopping = True
        review, machine = self.review, self.jobs.panel.local_machine
        self.status.setText("Requesting Stop for the listed jobs…")
        self.keep_button.setText("Close without waiting")
        self._run(lambda: stop_reviewed_jobs(review, machine=machine), self._stop_ready, mutation=True)

    def _stop_ready(self, _value):
        self.status.setText("Stop requested. Waiting for worker retirement confirmation…")
        self.keep_button.setText("Close without waiting")
        self.timer.start()

    @Slot()
    def _check_retirement(self):
        if self.thread is not None or not self._stopping:
            return
        self._run(self._review_jobs, self._retirement_ready)

    def _retirement_ready(self, review):
        if not review["jobs"] and not review["errors"]:
            self.timer.stop()
            self._approved = True
        else:
            self.status.setText("Stop requested. Waiting for worker retirement confirmation."
                + ("\n" + "\n".join(dict.fromkeys(row["retirement_note"] for row in review["jobs"]
                    if row.get("retirement_note"))) if any(row.get("retirement_note") for row in review["jobs"]) else "")
                + ("\n" + "\n".join(review["errors"]) if review["errors"] else ""))

    @Slot()
    def keep_and_close(self):
        self._approved = True
        self.timer.stop()
        self._cancel_read()
        self._finish_close()

    def _cancel_read(self):
        """Discard only passive status inspection; an actual Stop write must finish."""
        if self.thread is not None and self._reading:
            self.thread.cancel()
            self.thread.deleteLater()
            self.thread = self.worker = None
            self._reading = False

    def _finish_close(self):
        if not self._approved or self.thread is not None:
            return
        prepare = getattr(self.jobs, 'prepare_close', None)
        if callable(prepare):
            prepare()
        ready = getattr(self.jobs, 'can_close', None)
        if callable(ready) and not ready():
            self.status.setText('Finishing the current local file or launch operation before closing. No new status checks will start.')
            self.close_timer.start()
            return
        self.window.close()
        if self.window.isVisible():
            self.close_timer.start()
        else:
            self.close_timer.stop()

    @Slot()
    def cancel(self):
        self._cancelled = True
        self._approved = self._stopping = False
        self.timer.stop()
        self.close_timer.stop()
        self._cancel_read()
        resume = getattr(self.jobs, 'cancel_close', None)
        if callable(resume):
            resume()
        self._restore_page()

