"""UI-only resource facts and explicit worker choice for the existing Jobs page.

Snapshots and historical receipts are supplied by the owner. Opening this panel
performs no discovery, hardware probe, file write, worker launch or SSH call.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re
import traceback

from PySide6.QtCore import (
    QObject,
    QSignalBlocker,
    QStandardPaths,
    QThread,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from .device_choice import DEFAULT_SHAPE, receipt_rows
from .measurement_receipts import MeasurementRegistry

GIB = 1024**3
UNKNOWN = "Not recorded"
GPU_SPEED_NOTE = (
    "GPU speed: the mass kernel timed on this machine, median seconds per frame "
    "including upload and download, beside the CPU path of the same stage. A job "
    "with several GPUs uses the fastest one that can run it; an explicit device "
    "always wins, and the backend is never switched automatically. The timings "
    "are kept until a device or driver changes."
)


def _number(value, *, positive=False):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and value >= (0 if not positive else 1e-99)
    )


def _bytes(value):
    return f"{value / GIB:.2f} GiB" if type(value) is int and value >= 0 else UNKNOWN


def _seconds(value, frames):
    return (
        f"{value:.2f} s total / {value / frames:.2f} s per frame"
        if _number(value) and type(frames) is int and frames > 0
        else UNKNOWN
    )


def _bound_source(value):
    return (
        isinstance(value, dict)
        and isinstance(value.get("path"), str)
        and re.fullmatch(r"[a-f0-9]{64}", value.get("sha256", "")) is not None
        and type(value.get("size_bytes")) is int
        and value["size_bytes"] >= 0
    )


def resource_advice(machine, job, backend, threads, *, policy_budget=False,
                    reserve_bytes=40 * GIB, measured_memory=None, run_limit=None):
    """Pure display arithmetic; precision, CPU RAM and device VRAM stay separate.

    ``measured_memory`` ({"bytes", "basis"}) is a memory measurement made
    elsewhere, such as a published dataset's reference machine; it replaces
    this machine's matching samples. ``reserve_bytes`` replaces the shared
    machine's host-RAM reserve, and ``run_limit`` caps the count at the runs in
    scope. The defaults are the Jobs panel's rules.
    """
    if type(threads) is not int or threads < 1 or backend not in {"cpu", "cuda"}:
        raise ValueError(
            "Explicit CPU/CUDA backend and positive worker thread count required"
        )
    unknown = lambda reason: {
        "workers": None,
        "reason": reason,
        "memory_basis": "unknown",
    }
    precision = job.get("precision")
    if precision not in {"float32", "float64"}:
        return unknown("Choose a job with an explicit precision first.")
    inventory = machine.get("inventory") or {}
    cpu = inventory.get("cpu") or {}
    cores = cpu.get("physical_cores") or cpu.get("logical_cores")
    if type(cores) is not int or cores < 1:
        return unknown("Core count is not recorded for this machine.")
    available = (inventory.get("memory") or {}).get("available_bytes")
    if type(available) is not int or available < 0:
        return unknown(
            "Available host RAM is not recorded; total RAM is not a free-memory measurement."
        )
    history = machine.get("history") or []
    matching = [
        s
        for s in history
        if isinstance(s, dict)
        and s.get("machine") == machine.get("machine")
        and s.get("backend") == backend
        and s.get("precision") == precision
        and type(s.get("threads")) is int
        and s["threads"] == threads
        and s.get("pipeline") == job.get("pipeline")
        and job.get("pipeline")
        and s.get("git_commit") == job.get("commit")
        and job.get("commit")
        and _bound_source(s.get("source"))
    ]
    rss = [
        s["peak_worker_rss_bytes"]
        for s in matching
        if type(s.get("peak_worker_rss_bytes")) is int
        and s["peak_worker_rss_bytes"] > 0
    ]
    if policy_budget:
        memory, basis = (
            20 * GIB,
            "Policy budget: 20 GiB/worker; this is not measured RSS.",
        )
    elif measured_memory is not None:
        memory, basis = measured_memory["bytes"], measured_memory["basis"]
        if type(memory) is not int or memory < 1:
            return unknown("The supplied memory measurement is not a positive byte count.")
    elif rss:
        memory, basis = (
            max(rss),
            "Largest recorded peak worker RSS in matching saved samples.",
        )
    else:
        return unknown(
            "No peak worker RSS measurement matches this machine, backend, precision, threads, pipeline and commit. Historical timings do not measure memory. Choose an explicit policy budget or override the worker count."
        )
    core_limit = cores // threads
    ram_limit = max(0, available - reserve_bytes) // memory
    count = min(core_limit, ram_limit)
    reason = f"min({cores} cores / {threads} threads = {core_limit}, available RAM minus {reserve_bytes / GIB:g} GiB reserve / worker memory = {ram_limit}). {basis}"
    if backend == "cuda":
        device_id = str(machine.get("device_id", ""))
        devices = [
            d
            for d in inventory.get("accelerators") or []
            if d.get("family") == "cuda" and str(d.get("device_id")) == device_id
        ]
        device = devices[0] if len(devices) == 1 else {}
        free_vram = device.get("available_bytes")
        vram = [
            s["peak_device_memory_bytes"]
            for s in matching
            if str(s.get("device_id", "")) == device_id
            and type(s.get("peak_device_memory_bytes")) is int
            and s["peak_device_memory_bytes"] > 0
        ]
        if type(free_vram) is not int or free_vram < 0 or not vram:
            return unknown(
                reason
                + " CUDA also needs separately recorded available device VRAM and a matching peak device-memory measurement; host RAM and the 20 GiB host budget are not VRAM."
            )
        vram_limit = free_vram // max(vram)
        count = min(count, vram_limit)
        reason += f" Separate measured device-memory limit = {vram_limit}."
    if run_limit is not None and run_limit < count:
        count = run_limit
        reason += f" At most {run_limit} run(s) are in scope, and a run never spans workers."
    return {
        "workers": count,
        "reason": reason,
        "memory_basis": basis,
        "core_limit": core_limit,
        "ram_limit": ram_limit,
    }


class _InventoryWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, provider):
        super().__init__()
        self.provider = provider

    @Slot()
    def run(self):
        try:
            self.finished.emit(self.provider())
        except Exception:
            self.failed.emit(traceback.format_exc())


class ResourceAdvisorPanel(QWidget):
    """Jobs child panel. All worker values are choices, never execution actions.

    ``machines_provider()`` returns explicit saved snapshots:
    ``[{machine, inventory, history, backend, threads, device_id?}]``.
    ``set_context(job, series_label=...)`` takes a read-only copy of job facts.
    Owner supplies ``precision``, ``commit`` and actual ``pipeline`` for measured
    compatibility. Missing facts remain unknown. ``workers_selected`` emits a
    dict only after the user clicks Use worker count. The owner may persist that
    UI choice; this component never writes job/scientific files.

    The GPU speed table shows this machine's stored GPU benchmark receipt. Only
    Measure GPU speed runs ``device_benchmark(precision, shape_rc)``, which
    times the devices in child processes and stores the receipt.
    """

    workers_selected = Signal(object)
    hardware_inspected = Signal(object)
    hardware_inspection_failed = Signal(str)
    measurement_imported = Signal(object)
    measurement_import_failed = Signal(str)
    gpu_speed_measured = Signal(object)

    def __init__(
        self,
        parent=None,
        *,
        machines_provider=None,
        inventory_provider=None,
        measurement_registry=None,
        device_receipt_path=None,
        device_benchmark=None,
    ):
        super().__init__(parent)
        self.machines_provider, self.inventory_provider = (
            machines_provider,
            inventory_provider,
        )
        self.device_receipt_path = device_receipt_path
        self.device_benchmark = device_benchmark
        self.job, self.machines, self.series_label = {}, [], ""
        self._probe_thread = self._probe_worker = None
        self._history_thread = self._history_worker = None
        self._gpu_thread = self._gpu_worker = None
        self.measurement_registry = measurement_registry or MeasurementRegistry(
            Path(
                QStandardPaths.writableLocation(
                    QStandardPaths.StandardLocation.AppLocalDataLocation
                )
            )
            / "job_measurements"
        )
        self._advice = {"workers": None}
        layout = QVBoxLayout(self)
        self.context_label = self._label("No job selected")
        layout.addWidget(self.context_label)
        self.details_tabs = QTabWidget()
        hardware, measurements, gpu = QWidget(), QWidget(), QWidget()
        hardware_layout = QVBoxLayout(hardware)
        measurements_layout = QVBoxLayout(measurements)
        gpu_layout = QVBoxLayout(gpu)
        self.details_tabs.addTab(hardware, "Hardware")
        self.details_tabs.addTab(measurements, "Measurements")
        self.details_tabs.addTab(gpu, "GPU speed")
        bar = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh saved machine facts")
        self.inspect_button = QPushButton("Inspect this machine…")
        self.import_measurement_button = QPushButton("Load measurement receipt…")
        self.refresh_button.clicked.connect(self.refresh_known_machines)
        self.inspect_button.clicked.connect(self.inspect_local_hardware)
        self.import_measurement_button.clicked.connect(self.choose_measurement_receipt)
        bar.addWidget(self.refresh_button)
        bar.addWidget(self.inspect_button)
        bar.addStretch()
        hardware_layout.addLayout(bar)
        self.hardware_table = self._table(
            [
                "Machine",
                "Cores",
                "Host RAM: available / total",
                "GPU",
                "Device VRAM total",
                "Recorded",
            ]
        )
        hardware_layout.addWidget(self.hardware_table, 1)
        grid = QGridLayout()
        self.machine_combo, self.backend_combo = QComboBox(), QComboBox()
        self.backend_combo.addItem("CPU", "cpu")
        self.backend_combo.addItem("CUDA", "cuda")
        self.threads_spin, self.workers_spin = QSpinBox(), QSpinBox()
        self.threads_spin.setRange(1, 4096)
        self.workers_spin.setRange(1, 256)
        for column, (text, widget) in enumerate(
            [
                ("Machine", self.machine_combo),
                ("Backend", self.backend_combo),
                ("Threads per worker", self.threads_spin),
                ("Worker count (editable)", self.workers_spin),
            ]
        ):
            grid.addWidget(self._label(text), 0, column)
            grid.addWidget(widget, 1, column)
        layout.addLayout(grid)
        self.policy_budget = QCheckBox(
            "Use policy budget for worker recommendation"
        )
        self.policy_budget.setToolTip(
            "20 GiB host RAM per worker; 40 GiB host RAM reserve. "
            "This policy budget is not measured RSS or device VRAM."
        )
        layout.addWidget(self.policy_budget)
        self.advice_label, self.reason_label = self._label(
            "Recommendation: unknown"
        ), self._label("")
        layout.addWidget(self.advice_label)
        layout.addWidget(self.reason_label)
        buttons = QHBoxLayout()
        self.suggested_button, self.use_button = QPushButton(
            "Copy recommendation"
        ), QPushButton("Use worker count")
        buttons.addWidget(self.suggested_button)
        buttons.addWidget(self.use_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.selection_label = self._label("")
        layout.addWidget(self.selection_label)
        measurements_layout.addWidget(self.import_measurement_button)
        measurements_layout.addWidget(
            self._label(
                "Historical measurements — their own precision, backend, threads and pipeline; not a current-job speed prediction"
            )
        )
        self.history_table = self._table(
            [
                "Machine",
                "Backend",
                "Precision",
                "Threads",
                "Frames",
                "Wall time",
                "Input + compute",
                "Publication + checkpoints",
                "Peak worker RSS",
                "Peak device memory",
                "Pipeline / commit",
                "Source",
            ]
        )
        self.history_table.horizontalHeader().setStretchLastSection(False)
        measurements_layout.addWidget(self.history_table, 1)
        gpu_layout.addWidget(self._label(GPU_SPEED_NOTE))
        gpu_bar = QHBoxLayout()
        self.measure_gpu_button = QPushButton("Measure GPU speed…")
        self.measure_gpu_button.clicked.connect(self.measure_gpu_speed)
        gpu_bar.addWidget(self.measure_gpu_button)
        gpu_bar.addStretch()
        gpu_layout.addLayout(gpu_bar)
        self.gpu_table = self._table(
            [
                "Device",
                "Driver",
                "Precision",
                "Frame (px)",
                "Seconds per frame",
                "Peak device memory",
                "Fastest of its backend",
                "Status",
                "Measured",
            ]
        )
        gpu_layout.addWidget(self.gpu_table, 1)
        self.gpu_label = self._label("")
        gpu_layout.addWidget(self.gpu_label)
        self.probe_label = self._label(
            "No hardware inspection requested. Remote facts are supplied by the registered-machine provider; this panel never starts SSH."
        )
        hardware_layout.addWidget(self.probe_label)
        self.measurement_label = self._label(
            "Load an original completed.json or trial summary explicitly. Only recorded metadata is read; missing memory and timing subdivisions stay unknown."
        )
        measurements_layout.addWidget(self.measurement_label)
        layout.addWidget(self.details_tabs, 1)
        self.machine_combo.currentIndexChanged.connect(self._machine_changed)
        self.backend_combo.currentIndexChanged.connect(self._refresh_advice)
        self.threads_spin.valueChanged.connect(self._refresh_advice)
        self.policy_budget.toggled.connect(self._refresh_advice)
        self.suggested_button.clicked.connect(self._copy_recommendation)
        self.use_button.clicked.connect(self._choose_workers)
        self._refresh_advice()
        self.refresh_device_measurements()

    @staticmethod
    def _label(text):
        label = QLabel(text)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        return label

    @staticmethod
    def _table(headers):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setWordWrap(False)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        # Keep one multiline source/SHA record readable on its page.
        table.setMinimumHeight(
            table.horizontalHeader().sizeHint().height()
            + 3 * table.fontMetrics().lineSpacing()
            + table.horizontalScrollBar().sizeHint().height()
            + 2 * table.frameWidth()
        )
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def set_context(self, job=None, *, series_label="", machines_provider=None):
        self.selection_label.clear()
        self.job = deepcopy(dict(job or {}))
        self.series_label = str(series_label)
        if machines_provider is not None:
            self.machines_provider = machines_provider
        self.context_label.setText(
            f"Series: {self.series_label or 'Not selected'} · Job: {self.job.get('name') or self.job.get('job_id') or 'Not selected'} · Precision: {self.job.get('precision') or 'Not recorded'}"
        )
        if (
            self.machines_provider is not None
            or self.measurement_registry.root.exists()
        ):
            self.refresh_known_machines()
        else:
            self._refresh_advice()

    def refresh_known_machines(self, *_):
        try:
            rows = (
                self.machines_provider()
                if self.machines_provider is not None
                else self.machines
            )
            self.set_machines(self.measurement_registry.merge_machines(rows))
            self.probe_label.setText(
                f"Loaded {len(self.machines)} saved machine snapshot(s). "
                "No hardware inspection or worker was started."
            )
        except Exception:
            # Do not leave another series/job's machine facts looking current.
            self.set_machines([])
            self.probe_label.setText(traceback.format_exc())
        self.refresh_device_measurements()

    def set_machines(self, machines):
        rows = deepcopy(list(machines or []))
        if any(
            not isinstance(row, dict)
            or not isinstance(row.get("machine"), str)
            or not row["machine"].strip()
            for row in rows
        ):
            raise ValueError(
                "Each registered-machine snapshot needs an explicit machine name"
            )
        if len({row["machine"] for row in rows}) != len(rows):
            raise ValueError("Duplicate machine snapshots")
        for row in rows:
            inventory = row.get("inventory") or {}
            history = row.get("history") or []
            if (
                not isinstance(inventory, dict)
                or not isinstance(history, list)
                or any(not isinstance(s, dict) for s in history)
            ):
                raise ValueError(
                    "Machine inventory must be an object and history a list of sample objects"
                )
            if any(
                not isinstance(inventory.get(key) or {}, dict)
                for key in ("cpu", "memory")
            ):
                raise ValueError("CPU and host RAM facts must be objects")
            devices = inventory.get("accelerators") or []
            if not isinstance(devices, list) or any(
                not isinstance(device, dict) for device in devices
            ):
                raise ValueError("Accelerator facts must be a list of device objects")
        selected = self.machine_combo.currentData()
        self.machines = rows
        with QSignalBlocker(self.machine_combo):
            self.machine_combo.clear()
            for row in rows:
                self.machine_combo.addItem(row["machine"], row["machine"])
            index = self.machine_combo.findData(selected)
            self.machine_combo.setCurrentIndex(
                index if index >= 0 else (0 if rows else -1)
            )
        self.hardware_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            inventory = row.get("inventory") or {}
            cpu = inventory.get("cpu") or {}
            memory = inventory.get("memory") or {}
            devices = inventory.get("accelerators") or []
            cores = f"{cpu.get('physical_cores') or '?'} physical / {cpu.get('logical_cores') or '?'} logical"
            values = [
                row["machine"],
                cores,
                f"{_bytes(memory.get('available_bytes'))} / {_bytes(memory.get('total_bytes'))}",
                "; ".join(
                    str(d.get("name") or d.get("family") or UNKNOWN) for d in devices
                )
                or UNKNOWN,
                "; ".join(
                    f"{d.get('name') or d.get('device_id') or '?'}: {_bytes(d.get('memory_bytes'))}"
                    for d in devices
                )
                or UNKNOWN,
                str(inventory.get("recorded_utc") or UNKNOWN),
            ]
            self._fill_row(self.hardware_table, index, values)
        self._machine_changed()

    @staticmethod
    def _fill_row(table, index, values):
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(index, column, item)
        table.resizeRowToContents(index)

    def _selected_machine(self):
        return next(
            (
                row
                for row in self.machines
                if row["machine"] == self.machine_combo.currentData()
            ),
            {},
        )

    def _machine_changed(self, *_):
        row = self._selected_machine()
        with QSignalBlocker(self.backend_combo), QSignalBlocker(self.threads_spin):
            self.backend_combo.setCurrentIndex(
                max(0, self.backend_combo.findData(row.get("backend", "cpu")))
            )
            value = row.get("threads")
            self.threads_spin.setValue(value if type(value) is int and value > 0 else 1)
        self._refresh_advice()

    def _refresh_advice(self, *_):
        row = self._selected_machine()
        backend = self.backend_combo.currentData()
        self._advice = resource_advice(
            row,
            self.job,
            backend,
            self.threads_spin.value(),
            policy_budget=self.policy_budget.isChecked() or self.job.get("execution_mode") == "current_build",
            run_limit=(len(self.job["expected_runs"]) if self.job.get("execution_mode") == "current_build" else None),
        )
        value = self._advice["workers"]
        self.advice_label.setText(
            f"Recommendation: {value} worker(s)"
            if value is not None
            else "Recommendation: unknown"
        )
        self.reason_label.setText(self._advice["reason"])
        self.suggested_button.setEnabled(type(value) is int and value > 0)
        self.use_button.setEnabled(
            bool(row) and self.job.get("precision") in {"float32", "float64"}
        )
        history = row.get("history") or []
        self.history_table.setRowCount(len(history))
        for index, sample in enumerate(history):
            frames = sample.get("frames")
            source = sample.get("source") or {}
            values = [
                sample.get("machine", UNKNOWN),
                sample.get("backend", UNKNOWN),
                sample.get("precision", UNKNOWN),
                (
                    sample.get("threads")
                    if type(sample.get("threads")) is int
                    else UNKNOWN
                ),
                frames if type(frames) is int else UNKNOWN,
                _seconds(sample.get("wall_seconds"), frames),
                _seconds(sample.get("compute_and_input_seconds"), frames),
                _seconds(sample.get("publication_and_checkpoint_seconds"), frames),
                _bytes(sample.get("peak_worker_rss_bytes")),
                _bytes(sample.get("peak_device_memory_bytes")),
                f"{sample.get('pipeline') or UNKNOWN} / {sample.get('git_commit') or UNKNOWN}",
                f"{source.get('path') or UNKNOWN}\nSHA256: {source.get('sha256') or UNKNOWN}",
            ]
            self._fill_row(self.history_table, index, values)

    def _copy_recommendation(self):
        value = self._advice.get("workers")
        if type(value) is int and value > 0:
            self.workers_spin.setValue(min(value, self.workers_spin.maximum()))
            self.selection_label.setText(
                f"Copied {self.workers_spin.value()} worker(s) into the editable count. "
                "Use worker count applies it to the job controls."
            )

    def _choose_workers(self):
        if not self.use_button.isEnabled():
            return
        count = self.workers_spin.value()
        self.selection_label.setText(
            f"Selected {count} worker(s). "
            "Jobs confirms whether its controls were updated. No worker was started."
        )
        self.workers_selected.emit(
            dict(
                machine=self.machine_combo.currentData(),
                backend=self.backend_combo.currentData(),
                precision=self.job["precision"],
                threads=self.threads_spin.value(),
                workers=count,
                job_id=self.job.get("job_id"),
                override=count != self._advice.get("workers"),
                recommendation=deepcopy(self._advice),
                starts_job=False,
            )
        )

    def inspect_local_hardware(self, *_):
        if self._probe_thread is not None:
            return
        provider = self.inventory_provider
        if provider is None:
            from darsia.fluidflower.execution import detect_hardware

            provider = detect_hardware
        self.probe_label.setText("Inspecting this machine; no worker is started.")
        self.inspect_button.setEnabled(False)
        from .ui_metadata_preview import MetadataPreview
        preview = MetadataPreview(self)
        def returned(value, error):
            try:
                if error:
                    self._hardware_failed(error)
                else:
                    self._hardware_ready(value)
            finally:
                self._probe_closed()
                preview.deleteLater()
        preview.completed.connect(returned)
        self._probe_thread = self._probe_worker = preview
        preview.start(lambda cancelled: provider())

    @Slot(object)
    def _hardware_ready(self, inventory):
        if not isinstance(inventory, dict):
            self._hardware_failed(
                "Hardware provider did not return an inventory object."
            )
            return
        self.probe_label.setText(
            f"Local inventory returned for {inventory.get('host') or UNKNOWN}. The owner may register these facts; remote snapshots are not overwritten."
        )
        self.hardware_inspected.emit(deepcopy(inventory))

    @Slot(str)
    def _hardware_failed(self, text):
        self.probe_label.setText(text)
        self.hardware_inspection_failed.emit(text)

    @Slot()
    def _probe_closed(self):
        self._probe_thread = self._probe_worker = None
        self.inspect_button.setEnabled(True)

    def choose_measurement_receipt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load original measured run receipt", "", "JSON (*.json)"
        )
        if path:
            self.import_measurement_receipt(path)

    def import_measurement_receipt(self, path):
        if self._history_thread is not None:
            return
        self.import_measurement_button.setEnabled(False)
        self.measurement_label.setText(
            "Reading the selected original metadata receipt; no job or hardware probe starts."
        )
        thread = QThread(self)
        worker = _InventoryWorker(
            lambda: self.measurement_registry.import_receipt(path)
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._measurement_ready)
        worker.failed.connect(self._measurement_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._measurement_closed)
        thread.finished.connect(thread.deleteLater)
        self._history_thread, self._history_worker = thread, worker
        thread.start()

    @Slot(object)
    def _measurement_ready(self, receipt):
        self.refresh_known_machines()
        sample = receipt["sample"]
        index = self.machine_combo.findData(sample["machine"])
        if index >= 0:
            self.machine_combo.setCurrentIndex(index)
        self.measurement_label.setText(
            f"Imported original {sample['machine']} / {sample['backend']} / {sample['precision']} / {sample['frames']} frames. Source: {sample['source']['path']} · SHA256: {sample['source']['sha256']}. Sources are reverified on refresh; this is historical timing, not a current-job prediction."
        )
        self.measurement_imported.emit(deepcopy(receipt))

    @Slot(str)
    def _measurement_failed(self, text):
        self.measurement_label.setText(text)
        self.measurement_import_failed.emit(text)

    @Slot()
    def _measurement_closed(self):
        if self._history_thread is not None:
            self._history_thread.wait()
        self._history_thread = self._history_worker = None
        self.import_measurement_button.setEnabled(True)

    def _gpu_receipt_path(self):
        """Return the receipt file shown here: the given one, else this machine's."""
        if self.device_receipt_path is not None:
            return Path(self.device_receipt_path)
        from .device_choice import default_receipt_path

        return default_receipt_path()

    def refresh_device_measurements(self, *_):
        """Show the stored GPU benchmark receipt; nothing is measured."""
        from .device_choice import load_receipt

        receipt = load_receipt(self._gpu_receipt_path())
        rows = receipt_rows(receipt)
        self.gpu_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            seconds = row["seconds_per_frame"]
            peak = _bytes(row["peak_device_bytes"])
            values = [
                row["device"],
                row["driver"] or UNKNOWN,
                row["precision"],
                row["frame"],
                f"{seconds:.4f}" if _number(seconds) else UNKNOWN,
                "Host memory" if row["family"] == "cpu" else peak,
                "Yes" if row["fastest"] else "",
                row["status"],
                row["recorded_utc"] or UNKNOWN,
            ]
            self._fill_row(self.gpu_table, index, values)
        count = len((receipt or {}).get("devices") or [])
        self.gpu_label.setText(
            "No GPU speed measurement is stored on this machine yet."
            if receipt is None
            else f"Stored receipt for {count} GPU device(s); it is measured again "
            "when the device list or a driver changes."
        )
        return receipt

    def measure_gpu_speed(self, *_):
        """Time every GPU and the CPU on the mass kernel now and store the receipt.

        Uses the job's precision and frame size when known. The devices are
        timed in short child processes; no job or worker is started.
        """
        if self._gpu_thread is not None:
            return
        precision = self.job.get("precision")
        precision = precision if precision in {"float32", "float64"} else "float32"
        shape = self.job.get("frame_shape_rc")
        if not (
            isinstance(shape, (list, tuple))
            and len(shape) == 2
            and all(type(value) is int and value > 0 for value in shape)
        ):
            shape = DEFAULT_SHAPE
        shape = (int(shape[0]), int(shape[1]))
        benchmark = self.device_benchmark
        if benchmark is None:
            from .device_choice import measure_and_store

            path = self._gpu_receipt_path()

            def benchmark(precision, shape):
                return measure_and_store(
                    precision=precision, shape=shape, receipt_path=path
                )

        self.gpu_label.setText(
            f"Measuring every GPU and the CPU at {precision}, "
            f"{shape[0]}x{shape[1]} px, each in a short child process; "
            "no job is started."
        )
        self.measure_gpu_button.setEnabled(False)
        thread = QThread(self)
        worker = _InventoryWorker(lambda: benchmark(precision, shape))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._gpu_ready)
        worker.failed.connect(self._gpu_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        # No deleteLater for the worker. Deleted in its own thread, it needs the
        # GIL while Qt holds a signal-slot lock (the Python QThread's
        # disconnectNotify), and a GUI thread that holds the GIL and waits for
        # that lock deadlocks. _gpu_closed drops its Python owner after wait(),
        # which frees it in the GUI thread.
        thread.finished.connect(self._gpu_closed)
        thread.finished.connect(thread.deleteLater)
        self._gpu_thread, self._gpu_worker = thread, worker
        thread.start()

    @Slot(object)
    def _gpu_ready(self, receipt):
        """Show the new receipt, or say that it could not be stored."""
        self.refresh_device_measurements()
        error = receipt.get("store_error") if isinstance(receipt, dict) else ""
        if error:
            self.gpu_label.setText(f"Measured, but the receipt was not stored: {error}")
        self.gpu_speed_measured.emit(deepcopy(receipt))

    @Slot(str)
    def _gpu_failed(self, text):
        """Show why the GPU speed measurement failed."""
        self.gpu_label.setText(text)

    @Slot()
    def _gpu_closed(self):
        """Release the finished measurement thread and allow a new measurement."""
        if self._gpu_thread is not None:
            self._gpu_thread.wait()
        self._gpu_thread = self._gpu_worker = None
        self.measure_gpu_button.setEnabled(True)

    def prepare_close(self):
        """Discard inventory reads, retaining receipt-import and benchmark writes."""
        if self._probe_thread is not None:
            self._probe_thread.cancel()
            self._probe_thread.deleteLater()
            self._probe_thread = self._probe_worker = None

    def can_close(self):
        return (
            self._probe_thread is None
            and self._history_thread is None
            and self._gpu_thread is None
        )

    def closeEvent(self, event):
        self.prepare_close()
        if not self.can_close():
            self.probe_label.setText(
                "Hardware inspection, measurement import or GPU speed measurement "
                "is still finishing; close after it returns."
            )
            event.ignore()
            return
        super().closeEvent(event)
