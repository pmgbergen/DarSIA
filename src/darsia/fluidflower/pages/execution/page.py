"""Qt production preflight, execution, hardware, and worker page."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Mapping

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.device_choice import (
    backend_devices,
    requested_device_resources,
)
from darsia.fluidflower.execution import (
    BACKEND_CONTRACTS,
    AdapterRegistry,
    ExecutionQueue,
    WorkerProcessHandle,
    current_plan_reference,
    default_adapter_registry,
    default_execution_cache_root,
    detect_hardware,
    recommend_resources,
    spawn_worker_processes,
    task_for_current_plan,
)
from darsia.fluidflower.production_preflight import summarize_production_plan
from darsia.fluidflower.project import FluidFlowerProject, slugify
from darsia.fluidflower.ui.common import PathField, deepcopy_settings


class ExecutionHardwareWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    @Slot()
    def run(self) -> None:
        try:
            self.finished.emit(detect_hardware())
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ExecutionQueueWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        queue_root: Path,
        *,
        recover_timeout_seconds: float | None = None,
    ) -> None:
        super().__init__()
        self.queue_root = Path(queue_root)
        self.recover_timeout_seconds = recover_timeout_seconds

    @Slot()
    def run(self) -> None:
        try:
            queue = ExecutionQueue(self.queue_root)
            recovered = 0
            if self.recover_timeout_seconds is not None:
                recovered = queue.requeue_stale(self.recover_timeout_seconds)
            snapshot = queue.snapshot()
            snapshot["recovered_tasks"] = recovered
            self.finished.emit(snapshot)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ExecutionPage(QWidget):
    """Cross-cutting task queue, hardware recommendation, and worker monitor."""

    _MODULE_LABELS: ClassVar[dict[str, str]] = {
        "calibration": "Calibration",
        "image_correction": "Image correction",
        "mass_analysis": "Mass analysis",
        "wasserstein": "Wasserstein analysis",
        "post_analysis": "Post-analysis",
    }

    def __init__(self, project: FluidFlowerProject) -> None:
        super().__init__()
        self.project = project
        self.adapter_registry: AdapterRegistry = default_adapter_registry()
        self.inventory: dict[str, Any] = {}
        self.recommendations: dict[str, dict[str, Any]] = {}
        self.worker_processes: list[WorkerProcessHandle] = []
        self._hardware_thread: QThread | None = None
        self._hardware_worker: ExecutionHardwareWorker | None = None
        self._queue_thread: QThread | None = None
        self._queue_worker: ExecutionQueueWorker | None = None
        self._production_preflight_summary: dict[str, Any] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        title = QLabel("Production")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        description = QLabel(
            "Review the selected phase and calibration basis, then run an immutable "
            "analysis plan through the existing local or shared task queue. "
            "The queue records input fingerprints, worker identity, checkpoints, "
            "retries, and result publication."
        )
        description.setObjectName("secondaryText")
        description.setWordWrap(True)
        layout.addWidget(description)

        preflight_panel = QFrame()
        preflight_panel.setFrameShape(QFrame.Shape.StyledPanel)
        preflight_layout = QVBoxLayout(preflight_panel)
        preflight_header = QHBoxLayout()
        preflight_title = QLabel("Production preflight · read only")
        preflight_title.setObjectName("sectionTitle")
        preflight_header.addWidget(preflight_title)
        preflight_header.addStretch(1)
        self.production_refresh = QPushButton("Refresh selected plan")
        self.production_refresh.clicked.connect(self._refresh_preflight)
        preflight_header.addWidget(self.production_refresh)
        preflight_layout.addLayout(preflight_header)
        self.production_plan = QLabel()
        self.production_scope = QLabel()
        self.production_bindings = QLabel()
        self.production_photometry = QLabel()
        self.production_runtime = QLabel()
        self.production_limitations = QLabel()
        for label in (
            self.production_plan,
            self.production_scope,
            self.production_bindings,
            self.production_photometry,
            self.production_runtime,
            self.production_limitations,
        ):
            label.setObjectName("secondaryText")
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            preflight_layout.addWidget(label)
        layout.addWidget(preflight_panel)

        hardware_panel = QFrame()
        hardware_panel.setFrameShape(QFrame.Shape.StyledPanel)
        hardware_layout = QVBoxLayout(hardware_panel)
        hardware_header = QHBoxLayout()
        hardware_title = QLabel("Hardware and recommendations")
        hardware_title.setObjectName("sectionTitle")
        hardware_header.addWidget(hardware_title)
        hardware_header.addStretch(1)
        self.execution_detect_hardware = QPushButton("Detect hardware")
        self.execution_detect_hardware.clicked.connect(self._detect_hardware)
        hardware_header.addWidget(self.execution_detect_hardware)
        hardware_layout.addLayout(hardware_header)
        self.execution_hardware_summary = QLabel(
            "Hardware has not been measured in this session."
        )
        self.execution_hardware_summary.setObjectName("secondaryText")
        self.execution_hardware_summary.setWordWrap(True)
        hardware_layout.addWidget(self.execution_hardware_summary)
        self.execution_devices = QTableWidget(0, 5)
        self.execution_devices.setHorizontalHeaderLabels(
            ["Device", "Backend", "Precision", "Memory", "Runtime / driver"]
        )
        self.execution_devices.verticalHeader().setVisible(False)
        self.execution_devices.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        device_header = self.execution_devices.horizontalHeader()
        device_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 5):
            device_header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.execution_devices.setMaximumHeight(190)
        hardware_layout.addWidget(self.execution_devices)
        self.execution_recommendations = QTableWidget(0, 6)
        self.execution_recommendations.setHorizontalHeaderLabels(
            ["Module", "Backend", "Device", "GPU workers", "CPU workers", "Prepare"]
        )
        self.execution_recommendations.verticalHeader().setVisible(False)
        self.execution_recommendations.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        recommendation_header = self.execution_recommendations.horizontalHeader()
        recommendation_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 6):
            recommendation_header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.execution_recommendations.setMaximumHeight(205)
        hardware_layout.addWidget(self.execution_recommendations)
        layout.addWidget(hardware_panel)

        adapter_panel = QFrame()
        adapter_panel.setFrameShape(QFrame.Shape.StyledPanel)
        adapter_layout = QVBoxLayout(adapter_panel)
        adapter_title = QLabel("Installed execution adapters")
        adapter_title.setObjectName("sectionTitle")
        adapter_layout.addWidget(adapter_title)
        self.execution_adapters = QTableWidget(0, 4)
        self.execution_adapters.setHorizontalHeaderLabels(
            ["Adapter", "Version", "Modules", "Backends"]
        )
        self.execution_adapters.verticalHeader().setVisible(False)
        self.execution_adapters.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        adapter_header = self.execution_adapters.horizontalHeader()
        adapter_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        adapter_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        adapter_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        adapter_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.execution_adapters.setMaximumHeight(155)
        adapter_layout.addWidget(self.execution_adapters)
        adapter_note = QLabel(
            "Only installed Python adapters can execute tasks; queue documents "
            "cannot supply shell commands. Verification tests the queue and publish "
            "path without creating results. Post-analysis creates the selected "
            "figures, tables, movies, and provenance directly."
        )
        adapter_note.setObjectName("secondaryText")
        adapter_note.setWordWrap(True)
        adapter_layout.addWidget(adapter_note)
        layout.addWidget(adapter_panel)

        queue_panel = QFrame()
        queue_panel.setFrameShape(QFrame.Shape.StyledPanel)
        queue_layout = QGridLayout(queue_panel)
        execution_settings = dict(
            (project.manifest.get("tools") or {}).get("execution") or {}
        )
        configured_module_roots = execution_settings.get("module_result_roots") or {}
        if isinstance(configured_module_roots, Mapping):
            self._module_result_roots = {
                str(module): str(path)
                for module, path in configured_module_roots.items()
                if str(module).strip() and str(path).strip()
            }
        else:
            self._module_result_roots = {}
        configured_default_root = str(
            execution_settings.get("default_result_root") or ""
        ).strip()
        legacy_result_root = str(execution_settings.get("result_root") or "").strip()
        from darsia.fluidflower.series_paths import series_folder_path
        self._calibration_default_root = (series_folder_path(project.root, "calibration")
            if not configured_default_root and not legacy_result_root else None)
        self._default_result_root = (
            Path(configured_default_root or project.default_result_root())
            .expanduser()
            .resolve()
        )
        if (
            legacy_result_root
            and not configured_default_root
            and not self._module_result_roots
        ):
            legacy_root = Path(legacy_result_root).expanduser().resolve()
            self._module_result_roots = {
                module: str(legacy_root / slugify(module, fallback="module"))
                for module in self._MODULE_LABELS
            }
        self._active_result_module = ""
        self.execution_queue_mode = QComboBox()
        self.execution_queue_mode.addItem("Local project queue", "local")
        self.execution_queue_mode.addItem("Shared network queue", "shared")
        self.execution_queue_mode.currentIndexChanged.connect(
            self._execution_queue_mode_changed
        )
        self.execution_queue_path = PathField(
            directory=True,
            placeholder="Local folder or trusted network share",
        )
        self.execution_queue_path.setText(
            str(
                execution_settings.get("queue_root")
                or project.root / "cache" / "execution" / "queue"
            )
        )
        self.execution_cache_path = PathField(
            directory=True,
            placeholder="Host-local staging and read-through cache",
        )
        self.execution_cache_path.setText(
            str(
                execution_settings.get("local_cache_root")
                or default_execution_cache_root() / project.project_id
            )
        )
        self.execution_result_path = PathField(
            directory=True,
            placeholder="Generated-result folder for the selected analysis",
        )
        self.execution_result_path.setText(
            str(self._configured_module_result_root("calibration"))
        )
        self.execution_result_path.setToolTip(
            "Generated output for the selected analysis is written here. This does "
            "not change the registered raw-data sources."
        )
        self.execution_reset_result_path = QPushButton("Use project default")
        self.execution_reset_result_path.setToolTip(
            "Restore the selected analysis to its standard folder under the DarSIA "
            "project."
        )
        self.execution_reset_result_path.clicked.connect(self._reset_result_destination)
        self.execution_save_paths = QPushButton("Save locations")
        self.execution_save_paths.setToolTip(
            "Store the queue, local cache, and per-analysis result locations in the "
            "DarSIA project."
        )
        self.execution_save_paths.clicked.connect(self._save_execution_paths)
        saved_mode = str(execution_settings.get("queue_mode") or "local")
        saved_mode_index = self.execution_queue_mode.findData(saved_mode)
        if saved_mode_index >= 0:
            self.execution_queue_mode.setCurrentIndex(saved_mode_index)
        queue_layout.addWidget(QLabel("Queue mode"), 0, 0)
        queue_layout.addWidget(self.execution_queue_mode, 0, 1)
        queue_layout.addWidget(QLabel("Task queue"), 1, 0)
        queue_layout.addWidget(self.execution_queue_path, 1, 1, 1, 3)
        queue_layout.addWidget(QLabel("Local cache"), 2, 0)
        queue_layout.addWidget(self.execution_cache_path, 2, 1, 1, 3)
        queue_layout.addWidget(QLabel("Selected analysis results"), 3, 0)
        queue_layout.addWidget(self.execution_result_path, 3, 1)
        result_actions = QHBoxLayout()
        result_actions.addWidget(self.execution_reset_result_path)
        result_actions.addWidget(self.execution_save_paths)
        queue_layout.addLayout(result_actions, 3, 2, 1, 2)
        queue_note = QLabel(
            "The task queue is a small control area and may be a shared folder on "
            "the coordinator. Workers stage files in their local cache and publish "
            "verified results to the selected analysis destination. Each analysis "
            "defaults to a structured folder under the DarSIA project, but large "
            "products such as mass-analysis NPZ files may be redirected independently. "
            "Raw experiment data and the queue folder are never used for results."
        )
        queue_note.setObjectName("secondaryText")
        queue_note.setWordWrap(True)
        queue_layout.addWidget(queue_note, 4, 0, 1, 4)
        layout.addWidget(queue_panel)

        task_panel = QFrame()
        task_panel.setFrameShape(QFrame.Shape.StyledPanel)
        task_layout = QGridLayout(task_panel)
        self.execution_module = QComboBox()
        self.execution_module.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.execution_module.setMinimumContentsLength(12)
        for key, label in self._MODULE_LABELS.items():
            self.execution_module.addItem(label, key)
        self.execution_module.currentIndexChanged.connect(self._module_changed)
        self.execution_adapter = QComboBox()
        self.execution_adapter.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.execution_adapter.setMinimumContentsLength(16)
        self.execution_adapter.currentIndexChanged.connect(self._adapter_changed)
        self.execution_backend = QComboBox()
        self.execution_backend.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.execution_backend.setMinimumContentsLength(14)
        self.execution_backend.currentIndexChanged.connect(
            self._selected_backend_changed
        )
        self.execution_device = QComboBox()
        self.execution_device.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.execution_device.setMinimumContentsLength(14)
        self.execution_device.setToolTip(
            "Fastest: before the job starts, the worker times every device of this "
            "backend on the mass kernel and uses the fastest one that can run the "
            "job; the timings are kept per machine until a device or driver "
            "changes. Choose a device to decide yourself."
        )
        self.execution_priority = QSpinBox()
        self.execution_priority.setRange(0, 999)
        self.execution_priority.setValue(50)
        self.execution_workers = QSpinBox()
        self.execution_workers.setRange(1, 64)
        self.execution_workers.setValue(1)
        self.execution_actor = QLineEdit(str(project.manifest.get("created_by") or ""))
        self.execution_actor.setPlaceholderText("Required for provenance")
        task_layout.addWidget(QLabel("Saved plan"), 0, 0)
        task_layout.addWidget(self.execution_module, 0, 1)
        task_layout.addWidget(QLabel("Adapter"), 0, 2)
        task_layout.addWidget(self.execution_adapter, 0, 3)
        task_layout.addWidget(QLabel("Backend"), 1, 0)
        backend_row = QHBoxLayout()
        backend_row.addWidget(self.execution_backend, 1)
        backend_row.addWidget(QLabel("Device"))
        backend_row.addWidget(self.execution_device, 1)
        task_layout.addLayout(backend_row, 1, 1)
        task_layout.addWidget(QLabel("Worker processes"), 1, 2)
        task_layout.addWidget(self.execution_workers, 1, 3)
        task_layout.addWidget(QLabel("Priority"), 2, 0)
        task_layout.addWidget(self.execution_priority, 2, 1)
        task_layout.addWidget(QLabel("Submitted by"), 2, 2)
        task_layout.addWidget(self.execution_actor, 2, 3)
        actions = QGridLayout()
        self.execution_enqueue = QPushButton("Queue saved plan")
        self.execution_enqueue.clicked.connect(self._enqueue_plan)
        actions.addWidget(self.execution_enqueue, 0, 0)
        self.execution_start_workers = QPushButton("Start workers")
        self.execution_start_workers.clicked.connect(self._start_workers)
        actions.addWidget(self.execution_start_workers, 0, 1)
        self.execution_stop_workers = QPushButton("Stop after current task")
        self.execution_stop_workers.clicked.connect(self._stop_workers)
        actions.addWidget(self.execution_stop_workers, 0, 2)
        self.execution_pause_task = QPushButton("Pause selected task")
        self.execution_pause_task.clicked.connect(self._pause_selected_task)
        actions.addWidget(self.execution_pause_task, 1, 0)
        self.execution_resume_task = QPushButton("Resume selected task")
        self.execution_resume_task.clicked.connect(self._resume_selected_task)
        actions.addWidget(self.execution_resume_task, 1, 1)
        self.execution_recover = QPushButton("Recover stale tasks")
        self.execution_recover.clicked.connect(self._recover_stale)
        actions.addWidget(self.execution_recover, 2, 0)
        self.execution_refresh = QPushButton("Refresh")
        self.execution_refresh.clicked.connect(self._request_queue_snapshot)
        actions.addWidget(self.execution_refresh, 2, 1)
        actions.setColumnStretch(2, 1)
        task_layout.addLayout(actions, 3, 0, 1, 4)
        self.execution_message = QLabel("Queue has not been initialized.")
        self.execution_message.setObjectName("integrationNotice")
        self.execution_message.setWordWrap(True)
        task_layout.addWidget(self.execution_message, 4, 0, 1, 4)
        layout.addWidget(task_panel)

        monitor_panel = QFrame()
        monitor_panel.setFrameShape(QFrame.Shape.StyledPanel)
        monitor_layout = QVBoxLayout(monitor_panel)
        self.execution_counts = QLabel(
            "Pending 0 | Running 0 | Paused 0 | Completed 0 | Failed 0"
        )
        self.execution_counts.setObjectName("sectionTitle")
        monitor_layout.addWidget(self.execution_counts)
        self.execution_tasks = QTableWidget(0, 9)
        self.execution_tasks.setHorizontalHeaderLabels(
            [
                "Status",
                "Module",
                "Task",
                "Backend",
                "Numerics",
                "Adapter",
                "Attempts",
                "Worker",
                "Runtime",
            ]
        )
        self.execution_tasks.verticalHeader().setVisible(False)
        self.execution_tasks.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.execution_tasks.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.execution_tasks.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        tasks_header = self.execution_tasks.horizontalHeader()
        tasks_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        tasks_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        tasks_header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        for column in (0, 1, 3, 6, 7, 8):
            tasks_header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.execution_tasks.setMaximumHeight(260)
        monitor_layout.addWidget(self.execution_tasks)
        self.execution_heartbeats = QTableWidget(0, 5)
        self.execution_heartbeats.setHorizontalHeaderLabels(
            ["Worker", "Host", "State", "Task", "Last heartbeat"]
        )
        self.execution_heartbeats.verticalHeader().setVisible(False)
        self.execution_heartbeats.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        heartbeat_header = self.execution_heartbeats.horizontalHeader()
        heartbeat_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        heartbeat_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 4):
            heartbeat_header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.execution_heartbeats.setMaximumHeight(190)
        monitor_layout.addWidget(self.execution_heartbeats)
        layout.addWidget(monitor_panel)
        layout.addStretch(1)

        self.execution_timer = QTimer(self)
        self.execution_timer.setInterval(5000)
        self.execution_timer.timeout.connect(self._request_queue_snapshot)
        self._populate_adapters()
        self._module_changed()
        self.execution_result_path.changed.connect(self._refresh_runtime_preflight)
        self.execution_queue_path.changed.connect(self._refresh_runtime_preflight)
        self.execution_cache_path.changed.connect(self._refresh_runtime_preflight)
        self.execution_queue_mode.currentIndexChanged.connect(
            self._refresh_runtime_preflight
        )
        self.execution_workers.valueChanged.connect(self._refresh_runtime_preflight)

    @staticmethod
    def _short_identity(value: object) -> str:
        text = str(value or "").strip()
        return text[:16] + ("…" if len(text) > 16 else "") if text else "unknown"

    def _refresh_preflight(self) -> None:
        """Describe the frozen plan without running or changing any analysis."""

        module = str(self.execution_module.currentData() or "")
        self._production_preflight_summary = {}
        try:
            reference = current_plan_reference(self.project, module)
            plan = json.loads(Path(str(reference["path"])).read_text(encoding="utf-8-sig"))
            if not isinstance(plan, dict):
                raise TypeError("Saved plan must contain one JSON object")
            branch = None
            if module == "mass_analysis":
                from darsia.fluidflower.mass_analysis import current_optical_branch

                branch = current_optical_branch(self.project)
            summary = summarize_production_plan(
                module, plan, current_optical_branch=branch
            )
            summary["plan_file_sha256"] = str(reference.get("file_sha256") or "")
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError) as exc:
            self.production_plan.setText(
                f"Saved immutable plan: unavailable for {self._MODULE_LABELS.get(module, module)} "
                f"({exc})."
            )
            self.production_scope.setText("Runs and images: unknown until a plan is saved.")
            self.production_bindings.setText(
                "Correction, phase, and calibration identities: not established."
            )
            self.production_photometry.setText(
                "Production photometric consistency: unverified."
            )
        else:
            self._production_preflight_summary = summary
            status = (
                "existing execution gate passed"
                if summary["stored_ready"]
                else "existing execution gate not passed"
            )
            self.production_plan.setText(
                f"Saved immutable plan: {self._MODULE_LABELS.get(module, module)} · "
                f"{self._short_identity(summary['plan_id'])} · SHA-256 "
                f"{self._short_identity(summary['plan_sha256'])} · {status}."
            )
            if module == "mass_analysis":
                runs = summary["run_count"]
                times = summary["selected_time_count"]
                scope = (
                    f"{runs} experiments × {times} synchronized times = "
                    f"{summary['planned_images']} planned image analyses"
                    if runs and times
                    else "run/frame count unknown in the saved plan"
                )
                self.production_scope.setText(f"Scope: {scope}.")
                self.production_bindings.setText(
                    "Frozen bindings: image correction "
                    f"{self._short_identity(summary['correction_revision_id'])} / "
                    f"{self._short_identity(summary['correction_revision_sha256'])}; "
                    "visible phase "
                    f"{self._short_identity(summary['phase_revision_id'])} / "
                    f"{self._short_identity(summary['phase_revision_sha256'])}; "
                    "calibration plan "
                    f"{self._short_identity(summary['calibration_plan_id'])}; "
                    "calibration result SHA-256 "
                    f"{self._short_identity(summary['calibration_result_sha256'])}; "
                    "prepared inputs SHA-256 "
                    f"{self._short_identity(summary['mass_input_sha256'])}."
                )
                if summary["photometry"] == "revision_linked":
                    link_state = "the existing mass-plan correction/phase revision link matches"
                elif summary["photometry"] == "mismatch":
                    link_state = (
                        "the existing mass-plan correction/phase revision contract "
                        "is not satisfied"
                    )
                else:
                    link_state = "the existing mass-plan correction/phase revision link is unverified"
                phase_state = summary["phase_training_photometry"]
                concentration_state = summary["fit_application_photometry"]
                grid_state = summary["registered_grid"]
                cross_state = summary["cross_chain_photometry"]
                full_state = summary["end_to_end_photometry"]
                self.production_photometry.setText(
                    f"Photometry: {link_state}. Phase training → phase inference: "
                    f"{phase_state}; concentration clearwater/reference → fit → "
                    f"inference: {concentration_state}; registered grid: {grid_state}. "
                    f"Cross-chain correction: {cross_state} (informational; distinct "
                    "phase and concentration chains are allowed when each chain is "
                    "internally consistent). End-to-end photometric evidence: "
                    f"{full_state}. "
                    + (
                        "A fixed categorical phase-map variant has a complete "
                        "declared stage contract and can be queued. "
                        if summary["fixed_phase_variant_ready"]
                        else ""
                    )
                    + "Stage hashes are declared by the saved plan, not independently "
                    "derived from its source images here. Unknown legacy provenance "
                    "is provisional; conflicting stage hashes block mass execution."
                )
            else:
                self.production_scope.setText(
                    "Scope: run/frame count is not specified by this selected plan."
                )
                self.production_bindings.setText(
                    "Correction, visible-phase, and calibration bindings are shown "
                    "when a saved mass-analysis plan is selected."
                )
                self.production_photometry.setText(
                    "Full correction-chain consistency is not certified by this preflight."
                )
        self._update_queue_action_label()
        self._refresh_runtime_preflight()

    def _update_queue_action_label(self) -> None:
        adapter_key = str(self.execution_adapter.currentData() or "")
        module = str(self.execution_module.currentData() or "")
        if adapter_key == "plan-verification-v1":
            label = "Verify saved plan"
        elif (
            module == "mass_analysis"
            and self._production_preflight_summary.get("end_to_end_photometry") != "ready"
        ):
            label = "Queue provisional analysis"
        else:
            label = "Queue production plan"
        self.execution_enqueue.setText(label)

    def _refresh_runtime_preflight(self) -> None:
        if not hasattr(self, "production_runtime"):
            return
        mode = str(self.execution_queue_mode.currentData() or "local")
        result = self.execution_result_path.text() or "unselected"
        adapter_key = str(self.execution_adapter.currentData() or "")
        result_base = (
            str(Path(result) / slugify(adapter_key, fallback="adapter"))
            if result != "unselected" and adapter_key
            else result
        )
        queue = self.execution_queue_path.text() or "unselected"
        cache = self.execution_cache_path.text() or "unselected"
        backend = str(self.execution_backend.currentData() or "unselected")
        if self.inventory:
            available = set(self.inventory.get("available_backends") or [])
            gpu = (
                "detected"
                if any(value.startswith(("cuda-", "opencl-", "metal-")) for value in available)
                else "not detected"
            )
        else:
            gpu = "not checked on this host"
        location = (
            "UNC network path"
            if result.startswith("\\\\")
            else "confirm this project path is reachable from each selected worker"
        )
        self.production_runtime.setText(
            f"Execution: {backend}, {self.execution_workers.value()} local worker "
            f"process(es) if started here; CPU available, GPU {gpu}. "
            "Workers on other hosts are configured separately. "
            f"Queue: {mode} · {queue}. "
            f"Results base: {result_base} ({location}); the immutable task ID is "
            f"added on enqueue. Local staging: {cache}."
        )
        scientific = {
            descriptor.key: set(descriptor.modules)
            for descriptor in self.adapter_registry.descriptors()
            if descriptor.key != "plan-verification-v1"
        }
        unavailable = [
            name
            for module, name in (
                ("phase_interpretation", "phase inference"),
                ("calibration", "calibration fitting"),
            )
            if not any(module in modules for modules in scientific.values())
        ]
        limitation = (
            "Scientific execution adapter unavailable here for "
            + ", ".join(unavailable)
            + "; plan verification alone does not produce scientific results. "
            if unavailable
            else ""
        )
        self.production_limitations.setText(
            limitation
            + "One mass-analysis task is claimed by one host; a shared queue does not "
            "split that task across machines. Cross-host run partitioning is not yet wired."
        )

    def _populate_adapters(self) -> None:
        descriptors = self.adapter_registry.descriptors()
        self.execution_adapters.setRowCount(len(descriptors))
        for row, descriptor in enumerate(descriptors):
            values = (
                descriptor.title,
                descriptor.version,
                ", ".join(
                    self._MODULE_LABELS.get(key, key) for key in descriptor.modules
                ),
                ", ".join(descriptor.backends),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(descriptor.description)
                self.execution_adapters.setItem(row, column, item)

    def _detect_hardware(self) -> None:
        if self._hardware_thread is not None:
            return
        self.execution_detect_hardware.setEnabled(False)
        self.execution_hardware_summary.setText(
            "Inspecting CPU, memory, and accelerators..."
        )
        thread = QThread(self)
        worker = ExecutionHardwareWorker()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._hardware_ready)
        worker.failed.connect(self._hardware_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._hardware_thread_finished)
        self._hardware_thread = thread
        self._hardware_worker = worker
        thread.start()

    def _hardware_thread_finished(self) -> None:
        if self._hardware_thread is not None:
            # After wait() the worker is freed here, in the GUI thread (Q185).
            self._hardware_thread.wait()
            self._hardware_thread.deleteLater()
        self._hardware_worker = None
        self._hardware_thread = None
        self.execution_detect_hardware.setEnabled(True)

    def _hardware_failed(self, message: str) -> None:
        self.execution_hardware_summary.setText(f"Hardware detection failed: {message}")

    def _hardware_ready(self, raw_inventory: object) -> None:
        self.inventory = deepcopy_settings(raw_inventory)
        cpu = dict(self.inventory.get("cpu") or {})
        memory = dict(self.inventory.get("memory") or {})
        total_memory = memory.get("total_bytes")
        memory_text = (
            f"{float(total_memory) / 1024**3:.1f} GiB"
            if isinstance(total_memory, int)
            else "unknown RAM"
        )
        self.execution_hardware_summary.setText(
            f"{self.inventory.get('host', '')}: "
            f"{cpu.get('physical_cores') or '?'} physical / "
            f"{cpu.get('logical_cores') or '?'} logical CPU cores, {memory_text}, "
            f"{len(self.inventory.get('accelerators') or [])} accelerator interfaces."
        )
        accelerators = [
            dict(row)
            for row in self.inventory.get("accelerators") or []
            if isinstance(row, Mapping)
        ]
        rows = [
            {
                "name": str(cpu.get("processor") or "CPU"),
                "backend": "cpu-float64",
                "supported_backends": ["cpu-float64", "cpu-float32"],
                "memory_bytes": total_memory,
                "runtime": self.inventory.get("platform", {}).get("python", ""),
                "driver": "",
            }
        ] + accelerators
        self.execution_devices.setRowCount(len(rows))
        for row, device in enumerate(rows):
            backend = str(device.get("backend") or "")
            raw_supported = device.get("supported_backends")
            if raw_supported is None:
                raw_supported = [backend]
            supported_backends = [str(value) for value in raw_supported if str(value)]
            precisions = sorted(
                {
                    BACKEND_CONTRACTS[value].precision
                    for value in supported_backends
                    if value in BACKEND_CONTRACTS
                }
            )
            memory_bytes = device.get("memory_bytes")
            memory_label = (
                f"{float(memory_bytes) / 1024**3:.1f} GiB"
                if isinstance(memory_bytes, int)
                else "Unknown"
            )
            values = (
                str(device.get("name") or ""),
                ", ".join(supported_backends),
                ", ".join(precisions),
                memory_label,
                " | ".join(
                    value
                    for value in (
                        str(device.get("runtime") or ""),
                        str(device.get("driver") or ""),
                        (
                            "runtime ready"
                            if device.get("runtime_ready") is not False
                            else "runtime dependency missing"
                        ),
                    )
                    if value
                ),
            )
            for column, value in enumerate(values):
                self.execution_devices.setItem(row, column, QTableWidgetItem(value))
        self.recommendations = {
            module: recommend_resources(module, self.inventory)
            for module in self._MODULE_LABELS
        }
        self.execution_recommendations.setRowCount(len(self.recommendations))
        for row, (module, recommendation) in enumerate(self.recommendations.items()):
            values = (
                self._MODULE_LABELS[module],
                str(recommendation["selected_backend"]),
                str(recommendation["device_name"]),
                str(recommendation["accelerator_workers"]),
                str(recommendation["cpu_workers"]),
                str(recommendation["preparation_workers"]),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(str(recommendation.get("reason") or ""))
                self.execution_recommendations.setItem(row, column, item)
        self._module_changed()
        self._refresh_runtime_preflight()

    def _module_changed(self) -> None:
        module = str(self.execution_module.currentData() or "calibration")
        self._select_result_module(module)
        current_adapter = str(self.execution_adapter.currentData() or "")
        try:
            plan_reference = current_plan_reference(self.project, module)
            planned_backend = str(
                (plan_reference.get("quality_contract") or {}).get("backend") or ""
            )
        except (FileNotFoundError, OSError, TypeError, ValueError):
            planned_backend = ""
        self._planned_backend = planned_backend
        self.execution_adapter.blockSignals(True)
        self.execution_adapter.clear()
        for descriptor in self.adapter_registry.descriptors():
            if module not in descriptor.modules:
                continue
            self.execution_adapter.addItem(descriptor.title, descriptor.key)
            self.execution_adapter.setItemData(
                self.execution_adapter.count() - 1,
                descriptor.description,
                Qt.ItemDataRole.ToolTipRole,
            )
        if current_adapter:
            index = self.execution_adapter.findData(current_adapter)
            if index >= 0:
                self.execution_adapter.setCurrentIndex(index)
        if (
            planned_backend
            or not current_adapter
            or current_adapter == "plan-verification-v1"
            or self.execution_adapter.findData(current_adapter) < 0
        ):
            recommended_backend = str(
                planned_backend
                or (self.recommendations.get(module) or {}).get("selected_backend")
                or "cpu-float64"
            )
            preferred_index = -1
            fallback_index = -1
            for index in range(self.execution_adapter.count()):
                key = str(self.execution_adapter.itemData(index) or "")
                if key == "plan-verification-v1":
                    continue
                if fallback_index < 0:
                    fallback_index = index
                if (
                    recommended_backend
                    in self.adapter_registry.get(key).descriptor.backends
                ):
                    preferred_index = index
                    break
            target_index = preferred_index if preferred_index >= 0 else fallback_index
            if target_index < 0:
                target_index = self.execution_adapter.findData("plan-verification-v1")
            self.execution_adapter.setCurrentIndex(target_index)
        self.execution_adapter.blockSignals(False)
        self._adapter_changed()
        self._refresh_preflight()

    def _adapter_changed(self) -> None:
        adapter_key = str(self.execution_adapter.currentData() or "")
        self._update_queue_action_label()
        module = str(self.execution_module.currentData() or "calibration")
        current_backend = str(self.execution_backend.currentData() or "")
        self.execution_backend.blockSignals(True)
        self.execution_backend.clear()
        if adapter_key:
            descriptor = self.adapter_registry.get(adapter_key).descriptor
            available = set(
                self.inventory.get("available_backends")
                or ["cpu-float64", "cpu-float32"]
            )
            for backend in descriptor.backends:
                self.execution_backend.addItem(
                    backend if backend in available else f"{backend} (not detected)",
                    backend,
                )
                if backend not in available:
                    model_item = self.execution_backend.model().item(
                        self.execution_backend.count() - 1
                    )
                    if model_item is not None:
                        model_item.setEnabled(False)
            selected_index = self.execution_backend.findData(current_backend)
            if selected_index < 0:
                recommended_backend = str(
                    getattr(self, "_planned_backend", "")
                    or (self.recommendations.get(module) or {}).get("selected_backend")
                    or "cpu-float64"
                )
                selected_index = self.execution_backend.findData(recommended_backend)
            if selected_index < 0:
                for index in range(self.execution_backend.count()):
                    model_item = self.execution_backend.model().item(index)
                    if model_item is None or model_item.isEnabled():
                        selected_index = index
                        break
            if selected_index >= 0:
                self.execution_backend.setCurrentIndex(selected_index)
        self.execution_backend.blockSignals(False)
        self._selected_backend_changed()

    def _selected_backend_changed(self) -> None:
        module = str(self.execution_module.currentData() or "calibration")
        recommendation = self.recommendations.get(module)
        backend = str(self.execution_backend.currentData() or "cpu-float64")
        if recommendation is None:
            self.execution_workers.setValue(1)
        elif backend.startswith("cpu-"):
            self.execution_workers.setValue(
                min(64, max(1, int(recommendation.get("cpu_workers") or 1)))
            )
        else:
            self.execution_workers.setValue(
                min(
                    64,
                    max(1, int(recommendation.get("accelerator_workers") or 1)),
                )
            )
        self._refresh_device_choices()
        self._refresh_runtime_preflight()

    def _refresh_device_choices(self) -> None:
        """List the selected backend's devices for an explicit choice.

        The first entry leaves the choice to the measurement at job start.
        """

        backend = str(self.execution_backend.currentData() or "")
        current = str(self.execution_device.currentData() or "")
        devices = (
            []
            if not backend or backend.startswith("cpu-")
            else backend_devices(self.inventory, backend)
        )
        self.execution_device.blockSignals(True)
        self.execution_device.clear()
        if devices:
            self.execution_device.addItem(
                "Fastest, measured at job start" if len(devices) > 1 else "Only device",
                "",
            )
            for device in devices:
                self.execution_device.addItem(
                    f"{device.get('device_id')}: {device.get('name') or backend}",
                    str(device.get("device_id")),
                )
            self.execution_device.setCurrentIndex(
                max(0, self.execution_device.findData(current))
            )
        else:
            self.execution_device.addItem(
                "CPU" if backend.startswith("cpu-") else "No device detected", ""
            )
        self.execution_device.setEnabled(len(devices) > 1)
        self.execution_device.blockSignals(False)

    def _execution_queue_mode_changed(self) -> None:
        if self.execution_queue_mode.currentData() == "local":
            self.execution_queue_path.setText(
                str(self.project.root / "cache" / "execution" / "queue")
            )
        self.execution_message.setText(
            "Choose the queue folder, then queue a saved immutable plan or refresh status."
        )

    def _queue(self) -> ExecutionQueue:
        raw_path = self.execution_queue_path.text()
        if not raw_path:
            raise ValueError("Select a local or shared queue folder first")
        return ExecutionQueue(Path(raw_path))

    def _default_module_result_root(self, module: str) -> Path:
        if module == "calibration" and getattr(self, "_calibration_default_root", None) is not None:
            return self._calibration_default_root
        return self._default_result_root / slugify(module, fallback="module")

    def _configured_module_result_root(self, module: str) -> Path:
        configured = self._module_result_roots.get(module, "").strip()
        if configured:
            return Path(configured).expanduser().resolve()
        return self._default_module_result_root(module)

    def _remember_active_result_path(self) -> None:
        module = self._active_result_module
        raw_path = self.execution_result_path.text()
        if not module or not raw_path:
            return
        selected = Path(raw_path).expanduser().resolve()
        if selected == self._default_module_result_root(module):
            self._module_result_roots.pop(module, None)
        else:
            self._module_result_roots[module] = str(selected)

    def _select_result_module(self, module: str) -> None:
        if self._active_result_module == module:
            return
        self._remember_active_result_path()
        self._active_result_module = module
        self.execution_result_path.setText(
            str(self._configured_module_result_root(module))
        )

    def _reset_result_destination(self) -> None:
        module = str(self.execution_module.currentData() or "calibration")
        self._module_result_roots.pop(module, None)
        self._active_result_module = module
        self.execution_result_path.setText(
            str(self._default_module_result_root(module))
        )
        self.execution_message.setText(
            f"{self._MODULE_LABELS.get(module, module)} will use the project default."
        )

    def _result_root(self) -> Path:
        raw_path = self.execution_result_path.text()
        if not raw_path:
            raise ValueError("Select a result destination first")
        return Path(raw_path).expanduser().resolve()

    def _persist_execution_paths(self) -> None:
        self._remember_active_result_path()
        self.project.update_metadata(
            "tools",
            {
                "execution": {
                    "queue_mode": str(self.execution_queue_mode.currentData()),
                    "queue_root": str(self._queue().root),
                    "local_cache_root": str(
                        Path(self.execution_cache_path.text()).expanduser().resolve()
                    ),
                    "default_result_root": str(self._default_result_root),
                    "result_root": str(self._default_result_root),
                    "module_result_roots": dict(
                        sorted(self._module_result_roots.items())
                    ),
                }
            },
            updated_by=self.execution_actor.text().strip(),
            reason="Updated execution storage locations.",
        )

    def _save_execution_paths(self) -> None:
        try:
            self._persist_execution_paths()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not save locations", str(exc))
            return
        self.execution_message.setText(
            "Queue, cache, and per-analysis result locations were saved."
        )

    def _enqueue_plan(self) -> None:
        actor = self.execution_actor.text().strip()
        if not actor:
            QMessageBox.warning(
                self, "Submitted by is required", "Enter a name for provenance."
            )
            return
        module = str(self.execution_module.currentData() or "")
        adapter_key = str(self.execution_adapter.currentData() or "")
        backend = str(self.execution_backend.currentData() or "")
        try:
            self.adapter_registry.get(adapter_key)
            plan_reference = current_plan_reference(self.project, module)
            preflight_settings: dict[str, Any] = {}
            if module == "mass_analysis" and adapter_key != "plan-verification-v1":
                self._refresh_preflight()
                summary = self._production_preflight_summary
                if str(summary.get("plan_file_sha256") or "") != str(
                    plan_reference.get("file_sha256") or ""
                ):
                    raise ValueError(
                        "The saved mass plan changed while preflight was being "
                        "refreshed. Refresh the plan and try again."
                    )
                if not summary.get("ready_for_mass_enqueue"):
                    raise ValueError(
                        "The saved mass plan is not eligible under its existing "
                        "revision contract or an explicitly documented fixed-phase "
                        "variant. Review the Production preflight and save a "
                        "compatible plan."
                    )
                if summary.get("end_to_end_photometry") == "mismatch":
                    raise ValueError(
                        "Phase training/inference, concentration reference/fit/"
                        "inference, or registered-grid identities explicitly "
                        "conflict. Resolve the mismatch before production."
                    )
                preflight_settings = {
                    "production_photometry_preflight": {
                        "format": "darsia-fluidflower-production-preflight-v1",
                        "plan_file_sha256": summary["plan_file_sha256"],
                        "basis": (
                            "fixed_categorical_phase_variant"
                            if summary["fixed_phase_variant_ready"]
                            else "legacy_revision_link"
                        ),
                        "status": (
                            "matched_declared_stage_hashes"
                            if summary["end_to_end_photometry"] == "ready"
                            else "provisional_unverified_stage_hashes"
                        ),
                        "stage_evidence": summary["photometry_evidence"],
                    }
                }
            planned_backend = str(
                (plan_reference.get("quality_contract") or {}).get("backend") or ""
            )
            if (
                adapter_key != "plan-verification-v1"
                and planned_backend
                and backend != planned_backend
            ):
                raise ValueError(
                    f"The saved {module} plan requires {planned_backend}; "
                    f"select its matching execution adapter and backend."
                )
            if self.inventory and backend not in set(
                self.inventory.get("available_backends") or []
            ):
                raise ValueError(f"Backend is not available on this host: {backend}")
            recommendation = deepcopy_settings(
                self.recommendations.get(module)
                or {
                    "module": module,
                    "selected_backend": backend,
                    "reference_backend": "cpu-float64",
                    "accelerator_workers": 0 if backend.startswith("cpu-") else 1,
                    "cpu_workers": self.execution_workers.value(),
                    "preparation_workers": 1,
                    "local_read_through_cache": True,
                    "atomic_publish": True,
                }
            )
            recommendation["selected_backend"] = backend
            if backend.startswith("cpu-"):
                recommendation["cpu_workers"] = self.execution_workers.value()
                recommendation["accelerator_workers"] = 0
                recommendation["preparation_workers"] = 1
                recommendation["device_id"] = ""
                recommendation["device_name"] = "CPU"
                recommendation.pop("device_choice", None)
            else:
                recommendation["accelerator_workers"] = self.execution_workers.value()
                recommendation["preparation_workers"] = self.execution_workers.value()
                recommendation.update(
                    requested_device_resources(
                        self.inventory,
                        backend,
                        str(self.execution_device.currentData() or ""),
                    )
                )
            recommendation.pop("content_sha256", None)
            task = task_for_current_plan(
                self.project,
                module=module,
                adapter_key=adapter_key,
                backend=backend,
                created_by=actor,
                resources=recommendation,
                output_root=(
                    self._result_root() / slugify(adapter_key, fallback="adapter")
                ),
                priority=self.execution_priority.value(),
                registry=self.adapter_registry,
                adapter_settings=preflight_settings,
            )
            if module == "mass_analysis" and adapter_key != "plan-verification-v1":
                if dict(task.get("plan_reference") or {}) != dict(plan_reference):
                    raise ValueError(
                        "The saved mass plan changed while the queue task was being "
                        "built. Refresh the plan and try again."
                    )
            queue = self._queue()
            queued = queue.enqueue(task)
            self._persist_execution_paths()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not queue saved plan", str(exc))
            return
        self.execution_message.setText(
            f"{'Queued' if queued else 'Already present'}: {task['task_id']}"
        )
        self.execution_timer.start()
        self._request_queue_snapshot()

    def _start_workers(self) -> None:
        backend = str(self.execution_backend.currentData() or "")
        if not backend:
            QMessageBox.warning(
                self, "No backend", "Select an available backend first."
            )
            return
        try:
            queue = self._queue()
            queue.ensure_layout()
            cache = Path(self.execution_cache_path.text()).expanduser().resolve()
            cache.mkdir(parents=True, exist_ok=True)
            self._result_root().mkdir(parents=True, exist_ok=True)
            self._persist_execution_paths()
            handles = spawn_worker_processes(
                queue_root=queue.root,
                cache_root=cache,
                workers=self.execution_workers.value(),
                backends=[backend],
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not start workers", str(exc))
            return
        self.worker_processes.extend(handles)
        self.execution_message.setText(
            f"Started {len(handles)} hidden {backend} worker process(es)."
        )
        self.execution_timer.start()
        self._request_queue_snapshot()

    def _stop_workers(self) -> None:
        try:
            queue = self._queue()
            active = [
                handle
                for handle in self.worker_processes
                if handle.process.poll() is None
            ]
            for handle in active:
                queue.request_worker_stop(
                    handle.worker_id,
                    actor=self.execution_actor.text().strip(),
                )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not stop workers", str(exc))
            return
        self.execution_message.setText(
            f"Stop requested for {len(active)} local worker process(es)."
        )

    def _selected_task_id(self) -> str:
        row = self.execution_tasks.currentRow()
        if row < 0:
            raise ValueError("Select a task in the monitor first")
        item = self.execution_tasks.item(row, 2)
        task_id = str(item.data(Qt.ItemDataRole.UserRole) or "") if item else ""
        if not task_id:
            raise ValueError("The selected row has no task identity")
        return task_id

    def _pause_selected_task(self) -> None:
        try:
            task_id = self._selected_task_id()
            self._queue().request_task_pause(
                task_id,
                actor=self.execution_actor.text().strip(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not pause task", str(exc))
            return
        self.execution_message.setText(
            f"Pause requested for {task_id}; active work stops after its current chunk."
        )
        self._request_queue_snapshot()

    def _resume_selected_task(self) -> None:
        try:
            task_id = self._selected_task_id()
            resumed = self._queue().resume_task(
                task_id,
                actor=self.execution_actor.text().strip(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not resume task", str(exc))
            return
        self.execution_message.setText(
            f"{'Resumed' if resumed else 'No paused record found for'} {task_id}."
        )
        self._request_queue_snapshot()

    def _recover_stale(self) -> None:
        self._request_queue_snapshot(recover_timeout_seconds=15 * 60)

    def _request_queue_snapshot(
        self,
        _checked: bool = False,
        *,
        recover_timeout_seconds: float | None = None,
    ) -> None:
        if self._queue_thread is not None:
            return
        try:
            queue_root = self._queue().root
        except Exception as exc:  # noqa: BLE001
            self.execution_message.setText(str(exc))
            return
        self.execution_refresh.setEnabled(False)
        thread = QThread(self)
        worker = ExecutionQueueWorker(
            queue_root,
            recover_timeout_seconds=recover_timeout_seconds,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._queue_snapshot_ready)
        worker.failed.connect(self._queue_snapshot_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._queue_thread_finished)
        self._queue_thread = thread
        self._queue_worker = worker
        thread.start()

    def _queue_thread_finished(self) -> None:
        if self._queue_thread is not None:
            # After wait() the worker is freed here, in the GUI thread (Q185).
            self._queue_thread.wait()
            self._queue_thread.deleteLater()
        self._queue_worker = None
        self._queue_thread = None
        self.execution_refresh.setEnabled(True)

    def _queue_snapshot_failed(self, message: str) -> None:
        self.execution_message.setText(f"Queue unavailable: {message}")

    def _queue_snapshot_ready(self, raw_snapshot: object) -> None:
        snapshot = deepcopy_settings(raw_snapshot)
        counts = dict(snapshot.get("counts") or {})
        self.execution_counts.setText(
            f"Pending {counts.get('pending', 0)} | "
            f"Running {counts.get('in_progress', 0)} | "
            f"Paused {counts.get('paused', 0)} | "
            f"Completed {counts.get('completed', 0)} | "
            f"Failed {counts.get('failed', 0)}"
        )
        recovered = int(snapshot.get("recovered_tasks") or 0)
        if recovered:
            self.execution_message.setText(
                f"Recovered {recovered} stale task(s) for work stealing."
            )
        tasks = [
            dict(row) for row in snapshot.get("tasks") or [] if isinstance(row, Mapping)
        ]
        self.execution_tasks.setRowCount(len(tasks))
        for row, task in enumerate(tasks):
            runtime = task.get("runtime_seconds")
            quality = dict(task.get("quality_contract") or {})
            values = (
                str(task.get("status") or ""),
                self._MODULE_LABELS.get(
                    str(task.get("module") or ""),
                    str(task.get("module") or ""),
                ),
                str(task.get("task_id") or ""),
                str(task.get("backend") or ""),
                str(quality.get("label") or "not declared"),
                str(task.get("adapter") or ""),
                str(task.get("attempts") or 0),
                str(task.get("worker_id") or ""),
                f"{float(runtime):.1f} s" if isinstance(runtime, (int, float)) else "",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(str(task.get("destination") or task.get("path") or ""))
                item.setData(Qt.ItemDataRole.UserRole, str(task.get("task_id") or ""))
                self.execution_tasks.setItem(row, column, item)
        heartbeats = [
            dict(row)
            for row in snapshot.get("heartbeats") or []
            if isinstance(row, Mapping)
        ]
        heartbeats.sort(
            key=lambda row: float(row.get("recorded_epoch") or 0.0), reverse=True
        )
        self.execution_heartbeats.setRowCount(len(heartbeats))
        for row, heartbeat in enumerate(heartbeats):
            values = (
                str(heartbeat.get("worker_id") or ""),
                str(heartbeat.get("host") or ""),
                str(heartbeat.get("status") or ""),
                str(heartbeat.get("task_id") or ""),
                str(heartbeat.get("recorded_utc") or ""),
            )
            for column, value in enumerate(values):
                self.execution_heartbeats.setItem(row, column, QTableWidgetItem(value))

    def refresh(self) -> None:
        self.worker_processes = [
            handle for handle in self.worker_processes if handle.process.poll() is None
        ]
        if self.isVisible():
            self._refresh_preflight()

    def closeEvent(self, event) -> None:  # noqa: N802, ANN001
        self.execution_timer.stop()
        super().closeEvent(event)
