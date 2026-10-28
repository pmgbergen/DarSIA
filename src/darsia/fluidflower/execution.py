"""Portable execution contracts and workers for FluidFlower analyses.

The Qt application only creates immutable plans.  This module turns those plans
into versioned tasks and executes them through the same directory-backed queue
locally or on a trusted network share.  Scientific implementations are loaded
through a small adapter registry; queue files can therefore never inject an
arbitrary shell command.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Protocol
import uuid

from darsia.fluidflower.project import FluidFlowerProject, sha256_file, utc_now
from darsia.fluidflower.quality_budget import summarize_quality_contract


LOG = logging.getLogger("darsia.fluidflower.execution")

TASK_DOCUMENT_FORMAT = "darsia-fluidflower-task-v1"
TASK_RESULT_FORMAT = "darsia-fluidflower-task-result-v1"
TASK_FAILURE_FORMAT = "darsia-fluidflower-task-failure-v1"
QUEUE_FORMAT = "darsia-fluidflower-execution-queue-v1"
HARDWARE_INVENTORY_FORMAT = "darsia-fluidflower-hardware-v1"
RESOURCE_RECOMMENDATION_FORMAT = "darsia-fluidflower-resource-recommendation-v1"
PUBLISH_MANIFEST_FORMAT = "darsia-fluidflower-publish-manifest-v1"
CHECKPOINT_FORMAT = "darsia-fluidflower-checkpoint-v1"
CHUNK_MANIFEST_FORMAT = "darsia-fluidflower-chunk-v1"

PENDING = "pending"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"
FAILED = "failed"
PAUSED = "paused"
HEARTBEATS = "heartbeats"
CLAIMS = "claims"
CHECKPOINTS = "checkpoints"
CONTROL = "control"
LOGS = "logs"
QUEUE_DIRECTORIES = (
    PENDING,
    IN_PROGRESS,
    COMPLETED,
    FAILED,
    PAUSED,
    HEARTBEATS,
    CLAIMS,
    CHECKPOINTS,
    CONTROL,
    LOGS,
)

STORAGE_RETRY_DELAYS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0)
ORPHANED_CLAIM_SECONDS = 60.0


class TaskPauseRequested(RuntimeError):
    """Cooperative pause raised only after an atomic chunk is preserved."""


@dataclass(frozen=True)
class BackendContract:
    """Numerical semantics that must remain explicit in every task."""

    name: str
    family: str
    precision: str
    semantic_contract: str
    authoritative_reference: bool = False


BACKEND_CONTRACTS: dict[str, BackendContract] = {
    "cpu-float32": BackendContract(
        "cpu-float32",
        "cpu",
        "float32",
        "fluidflower-phase-mass-float32-hard-mask-v1",
    ),
    "cpu-float64": BackendContract(
        "cpu-float64",
        "cpu",
        "float64",
        "fluidflower-cpu-float64-reference-v1",
        authoritative_reference=True,
    ),
    "cuda-float32": BackendContract(
        "cuda-float32",
        "cuda",
        "float32",
        "fluidflower-phase-mass-float32-hard-mask-v1",
    ),
    "cuda-float64": BackendContract(
        "cuda-float64",
        "cuda",
        "float64",
        "fluidflower-phase-mass-float64-hard-mask-v1",
    ),
    "opencl-float32": BackendContract(
        "opencl-float32",
        "opencl",
        "float32",
        "fluidflower-phase-mass-float32-hard-mask-v1",
    ),
    "opencl-float64": BackendContract(
        "opencl-float64",
        "opencl",
        "float64",
        "fluidflower-phase-mass-float64-hard-mask-v1",
    ),
    "metal-float32": BackendContract(
        "metal-float32",
        "metal",
        "float32",
        "fluidflower-phase-mass-float32-hard-mask-v1",
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _retry_storage_io(
    action: Callable[[], Any],
    description: str,
    *,
    retry_not_found: bool = True,
) -> Any:
    """Retry a filesystem operation across short SMB/NAS interruptions."""

    for attempt, delay in enumerate((*STORAGE_RETRY_DELAYS, None), start=1):
        try:
            return action()
        except FileNotFoundError:
            if not retry_not_found:
                raise
            if delay is None:
                raise
            LOG.warning(
                "Storage path unavailable for %s (attempt %d); retrying in %.2f s",
                description,
                attempt,
                delay,
            )
            time.sleep(delay)
        except OSError:
            if delay is None:
                raise
            LOG.warning(
                "Storage operation failed for %s (attempt %d); retrying in %.2f s",
                description,
                attempt,
                delay,
            )
            time.sleep(delay)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"

    def write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except (FileNotFoundError, OSError):
                pass

    _retry_storage_io(write, str(path))


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read one UTF-8-sig JSON object with the execution reader's original error."""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    def read() -> dict[str, Any]:
        return _read_json_object(path)

    return _retry_storage_io(read, str(path), retry_not_found=False)


def _safe_identifier(value: str, fallback: str = "worker") -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in str(value).strip()
    ).strip("-.")
    return cleaned[:96] or fallback


def _host_name() -> str:
    return platform.node() or socket.gethostname() or "host"


def default_execution_cache_root() -> Path:
    """Return a host-local cache path that is never placed in a project share."""

    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "DarSIA" / "FluidFlower" / "execution"


def _subprocess_options() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": int(getattr(subprocess, "CREATE_NO_WINDOW", 0))}
    return {}


def _memory_bytes() -> tuple[int | None, int | None]:
    """Return total and currently available memory without a hard dependency."""

    if importlib.util.find_spec("psutil") is not None:
        try:
            import psutil  # type: ignore[import-not-found]

            memory = psutil.virtual_memory()
            return int(memory.total), int(memory.available)
        except Exception:  # noqa: BLE001
            pass
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_physical), int(status.available_physical)
        except Exception:  # noqa: BLE001
            pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total = page_size * int(os.sysconf("SC_PHYS_PAGES"))
        available = page_size * int(os.sysconf("SC_AVPHYS_PAGES"))
        return total, available
    except (AttributeError, OSError, ValueError):
        return None, None


def _physical_cpu_count() -> int | None:
    if importlib.util.find_spec("psutil") is None:
        return None
    try:
        import psutil  # type: ignore[import-not-found]

        value = psutil.cpu_count(logical=False)
        return int(value) if value else None
    except Exception:  # noqa: BLE001
        return None


def _nvidia_devices() -> list[dict[str, Any]]:
    query = "index,uuid,name,memory.total,driver_version"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
            **_subprocess_options(),
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []
    devices: list[dict[str, Any]] = []
    runtime_ready = importlib.util.find_spec("cupy") is not None
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            memory_bytes = int(float(parts[3]) * 1024 * 1024)
        except ValueError:
            memory_bytes = None
        devices.append(
            {
                "backend": "cuda-float32",
                "supported_backends": (
                    ["cuda-float32", "cuda-float64"] if runtime_ready else []
                ),
                "runtime_ready": runtime_ready,
                "family": "cuda",
                "device_id": parts[0],
                "stable_id": parts[1],
                "name": parts[2],
                "memory_bytes": memory_bytes,
                "driver": parts[4],
                "runtime": "nvidia-smi",
            }
        )
    return devices


def _opencl_devices() -> list[dict[str, Any]]:
    if importlib.util.find_spec("pyopencl") is None:
        return []
    try:
        import pyopencl as cl  # type: ignore[import-not-found]

        devices: list[dict[str, Any]] = []
        index = 0
        for opencl_platform in cl.get_platforms():
            for device in opencl_platform.get_devices():
                device_type = int(device.type)
                if not device_type & int(cl.device_type.GPU):
                    continue
                devices.append(
                    {
                        "backend": "opencl-float32",
                        "supported_backends": [
                            "opencl-float32",
                            *(
                                ["opencl-float64"]
                                if int(getattr(device, "double_fp_config", 0)) != 0
                                or "cl_khr_fp64" in str(device.extensions).split()
                                else []
                            ),
                        ],
                        "family": "opencl",
                        "device_id": str(index),
                        "stable_id": f"{opencl_platform.name}:{device.name}",
                        "name": str(device.name).strip(),
                        "memory_bytes": int(device.global_mem_size),
                        "compute_units": int(device.max_compute_units),
                        "driver": str(device.driver_version),
                        "runtime": str(opencl_platform.version),
                        "runtime_ready": True,
                    }
                )
                index += 1
        return devices
    except Exception:  # noqa: BLE001
        return []


def _metal_devices(total_memory: int | None) -> list[dict[str, Any]]:
    if sys.platform != "darwin" or platform.machine().casefold() not in {
        "arm64",
        "aarch64",
    }:
        return []
    runtime_ready = importlib.util.find_spec("torch") is not None
    return [
        {
            "backend": "metal-float32",
            "supported_backends": ["metal-float32"] if runtime_ready else [],
            "runtime_ready": runtime_ready,
            "family": "metal",
            "device_id": "0",
            "stable_id": f"apple-silicon:{platform.machine()}",
            "name": "Apple Silicon GPU",
            "memory_bytes": total_memory,
            "shared_memory": True,
            "driver": platform.mac_ver()[0],
            "runtime": "Metal",
        }
    ]


def detect_host_resources() -> dict[str, Any]:
    """Return CPU and memory facts without probing accelerator runtimes."""

    total_memory, available_memory = _memory_bytes()
    logical_cpus = max(1, int(os.cpu_count() or 1))
    physical_cpus = _physical_cpu_count()
    return {
        "cpu": {
            "logical_cores": logical_cpus,
            "physical_cores": physical_cpus,
            "processor": platform.processor(),
        },
        "memory": {
            "total_bytes": total_memory,
            "available_bytes": available_memory,
        },
    }


def detect_hardware() -> dict[str, Any]:
    """Return a portable inventory used for recommendations and provenance."""

    host_resources = detect_host_resources()
    total_memory = (host_resources.get("memory") or {}).get("total_bytes")
    accelerators = _nvidia_devices() + _opencl_devices() + _metal_devices(total_memory)
    backends = ["cpu-float64", "cpu-float32"]
    for device in accelerators:
        supported = device.get("supported_backends")
        if supported is None:
            supported = [device["backend"]]
        for raw_backend in supported:
            backend = str(raw_backend)
            if backend not in backends:
                backends.append(backend)
    payload: dict[str, Any] = {
        "schema": 1,
        "format": HARDWARE_INVENTORY_FORMAT,
        "recorded_utc": _utc_now(),
        "host": _host_name(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        **host_resources,
        "accelerators": accelerators,
        "available_backends": backends,
    }
    payload["content_sha256"] = _json_digest(payload)
    return payload


def _gpu_worker_count(device: Mapping[str, Any], bytes_per_worker: int) -> int:
    memory = device.get("memory_bytes")
    if not isinstance(memory, int) or memory <= 0:
        return 1
    # Keep both an absolute reserve and at least 20 percent headroom.
    usable = max(0, min(memory - 2 * 1024**3, int(memory * 0.8)))
    return max(1, min(2, usable // max(1, bytes_per_worker)))


def recommend_resources(
    module: str,
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a conservative, editable recommendation for one module."""

    module = str(module)
    logical = max(1, int((inventory.get("cpu") or {}).get("logical_cores") or 1))
    physical = int((inventory.get("cpu") or {}).get("physical_cores") or logical)
    available_memory = (inventory.get("memory") or {}).get("available_bytes")
    devices = [
        dict(device)
        for device in inventory.get("accelerators") or []
        if isinstance(device, Mapping) and device.get("runtime_ready") is not False
    ]
    selected_device: dict[str, Any] | None = None
    preferred_order = ("cuda-float32", "opencl-float32", "metal-float32")
    if module in {"calibration", "mass_analysis"}:
        for backend in preferred_order:
            selected_device = next(
                (
                    device
                    for device in devices
                    if backend
                    in (
                        device.get("supported_backends")
                        if device.get("supported_backends") is not None
                        else [device.get("backend")]
                    )
                ),
                None,
            )
            if selected_device is not None:
                break
    selected_backend = (
        str(selected_device["backend"]) if selected_device else "cpu-float64"
    )
    device_fields = {"device_id": "", "device_name": "CPU"}
    if selected_device is not None:
        from darsia.fluidflower.device_choice import requested_device_resources

        # Several devices are not named here: the worker measures them at job start.
        device_fields = requested_device_resources(
            {"accelerators": devices}, selected_backend
        )
    accelerator_workers = 0
    if selected_device is not None:
        estimated = 6 * 1024**3 if module == "calibration" else 5 * 1024**3
        accelerator_workers = _gpu_worker_count(selected_device, estimated)
    reserve = 2 if logical >= 8 else 1
    cpu_workers = max(1, min(24, physical - reserve))
    if module == "wasserstein":
        selected_backend = "cpu-float64"
        accelerator_workers = 0
        cpu_workers = min(cpu_workers, 12)
        if isinstance(available_memory, int) and available_memory > 0:
            cpu_workers = min(cpu_workers, max(1, available_memory // (2 * 1024**3)))
    elif module in {"setup", "data_integrity"}:
        selected_backend = "cpu-float64"
        accelerator_workers = 0
        cpu_workers = min(cpu_workers, 12)
    elif module == "post_analysis":
        selected_backend = "cpu-float64"
        accelerator_workers = 0
        cpu_workers = min(24, max(1, logical - reserve))
    elif selected_device is not None:
        # CPU preparation runs beside the accelerator, but is bounded so image
        # correction and storage do not starve the GPU workers.
        cpu_workers = min(max(accelerator_workers, 2), max(1, physical // 2))

    explanation = {
        "calibration": "Accelerator evaluation with CPU float64 parity samples.",
        "mass_analysis": "Accelerator phase algebra with bounded CPU image preparation.",
        "wasserstein": "CPU float64 is selected because transport solves are CPU-bound.",
        "post_analysis": "Parallel CPU frame preparation; video encoding is selected separately.",
    }.get(module, "CPU execution with conservative memory headroom.")
    payload: dict[str, Any] = {
        "schema": 1,
        "format": RESOURCE_RECOMMENDATION_FORMAT,
        "module": module,
        "selected_backend": selected_backend,
        "reference_backend": "cpu-float64",
        **device_fields,
        "accelerator_workers": accelerator_workers,
        "cpu_workers": cpu_workers,
        "preparation_workers": (
            max(1, accelerator_workers) if accelerator_workers else 1
        ),
        "local_read_through_cache": True,
        "atomic_publish": True,
        "reason": explanation,
        "inventory_sha256": str(inventory.get("content_sha256") or ""),
    }
    payload["content_sha256"] = _json_digest(payload)
    return payload


@dataclass(frozen=True)
class AdapterDescriptor:
    key: str
    version: str
    title: str
    modules: tuple[str, ...]
    backends: tuple[str, ...]
    description: str = ""


@dataclass
class WorkerProcessHandle:
    worker_id: str
    process: subprocess.Popen[Any]
    cache_observation_error: dict[str, Any] | None = None


class TaskExecutionContext:
    """Adapter-visible staging, checkpoint, cancellation, and provenance API."""

    def __init__(
        self,
        *,
        task: Mapping[str, Any],
        queue: "ExecutionQueue",
        worker_id: str,
        workspace: Path,
        inventory: Mapping[str, Any],
    ) -> None:
        self.task = deepcopy(dict(task))
        self.queue = queue
        self.worker_id = worker_id
        self.workspace = workspace
        self.staging = workspace / "artifacts"
        self.staging.mkdir(parents=True, exist_ok=True)
        self.inventory = deepcopy(dict(inventory))
        self.local_checkpoint_path = workspace / "checkpoint.json"
        self.shared_checkpoint_path = (
            queue.directory(CHECKPOINTS) / str(task["task_id"]) / "current.json"
        )
        destination = Path(str((task.get("output") or {}).get("destination") or ""))
        if not str(destination):
            raise ValueError("Task output destination is required for resumable chunks")
        self.partial_root = (
            destination.expanduser().resolve().parent
            / ".darsia-partial"
            / _safe_identifier(str(task["task_id"]), "task")
        )
        self.chunk_root = self.partial_root / "chunks"
        self.local_chunk_staging = workspace / "chunk-staging"

    def checkpoint(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        record = {
            "schema": 1,
            "format": CHECKPOINT_FORMAT,
            "task_id": str(self.task["task_id"]),
            "task_sha256": str(self.task["content_sha256"]),
            "worker_id": self.worker_id,
            "recorded_utc": _utc_now(),
            "state": deepcopy(dict(payload)),
        }
        record["content_sha256"] = _json_digest(record)
        _atomic_json(self.local_checkpoint_path, record)
        _atomic_json(self.shared_checkpoint_path, record)
        return record

    def load_checkpoint(self) -> dict[str, Any]:
        for path in (self.local_checkpoint_path, self.shared_checkpoint_path):
            try:
                record = _read_json(path)
            except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
                continue
            if record.get("format") == CHECKPOINT_FORMAT and record.get(
                "task_sha256"
            ) == self.task.get("content_sha256"):
                return record
        return {}

    def cancellation_requested(self) -> bool:
        task_marker = (
            self.queue.directory(CONTROL) / "cancel" / (f"{self.task['task_id']}.json")
        )
        return task_marker.is_file()

    def pause_requested(self) -> bool:
        task_marker = (
            self.queue.directory(CONTROL)
            / "pause"
            / f"{_safe_identifier(str(self.task['task_id']), 'task')}.json"
        )
        return task_marker.is_file()

    def raise_if_interrupted(self) -> None:
        if self.cancellation_requested():
            raise RuntimeError("Task cancellation was requested")
        if self.pause_requested():
            raise TaskPauseRequested("Task pause was requested")

    def prepare_chunk(self, chunk_id: str) -> Path:
        """Return an empty host-local directory for one atomic work unit."""

        safe_id = _safe_identifier(chunk_id, "chunk")
        path = self.local_chunk_staging / safe_id
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def committed_chunk(self, chunk_id: str) -> dict[str, Any]:
        """Return a verified durable chunk, or an empty mapping."""

        safe_id = _safe_identifier(chunk_id, "chunk")
        root = self.chunk_root / safe_id
        marker = root / ".darsia-chunk.json"
        try:
            record = _read_json(marker)
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            return {}
        if (
            record.get("format") != CHUNK_MANIFEST_FORMAT
            or record.get("task_sha256") != self.task.get("content_sha256")
            or record.get("chunk_id") != chunk_id
        ):
            return {}
        actual = _chunk_artifact_manifest(root)
        if actual.get("content_sha256") != (
            record.get("artifacts") or {}
        ).get("content_sha256"):
            return {}
        return {**record, "root": str(root)}

    def commit_chunk(
        self,
        chunk_id: str,
        source: Path,
        *,
        state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Checksum, copy, and atomically expose one durable partial result."""

        source = Path(source).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Chunk staging directory is unavailable: {source}")
        safe_id = _safe_identifier(chunk_id, "chunk")
        destination = self.chunk_root / safe_id
        source_manifest = _chunk_artifact_manifest(source)
        record = {
            "schema": 1,
            "format": CHUNK_MANIFEST_FORMAT,
            "task_id": str(self.task["task_id"]),
            "task_sha256": str(self.task["content_sha256"]),
            "chunk_id": str(chunk_id),
            "committed_utc": _utc_now(),
            "worker_id": self.worker_id,
            "state": deepcopy(dict(state or {})),
            "artifacts": source_manifest,
        }
        record["content_sha256"] = _json_digest(record)

        def publish_chunk() -> None:
            existing = self.committed_chunk(chunk_id)
            if existing:
                if (
                    (existing.get("artifacts") or {}).get("content_sha256")
                    != source_manifest["content_sha256"]
                ):
                    raise RuntimeError(
                        f"Committed chunk differs for immutable chunk {chunk_id}"
                    )
                return
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.parent / f".{safe_id}.{uuid.uuid4().hex}.incoming"
            try:
                shutil.copytree(source, temporary)
                copied = _chunk_artifact_manifest(temporary)
                if copied["content_sha256"] != source_manifest["content_sha256"]:
                    raise OSError("Copied chunk checksum does not match local staging")
                _atomic_json(temporary / ".darsia-chunk.json", record)
                if destination.exists():
                    shutil.rmtree(destination)
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary, ignore_errors=True)

        _retry_storage_io(publish_chunk, f"commit chunk {chunk_id}")
        return self.committed_chunk(chunk_id)

    def restore_chunk(self, chunk_id: str, destination: Path) -> dict[str, Any]:
        """Copy a verified chunk into final local staging without its marker."""

        record = self.committed_chunk(chunk_id)
        if not record:
            raise FileNotFoundError(f"No verified chunk exists: {chunk_id}")
        root = Path(str(record["root"]))
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for row in (record.get("artifacts") or {}).get("files") or []:
            relative = Path(str(row["path"]))
            source = root / relative
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if sha256_file(target) != str(row["sha256"]):
                raise OSError(f"Restored chunk failed checksum verification: {relative}")
        return record

    def committed_chunks(self) -> dict[str, dict[str, Any]]:
        if not self.chunk_root.is_dir():
            return {}
        output: dict[str, dict[str, Any]] = {}
        for path in sorted(candidate for candidate in self.chunk_root.iterdir() if candidate.is_dir()):
            marker = path / ".darsia-chunk.json"
            try:
                raw = _read_json(marker)
            except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
                continue
            chunk_id = str(raw.get("chunk_id") or "")
            verified = self.committed_chunk(chunk_id)
            if verified:
                output[chunk_id] = verified
        return output


class ExecutionAdapter(Protocol):
    descriptor: AdapterDescriptor

    def execute(
        self,
        task: Mapping[str, Any],
        context: TaskExecutionContext,
    ) -> Mapping[str, Any]: ...


class AdapterRegistry:
    """Trusted adapters installed in this Python environment."""

    def __init__(self) -> None:
        self._adapters: dict[str, ExecutionAdapter] = {}

    def register(self, adapter: ExecutionAdapter) -> None:
        descriptor = adapter.descriptor
        if not descriptor.key or not descriptor.version:
            raise ValueError("Execution adapters require a key and version")
        if descriptor.key in self._adapters:
            raise ValueError(f"Duplicate execution adapter: {descriptor.key}")
        unknown = set(descriptor.backends) - set(BACKEND_CONTRACTS)
        if unknown:
            raise ValueError(
                f"Adapter {descriptor.key} uses unknown backends: {unknown}"
            )
        self._adapters[descriptor.key] = adapter

    def get(self, key: str) -> ExecutionAdapter:
        try:
            return self._adapters[str(key)]
        except KeyError as exc:
            raise KeyError(f"Execution adapter is not installed: {key}") from exc

    def descriptors(self) -> tuple[AdapterDescriptor, ...]:
        return tuple(self._adapters[key].descriptor for key in sorted(self._adapters))

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def supports(self, key: str, module: str, backend: str) -> bool:
        try:
            descriptor = self.get(key).descriptor
        except KeyError:
            return False
        return module in descriptor.modules and backend in descriptor.backends


class PlanVerificationAdapter:
    """Built-in adapter proving the full queue and publication path."""

    descriptor = AdapterDescriptor(
        key="plan-verification-v1",
        version="1.0.0",
        title="Immutable plan verification",
        modules=(
            "image_correction",
            "calibration",
            "mass_analysis",
            "wasserstein",
            "post_analysis",
        ),
        backends=("cpu-float64",),
        description="Verifies plan and project fingerprints and publishes provenance.",
    )

    def execute(
        self,
        task: Mapping[str, Any],
        context: TaskExecutionContext,
    ) -> Mapping[str, Any]:
        plan, plan_path, actual_file_sha256 = load_task_plan(task)
        project_root = Path(str(task["project"]["root"])).expanduser().resolve()
        series_path = project_root / "series.json"
        source_manifest_path = project_root / "source_manifest.json"
        expected_source_sha256 = str(
            (task.get("project") or {}).get("source_manifest_sha256") or ""
        )
        if expected_source_sha256:
            if not source_manifest_path.is_file():
                raise FileNotFoundError(
                    f"Project source manifest is unavailable: {source_manifest_path}"
                )
            if sha256_file(source_manifest_path) != expected_source_sha256:
                raise RuntimeError(
                    "Project source manifest changed after task creation"
                )
        context.checkpoint({"stage": "fingerprints_verified"})
        verification = {
            "schema": 1,
            "verified_utc": _utc_now(),
            "task_id": str(task["task_id"]),
            "plan_path": str(plan_path),
            "plan_file_sha256": actual_file_sha256,
            "plan_content_sha256": str(plan.get("content_sha256") or ""),
            "series_sha256": sha256_file(series_path) if series_path.is_file() else "",
            "source_manifest_sha256": (
                sha256_file(source_manifest_path)
                if source_manifest_path.is_file()
                else ""
            ),
            "backend": deepcopy(dict(task.get("backend") or {})),
        }
        _atomic_json(context.staging / "verification.json", verification)
        _atomic_json(context.staging / "task.json", task)
        _atomic_json(context.staging / "plan.json", plan)
        _atomic_json(context.staging / "hardware.json", context.inventory)
        context.checkpoint({"stage": "artifacts_ready"})
        return {
            "verification": verification,
            "scientific_results_created": False,
        }


def default_adapter_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register(PlanVerificationAdapter())
    from darsia.fluidflower.image_correction_materialization import (
        ImageCorrectionAdapter,
    )
    from darsia.fluidflower.mass_materialization import (
        AcceleratedMassAnalysisAdapter,
        CpuFloat32MassAnalysisAdapter,
        MassAnalysisAdapter,
    )
    from darsia.fluidflower.post_materialization import PostAnalysisAdapter
    from darsia.fluidflower.wasserstein_materialization import (
        WassersteinAnalysisAdapter,
    )

    registry.register(ImageCorrectionAdapter())
    registry.register(MassAnalysisAdapter())
    registry.register(CpuFloat32MassAnalysisAdapter())
    registry.register(AcceleratedMassAnalysisAdapter())
    registry.register(PostAnalysisAdapter())
    registry.register(WassersteinAnalysisAdapter())
    try:
        entry_points = importlib.metadata.entry_points(
            group="darsia.fluidflower.execution_adapters"
        )
    except TypeError:
        entry_points = importlib.metadata.entry_points().select(
            group="darsia.fluidflower.execution_adapters"
        )
    for entry_point in entry_points:
        try:
            loaded = entry_point.load()
            adapter = loaded() if isinstance(loaded, type) else loaded
            registry.register(adapter)
        except Exception:  # noqa: BLE001
            LOG.exception("Could not load execution adapter %s", entry_point.name)
    return registry


def _seal_task(payload: Mapping[str, Any]) -> dict[str, Any]:
    task = deepcopy(dict(payload))
    task.pop("content_sha256", None)
    task["content_sha256"] = _json_digest(task)
    return task


def validate_task_document(task: Mapping[str, Any]) -> None:
    if task.get("format") != TASK_DOCUMENT_FORMAT or int(task.get("schema") or 0) != 1:
        raise ValueError("Unsupported FluidFlower task document")
    task_id = str(task.get("task_id") or "")
    if not task_id:
        raise ValueError("Task document has no task_id")
    expected = str(task.get("content_sha256") or "")
    candidate = deepcopy(dict(task))
    candidate.pop("content_sha256", None)
    if expected != _json_digest(candidate):
        raise ValueError(f"Task document fingerprint mismatch: {task_id}")
    backend = str((task.get("backend") or {}).get("name") or "")
    if backend not in BACKEND_CONTRACTS:
        raise ValueError(f"Unknown task backend: {backend}")
    expected_backend = asdict(BACKEND_CONTRACTS[backend])
    if dict(task.get("backend") or {}) != expected_backend:
        raise ValueError(f"Backend contract mismatch for {task_id}")


def load_task_plan(task: Mapping[str, Any]) -> tuple[dict[str, Any], Path, str]:
    """Load and verify the exact immutable plan referenced by a task."""

    plan_reference = dict(task.get("plan_reference") or {})
    plan_path = Path(str(plan_reference.get("path") or "")).expanduser().resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"Saved plan is unavailable: {plan_path}")
    actual_file_sha256 = sha256_file(plan_path)
    expected_file_sha256 = str(plan_reference.get("file_sha256") or "")
    if expected_file_sha256 and actual_file_sha256 != expected_file_sha256:
        raise RuntimeError("Saved plan changed after the task was created")
    plan = _read_json(plan_path)
    expected_content = str(plan_reference.get("content_sha256") or "")
    if expected_content and str(plan.get("content_sha256") or "") != expected_content:
        raise RuntimeError("Saved plan content contract no longer matches the task")
    contract = dict(plan.get("task_contract") or plan.get("objective_contract") or {})
    expected_contract = str(plan_reference.get("contract_sha256") or "")
    if (
        expected_contract
        and str(contract.get("content_sha256") or "") != expected_contract
    ):
        raise RuntimeError("Saved plan task contract no longer matches the task")
    return plan, plan_path, actual_file_sha256


def verify_file_reference(
    path_value: str | Path,
    expected_sha256: str,
    *,
    label: str = "Referenced file",
    relative_to: Path | None = None,
) -> Path:
    """Resolve and hash-check one immutable file across transient storage loss."""

    path = Path(path_value).expanduser()
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    path = path.resolve()

    def verify() -> Path:
        if not path.is_file():
            raise FileNotFoundError(f"{label} is unavailable: {path}")
        actual = sha256_file(path)
        if expected_sha256 and actual != expected_sha256:
            raise RuntimeError(f"{label} failed its SHA-256 check: {path}")
        return path

    return _retry_storage_io(verify, f"verify {label}: {path}")


def build_task_document(
    *,
    project: FluidFlowerProject,
    module: str,
    operation: str,
    adapter: AdapterDescriptor,
    plan_reference: Mapping[str, Any],
    backend: str,
    resource_request: Mapping[str, Any],
    output_root: Path,
    created_by: str,
    priority: int = 50,
    max_attempts: int = 3,
    adapter_settings: Mapping[str, Any] | None = None,
    rerun_nonce: str = "",
) -> dict[str, Any]:
    """Build an immutable and idempotent task for an already saved plan."""

    if backend not in BACKEND_CONTRACTS:
        raise ValueError(f"Unknown execution backend: {backend}")
    if module not in adapter.modules or backend not in adapter.backends:
        raise ValueError(
            f"Adapter {adapter.key} does not support {module} on {backend}"
        )
    source_manifest_path = project.root / "source_manifest.json"
    project_reference = {
        "project_id": project.project_id,
        "series_id": str(project.manifest.get("series_id") or ""),
        "root": str(project.root),
        "series_snapshot_sha256": (
            sha256_file(project.manifest_path)
            if project.manifest_path.is_file()
            else ""
        ),
        "source_manifest_sha256": (
            sha256_file(source_manifest_path) if source_manifest_path.is_file() else ""
        ),
    }
    identity = {
        "project_id": project.project_id,
        "source_manifest_sha256": project_reference["source_manifest_sha256"],
        "module": str(module),
        "operation": str(operation),
        "adapter": {"key": adapter.key, "version": adapter.version},
        "plan_reference": deepcopy(dict(plan_reference)),
        "backend": asdict(BACKEND_CONTRACTS[backend]),
        "adapter_settings": deepcopy(dict(adapter_settings or {})),
        "rerun_nonce": str(rerun_nonce),
    }
    idempotency_key = _json_digest(identity)
    task_id = f"{_safe_identifier(module, 'task')}-{idempotency_key[:24]}"
    destination = Path(output_root).expanduser().resolve() / task_id
    payload = {
        "schema": 1,
        "format": TASK_DOCUMENT_FORMAT,
        "task_id": task_id,
        "idempotency_key": idempotency_key,
        "created_utc": utc_now(),
        "created_by": str(created_by).strip(),
        "priority": max(0, min(999, int(priority))),
        "maximum_attempts": max(1, int(max_attempts)),
        "project": project_reference,
        "module": str(module),
        "operation": str(operation),
        "plan_reference": deepcopy(dict(plan_reference)),
        "adapter": {
            "key": adapter.key,
            "version": adapter.version,
            "settings": deepcopy(dict(adapter_settings or {})),
        },
        "backend": asdict(BACKEND_CONTRACTS[backend]),
        "resources": deepcopy(dict(resource_request)),
        "output": {
            "destination": str(destination),
            "atomic_publish": True,
            "remove_local_staging_after_publish": True,
        },
    }
    return _seal_task(payload)


def _plan_store(project: FluidFlowerProject, module: str) -> Any:
    if module == "image_correction":
        from darsia.fluidflower.image_correction import CorrectionRevisionStore

        return CorrectionRevisionStore(project)
    if module == "calibration":
        from darsia.fluidflower.calibration import CalibrationPlanStore

        return CalibrationPlanStore(project)
    if module == "mass_analysis":
        from darsia.fluidflower.mass_analysis import MassAnalysisPlanStore

        return MassAnalysisPlanStore(project)
    if module == "wasserstein":
        from darsia.fluidflower.wasserstein import WassersteinPlanStore

        return WassersteinPlanStore(project)
    if module == "post_analysis":
        from darsia.fluidflower.post_analysis import PostAnalysisPlanStore

        return PostAnalysisPlanStore(project)
    raise ValueError(f"Module has no immutable execution plan: {module}")


def current_plan_reference(
    project: FluidFlowerProject,
    module: str,
) -> dict[str, Any]:
    store = _plan_store(project, module)
    plan = store.current()
    if not plan:
        raise FileNotFoundError(f"No saved {module} plan exists")
    plan_id = str(plan.get("plan_id") or "")
    if not plan_id:
        raise ValueError(f"The saved {module} plan has no plan_id")
    path = Path(store.artifact_root) / plan_id / "plan.json"
    if not path.is_file():
        raise FileNotFoundError(f"Immutable plan artifact is unavailable: {path}")
    contract = dict(plan.get("task_contract") or plan.get("objective_contract") or {})
    reference = {
        "plan_id": plan_id,
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "content_sha256": str(plan.get("content_sha256") or ""),
        "contract_format": str(contract.get("format") or ""),
        "contract_sha256": str(contract.get("content_sha256") or ""),
    }
    configuration = dict(plan.get("configuration") or {})
    quality_profile = dict(
        contract.get("quality_profile")
        or configuration.get("quality_profile")
        or {}
    )
    backend = str(
        (contract.get("execution_profile") or {}).get("primary_backend")
        or (configuration.get("execution_profile") or {}).get("primary_backend")
        or ""
    )
    if quality_profile and backend:
        reference["quality_contract"] = summarize_quality_contract(
            quality_profile,
            backend=backend,
        )
    return reference


def verification_task_for_current_plan(
    project: FluidFlowerProject,
    module: str,
    *,
    created_by: str,
    output_root: Path | None = None,
    priority: int = 50,
) -> dict[str, Any]:
    registry = default_adapter_registry()
    adapter = registry.get("plan-verification-v1").descriptor
    inventory = detect_hardware()
    recommendation = recommend_resources(module, inventory)
    resources = {
        **recommendation,
        "selected_backend": "cpu-float64",
        "cpu_workers": 1,
        "accelerator_workers": 0,
        "preparation_workers": 1,
    }
    resources.pop("content_sha256", None)
    resources["content_sha256"] = _json_digest(resources)
    return build_task_document(
        project=project,
        module=module,
        operation="verify_saved_plan",
        adapter=adapter,
        plan_reference=current_plan_reference(project, module),
        backend="cpu-float64",
        resource_request=resources,
        output_root=(
            Path(output_root)
            if output_root is not None
            else project.root / "results" / "execution" / "verified_plans"
        ),
        created_by=created_by,
        priority=priority,
    )


def task_for_current_plan(
    project: FluidFlowerProject,
    module: str,
    *,
    adapter_key: str,
    backend: str,
    created_by: str,
    resources: Mapping[str, Any] | None = None,
    output_root: Path | None = None,
    priority: int = 50,
    registry: AdapterRegistry | None = None,
    adapter_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an execution task using an installed, trusted adapter."""

    registry = registry or default_adapter_registry()
    descriptor = registry.get(adapter_key).descriptor
    resource_request = deepcopy(dict(resources or {}))
    if not resource_request:
        resource_request = recommend_resources(module, detect_hardware())
    plan_reference = current_plan_reference(project, module)
    resolved_output_root = Path(output_root) if output_root is not None else None
    if resolved_output_root is None and adapter_key != "plan-verification-v1":
        plan = _read_json(Path(str(plan_reference["path"])))
        contract = dict(plan.get("task_contract") or {})
        planned_root = str(contract.get("output_root") or "").strip()
        if planned_root:
            resolved_output_root = Path(planned_root)
    if resolved_output_root is None:
        resolved_output_root = (
            project.root
            / "results"
            / "execution"
            / _safe_identifier(adapter_key, "adapter")
        )
    return build_task_document(
        project=project,
        module=module,
        operation=(
            "verify_saved_plan"
            if adapter_key == "plan-verification-v1"
            else "execute_saved_plan"
        ),
        adapter=descriptor,
        plan_reference=plan_reference,
        backend=backend,
        resource_request=resource_request,
        output_root=resolved_output_root,
        created_by=created_by,
        priority=priority,
        adapter_settings=adapter_settings,
    )


class ExecutionQueue:
    """One-task-per-file queue for local disks and trusted SMB/NAS shares."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def directory(self, name: str) -> Path:
        return self.root / name

    @property
    def metadata_path(self) -> Path:
        return self.root / "queue.json"

    def ensure_layout(self) -> None:
        def ensure() -> None:
            self.root.mkdir(parents=True, exist_ok=True)
            for name in QUEUE_DIRECTORIES:
                self.directory(name).mkdir(parents=True, exist_ok=True)
            (self.directory(CONTROL) / "cancel").mkdir(parents=True, exist_ok=True)
            (self.directory(CONTROL) / "pause").mkdir(parents=True, exist_ok=True)
            (self.directory(CONTROL) / "stop").mkdir(parents=True, exist_ok=True)

        _retry_storage_io(ensure, f"execution queue layout {self.root}")
        if not self.metadata_path.is_file():
            _atomic_json(
                self.metadata_path,
                {
                    "schema": 1,
                    "format": QUEUE_FORMAT,
                    "created_utc": _utc_now(),
                    "trust_boundary": "explicitly-selected-local-or-network-share",
                    "arbitrary_commands_allowed": False,
                },
            )
        metadata = _read_json(self.metadata_path)
        if metadata.get("format") != QUEUE_FORMAT:
            raise ValueError(f"Unsupported execution queue: {self.root}")

    def _task_glob(self, task_id: str) -> Iterable[Path]:
        safe_task_id = _safe_identifier(task_id, "task")
        for name in (PENDING, IN_PROGRESS, PAUSED, COMPLETED, FAILED):
            yield from self.directory(name).glob(f"*{safe_task_id}*.json")

    def find_task(self, task_id: str) -> Path | None:
        self.ensure_layout()
        return next(self._task_glob(task_id), None)

    def enqueue(self, task: Mapping[str, Any]) -> bool:
        self.ensure_layout()
        validate_task_document(task)
        task_id = str(task["task_id"])
        if self.find_task(task_id) is not None:
            return False
        priority = max(0, min(999, int(task.get("priority") or 0)))
        rank = 999 - priority
        created = str(task.get("created_utc") or "").replace(":", "").replace("-", "")
        name = f"p{rank:03d}_{created[:15]}_{_safe_identifier(task_id)}.json"
        record = {
            "task": deepcopy(dict(task)),
            "attempts": 0,
            "failures": [],
            "enqueued_utc": _utc_now(),
        }
        destination = self.directory(PENDING) / name
        _atomic_json(destination, record)
        return True

    def counts(self) -> dict[str, int]:
        self.ensure_layout()

        def read_counts() -> dict[str, int]:
            return {
                name: sum(1 for _ in self.directory(name).glob("*.json"))
                for name in (PENDING, IN_PROGRESS, PAUSED, COMPLETED, FAILED)
            }

        return _retry_storage_io(read_counts, f"queue counts {self.root}")

    def heartbeats(self) -> list[dict[str, Any]]:
        self.ensure_layout()
        rows: list[dict[str, Any]] = []
        for path in sorted(self.directory(HEARTBEATS).glob("*.json")):
            try:
                rows.append(_read_json(path))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return rows

    def snapshot(self, *, maximum_tasks: int = 250) -> dict[str, Any]:
        """Return a compact monitoring snapshot without changing queue state."""

        self.ensure_layout()
        tasks: list[dict[str, Any]] = []
        for status in (IN_PROGRESS, PENDING, PAUSED, FAILED, COMPLETED):
            for path in sorted(
                self.directory(status).glob("*.json"),
                key=lambda candidate: candidate.stat().st_mtime,
                reverse=True,
            ):
                if len(tasks) >= max(1, int(maximum_tasks)):
                    break
                try:
                    record = _read_json(path)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                task = dict(record.get("task") or {})
                result = dict(record.get("result") or {})
                tasks.append(
                    {
                        "status": status,
                        "task_id": str(task.get("task_id") or ""),
                        "module": str(task.get("module") or ""),
                        "operation": str(task.get("operation") or ""),
                        "backend": str((task.get("backend") or {}).get("name") or ""),
                        "adapter": str((task.get("adapter") or {}).get("key") or ""),
                        "quality_contract": deepcopy(
                            dict(
                                (task.get("plan_reference") or {}).get(
                                    "quality_contract"
                                )
                                or {}
                            )
                        ),
                        "attempts": int(record.get("attempts") or 0),
                        "created_utc": str(task.get("created_utc") or ""),
                        "worker_id": str(
                            record.get("worker_id")
                            or (result.get("worker") or {}).get("worker_id")
                            or ""
                        ),
                        "runtime_seconds": result.get("runtime_seconds"),
                        "destination": str(
                            result.get("destination")
                            or (task.get("output") or {}).get("destination")
                            or ""
                        ),
                        "path": str(path),
                    }
                )
        tasks.sort(key=lambda row: str(row.get("created_utc") or ""), reverse=True)
        return {
            "root": str(self.root),
            "counts": self.counts(),
            "heartbeats": self.heartbeats(),
            "tasks": tasks[: max(1, int(maximum_tasks))],
        }

    def _claim_marker(self, source: Path) -> Path:
        return self.directory(CLAIMS) / f"{source.stem}.claim"

    def _remove_orphaned_marker(self, source: Path, marker: Path) -> bool:
        try:
            age = time.time() - marker.stat().st_mtime
        except FileNotFoundError:
            return True
        if age < ORPHANED_CLAIM_SECONDS:
            return False
        if any(self.directory(IN_PROGRESS).glob(f"{source.stem}__*.json")):
            return False
        try:
            marker.unlink()
            return True
        except (FileNotFoundError, PermissionError, OSError):
            return False

    def _acquire_marker(self, source: Path, worker_id: str) -> Path | None:
        marker = self._claim_marker(source)
        for attempt in range(2):
            try:
                descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if attempt == 0 and self._remove_orphaned_marker(source, marker):
                    continue
                return None
            except OSError:
                return None
            try:
                payload = json.dumps(
                    {
                        "worker_id": worker_id,
                        "source": source.name,
                        "claimed_utc": _utc_now(),
                    }
                ).encode("utf-8")
                os.write(descriptor, payload)
            finally:
                os.close(descriptor)
            return marker
        return None

    def claim(
        self,
        worker_id: str,
        *,
        backends: Sequence[str],
        adapters: Sequence[str],
    ) -> "TaskLease | None":
        self.ensure_layout()
        backend_set = set(backends)
        adapter_set = set(adapters)
        try:
            entries = sorted(self.directory(PENDING).glob("*.json"))
        except OSError:
            return None
        for source in entries:
            try:
                record = _read_json(source)
                task = dict(record.get("task") or {})
                validate_task_document(task)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            backend = str((task.get("backend") or {}).get("name") or "")
            adapter = str((task.get("adapter") or {}).get("key") or "")
            if backend not in backend_set or adapter not in adapter_set:
                continue
            marker = self._acquire_marker(source, worker_id)
            if marker is None:
                continue
            claimed = self.directory(IN_PROGRESS) / (
                f"{source.stem}__{_safe_identifier(worker_id)}.json"
            )
            try:
                os.replace(source, claimed)
            except OSError:
                try:
                    marker.unlink()
                except OSError:
                    pass
                continue
            return TaskLease(
                queue=self,
                path=claimed,
                marker=marker,
                worker_id=worker_id,
                record=record,
            )
        return None

    def write_heartbeat(
        self,
        worker_id: str,
        status: str,
        *,
        task_id: str = "",
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        payload = {
            "schema": 1,
            "worker_id": worker_id,
            "host": _host_name(),
            "pid": os.getpid(),
            "status": status,
            "task_id": task_id,
            "recorded_utc": _utc_now(),
            "recorded_epoch": time.time(),
        }
        payload.update(dict(extra or {}))
        try:
            _atomic_json(
                self.directory(HEARTBEATS) / f"{_safe_identifier(worker_id)}.json",
                payload,
            )
        except OSError:
            # An active calculation should survive a short share outage.  The
            # lease file is also touched and completion publishing is retried.
            pass

    def request_task_cancellation(self, task_id: str, *, actor: str = "") -> None:
        self.ensure_layout()
        _atomic_json(
            self.directory(CONTROL) / "cancel" / f"{_safe_identifier(task_id)}.json",
            {"task_id": task_id, "actor": actor, "requested_utc": _utc_now()},
        )

    def request_task_pause(self, task_id: str, *, actor: str = "") -> None:
        """Pause pending work immediately or active work at its next chunk boundary."""

        self.ensure_layout()
        safe_id = _safe_identifier(task_id, "task")
        _atomic_json(
            self.directory(CONTROL) / "pause" / f"{safe_id}.json",
            {"task_id": task_id, "actor": actor, "requested_utc": _utc_now()},
        )
        for source in self.directory(PENDING).glob(f"*{safe_id}*.json"):
            destination = self.directory(PAUSED) / source.name
            try:
                os.replace(source, destination)
            except (FileNotFoundError, OSError):
                continue

    def resume_task(self, task_id: str, *, actor: str = "") -> bool:
        """Return a paused immutable task to the pending queue."""

        self.ensure_layout()
        safe_id = _safe_identifier(task_id, "task")
        marker = self.directory(CONTROL) / "pause" / f"{safe_id}.json"
        try:
            marker.unlink()
        except FileNotFoundError:
            pass
        for source in self.directory(PAUSED).glob(f"*{safe_id}*.json"):
            record = _read_json(source)
            record["resumed_utc"] = _utc_now()
            record["resumed_by"] = str(actor).strip()
            destination = self.directory(PENDING) / source.name
            _atomic_json(destination, record)
            source.unlink()
            return True
        return False

    def request_worker_stop(self, worker_id: str, *, actor: str = "") -> None:
        self.ensure_layout()
        _atomic_json(
            self.directory(CONTROL) / "stop" / f"{_safe_identifier(worker_id)}.json",
            {"worker_id": worker_id, "actor": actor, "requested_utc": _utc_now()},
        )

    def requeue_stale(self, lease_timeout_seconds: float) -> int:
        self.ensure_layout()
        cutoff = time.time() - max(1.0, float(lease_timeout_seconds))
        moved = 0
        for path in self.directory(IN_PROGRESS).glob("*.json"):
            try:
                if path.stat().st_mtime > cutoff:
                    continue
                record = _read_json(path)
            except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
                continue
            worker_id = path.stem.rsplit("__", 1)[-1]
            heartbeat_path = self.directory(HEARTBEATS) / f"{worker_id}.json"
            try:
                heartbeat = _read_json(heartbeat_path)
                if float(heartbeat.get("recorded_epoch") or 0.0) > cutoff:
                    continue
            except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
                pass
            lease = TaskLease(
                queue=self,
                path=path,
                marker=self.directory(CLAIMS) / f"{path.stem.split('__', 1)[0]}.claim",
                worker_id=worker_id,
                record=record,
            )
            lease.fail("Lease expired; task recovered for work stealing", stale=True)
            moved += 1
        return moved


@dataclass
class TaskLease:
    queue: ExecutionQueue
    path: Path
    marker: Path
    worker_id: str
    record: dict[str, Any]

    @property
    def task(self) -> dict[str, Any]:
        return dict(self.record.get("task") or {})

    def touch(self) -> None:
        try:
            os.utime(self.path, None)
        except (FileNotFoundError, OSError):
            pass

    def _remove_claim(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        try:
            self.marker.unlink()
        except FileNotFoundError:
            pass

    def complete(self, result: Mapping[str, Any]) -> None:
        task_id = str(self.task["task_id"])
        destination = self.queue.directory(COMPLETED) / f"{task_id}.json"
        payload = {
            "task": self.task,
            "attempts": int(self.record.get("attempts") or 0) + 1,
            "failures": deepcopy(list(self.record.get("failures") or [])),
            "result": deepcopy(dict(result)),
            "worker_id": self.worker_id,
            "finished_utc": _utc_now(),
        }
        if destination.is_file():
            existing = _read_json(destination)
            existing_result = dict(existing.get("result") or {})
            if existing_result.get("task_sha256") != self.task.get("content_sha256"):
                raise RuntimeError(f"Conflicting completed result for {task_id}")
        else:
            _atomic_json(destination, payload)
        self._remove_claim()

    def pause(self, reason: str = "Pause requested") -> None:
        """Preserve attempt accounting and move this lease to paused."""

        original_name = self.path.name.rsplit("__", 1)[0] + ".json"
        record = {
            **deepcopy(self.record),
            "paused_utc": _utc_now(),
            "paused_by_worker": self.worker_id,
            "pause_reason": str(reason),
        }
        _atomic_json(self.queue.directory(PAUSED) / original_name, record)
        self._remove_claim()

    def fail(self, error: str, *, stale: bool = False) -> None:
        task = self.task
        attempts = int(self.record.get("attempts") or 0) + 1
        failures = deepcopy(list(self.record.get("failures") or []))
        failures.append(
            {
                "attempt": attempts,
                "worker_id": self.worker_id,
                "failed_utc": _utc_now(),
                "error": str(error),
                "stale_lease": bool(stale),
            }
        )
        record = {
            "task": task,
            "attempts": attempts,
            "failures": failures,
            "enqueued_utc": self.record.get("enqueued_utc"),
        }
        maximum_attempts = max(1, int(task.get("maximum_attempts") or 1))
        if attempts >= maximum_attempts:
            destination = self.queue.directory(FAILED) / f"{task['task_id']}.json"
            _atomic_json(
                destination,
                {
                    **record,
                    "format": TASK_FAILURE_FORMAT,
                    "failed_utc": _utc_now(),
                },
            )
        else:
            original_name = self.path.name.rsplit("__", 1)[0] + ".json"
            _atomic_json(self.queue.directory(PENDING) / original_name, record)
        self._remove_claim()


def _directory_artifact_manifest(
    root: Path,
    *,
    ignored_names: Sequence[str] = (),
) -> dict[str, Any]:
    root = Path(root)
    ignored = set(ignored_names)
    files: list[dict[str, Any]] = []
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        if path.name in ignored:
            continue
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {"files": files, "file_count": len(files)}
    payload["content_sha256"] = _json_digest(payload)
    return payload


def _chunk_artifact_manifest(root: Path) -> dict[str, Any]:
    return _directory_artifact_manifest(
        root,
        ignored_names=(".darsia-chunk.json",),
    )


def artifact_manifest(root: Path) -> dict[str, Any]:
    return _directory_artifact_manifest(
        root,
        ignored_names=(".darsia-publish.json",),
    )


def publish_directory_atomic(
    staging: Path,
    destination: Path,
    *,
    task_id: str,
    task_sha256: str,
) -> dict[str, Any]:
    """Copy, verify, and atomically expose a result directory."""

    staging = Path(staging).resolve()
    destination = Path(destination).expanduser().resolve()
    if not staging.is_dir():
        raise FileNotFoundError(f"Task staging directory is unavailable: {staging}")
    source_manifest = artifact_manifest(staging)
    manifest_payload = {
        "schema": 1,
        "format": PUBLISH_MANIFEST_FORMAT,
        "task_id": task_id,
        "task_sha256": task_sha256,
        "published_utc": _utc_now(),
        "artifacts": source_manifest,
    }
    if destination.is_dir():
        published_path = destination / ".darsia-publish.json"
        existing = _read_json(published_path)
        if (
            existing.get("task_id") == task_id
            and existing.get("task_sha256") == task_sha256
        ):
            # A stale worker may finish after another host has already won the
            # publish race.  The first verified result for this immutable task
            # remains authoritative; later workers reuse it rather than
            # replacing or merging files.
            return existing
        raise FileExistsError(
            f"Result destination already contains another task: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    def publish_once() -> None:
        temporary = (
            destination.parent / f".{destination.name}.{uuid.uuid4().hex}.incoming"
        )
        try:
            shutil.copytree(staging, temporary)
            copied_manifest = artifact_manifest(temporary)
            if copied_manifest["content_sha256"] != source_manifest["content_sha256"]:
                raise OSError("Copied artifact checksum does not match local staging")
            _atomic_json(temporary / ".darsia-publish.json", manifest_payload)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)

    _retry_storage_io(publish_once, f"publish task {task_id} to {destination}")
    return _read_json(destination / ".darsia-publish.json")


def _worker_stop_requested(queue: ExecutionQueue, worker_id: str) -> bool:
    marker = queue.directory(CONTROL) / "stop" / f"{_safe_identifier(worker_id)}.json"
    return marker.is_file()


def execute_task_lease(
    lease: TaskLease,
    *,
    registry: AdapterRegistry,
    local_cache_root: Path,
    inventory: Mapping[str, Any],
    heartbeat_seconds: float = 30.0,
) -> dict[str, Any]:
    from darsia._opencv_runtime import scientific_thread, completed_frame
    ipp_before = scientific_thread("execute_task_lease:before")
    task = lease.task
    validate_task_document(task)
    task_id = str(task["task_id"])
    adapter_key = str((task.get("adapter") or {}).get("key") or "")
    backend = str((task.get("backend") or {}).get("name") or "")
    adapter = registry.get(adapter_key)
    if not registry.supports(adapter_key, str(task["module"]), backend):
        raise RuntimeError(
            f"Adapter {adapter_key} does not support {task['module']} on {backend}"
        )
    attempt = int(lease.record.get("attempts") or 0) + 1
    workspace = (
        Path(local_cache_root).expanduser().resolve()
        / task_id
        / f"attempt-{attempt:03d}"
    )
    workspace.mkdir(parents=True, exist_ok=True)
    context = TaskExecutionContext(
        task=task,
        queue=lease.queue,
        worker_id=lease.worker_id,
        workspace=workspace,
        inventory=inventory,
    )
    stop_heartbeat = threading.Event()

    def pulse() -> None:
        interval = max(0.2, float(heartbeat_seconds))
        while not stop_heartbeat.wait(interval):
            lease.touch()
            lease.queue.write_heartbeat(
                lease.worker_id,
                "busy",
                task_id=task_id,
                extra={"attempt": attempt, "backend": backend},
            )

    lease.touch()
    lease.queue.write_heartbeat(
        lease.worker_id,
        "busy",
        task_id=task_id,
        extra={"attempt": attempt, "backend": backend},
    )
    heartbeat = threading.Thread(
        target=pulse,
        name=f"heartbeat-{lease.worker_id}",
        daemon=True,
    )
    heartbeat.start()
    started = time.time()
    try:
        adapter_result = dict(adapter.execute(task, context))
        if context.cancellation_requested():
            raise RuntimeError("Task cancellation was requested")
        ipp_runtime = completed_frame(ipp_before, "execute_task_lease:before-publication")
        destination = Path(str((task.get("output") or {})["destination"]))
        publish_manifest = publish_directory_atomic(
            context.staging,
            destination,
            task_id=task_id,
            task_sha256=str(task["content_sha256"]),
        )
        result: dict[str, Any] = {
            "schema": 1,
            "format": TASK_RESULT_FORMAT,
            "task_id": task_id,
            "task_sha256": str(task["content_sha256"]),
            "adapter": {
                "key": adapter.descriptor.key,
                "version": adapter.descriptor.version,
            },
            "backend": deepcopy(dict(task["backend"])),
            "worker": {
                "worker_id": lease.worker_id,
                "host": _host_name(),
                "pid": os.getpid(),
                "hardware_inventory_sha256": str(inventory.get("content_sha256") or ""),
            },
            "attempt": attempt,
            "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "finished_utc": _utc_now(),
            "runtime_seconds": time.time() - started,
            "destination": str(destination),
            "publish_manifest": publish_manifest,
            "adapter_result": adapter_result,
            "opencv_ipp": ipp_runtime,
            "opencv_ipp_scope": "current thread and new computations; restored chunks retain original provenance",
        }
        result["content_sha256"] = _json_digest(result)
        lease.complete(result)
        if bool((task.get("output") or {}).get("remove_local_staging_after_publish")):
            shutil.rmtree(workspace, ignore_errors=True)
        if context.partial_root.exists():
            shutil.rmtree(context.partial_root, ignore_errors=True)
        return result
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=2.0)


def run_worker(
    queue: ExecutionQueue,
    *,
    registry: AdapterRegistry | None = None,
    local_cache_root: Path,
    worker_id: str,
    backends: Sequence[str],
    poll_seconds: float = 2.0,
    heartbeat_seconds: float = 30.0,
    idle_exit_seconds: float | None = None,
    maximum_tasks: int | None = None,
) -> int:
    from darsia._opencv_runtime import scientific_thread
    scientific_thread("legacy-worker:start")
    from darsia_cache.adapters import bind_legacy_worker_identity
    bind_legacy_worker_identity(worker_id)
    registry = registry or default_adapter_registry()
    queue.ensure_layout()
    inventory = detect_hardware()
    available = set(inventory.get("available_backends") or [])
    requested = tuple(dict.fromkeys(str(value) for value in backends))
    unsupported = set(requested) - available
    if unsupported:
        raise RuntimeError(
            "Worker requested unavailable backends: " + ", ".join(sorted(unsupported))
        )
    completed = 0
    idle_since = time.time()
    while not _worker_stop_requested(queue, worker_id):
        lease = queue.claim(
            worker_id,
            backends=requested,
            adapters=registry.keys(),
        )
        if lease is None:
            queue.write_heartbeat(
                worker_id,
                "idle",
                extra={
                    "backends": list(requested),
                    "adapters": list(registry.keys()),
                    "completed": completed,
                },
            )
            if idle_exit_seconds is not None and (
                time.time() - idle_since >= float(idle_exit_seconds)
            ):
                return completed
            time.sleep(max(0.1, float(poll_seconds)))
            continue
        idle_since = time.time()
        try:
            execute_task_lease(
                lease,
                registry=registry,
                local_cache_root=local_cache_root,
                inventory=inventory,
                heartbeat_seconds=heartbeat_seconds,
            )
            completed += 1
        except TaskPauseRequested as exc:
            LOG.info("Task %s paused after a committed chunk", lease.task.get("task_id"))
            try:
                lease.pause(str(exc))
            except Exception:  # noqa: BLE001
                LOG.exception("Could not persist paused task state")
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Task %s failed", lease.task.get("task_id"))
            try:
                lease.fail(repr(exc))
            except Exception:  # noqa: BLE001
                LOG.exception("Could not persist task failure")
        if maximum_tasks is not None and completed >= maximum_tasks:
            return completed
    queue.write_heartbeat(worker_id, "stopped", extra={"completed": completed})
    return completed


def spawn_worker_processes(
    *,
    queue_root: Path,
    cache_root: Path,
    workers: int,
    backends: Sequence[str],
    idle_exit_seconds: float = 120.0,
) -> list[WorkerProcessHandle]:
    """Start hidden, independent workers suitable for GUI use on Windows."""

    from darsia_cache.adapters import cooperative_child_environment, record_cache_observation
    # This binds import inheritance only. Legacy workspace writers/cleanup below
    # remain explicitly unqualified; this hook is not an all-writer adapter.
    child_environment=cooperative_child_environment(parent_environment=os.environ)
    processes: list[WorkerProcessHandle] = []
    host = _safe_identifier(_host_name(), "host")
    for index in range(max(1, int(workers))):
        worker_id = f"{host}-{os.getpid()}-{index:02d}"
        command = [
            sys.executable,
            "-m",
            "darsia.fluidflower.execution",
            "worker",
            "--queue",
            str(Path(queue_root).expanduser().resolve()),
            "--cache",
            str(Path(cache_root).expanduser().resolve()),
            "--worker-id",
            worker_id,
            "--backends",
            *list(backends),
            "--idle-exit-seconds",
            str(float(idle_exit_seconds)),
        ]
        record_cache_observation("legacy_spawn_intent",dict(worker_id=worker_id))
        process = subprocess.Popen(command, env=child_environment, **_subprocess_options())
        handle=WorkerProcessHandle(worker_id=worker_id, process=process)
        processes.append(handle)  # Track before supplementary observation I/O.
        import psutil
        try:
            native=psutil.Process(process.pid)
            observed=dict(pid=native.pid,process_create_time=str(native.create_time()))
        except psutil.NoSuchProcess:
            observed=None
        try:
            record_cache_observation("legacy_spawn_observed",dict(worker_id=worker_id,
                original_native_identity=observed,epoch_unavailable=observed is None))
        except Exception as error:
            handle.cache_observation_error=dict(exception_type=type(error).__name__,
                message=str(error),observation_complete=False)
            try:
                LOG.error("Cache legacy spawn observation incomplete: %s",
                    handle.cache_observation_error,exc_info=True)
            except BaseException:
                handle.cache_observation_error["logging_failed"]=True
    return processes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DarSIA FluidFlower task runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--output", default="")
    initialize = subparsers.add_parser("init")
    initialize.add_argument("--queue", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--queue", required=True)
    worker.add_argument("--cache", required=True)
    worker.add_argument("--worker-id", required=True)
    worker.add_argument(
        "--backends",
        nargs="+",
        choices=tuple(BACKEND_CONTRACTS),
        default=["cpu-float64"],
    )
    worker.add_argument("--poll-seconds", type=float, default=2.0)
    worker.add_argument("--heartbeat-seconds", type=float, default=30.0)
    worker.add_argument("--idle-exit-seconds", type=float, default=None)
    worker.add_argument("--maximum-tasks", type=int, default=None)
    status = subparsers.add_parser("status")
    status.add_argument("--queue", required=True)
    recover = subparsers.add_parser("recover")
    recover.add_argument("--queue", required=True)
    recover.add_argument("--lease-timeout-seconds", type=float, default=900.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "inventory":
        payload = detect_hardware()
        if args.output:
            _atomic_json(Path(args.output).expanduser().resolve(), payload)
        else:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    queue = ExecutionQueue(Path(args.queue))
    if args.command == "init":
        queue.ensure_layout()
        print(str(queue.root))
        return 0
    if args.command == "status":
        print(
            json.dumps(
                {"counts": queue.counts(), "heartbeats": queue.heartbeats()},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "recover":
        print(queue.requeue_stale(args.lease_timeout_seconds))
        return 0
    if args.command == "worker":
        queue.ensure_layout()
        log_path = queue.directory(LOGS) / f"{_safe_identifier(args.worker_id)}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        )
        run_worker(
            queue,
            local_cache_root=Path(args.cache),
            worker_id=args.worker_id,
            backends=args.backends,
            poll_seconds=args.poll_seconds,
            heartbeat_seconds=args.heartbeat_seconds,
            idle_exit_seconds=args.idle_exit_seconds,
            maximum_tasks=args.maximum_tasks,
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
