"""Measured choice of the GPU device for accelerated mass analysis.

When several GPU devices could run an accelerated mass job, DarSIA times each
one on the stage it actually runs on the GPU and takes the fastest device that
can run the job, instead of silently taking device index 0.

* The benchmark is the fused mass kernel of :mod:`.mass_acceleration` on a
  synthetic frame of the job's size and precision. Each frame is timed end to
  end (host upload, kernel and host download) after one warm-up frame that
  also builds the kernel, and the result is the median of a few repeats. The
  CPU path of the same stage is timed too, so the user sees whether a GPU
  helps at all. The code never switches backend on its own.
* Only devices that can run the job count: the job's backend family, float64
  support for a float64 job, one field within the device's largest
  allocation, and enough free device memory for the measured peak of every
  worker (the resource advisor's VRAM rule).
* One receipt per machine stores the devices, drivers and times. It is reused
  until the device list, a driver, Python, NumPy, a GPU runtime library or the
  measured kernel file changes.
* An explicit device (``device_id``, ``DARSIA_OPENCL_DEVICE`` or
  ``DARSIA_CUDA_DEVICE``, or the GUI) always wins, and a single device is used
  without a measurement, as before.

Each device is measured in a short child process that loads only this file
and the mass-kernel file, never the ``darsia`` package. A driver fault or hang
therefore cannot stop the caller, and no GPU context outlives its measurement.
This module imports only the standard library and NumPy at module level.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import types
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

RECEIPT_FORMAT = "darsia-gpu-benchmark-receipt-v1"
CHOICE_FORMAT = "darsia-gpu-device-choice-v1"
FAMILIES = ("cuda", "opencl")
FAMILY_LABELS = {"cuda": "CUDA", "opencl": "OpenCL", "cpu": "CPU"}
# A 24-megapixel frame; a job always measures at its own analysis size.
DEFAULT_SHAPE = (4000, 6000)
MEASUREMENT_BUDGET_SECONDS = 3.0
MINIMUM_REPEATS = 3
MAXIMUM_REPEATS = 7
DEVICE_TIMEOUT_SECONDS = 180.0
LOCK_WAIT_SECONDS = 900.0
IDENTITY_FIELDS = (
    "family",
    "device_id",
    "name",
    "vendor",
    "platform",
    "platform_version",
    "driver",
    "display_driver",
    "fp64",
    "memory_bytes",
    "max_allocation_bytes",
    "unified_memory",
    "pci_bus_id",
)
PROTOCOL = {
    "stage": "fused mass kernel of darsia.fluidflower.mass_acceleration",
    "timed": "host upload, kernel and host download of one frame",
    "warmup_frames": 1,
    "statistic": "median",
    "minimum_repeats": MINIMUM_REPEATS,
    "maximum_repeats": MAXIMUM_REPEATS,
    "budget_seconds": MEASUREMENT_BUDGET_SECONDS,
    "inputs": "seeded synthetic float64 fields, titration LUT, smoothstep",
    "cpu": "NumPy session of the same mass algebra at the same precision",
}
_CHILD_CODE = """\
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("darsia_gpu_benchmark_child", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print(json.dumps(module.child_measurement(json.loads(sys.argv[2]))), flush=True)
"""
_THREAD_LOCK = threading.Lock()


def default_receipt_path() -> Path:
    """Return this machine's host-local GPU benchmark receipt file."""

    from .execution import default_execution_cache_root

    return default_execution_cache_root().parent / "gpu_benchmark" / "receipt.json"


def backend_family(backend: str) -> str | None:
    """Return ``cuda`` or ``opencl`` for a backend with a device choice, else None."""

    family = str(backend).split("-", 1)[0]
    return family if family in FAMILIES else None


def device_identity(device: Mapping[str, Any]) -> dict[str, Any]:
    """Return the fields that identify one device; free memory is not one of them."""

    return {key: device.get(key) for key in IDENTITY_FIELDS}


def enumerate_devices() -> dict[str, Any]:
    """Return this machine's CUDA and OpenCL GPU devices and their free memory now.

    The identity fields define the receipt fingerprint. ``free_bytes`` and
    ``free_source`` are read at this moment and never fingerprinted. A missing
    runtime gives no devices of its family and a note.
    """

    from . import mass_acceleration
    from .execution import _memory_bytes

    nvidia = _nvidia_smi()
    runtime: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "implementation_sha256": _sha256_file(Path(mass_acceleration.__file__)),
    }
    devices: list[dict[str, Any]] = []
    notes: dict[str, str] = {}
    for family, reader in (("cuda", _cuda_devices), ("opencl", _opencl_devices)):
        try:
            found, version = reader(mass_acceleration, nvidia)
        except Exception as exc:  # noqa: BLE001 - an absent runtime means no device
            found, version = [], ""
            notes[family] = f"{type(exc).__name__}: {exc}"
        devices.extend(found)
        runtime[family] = version
    host_available = _memory_bytes()[1]
    for device in devices:
        free = device.pop("_free_bytes", None)
        source = device.pop("_free_source", "")
        if free is None and device["pci_bus_id"] in nvidia:
            free, source = nvidia[device["pci_bus_id"]]["free_bytes"], "nvidia-smi"
        if free is None and device["unified_memory"] and host_available is not None:
            free, source = host_available, "available host memory (unified memory)"
        device["free_bytes"] = free
        device["free_source"] = source if free is not None else "not reported"
    return {"devices": devices, "runtime": runtime, "notes": notes}


def device_fingerprint(inventory: Mapping[str, Any]) -> str:
    """Return the digest that keeps a receipt valid: devices, drivers and runtime."""

    from .execution import _json_digest

    return _json_digest(
        {
            "devices": [
                device_identity(device) for device in inventory.get("devices") or []
            ],
            "runtime": dict(inventory.get("runtime") or {}),
        }
    )


def device_memory_fits(free_bytes: int, peak_bytes: int, workers: int = 1) -> bool:
    """Apply the advisor's VRAM rule: free memory over the peak covers every worker."""

    return int(peak_bytes) > 0 and int(free_bytes) // int(peak_bytes) >= max(
        1, int(workers)
    )


def measure_devices(
    inventory: Mapping[str, Any],
    *,
    precision: str,
    shape: Sequence[int],
    runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Time the mass stage on every device and on the CPU, each in its own child.

    A float64 measurement skips devices without float64 support.
    """

    runner = runner or run_measurement_child
    shape_rc = [int(shape[0]), int(shape[1])]
    rows: list[dict[str, Any]] = []
    for device in inventory.get("devices") or []:
        row = {
            "family": device["family"],
            "device_id": device["device_id"],
            "name": device["name"],
        }
        if precision == "float64" and not device.get("fp64"):
            rows.append(
                dict(row, status="skipped", reason="no float64 support (cl_khr_fp64)")
            )
            continue
        request = dict(
            family=device["family"],
            device_id=device["device_id"],
            precision=precision,
            shape_rc=shape_rc,
        )
        rows.append(dict(row, **runner(request)))
    cpu = runner(
        dict(family="cpu", device_id="cpu", precision=precision, shape_rc=shape_rc)
    )
    return {
        "precision": precision,
        "shape_rc": shape_rc,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": dict(PROTOCOL),
        "devices": rows,
        "cpu": cpu,
    }


def run_measurement_child(
    request: Mapping[str, Any],
    *,
    timeout: float = DEVICE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Measure one device in a fresh child process; a fault or hang only fails it."""

    from . import mass_acceleration
    from .execution import _subprocess_options

    implementation = Path(mass_acceleration.__file__).resolve()
    payload = dict(
        request,
        implementation=str(implementation),
        implementation_sha256=_sha256_file(implementation),
        budget_seconds=MEASUREMENT_BUDGET_SECONDS,
        minimum_repeats=MINIMUM_REPEATS,
        maximum_repeats=MAXIMUM_REPEATS,
    )
    command = [
        sys.executable,
        "-B",
        "-P",
        "-c",
        _CHILD_CODE,
        str(Path(__file__).resolve()),
        json.dumps(payload),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            **_subprocess_options(),
        )
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": f"no result within {timeout:.0f} s"}
    except OSError as exc:
        return {"status": "failed", "error": f"could not start the measurement: {exc}"}
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if completed.returncode != 0 or not lines:
        tail = (completed.stderr or completed.stdout).strip().splitlines()[-3:]
        return {
            "status": "failed",
            "error": f"measurement ended with exit code {completed.returncode}: "
            + " | ".join(tail),
        }
    return dict(json.loads(lines[-1]))


def child_measurement(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Measure one device inside the child process and return its timings.

    Only the mass-kernel file is loaded, from the exact bytes that were hashed;
    the ``darsia`` package is never imported here.
    """

    try:
        implementation = _load_implementation(
            str(payload["implementation"]), str(payload["implementation_sha256"])
        )
        result = _measure_session(implementation, payload)
    except Exception as exc:  # noqa: BLE001 - every failure is a measured fact
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    result["darsia_package_loaded"] = any(
        name == "darsia" or name.startswith("darsia.") for name in sys.modules
    )
    return result


def synthetic_frame(
    shape: Sequence[int],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Return seeded mass-stage inputs, active mask and voxel volume of one frame.

    The dynamic fields are float64, as the mass adapter passes them, so every
    session converts them to its own precision inside the timed frame.
    """

    rows, columns = int(shape[0]), int(shape[1])
    generator = np.random.default_rng(0)
    co2 = generator.random((rows, columns))
    inputs = {
        "acidification": generator.random((rows, columns)),
        "co2_probability": co2,
        "gas_probability": co2 * generator.random((rows, columns)),
        "solubility_co2": np.full((rows, columns), 1.7),
        "density_gaseous_co2": np.full((rows, columns), 1.9),
        "diffusion_factor": 0.9,
    }
    active = (generator.random((rows, columns)) < 0.9).astype(np.uint8)
    voxel_volume = np.full((rows, columns), 1.0e-9)
    return inputs, active, voxel_volume


def benchmark_configuration() -> dict[str, Any]:
    """Return the kernel settings of the benchmark: the titration LUT path."""

    signal = np.linspace(0.0, 1.0, 256)
    return dict(
        model="titration",
        transfer_shape="smoothstep",
        acid_low=0.1,
        acid_high=0.9,
        co2_onset=0.2,
        co2_high=0.8,
        gas_onset=0.3,
        gas_high=0.9,
        max_gas_saturation=0.8,
        lut_signal=signal,
        lut_concentration=np.sqrt(signal),
    )


def session_device_bytes(session: Any) -> int | None:
    """Return the device memory a mass session holds: its buffers or memory pool."""

    if hasattr(session, "output_buffers"):
        buffers = [
            session.active_buffer,
            session.voxel_buffer,
            session.lut_signal_buffer,
            session.lut_concentration_buffer,
            *session.dynamic_buffers,
            *session.output_buffers,
        ]
        return int(sum(int(buffer.size) for buffer in buffers))
    if hasattr(session, "cp"):
        return int(session.cp.get_default_memory_pool().total_bytes())
    return None


def _measure_session(module: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    family = str(payload["family"])
    precision = str(payload["precision"])
    inputs, active, voxel_volume = synthetic_frame(payload["shape_rc"])
    arguments = dict(
        active=active,
        voxel_volume_m3=voxel_volume,
        configuration=module.MassKernelConfiguration(**benchmark_configuration()),
    )
    started = time.perf_counter()
    if family == "cpu":
        session = module.NumpyMassComputeSession(
            backend=f"cpu-{precision}", **arguments
        )
    else:
        session = module.create_mass_compute_session(
            backend=f"{family}-{precision}",
            device_id=str(payload["device_id"]),
            **arguments,
        )
    try:
        session.compute_frame(**inputs)
        warmup = time.perf_counter() - started
        minimum = int(payload["minimum_repeats"])
        maximum = int(payload["maximum_repeats"])
        budget = float(payload["budget_seconds"])
        times: list[float] = []
        budget_started = time.perf_counter()
        while len(times) < maximum and (
            len(times) < minimum or time.perf_counter() - budget_started < budget
        ):
            frame_started = time.perf_counter()
            session.compute_frame(**inputs)
            times.append(time.perf_counter() - frame_started)
        peak = session_device_bytes(session)
        provenance = session.provenance()
    finally:
        session.close()
    return {
        "status": "measured",
        "seconds_per_frame_median": float(np.median(times)),
        "seconds_per_frame": [float(value) for value in times],
        "repeats": len(times),
        "warmup_seconds": float(warmup),
        "peak_device_bytes": peak,
        "device_name": str(provenance.get("device_name") or "CPU"),
        "device_driver": str(provenance.get("device_driver") or ""),
    }


def _load_implementation(path: str, expected_sha256: str) -> types.ModuleType:
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise RuntimeError("The mass-kernel file changed before its measurement")
    module = types.ModuleType("darsia_gpu_benchmark_kernel")
    module.__file__ = path
    # Dataclasses look their module up while the file executes.
    sys.modules[module.__name__] = module
    exec(compile(data, path, "exec"), module.__dict__)
    return module


def load_receipt(path: Path | str) -> dict[str, Any] | None:
    """Return a stored receipt whose content digest is intact, else None."""

    from .execution import _json_digest

    try:
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(receipt, dict) or receipt.get("format") != RECEIPT_FORMAT:
        return None
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    return receipt if receipt.get("content_sha256") == _json_digest(body) else None


def store_receipt(path: Path | str, receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Seal one receipt with its content digest, write it atomically, return it."""

    from .execution import _json_digest

    sealed = {key: value for key, value in receipt.items() if key != "content_sha256"}
    sealed["content_sha256"] = _json_digest(sealed)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(sealed, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sealed


def find_measurement(
    receipt: Mapping[str, Any] | None, precision: str, shape: Sequence[int]
) -> dict[str, Any] | None:
    """Return the receipt's measurement of one precision and frame size, if any."""

    wanted = (str(precision), [int(shape[0]), int(shape[1])])
    for measurement in (receipt or {}).get("measurements") or []:
        if (measurement.get("precision"), measurement.get("shape_rc")) == wanted:
            return dict(measurement)
    return None


def ensure_measurement(
    inventory: Mapping[str, Any],
    *,
    precision: str,
    shape: Sequence[int],
    path: Path | str,
    measure: Callable[..., dict[str, Any]] | None = None,
    force: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Return the receipt, the measurement and whether it was reused.

    A stored measurement is reused while the fingerprint matches. Otherwise the
    devices are measured, one process at a time per machine, and the receipt
    is rewritten; a receipt that cannot be stored is reported, never fatal.
    """

    measure = measure or measure_devices
    shape_rc = [int(shape[0]), int(shape[1])]
    fingerprint = device_fingerprint(inventory)
    with _receipt_lock(Path(path)):
        stored = load_receipt(path)
        if stored is not None and stored.get("fingerprint") == fingerprint:
            found = find_measurement(stored, precision, shape_rc)
            if found is not None and not force:
                return stored, found, True
            receipt = stored
        else:
            receipt = {
                "format": RECEIPT_FORMAT,
                "schema": 1,
                "fingerprint": fingerprint,
                "devices": [
                    device_identity(device) for device in inventory.get("devices") or []
                ],
                "runtime": dict(inventory.get("runtime") or {}),
                "measurements": [],
            }
        measurement = measure(inventory, precision=precision, shape=shape_rc)
        kept = [
            row
            for row in receipt.get("measurements") or []
            if (row.get("precision"), row.get("shape_rc")) != (precision, shape_rc)
        ]
        receipt = dict(receipt, measurements=[*kept, measurement])
        try:
            receipt = store_receipt(path, receipt)
        except OSError as exc:
            receipt = dict(receipt, store_error=f"{type(exc).__name__}: {exc}")
        return receipt, measurement, False


def measure_and_store(
    *,
    precision: str = "float32",
    shape: Sequence[int] = DEFAULT_SHAPE,
    receipt_path: Path | str | None = None,
    inventory: Mapping[str, Any] | None = None,
    measure: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Measure every device now at one precision and frame size; return the receipt.

    This is the resource advisor's explicit action; it replaces any stored
    measurement of the same precision and size.
    """

    inventory = enumerate_devices() if inventory is None else inventory
    path = Path(receipt_path) if receipt_path is not None else default_receipt_path()
    receipt, _, _ = ensure_measurement(
        inventory,
        precision=precision,
        shape=shape,
        path=path,
        measure=measure,
        force=True,
    )
    return receipt


def rate_candidates(
    devices: Sequence[Mapping[str, Any]],
    measurement: Mapping[str, Any],
    *,
    precision: str,
    shape: Sequence[int],
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Return every device of the job's family with its time and why it may not run.

    A device without reported free memory is checked against its total memory,
    and the record says so.
    """

    timings = {
        (row.get("family"), row.get("device_id")): row
        for row in measurement.get("devices") or []
    }
    field_bytes = int(shape[0]) * int(shape[1]) * np.dtype(precision).itemsize
    candidates = []
    for device in devices:
        timing = timings.get((device["family"], device["device_id"]), {})
        reasons = []
        if precision == "float64" and not device.get("fp64"):
            reasons.append("no float64 support (cl_khr_fp64)")
        if timing.get("status") != "measured":
            reasons.append(
                "not measured: "
                + str(timing.get("error") or timing.get("reason") or "no result")
            )
        largest = int(device.get("max_allocation_bytes") or 0)
        if field_bytes > largest:
            reasons.append(
                f"one field of {field_bytes} B exceeds the largest allocation "
                f"of {largest} B"
            )
        peak = timing.get("peak_device_bytes")
        free, source = device.get("free_bytes"), device.get("free_source")
        if free is None:
            free = device.get("memory_bytes")
            source = "total memory; free memory not reported by the driver"
        if (
            type(peak) is int
            and peak > 0
            and type(free) is int
            and not device_memory_fits(free, peak, workers)
        ):
            reasons.append(
                f"{free} B free cannot hold {max(1, int(workers))} worker(s) of the "
                f"measured peak {peak} B"
            )
        candidates.append(
            {
                "family": device["family"],
                "device_id": device["device_id"],
                "name": device.get("name"),
                "driver": device.get("driver"),
                "seconds_per_frame": timing.get("seconds_per_frame_median"),
                "peak_device_bytes": peak,
                "free_bytes": free,
                "free_source": source,
                "eligible": not reasons,
                "reasons": reasons,
            }
        )
    return candidates


def choose_device(
    *,
    family: str,
    precision: str,
    shape: Sequence[int],
    workers: int = 1,
    explicit: str = "",
    explicit_source: str = "device_id",
    receipt_path: Path | str | None = None,
    inventory: Mapping[str, Any] | None = None,
    measure: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Choose the device of one accelerator family and return the choice record.

    An explicit device, then the family's environment variable, always wins and
    is never filtered. One device is used without a measurement. With several
    devices the fastest device that can run the job wins; ties go to the lower
    index, and no eligible device is an error that names every reason.
    """

    if family not in FAMILIES:
        raise ValueError(f"No device choice exists for {family}")
    from .mass_acceleration import DEVICE_ENVIRONMENT_VARIABLES

    variable = DEVICE_ENVIRONMENT_VARIABLES[family]
    shape_rc = [int(shape[0]), int(shape[1])]
    if not explicit and os.environ.get(variable, ""):
        explicit, explicit_source = os.environ[variable], f"environment:{variable}"
    inventory = enumerate_devices() if inventory is None else inventory
    path = Path(receipt_path) if receipt_path is not None else default_receipt_path()
    devices = [
        dict(device)
        for device in inventory.get("devices") or []
        if device.get("family") == family
    ]
    record: dict[str, Any] = {
        "format": CHOICE_FORMAT,
        "family": family,
        "precision": precision,
        "shape_rc": shape_rc,
        "workers": max(1, int(workers)),
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }
    if explicit or len(devices) <= 1:
        if not explicit and not devices:
            raise RuntimeError(f"No {FAMILY_LABELS[family]} GPU device exists here")
        device_id = str(int(explicit)) if explicit else devices[0]["device_id"]
        device = next(
            (row for row in devices if row["device_id"] == device_id),
            {"device_id": device_id},
        )
        stored = load_receipt(path)
        found = (
            find_measurement(stored, precision, shape_rc)
            if stored is not None
            and stored.get("fingerprint") == device_fingerprint(inventory)
            else None
        )
        record.update(
            basis="explicit" if explicit else "single_device",
            source=explicit_source if explicit else "",
            device_id=device_id,
            device=device_identity(device),
            benchmark=_benchmark_summary(stored, found, True) if found else None,
        )
        return record
    receipt, measurement, reused = ensure_measurement(
        inventory, precision=precision, shape=shape_rc, path=path, measure=measure
    )
    candidates = rate_candidates(
        devices, measurement, precision=precision, shape=shape_rc, workers=workers
    )
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        raise RuntimeError(
            f"No {FAMILY_LABELS[family]} device can run this job: "
            + "; ".join(
                f"device {row['device_id']} ({row['name']}): "
                + ", ".join(row["reasons"])
                for row in candidates
            )
            + f". Choose a device explicitly (device_id or {variable})."
        )
    winner = min(
        eligible, key=lambda row: (row["seconds_per_frame"], int(row["device_id"]))
    )
    device = next(row for row in devices if row["device_id"] == winner["device_id"])
    record.update(
        basis="measured",
        source="",
        device_id=winner["device_id"],
        device=device_identity(device),
        candidates=candidates,
        benchmark=_benchmark_summary(receipt, measurement, reused),
    )
    return record


def committed_device(
    executions: Sequence[Mapping[str, Any]], family: str
) -> dict[str, Any] | None:
    """Return the one device that a job's committed runs used, or None.

    Older runs name their device by index, name and driver; newer runs also
    carry their choice record. Runs that disagree are an error.
    """

    used = []
    for execution in executions:
        if not execution:
            continue
        choice = dict(execution.get("device_choice") or {})
        device = dict(choice.get("device") or {})
        name = execution.get("device_name") or device.get("name") or ""
        driver = execution.get("device_driver") or device.get("driver") or ""
        used.append(
            (
                {
                    "family": family,
                    "device_id": str(execution.get("device_id", "")),
                    "name": str(name),
                    "driver": str(driver),
                },
                choice,
            )
        )
    if not used:
        return None
    first = used[0][0]
    if any(identity != first for identity, _ in used):
        raise RuntimeError(
            "This job's completed runs used different devices: "
            + "; ".join(sorted({repr(identity) for identity, _ in used}))
        )
    return dict(first, choice=used[0][1] or None)


def resolve_job_device(
    *,
    backend: str,
    shape: Sequence[int],
    workers: int = 1,
    explicit: str = "",
    explicit_source: str = "",
    committed: Sequence[Mapping[str, Any]] = (),
    receipt_path: Path | str | None = None,
    inventory: Mapping[str, Any] | None = None,
    measure: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Choose the device of an accelerated job once, before its first run.

    A job whose runs were already committed continues on the same device, so the
    device never changes within a job; if that device is gone, the job stops
    with the reason. An explicit device always wins. CPU and Metal jobs have no
    choice and return None.
    """

    family = backend_family(backend)
    if family is None:
        return None
    from .mass_acceleration import DEVICE_ENVIRONMENT_VARIABLES

    variable = DEVICE_ENVIRONMENT_VARIABLES[family]
    precision = str(backend).rsplit("-", 1)[-1]
    pinned = committed_device(committed, family)
    if pinned is not None and not explicit and not os.environ.get(variable, ""):
        inventory = enumerate_devices() if inventory is None else inventory
        match = next(
            (
                device
                for device in inventory.get("devices") or []
                if device.get("family") == family
                and str(device.get("device_id")) == pinned["device_id"]
                and str(device.get("name")).strip() == pinned["name"].strip()
                and (
                    not pinned["driver"]
                    or str(device.get("driver")).strip() == pinned["driver"].strip()
                )
            ),
            None,
        )
        if match is None:
            raise RuntimeError(
                f"This job's completed runs used {FAMILY_LABELS[family]} device "
                f"{pinned['device_id']} ({pinned['name']}, driver "
                f"{pinned['driver'] or 'not recorded'}), which is not available now. "
                f"Resume where it is, or choose a device explicitly (device_id or "
                f"{variable})."
            )
        return {
            "format": CHOICE_FORMAT,
            "family": family,
            "precision": precision,
            "shape_rc": [int(shape[0]), int(shape[1])],
            "workers": max(1, int(workers)),
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "basis": "resumed",
            "source": "",
            "device_id": match["device_id"],
            "device": device_identity(match),
            "earlier_choice": pinned["choice"],
        }
    return choose_device(
        family=family,
        precision=precision,
        shape=shape,
        workers=workers,
        explicit=explicit,
        explicit_source=explicit_source or "device_id",
        receipt_path=receipt_path,
        inventory=inventory,
        measure=measure,
    )


def verify_session_device(
    execution: Mapping[str, Any], choice: Mapping[str, Any]
) -> None:
    """Refuse a run whose session opened another device than the job's choice.

    The index must match, and so must the name and driver wherever the choice
    recorded them, so a changed device list cannot move a job to another GPU.
    """

    device = dict(choice.get("device") or {})
    opened = {
        "device_id": str(execution.get("device_id")),
        "name": str(execution.get("device_name") or "").strip(),
        "driver": str(execution.get("device_driver") or "").strip(),
    }
    wanted = {
        "device_id": str(choice.get("device_id")),
        "name": str(device.get("name") or "").strip(),
        "driver": str(device.get("driver") or "").strip(),
    }
    differs = [
        key
        for key, value in wanted.items()
        if (value or key == "device_id") and opened[key] != value
    ]
    if differs:
        raise RuntimeError(
            "The mass session opened another device than the one chosen for this "
            f"job: chose {wanted}, opened {opened}"
        )


def task_device_request(task: Mapping[str, Any]) -> tuple[str, str]:
    """Return a task's explicit device index and its source, or two empty strings.

    The adapter setting always counts. A resources entry counts unless the task
    marks it as an automatic choice (``single_device`` or ``measured``), which
    the worker makes again on the machine that runs the job.
    """

    settings = dict((task.get("adapter") or {}).get("settings") or {})
    if str(settings.get("device_id") or ""):
        return str(settings["device_id"]), "task adapter setting"
    resources = dict(task.get("resources") or {})
    requested = str(resources.get("device_id") or "")
    if requested and resources.get("device_choice", "explicit") == "explicit":
        return requested, "task resources"
    return "", ""


def backend_devices(inventory: Mapping[str, Any], backend: str) -> list[dict[str, Any]]:
    """Return the hardware-inventory devices that provide one execution backend."""

    return [
        dict(device)
        for device in inventory.get("accelerators") or []
        if isinstance(device, Mapping)
        and backend
        in (
            device.get("supported_backends")
            if device.get("supported_backends") is not None
            else [device.get("backend")]
        )
    ]


def requested_device_resources(
    inventory: Mapping[str, Any], backend: str, device_id: str = ""
) -> dict[str, str]:
    """Return the task resource fields that name the device of one GPU backend.

    A chosen device index is the user's explicit choice. Without one, a single
    device is named as before, and several devices leave the choice to the
    measurement at job start on the machine that runs the job.
    """

    devices = backend_devices(inventory, backend)
    if not devices:
        raise ValueError(f"No device provides backend {backend}")
    if str(device_id):
        match = next(
            (row for row in devices if str(row.get("device_id")) == str(device_id)),
            None,
        )
        if match is None:
            raise ValueError(f"No {backend} device has index {device_id}")
        return {
            "device_id": str(device_id),
            "device_name": str(match.get("name") or backend),
            "device_choice": "explicit",
        }
    if len(devices) == 1:
        return {
            "device_id": str(devices[0].get("device_id") or "0"),
            "device_name": str(devices[0].get("name") or backend),
            "device_choice": "single_device",
        }
    return {
        "device_id": "",
        "device_name": f"Fastest of {len(devices)} devices, measured at job start",
        "device_choice": "measured",
    }


def receipt_rows(receipt: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Return one display row per measured device and the CPU, newest first.

    ``fastest`` marks the fastest measured device of each backend family in
    its measurement; whether it can run a job also depends on free memory then.
    """

    devices = {
        (row.get("family"), row.get("device_id")): row
        for row in (receipt or {}).get("devices") or []
    }
    rows: list[dict[str, Any]] = []
    measurements = sorted(
        (receipt or {}).get("measurements") or [],
        key=lambda row: str(row.get("recorded_utc") or ""),
        reverse=True,
    )
    for measurement in measurements:
        timings = [
            *(measurement.get("devices") or []),
            dict(measurement.get("cpu") or {}, family="cpu", device_id="", name="CPU"),
        ]
        best: dict[str, float] = {}
        for timing in timings:
            seconds = timing.get("seconds_per_frame_median")
            if timing.get("status") == "measured" and timing.get("family") != "cpu":
                family = str(timing.get("family"))
                best[family] = min(best.get(family, float("inf")), float(seconds))
        for timing in timings:
            family = str(timing.get("family"))
            device = devices.get((family, timing.get("device_id")), {})
            status = str(timing.get("status") or "missing")
            seconds = timing.get("seconds_per_frame_median")
            driver = device.get("display_driver") or device.get("driver") or ""
            frame = "x".join(str(n) for n in measurement.get("shape_rc") or [])
            rows.append(
                {
                    "family": family,
                    "device": (
                        "CPU (NumPy, same mass algebra)"
                        if family == "cpu"
                        else f"{FAMILY_LABELS.get(family, family)} "
                        f"{timing.get('device_id')}: {timing.get('name')}"
                    ),
                    "driver": str(driver),
                    "precision": str(measurement.get("precision")),
                    "frame": frame,
                    "seconds_per_frame": seconds,
                    "peak_device_bytes": timing.get("peak_device_bytes"),
                    "status": (
                        "Measured"
                        if status == "measured"
                        else f"{status.capitalize()}: "
                        + str(timing.get("reason") or timing.get("error") or "")
                    ),
                    "fastest": status == "measured"
                    and family in best
                    and float(seconds) == best[family],
                    "recorded_utc": str(measurement.get("recorded_utc") or ""),
                }
            )
    return rows


def _benchmark_summary(
    receipt: Mapping[str, Any] | None,
    measurement: Mapping[str, Any],
    reused: bool,
) -> dict[str, Any]:
    return {
        "receipt_sha256": (receipt or {}).get("content_sha256"),
        "fingerprint": (receipt or {}).get("fingerprint"),
        "recorded_utc": measurement.get("recorded_utc"),
        "reused": bool(reused),
        "store_error": str((receipt or {}).get("store_error") or ""),
        "seconds_per_frame": {
            f"{row['family']}:{row['device_id']}": row.get("seconds_per_frame_median")
            for row in measurement.get("devices") or []
            if row.get("status") == "measured"
        },
        "cpu_seconds_per_frame": (measurement.get("cpu") or {}).get(
            "seconds_per_frame_median"
        ),
        "protocol": deepcopy(measurement.get("protocol")),
    }


def _cuda_devices(
    mass_acceleration: Any, nvidia: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    import cupy as cp

    count = int(cp.cuda.runtime.getDeviceCount())
    driver = mass_acceleration.cuda_driver_version(cp)
    runtime = int(cp.cuda.runtime.runtimeGetVersion())
    devices = []
    for index in range(count):
        properties = cp.cuda.runtime.getDeviceProperties(index)
        raw_name = properties.get("name", "CUDA device")
        name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        bus = (
            f"{int(properties.get('pciDomainID', 0)):08x}:"
            f"{int(properties.get('pciBusID', 0)):02x}:"
            f"{int(properties.get('pciDeviceID', 0)):02x}"
        )
        memory = int(properties.get("totalGlobalMem", 0))
        devices.append(
            {
                "family": "cuda",
                "device_id": str(index),
                "name": name.strip(),
                "vendor": "",
                "platform": "CUDA",
                "platform_version": f"CUDA runtime {runtime}",
                "driver": driver,
                "display_driver": str(nvidia.get(bus, {}).get("driver", "")),
                "fp64": True,
                "memory_bytes": memory,
                "max_allocation_bytes": memory,
                "unified_memory": bool(int(properties.get("integrated", 0))),
                "pci_bus_id": bus,
            }
        )
    return devices, str(cp.__version__)


def _opencl_devices(
    mass_acceleration: Any, nvidia: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    import pyopencl as cl

    devices = []
    for index, (opencl_platform, device) in enumerate(
        mass_acceleration.opencl_gpu_devices(cl)
    ):
        bus = ""
        nv_bus = _device_info(device, "pci_bus_id_nv")
        if nv_bus is not None:
            domain = int(_device_info(device, "pci_domain_id_nv") or 0)
            slot = int(_device_info(device, "pci_slot_id_nv") or 0)
            bus = f"{domain:08x}:{int(nv_bus):02x}:{slot:02x}"
        amd_free = _device_info(device, "global_free_memory_amd")
        devices.append(
            {
                "family": "opencl",
                "device_id": str(index),
                "name": str(device.name).strip(),
                "vendor": str(device.vendor).strip(),
                "platform": str(opencl_platform.name).strip(),
                "platform_version": str(opencl_platform.version).strip(),
                "driver": str(device.driver_version).strip(),
                "display_driver": str(nvidia.get(bus, {}).get("driver", "")),
                "fp64": bool(mass_acceleration.opencl_supports_float64(device)),
                "memory_bytes": int(device.global_mem_size),
                "max_allocation_bytes": int(device.max_mem_alloc_size),
                "unified_memory": bool(_device_info(device, "host_unified_memory")),
                "pci_bus_id": bus,
                "_free_bytes": int(amd_free[0]) * 1024 if amd_free else None,
                "_free_source": "OpenCL AMD free-memory query" if amd_free else "",
            }
        )
    return devices, str(getattr(cl, "VERSION_TEXT", ""))


def _device_info(device: Any, name: str) -> Any:
    try:
        return getattr(device, name)
    except Exception:  # noqa: BLE001 - vendor queries raise on other vendors
        return None


def _nvidia_smi() -> dict[str, dict[str, Any]]:
    from .execution import _subprocess_options

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=pci.bus_id,driver_version,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
            **_subprocess_options(),
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or parts[0].count(":") != 2:
            continue
        domain, bus, slot = parts[0].split(":")
        try:
            slot_number = int(slot.split(".")[0], 16)
            key = f"{int(domain, 16):08x}:{int(bus, 16):02x}:{slot_number:02x}"
            free = int(float(parts[2]) * 1024 * 1024)
        except ValueError:
            continue
        rows[key] = {"driver": parts[1], "free_bytes": free}
    return rows


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def _receipt_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_name(path.name + ".lock")
    with _THREAD_LOCK:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(lock_path, "a+b")
        except OSError:
            handle = None
        locked = False
        if handle is not None:
            deadline = time.monotonic() + LOCK_WAIT_SECONDS
            while not (locked := _try_lock(handle)) and time.monotonic() < deadline:
                time.sleep(0.25)
        try:
            yield
        finally:
            if handle is not None:
                if locked:
                    _unlock(handle)
                handle.close()


def _try_lock(handle: Any) -> bool:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


__all__ = [
    "CHOICE_FORMAT",
    "DEFAULT_SHAPE",
    "RECEIPT_FORMAT",
    "backend_devices",
    "backend_family",
    "benchmark_configuration",
    "child_measurement",
    "choose_device",
    "committed_device",
    "default_receipt_path",
    "device_fingerprint",
    "device_identity",
    "device_memory_fits",
    "ensure_measurement",
    "enumerate_devices",
    "find_measurement",
    "load_receipt",
    "measure_and_store",
    "measure_devices",
    "rate_candidates",
    "receipt_rows",
    "requested_device_resources",
    "resolve_job_device",
    "run_measurement_child",
    "session_device_bytes",
    "store_receipt",
    "synthetic_frame",
    "task_device_request",
    "verify_session_device",
]
