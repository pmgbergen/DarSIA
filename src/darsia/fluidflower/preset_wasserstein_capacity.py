"""Exact administrative admission for one original preset comparison time.

Measured qualification and reuse are disabled until an original writer,
lifetime-allocation and retirement verifier exists. Declared observations and
byte-pinned output metadata cannot establish that proof. First qualification
and the independently checked bounded administrative provider remain separate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
import socket
import subprocess

from .preset_wasserstein_protocol import (
    PIPELINE, RESULT_FORMAT, digest, expected_comparisons, validate_operation,
    validate_preset_wasserstein_task,
)

PROFILE_FORMAT = "darsia-preset-wasserstein-capacity-profile-v1"
MEASUREMENT_FORMAT = "darsia-preset-wasserstein-space-measurement-v1"
COMPLETION_FORMAT = "darsia-preset-wasserstein-space-completion-v1"
RESERVATION_MODEL = "preset-wasserstein-whole-time"
MEASURED_UNAVAILABLE = (
    "Original Wasserstein writer/lifetime/retirement verifier is unavailable; "
    "measured preset qualification and reuse are disabled"
)

# These are minimum boundary checks, not a substitute for the full observed
# producer/runtime closure, which must exactly equal the admitted ImportLease.
REQUIRED_SERVICE_SOURCES = tuple("src/darsia/fluidflower/" + name for name in (
    "preset_wasserstein_capacity.py", "preset_wasserstein_protocol.py",
    "preset_wasserstein_inputs.py", "preset_wasserstein_runner.py",
    "preset_wasserstein_results.py", "mass/n1_pipeline.py",
    "pipeline/worker.py", "pipeline/worker_io.py", "pipeline/run_io.py",
    "pipeline/capacity.py", "pipeline/runner.py", "pipeline/job_inputs.py",
    "pipeline/products_publication.py", "pipeline/publication_runtime.py",
)) + tuple("src/darsia/" + name for name in (
    "presets/workflows/comparison/comparison_wasserstein.py",
    "presets/workflows/rig.py", "presets/workflows/utils/mass.py",
    "presets/workflows/config/multi_fluidflower_config.py",
    "presets/workflows/config/fluidflower_config.py",
    "presets/workflows/config/wasserstein.py", "presets/workflows/config/roi.py",
    "presets/workflows/config/roi_registry.py", "presets/workflows/config/time_data.py",
    "experiment/experiment.py", "experiment/protocols.py", "image/imread.py",
    "image/image.py", "measure/wasserstein.py", "measure/integration.py",
    "restoration/resize.py", "utils/plotting.py", "_opencv_runtime.py",
)) + tuple("src/darsia_cache/" + name for name in (
    "control.py", "worker_adapter.py", "bootstrap.py", "policy.py",
    "preset_qualification.py", "first_qualification.py", "service_context.py",
    "owned_process.py", "service_lifecycle.py", "fresh_scope.py",
))


def _integer(value, where, *, positive=False):
    if type(value) is not int or value < int(positive):
        raise ValueError(where + " must be an explicit observed nonnegative integer")
    return value


def _key(path):
    return str(PureWindowsPath(path)).casefold()


def task_scope(task, *, requested_time_count):
    validate_preset_wasserstein_task(task)
    _integer(requested_time_count, "Original requested time count", positive=True)
    if task["comparison_time"]["order"] >= requested_time_count:
        raise ValueError("Preset measurement is outside the original requested times")
    return dict(commit=task["commit"], pipeline=PIPELINE, backend="cpu", precision="original",
        run=task["run"], operation_sha256=task["configuration"]["sha256"],
        preset_inputs_sha256=task["preset_inputs_sha256"],
        requested_time_count=requested_time_count, comparison_time=deepcopy(task["comparison_time"]),
        expected_comparisons=task["expected_comparisons"])


def operation_scope(task, operation):
    validate_operation(operation)
    from .pipeline.job_queue import _immutable_bytes
    raw = _immutable_bytes(operation)
    if (hashlib.sha256(raw).hexdigest() != task["configuration"]["sha256"]
            or len(raw) != task["configuration"]["bytes"]):
        raise ValueError("Observed original operation bytes differ from the exact task configuration")
    inputs = operation["preset_inputs"]
    if (operation["run"] != task["run"] or operation["comparison_time"] != task["comparison_time"]
            or digest(inputs) != task["preset_inputs_sha256"]
            or len(expected_comparisons(inputs, task["comparison_time"])) != task["expected_comparisons"]):
        raise ValueError("Measured comparison operation differs from the original consumed task")
    return task_scope(task, requested_time_count=len(inputs["requested_times"]))


@dataclass(frozen=True)
class PresetWassersteinEstimate:
    run: str
    case_count: int
    comparison_count: int
    products_estimated_bytes: int
    transient_estimated_bytes: int
    profile: dict
    measurement: dict
    completion: dict
    scope: dict
    measured_commit: str
    model: str = "exact observed original preset time; whole operation, no frame prefix"

    def needed(self, *, reserve_bytes):
        return self.products_estimated_bytes + self.transient_estimated_bytes + reserve_bytes


@dataclass(frozen=True)
class PresetWassersteinEntitlement:
    run: str
    case_count: int
    comparison_count: int
    products_estimated_bytes: int
    transient_estimated_bytes: int
    scope: dict
    request: dict
    administration: dict
    commit: str
    not_a_measured_profile: bool = True
    model: str = "explicit first preset whole-time administration; no measured capacity or scientific acceptance"

    def needed(self, *, reserve_bytes):
        return self.products_estimated_bytes + self.transient_estimated_bytes + reserve_bytes


@dataclass(frozen=True)
class PresetWassersteinBoundedEstimate(PresetWassersteinEntitlement):
    profile: dict | None = None
    completion: dict | None = None
    observed_closed_output_bytes: int | None = None
    model: str = "qualified physical-bound preset administration; observed closed output, explicit whole-time allowance, no lifetime-peak claim"


def require_estimate(task, estimate, *, allow_administration=False):
    if allow_administration and isinstance(estimate, PresetWassersteinEntitlement):
        if (estimate.case_count != 1 or estimate.comparison_count != task["expected_comparisons"]
                or estimate.scope != task_scope(task, requested_time_count=estimate.scope["requested_time_count"])
                or estimate.run != task["run"] or estimate.commit != task["commit"]
                or estimate.not_a_measured_profile is not True):
            raise ValueError("Exact independently released preset administration required")
        _integer(estimate.products_estimated_bytes, "Explicit whole-time output entitlement", positive=True)
        _integer(estimate.transient_estimated_bytes, "Explicit whole-time transient entitlement")
        return True
    raise ValueError(MEASURED_UNAVAILABLE)


def first_entitlement(runtime, task, operation):
    from darsia_cache.preset_qualification import PresetQualification
    lease = runtime.cache_adapter.lease
    first = getattr(lease, "first_qualification", None)
    if not isinstance(first, PresetQualification):
        raise ValueError("A separate original preset first qualification is required")
    first.require_task(lease, task)
    scope = operation_scope(task, operation)
    if (operation != first.configuration
            or scope != {key: first.provider["scope"][key] for key in scope}
            or runtime.backend != "cpu" or runtime.commit != task["commit"]
            or runtime.stores.machine_id != first.provider["scope"]["target_machine"]):
        raise ValueError("Preset first entitlement differs from actual consumed original operation")
    require_repository_sources(runtime.repository, task["commit"], first.provider)
    plan = first.output_entitlement
    return PresetWassersteinEntitlement(task["run"], 1, task["expected_comparisons"],
        plan["maximum_output_bytes"], plan["maximum_transient_bytes"], scope,
        first.binding, first.request["output_entitlement_plan"], task["commit"])


def require_bounded_estimate(task, estimate, profile, binding):
    from darsia_cache.preset_qualification import BOUNDED_PROFILE
    require_estimate(task, estimate, allow_administration=True)
    if (not isinstance(estimate, PresetWassersteinBoundedEstimate)
            or profile.get("format") != BOUNDED_PROFILE or profile.get("provider") != binding
            or estimate.profile != binding or estimate.completion != profile.get("completion")
            or estimate.scope != {key: profile.get("scope", {}).get(key) for key in estimate.scope}
            or str(profile.get("scope", {}).get("physical_host", "")).casefold() != socket.gethostname().casefold()
            or estimate.products_estimated_bytes != profile.get("maximum_output_bytes")
            or estimate.transient_estimated_bytes != profile.get("maximum_transient_bytes")
            or estimate.request != profile.get("first_request")
            or estimate.administration != profile.get("output_administration")
            or profile.get("not_a_measured_profile") is not True
            or profile.get("actual_writer_IO_proven") is not False
            or profile.get("sampled_highwater_is_lifetime_peak") is not False):
        raise ValueError("Exact qualified preset bounded administration required for this machine, input and requested time")
    return True


def _bounded_estimate(task, operation, profile, binding, repository):
    scope = operation_scope(task, operation)
    estimate = PresetWassersteinBoundedEstimate(task["run"], 1, task["expected_comparisons"],
        profile["maximum_output_bytes"], profile["maximum_transient_bytes"], scope,
        profile["first_request"], profile["output_administration"], task["commit"],
        profile=dict(binding), completion=profile["completion"],
        observed_closed_output_bytes=profile["closed_output_logical_bytes"])
    require_bounded_estimate(task, estimate, profile, binding)
    require_repository_sources(repository, task["commit"], profile)
    return estimate


def _receipt_metadata(receipt, pin, task):
    from .preset_wasserstein_results import (
        RECEIPT_FORMAT, validate_preset_wasserstein_result,
    )
    if receipt.get("format") != RECEIPT_FORMAT:
        raise ValueError("Real original comparison execution receipt required")
    result = dict(receipt, format=RESULT_FORMAT, frames=0, execution_receipt=pin)
    validate_preset_wasserstein_result(result)
    # A qualification execution may have another job ID. Only an identical
    # original input/operation/time/count is applicable to a future job; this
    # evidence is never presented as that future job's execution receipt.
    if (any(result[key] != task[key] for key in
            ("run", "backend", "precision", "comparison_time", "preset_inputs_sha256"))
            or result["expected_count"] != task["expected_comparisons"]):
        raise ValueError("Observed comparison receipt is outside the exact task input/time/count scope")
    return result


def _retained_output_bytes(measured, result, *, verify_payloads=False):
    """Count the entire closed attempt, separately from scientific products.

    Ordinary projection checks pinned metadata only. The explicit qualification
    builder checks the actual original owned root and every retained file once.
    """
    from .pipeline.job_queue import descriptor
    inventory = measured.get("retained_output_inventory")
    if (not isinstance(inventory, dict) or set(inventory) != {"root", "files"}
            or not isinstance(inventory["files"], dict) or not inventory["files"]):
        raise ValueError("Exact whole-attempt retained output inventory required")
    root = PureWindowsPath(inventory["root"])
    if root != PureWindowsPath(measured["execution_receipt"]["path"]).parent:
        raise ValueError("Retained inventory leaves the original comparison attempt")
    files, seen = inventory["files"], set()
    for key, pin in files.items():
        relative = PurePosixPath(key)
        if (not isinstance(key, str) or relative.is_absolute() or relative.as_posix() != key
                or ".." in relative.parts or "\\" in key or key.casefold() in seen
                or not isinstance(pin, dict) or set(pin) != {"path", "sha256", "size_bytes"}
                or _key(pin["path"]) != _key(root.joinpath(*relative.parts))):
            raise ValueError("Retained output needs distinct exact owned relative file descriptors")
        _integer(pin["size_bytes"], "Retained output bytes")
        if not isinstance(pin["sha256"], str) or len(pin["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in pin["sha256"]):
            raise ValueError("Retained output requires exact file hashes")
        seen.add(key.casefold())
    required = dict(result["files"], **{PureWindowsPath(measured["execution_receipt"]["path"]).name: measured["execution_receipt"]})
    if any(files.get(key) != pin for key, pin in required.items()):
        raise ValueError("Retained output inventory omits original JSON/VTK/strict receipt bytes")
    if "candidate_completed.json" not in files or "retirement.json" not in files:
        raise ValueError("Closed retained output must include actual candidate and retirement metadata")
    total = sum(pin["size_bytes"] for pin in files.values())
    if measured["result_bytes"] != total:
        raise ValueError("Measured result bytes omit or change retained attempt files")
    if verify_payloads:
        actual_root = Path(inventory["root"])
        if actual_root.is_symlink() or actual_root.resolve() != actual_root:
            raise ValueError("Qualification retained output must use its original owned root")
        actual = {}
        for path in actual_root.rglob("*"):
            if path.is_symlink() or path.resolve() != path:
                raise ValueError("Retained output cannot traverse links")
            if path.is_file():
                actual[path.relative_to(actual_root).as_posix()] = descriptor(path)
            elif not path.is_dir():
                raise ValueError("Unclassified retained output entry")
        if actual != files:
            raise ValueError("Retained attempt inventory differs from all actual original output bytes")
    return total


def _receipt(measured, task, reader):
    from .preset_wasserstein_results import _owned_files, _summaries
    pin = measured["execution_receipt"]
    receipt, _ = reader(pin)
    result = _receipt_metadata(receipt, pin, task)
    read, finish = _owned_files(result, Path(pin["path"]).parent)
    if any(result[key] != value for key, value in _summaries(result, read).items()):
        raise ValueError("Measured original scientific summaries differ from their actual bytes")
    finish()
    _retained_output_bytes(measured, result, verify_payloads=True)
    return receipt


def _evidence(entry, task, scope, *, reader, verify_payloads=False):
    # Neither valid scientific bytes nor declarations prove the original writer
    # lifetime or native retirement. Both qualification and reuse stop here.
    raise ValueError(MEASURED_UNAVAILABLE)


def resolve_profile(profile, task, scope, *, reader):
    """Reject measured reuse until original writer/lifetime/retirement proof exists."""
    if (not isinstance(profile, dict) or set(profile) != {
            "format", "commit", "pipeline", "backend", "precision", "entries", "qualification_host", "repository"}
            or profile["format"] != PROFILE_FORMAT or profile["commit"] != task["commit"]
            or profile["pipeline"] != PIPELINE or profile["backend"] != "cpu"
            or profile["precision"] != "original" or not isinstance(profile["entries"], list)
            or not profile["entries"]):
        raise ValueError("An actual CPU/original preset comparison measured profile is required; no native profile reuse")
    if (not isinstance(profile["qualification_host"], str)
            or profile["qualification_host"].casefold() != socket.gethostname().casefold()):
        raise ValueError("Preset qualification belongs to another machine; provide this machine's own observed profile and replay")
    scopes = [entry.get("scope") for entry in profile["entries"] if isinstance(entry, dict)]
    if len(scopes) != len(profile["entries"]) or len({digest(item) for item in scopes}) != len(scopes):
        raise ValueError("Distinct exact preset measurement scopes required")
    matches = [entry for entry in profile["entries"] if entry["scope"] == scope]
    if len(matches) != 1:
        raise ValueError("No measured profile for this original preset configuration, time and comparison count; no claim")
    entry = matches[0]
    measured, complete = _evidence(entry, task, scope, reader=reader)
    replay, _ = reader(entry["replay"])
    if _key(replay["repository"]) != _key(profile["repository"]):
        raise ValueError("Preset replay and profile source checkout differ")
    resolved = {key: deepcopy(profile[key]) for key in ("format", "commit", "pipeline", "backend", "precision")}
    resolved.update(deepcopy(entry), maximum_cache_bytes=measured["maximum_observed_cache_bytes"])
    return resolved, measured, complete


def require_service_sources(repository, producer_paths, runtime_paths):
    required = {_key(Path(repository) / relative) for relative in REQUIRED_SERVICE_SOURCES}
    if not required <= set(producer_paths):
        raise ValueError("Preset qualification omits an actual original reader/science/output/cache boundary")
    for suffix in ("numpy/lib/_npyio_impl.py", "scipy/sparse/linalg/_dsolve/linsolve.py"):
        # Older NumPy keeps the same NPZ implementation in npyio.py.
        candidates = (suffix, "numpy/lib/npyio.py") if suffix.startswith("numpy/") else (suffix,)
        if not any(path.replace("\\", "/").endswith(candidate)
                   for path in runtime_paths for candidate in candidates):
            raise ValueError("Preset runtime reader/solver is outside the startup source closure: " + suffix)
    return True


def require_repository_sources(repository, commit, coverage):
    """Capacity projection checks real current/committed producer bytes too."""
    from darsia_cache.worker_adapter import _source_pins
    producers = _source_pins(coverage["producer_source_closure"])
    runtime = _source_pins(coverage["runtime_source_closure"])
    require_service_sources(repository, producers, runtime)
    root = Path(repository).resolve()
    for pin in coverage["producer_source_closure"]:
        path = Path(pin["path"]).resolve()
        if not path.is_relative_to(root):
            continue  # The admitted service separately verifies external writers.
        relative = path.relative_to(root).as_posix()
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        version = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
        if (version(before) != version(after) or len(raw) != pin["size_bytes"]
                or hashlib.sha256(raw).hexdigest() != pin["sha256"]):
            raise ValueError("Actual preset producer source differs from its observed identity: " + relative)
        committed = subprocess.run(["git", "show", commit + ":" + relative], cwd=root,
            check=True, capture_output=True).stdout
        if committed.replace(b"\r\n", b"\n") != raw.replace(b"\r\n", b"\n"):
            raise ValueError("Preset producer source differs from the exact measured commit: " + relative)


def preset_estimate(stores, series_id, task, operation, *, repository):
    from .pipeline.capacity import read_bound, settings
    scope = operation_scope(task, operation)
    try:
        _state, _reserve, binding = settings(stores, series_id)
    except ValueError as error:
        raise ValueError("Preset comparison requires its own observed whole-time capacity profile: " + str(error)) from error
    profile, _ = read_bound(binding)
    from darsia_cache.preset_qualification import PROVIDER
    if profile.get("format") == PROVIDER:
        from darsia_cache.first_qualification import resolve_profile as resolve_provider
        resolved = resolve_provider(binding, "jobs", lambda pin: read_bound(pin)[0],
            stores_source={key: stores.source[key] for key in ("path", "sha256", "size_bytes")})
        return _bounded_estimate(task, operation, resolved, binding, repository)
    if _key(profile.get("repository", "")) != _key(Path(repository).resolve()):
        raise ValueError("Preset profile requires its exact locally qualified source checkout; no implicit remote source reads")
    selected, measured, _ = resolve_profile(profile, task, scope, reader=read_bound)
    require_repository_sources(repository, task["commit"], measured["worker_cache_service_coverage"])
    return PresetWassersteinEstimate(task["run"], 1, task["expected_comparisons"],
        measured["maximum_output_bytes"], measured["maximum_transient_bytes"], dict(binding),
        selected["measurement"], selected["completion"], scope, task["commit"])


def write_preset_wasserstein_profile(records, output, *, repository):
    """Reject measured qualification until the original proof verifier exists.

    The evidence gate refuses every record before any profile or replay write.
    Consistent declarations, hashes and scientific bytes are not that proof.
    """
    from .pipeline.capacity import read_bound
    from .pipeline.job_queue import descriptor, write_immutable
    if not isinstance(records, list) or not records:
        raise ValueError("Real observed preset task records required")
    entries, commit, replays = [], None, []
    path = Path(output).resolve()
    repository = Path(repository).resolve()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"task", "operation", "measurement", "completion"}:
            raise ValueError("Qualification records require actual task/operation and bound evidence")
        task = record["task"]
        scope = operation_scope(task, record["operation"])
        if commit is not None and commit != task["commit"]:
            raise ValueError("One exact measured source commit per preset profile required")
        commit = task["commit"]
        entry = dict(scope=scope, measurement=record["measurement"], completion=record["completion"], replay=None)
        measured, _ = _evidence(entry, task, scope, reader=read_bound, verify_payloads=True)
        require_repository_sources(repository, commit, measured["worker_cache_service_coverage"])
        coverage = measured["worker_cache_service_coverage"]
        namespace, _ = read_bound(coverage["namespace_admission"])
        if str(namespace.get("host", "")).casefold() != socket.gethostname().casefold():
            raise ValueError("Actual preset qualification namespace belongs to another physical host")
        receipt, _ = read_bound(measured["execution_receipt"])
        replay = dict(format="darsia-preset-wasserstein-qualified-replay-v1", scope=scope,
            measurement=record["measurement"], completion=record["completion"],
            execution_receipt=measured["execution_receipt"], receipt=receipt,
            qualification_host=socket.gethostname(), repository=str(repository),
            source_closure_sha256=digest(coverage), actual_original_bytes_verified=True)
        replay_path = path.with_name(path.stem + ".entry-" + str(len(entries)).zfill(8) + ".json")
        replays.append((replay_path, replay))
        entries.append(entry)
    if len({digest(entry["scope"]) for entry in entries}) != len(entries):
        raise ValueError("Repeated original preset qualification scope")
    value = dict(format=PROFILE_FORMAT, commit=commit, pipeline=PIPELINE,
                 backend="cpu", precision="original", entries=entries,
                 qualification_host=socket.gethostname(), repository=str(repository))
    for entry, (replay_path, replay) in zip(entries, replays):
        write_immutable(replay_path, replay)
        entry["replay"] = descriptor(replay_path)
    write_immutable(path, value)
    return descriptor(path)


def validate_retained_reservation(row):
    estimate = row.get("estimate", {})
    if (row.get("format") != "darsia-local-capacity-reservation-v1"
            or row.get("reservation_model") != RESERVATION_MODEL or row.get("case_count") != 1
            or row.get("ordered_case_ids") != [row.get("run")]
            or row.get("maximum_output_bytes") != estimate.get("products_estimated_bytes")
            or row.get("transient_estimated_bytes") != estimate.get("transient_estimated_bytes")
            or row.get("comparison_count") != estimate.get("comparison_count")
            or row.get("scope") != estimate.get("scope")
            or row.get("scope", {}).get("run") != row.get("run")
            or row.get("scope", {}).get("pipeline") != PIPELINE
            or row.get("comparison_count") != row.get("scope", {}).get("expected_comparisons")):
        raise ValueError("Original preset whole-time reservation identity differs")
    _integer(row["maximum_output_bytes"], "Reserved whole-time output", positive=True)
    _integer(row["transient_estimated_bytes"], "Reserved whole-time transient")
    return True
