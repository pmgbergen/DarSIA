"""Prepare the selected Setup service after an explicit GUI Start.

This stdlib entry does metadata preparation, then invokes the unchanged public
worker once. It does not Apply, Build, decode RAW, qualify a measured profile or
retry a worker. The original native supervisor and Setup sinks own execution.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import sysconfig
import tomllib
import uuid
from datetime import datetime, timezone
import traceback

from .policy import CachePolicyError, PHYSICAL_RESERVE_BYTES, _descriptor, cache_settings

CAPTURE_ENV = "DARSIA_SETUP_STARTUP_CAPTURE"
HOST_FORMAT = "darsia-setup-cache-host-v1"


def source_pin(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise CachePolicyError("Setup source changed while its launch was prepared")
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=after.st_size)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _toml_value(value):
    if isinstance(value, dict):
        return "{ " + ", ".join(json.dumps(str(key)) + " = " + _toml_value(item)
                                 for key, item in value.items()) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if type(value) in {str, int, bool}:
        return json.dumps(value, ensure_ascii=False)
    raise CachePolicyError("Unsupported value in the selected local stores metadata")


def stores_bytes(document):
    return ("\n".join(json.dumps(key) + " = " + _toml_value(value)
                     for key, value in document.items()) + "\n").encode("utf-8")


def read_host(stores_document, *, package_file=__file__):
    from .bootstrap import _exact
    block = stores_document.get("setup_cache")
    if not isinstance(block, dict) or set(block) != {"host_policy"}:
        raise CachePolicyError("Choose this machine's existing Setup cache host policy before Start")
    host = _exact(_descriptor(block["host_policy"]), package_file)
    if (set(host) != {"format", "cache", "capacity", "cache_bootstrap", "capacity_bound"}
            or host["format"] != HOST_FORMAT
            or set(host["cache"]) != {"root", "quota_bytes", "failed_retention_hours", "volume_cap"}
            or set(host["capacity"]) != {"state_directory", "reserve_bytes"}
            or host["capacity"]["reserve_bytes"] != PHYSICAL_RESERVE_BYTES
            or set(host["cache_bootstrap"]) != {"namespace_admission", "accounting_entry_limit"}):
        raise CachePolicyError("Exact existing capped Setup cache host configuration required")
    return host


def capture_startup():
    """Metadata-only pre-import probe; always exit before the application loads."""
    from .bootstrap import _exact
    from . import setup_session, owned_process, service_context, adapters  # Same pre-import dependencies.
    # Windows spawn imports these stdlib modules before unpickling the public
    # worker target. Declare that actual child entry as well as the controller.
    import multiprocessing.spawn
    import multiprocessing.queues
    import multiprocessing.popen_spawn_win32
    import multiprocessing.synchronize
    from .windows import WindowsNamespace
    # The ordinary metadata parser loads this codec before admission too.
    b"{}".decode("utf-8-sig")
    request = json.loads(os.environ[CAPTURE_ENV])
    host = _exact(_descriptor(request["host_policy"]), __file__)
    admission = _exact(host["cache_bootstrap"]["namespace_admission"], __file__)
    target = admission["receipts"]
    name = request["name"]
    if Path(name).name != name or not name.endswith(".startup.json"):
        raise CachePolicyError("One flat Setup startup metadata name is required")
    modules = {name: str(Path(module.__file__).resolve()) for name, module in tuple(sys.modules.items())
               if getattr(module, "__file__", None) and not name.startswith("darsia.")}
    paths = set(modules.values()) | {sys.executable, sys._base_executable}
    startup = dict(format="darsia-qualified-worker-startup-source-v1",
        python_executable=source_pin(sys.executable),
        sys_path=[str(Path(value or os.getcwd()).resolve()) for value in sys.path],
        include_system_site_packages=False, allowed_loaded_modules=modules,
        source_files=[source_pin(path) for path in sorted(paths)], capture_only=True)
    namespace = WindowsNamespace(target["path"], target["identity"])
    try:
        namespace.write_once(name, _json_bytes(startup), maximum=16 * 1024**2)
    finally:
        namespace.close()
    raise SystemExit(0)


def source_closures(repository, startup):
    """Bind current installed sources; never reuse another job's measurement."""
    repository = Path(repository).resolve(strict=True)
    files = subprocess.check_output(["git", "-C", str(repository), "ls-files", "-z", "--", "src"], text=True)
    producer_paths = {str((repository / name).resolve(strict=True)) for name in files.split("\0")
                      if name and Path(name).suffix in {".py", ".pyd"}}
    runtime_paths = {pin["path"] for pin in startup["source_files"]} - producer_paths
    roots = {Path(sysconfig.get_path("stdlib")), Path(sysconfig.get_path("purelib")),
             Path(sys.base_prefix) / "DLLs"}
    for root in roots:
        if not root.is_dir():
            continue
        for parent, dirs, names in os.walk(root):
            dirs[:] = [name for name in dirs if name not in {"__pycache__", ".git"}]
            for name in names:
                if Path(name).suffix.casefold() in {".py", ".pyd", ".pth"}:
                    runtime_paths.add(str((Path(parent) / name).resolve(strict=True)))
    runtime_paths -= producer_paths
    return ([source_pin(path) for path in sorted(producer_paths)],
            [source_pin(path) for path in sorted(runtime_paths)])


def _configuration_path(stores, job, task):
    ref = task["configuration"]
    relative = PurePosixPath(ref["relative_path"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in ref["relative_path"]:
        raise CachePolicyError("Setup configuration must retain its original portable locator")
    path = Path(stores["stores"][job["series_id"]][ref["root_alias"]]).joinpath(*relative.parts)
    pin = source_pin(path)
    if pin["sha256"] != ref["sha256"] or pin["size_bytes"] != ref["bytes"]:
        raise CachePolicyError("Queued Setup configuration bytes have changed")
    return pin


def prepare(command, *, stores_path, repository, environment=None):
    """Publish only bounded local launch metadata. No worker is started here."""
    from .bootstrap import _exact, _explicit_document, _owner
    from .adapters import _worker_metadata
    from .setup_qualification import setup_control_metadata
    from .windows import WindowsNamespace
    from .first_qualification import _digest, checked_required_capacity_bound, REQUEST_ENV
    from .setup_session import (SetupJobService, PROVIDER, REQUEST, RELEASE,
                                checked_provider, require_command)
    original, original_pin = _explicit_document(stores_path, __file__)
    host = read_host(original)
    environment = dict(os.environ if environment is None else environment)
    job_root = Path(command[command.index("--job") + 1])
    job, job_pin = _worker_metadata(job_root / "job.json")
    repository = Path(repository).resolve(strict=True)
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True)
    if job.get("pipeline") != "series-setup" or dirty or job.get("commit") != head:
        raise CachePolicyError("Prepare the Setup job again in the current clean application version before Start")
    pending = sorted((job_root / "queue").glob("*.json"))
    if not pending:
        raise CachePolicyError("This Setup job has no pending operation to start")
    task, _ = _worker_metadata(pending[0])
    task_path = job_root / "results" / task["run"] / "task.json"
    durable_task, task_pin = _worker_metadata(task_path)
    if durable_task != task:
        raise CachePolicyError("Pending Setup task differs from its original immutable task")
    config_pin = _configuration_path(original, job, task)
    config = setup_control_metadata(config_pin)
    # The GUI may use the hostname's display casing. The immutable task and
    # configuration own the execution identity; retain their exact spelling in
    # the command and provider after verifying the selected local store host.
    target = task.get("target_machine")
    selected_machine = command[command.index("--machine") + 1]
    if (not isinstance(target, str) or not target
            or config.get("target_machine") != target
            or original["machine"]["id"].casefold() != target.casefold()
            or selected_machine.casefold() != target.casefold()):
        raise CachePolicyError("Selected Setup machine differs from its original task and configuration")
    command = list(command)
    command[command.index("--machine") + 1] = target
    admission = _exact(host["cache_bootstrap"]["namespace_admission"], __file__)
    policy_document = dict(original, cache=dict(host["cache"], profiles={"jobs": task_pin}))
    policy = cache_settings(policy_document, required=True)
    from .control import require_volume_cap_binding
    require_volume_cap_binding(policy, admission)
    namespace = WindowsNamespace(policy.root, admission["cache_identity"])
    try:
        observed = namespace.verify_volume_cap(policy.volume_cap)
    finally:
        namespace.close()
    capacity = _exact(host["capacity_bound"], __file__)
    capacity["volume_cap_observation"] = observed
    capacity["volume_capacity_bytes"] = capacity["required_capacity_bound_bytes"] = observed["volume_total_bytes"]
    capacity["operational_admission_headroom_bytes"] = observed["volume_total_bytes"] - PHYSICAL_RESERVE_BYTES
    checked_required_capacity_bound(capacity)
    _actual_owner, context = _owner()
    if (admission.get("host") != context["host"]
            or context["service_account_sid"] not in admission.get("authorized_sids", [])
            or str(Path(admission["host_ledger"]["path"])) != str(Path(context["common_root"]))
            or original["machine"]["id"].casefold() != command[command.index("--machine") + 1].casefold()):
        raise CachePolicyError("Selected Setup host policy belongs to another physical host/account")
    target = admission["receipts"]
    namespace = WindowsNamespace(target["path"], target["identity"])
    token = uuid.uuid4().hex
    prefix = "setup-service-" + token
    def publish(suffix, value, *, raw=False):
        name = prefix + suffix
        content = value if raw else _json_bytes(value)
        namespace.write_once(name, content, maximum=16 * 1024**2)
        return dict(path=str(Path(target["path"]) / name), sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content))
    try:
        print("Preparing the managed Setup worker startup (no RAW reads)...", flush=True)
        capture_name = prefix + ".startup.json"
        capture_env = dict(environment)
        capture_env[CAPTURE_ENV] = json.dumps(dict(host_policy=original["setup_cache"]["host_policy"], name=capture_name))
        probe = subprocess.run(command, cwd=Path.cwd(), env=capture_env, check=False)
        if probe.returncode:
            raise CachePolicyError("Setup pre-import metadata capture failed; worker was not started")
        startup_path = Path(target["path"]) / capture_name
        startup = _worker_metadata(startup_path)[0]
        producers, runtime = source_closures(repository, startup)
        startup["module_closure"] = producers + runtime
        startup_pin = publish(".startup-bound.json", startup)
        scope = {key: job[key] for key in ("commit", "pipeline", "backend", "precision")}
        scope.update(target_machine=command[command.index("--machine") + 1], job=job_pin)
        provider = dict(format=PROVIDER, qualification_id=token, scope=scope,
            namespace_admission=host["cache_bootstrap"]["namespace_admission"], output_namespace=target,
            slots={key: prefix + "." + key + ".json" for key in ("observation", "native_completion", "completion")},
            python_executable=source_pin(command[0]), startup=startup_pin, module_closure=producers + runtime,
            producer_source_closure=producers, runtime_source_closure=runtime,
            workers=int(command[command.index("--workers") + 1]))
        checked_provider(provider)
        provider_pin = publish(".provider.json", provider)
        selected = deepcopy(original)
        selected.pop("setup_cache", None)
        selected["cache"] = dict(host["cache"], profiles={"jobs": provider_pin})
        selected["capacity"] = dict(host["capacity"], profiles={job["series_id"]: provider_pin})
        role = "setup_job_service"
        selected["cache_bootstrap"] = dict(host["cache_bootstrap"],
            entry_roles={"runtime": role, "worker": role, "tests": role},
            roles={role: dict(purpose="jobs", profile=provider_pin)})
        cache_settings(selected, required=True)
        stores_pin = publish(".stores.toml", stores_bytes(selected), raw=True)
        command = list(command)
        command[command.index("--stores") + 1] = stores_pin["path"]
        request = dict(format=REQUEST, execution_authorized=True, provider=provider_pin,
            stores_source=stores_pin, namespace_admission=provider["namespace_admission"], public_command=command,
            task=task_pin, job=job_pin, job_root=str(job_root), configuration=config_pin,
            capacity_bound_plan=publish(".capacity.json", capacity),
            budget="fresh-fixed-volume-headroom-with-original-reserves", cache_stat_period_seconds=30)
        require_command(command, request, provider)
        release = dict(format=RELEASE, execution_authorized=True, provider=provider_pin,
            stores_source=stores_pin, scope=scope, request_payload_sha256=_digest(request))
        request["root_release"] = publish(".release.json", release)
        request_pin = publish(".request.json", request)
        # Full original constructor checks job/catalogue/source/authority bytes;
        # it neither reserves cache space nor starts a native process.
        SetupJobService(request_pin, provider_binding=provider_pin, stores_source=stores_pin,
            namespace=provider["namespace_admission"], read=lambda pin: _exact(pin, __file__))
        if _explicit_document(stores_path, __file__)[1] != original_pin:
            raise CachePolicyError("Selected host stores changed during Setup launch preparation")
        environment.pop(CAPTURE_ENV, None)
        environment.pop("DARSIA_CACHE_BOOTSTRAP_RECEIPT", None)
        environment[REQUEST_ENV] = json.dumps(request_pin)
        environment["DARSIA_CACHE_BOOTSTRAP_EXPECTED_STORES"] = json.dumps(stores_pin)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return command, environment, request_pin
    finally:
        namespace.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stores", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--launch-directory")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    command = options.command[1:] if options.command[:1] == ["--"] else options.command
    try:
        selected, environment, request = prepare(command, stores_path=options.stores, repository=options.repository)
    except Exception:
        if options.launch_directory:
            record_preparation_failure(command, options.launch_directory, traceback.format_exc())
        raise
    if options.launch_directory:
        try:
            record_prepared_service(selected, options.launch_directory, request)
        except OSError as error:
            print("Setup launch status link could not be saved: " + str(error), file=sys.stderr, flush=True)
    print("Setup cache admission is prepared. Starting the original worker...", flush=True)
    return subprocess.run(selected, cwd=Path.cwd(), env=environment, check=False).returncode


def record_prepared_service(command, launch_directory, service_request):
    """Link this UI launch to its existing original service receipt namespace."""
    root, launch = Path(command[command.index("--job") + 1]), Path(launch_directory)
    def reference(path):
        pin = source_pin(path)
        return dict(relative_path=path.relative_to(root).as_posix(),
                    sha256=pin["sha256"], bytes=pin["size_bytes"])
    value = dict(format="darsia-job-prepared-service-v1", job=reference(root / "job.json"),
        request=reference(launch.with_name(launch.name + ".request.json")),
        service_request=service_request)
    with (launch / "service-request.json").open("xb") as stream:
        stream.write(_json_bytes(value))


def record_preparation_failure(command, launch_directory, text):
    """Record only this helper's failure before it can invoke a scientific worker."""
    root = Path(command[command.index("--job") + 1])
    launch = Path(launch_directory)
    if launch.parent != root / "ui" / "launches":
        raise CachePolicyError("Preparation failure must stay in its original launch directory")
    request_path = launch.with_name(launch.name + ".request.json")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    def reference(path):
        pin = source_pin(path)
        return dict(relative_path=path.relative_to(root).as_posix(), sha256=pin["sha256"], bytes=pin["size_bytes"])
    if (request.get("format") != "darsia-job-ui-launch-request-v2"
            or request.get("job") != reference(root / "job.json")
            or request.get("launch_directory") != launch.relative_to(root).as_posix()):
        raise CachePolicyError("Preparation failure differs from the actual UI request/job")
    value = dict(format="darsia-job-startup-preparation-failure-v1", job=request["job"],
        request=reference(request_path), stage="prepare", worker_started=False,
        cache_scope_allocated=False, traceback=text, utc=datetime.now(timezone.utc).isoformat())
    with (launch / "preparation-failure.json").open("xb") as stream:
        stream.write(_json_bytes(value))


if __name__ == "__main__":
    raise SystemExit(main())
