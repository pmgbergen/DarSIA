"""Explicit, isolated exact-runtime launch for original staged reproduction.

This file is also the standalone child driver: its top-level imports are all
stdlib. The selected interpreter starts with -I -S -B, checks its environment
and the selected checkout, and only then imports that checkout's science.
No checkout, environment, queue, worker guard, or numerical body is modified.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import re
import subprocess
import sys
import sysconfig
import traceback

FORMAT = "darsia-staged-runtimes-v1"
PROBE_FORMAT = "darsia-staged-runtime-probe-v1"
RECEIPT_FORMAT = "darsia-staged-runtime-result-v1"


def _json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate staged runtime JSON key: " + key)
            value[key] = item
        return value

    def constant(value):
        raise ValueError("Nonfinite staged runtime JSON value: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _absolute(value, *, directory=False):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("Select an explicit absolute runtime path")
    path = Path(value).absolute()
    if directory and not path.is_dir():
        raise ValueError("The selected runtime repository is not a directory")
    if not directory and not path.is_file():
        raise ValueError("The selected runtime file is unavailable")
    return path


def _relative(root, value):
    if (not isinstance(value, str) or not value or "\\" in value or "\x00" in value
            or PureWindowsPath(value).drive or value.startswith("/")
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError("A normalized portable relative asset path is required")
    base = Path(root).resolve()
    path = base.joinpath(*value.split("/"))
    if not path.resolve().is_relative_to(base):
        raise ValueError("A staged runtime asset escapes its explicit root")
    return path


def _read(path):
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    stamp = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if stamp(before) != stamp(after) or len(raw) != after.st_size:
        raise ValueError("A staged runtime file changed while being read")
    return raw


def _bound(bound, *, root=None):
    if not isinstance(bound, dict) or re.fullmatch(r"[0-9a-f]{64}", str(bound.get("sha256", ""))) is None:
        raise ValueError("An exact staged runtime file binding is required")
    path = _absolute(bound.get("path")) if root is None else _relative(root, bound.get("path"))
    raw = _read(path)
    if _sha(raw) != bound["sha256"] or ("size_bytes" in bound and (
            type(bound["size_bytes"]) is not int or len(raw) != bound["size_bytes"])):
        raise ValueError("A staged runtime file differs from its original SHA/size binding")
    return path, raw


def _asset(path):
    path = Path(path).absolute()
    raw = _read(path)
    return dict(path=str(path), sha256=_sha(raw), size_bytes=len(raw))


def _normal_name(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def _pins(raw):
    pins = {}
    for line in raw.decode("utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if match is None or _normal_name(match[1]) in pins:
            raise ValueError("The original runtime lock needs unique exact name==version pins")
        pins[_normal_name(match[1])] = match[2]
    if not pins:
        raise ValueError("The original runtime lock has no pinned libraries")
    return pins


def load_runtimes(mapping_path, stage_jobs):
    """Read an explicit complete stage-id -> {python, repository} selection."""
    value = _json(Path(mapping_path).read_bytes())
    stages = value.get("stages") if isinstance(value, dict) else None
    if (not isinstance(value, dict) or value.get("format") != FORMAT or not isinstance(stages, dict)
            or set(stages) != set(stage_jobs)):
        raise ValueError("Runtime selection must name every exact reproduction stage once")
    result = {}
    for stage, selected in stages.items():
        if not isinstance(selected, dict) or set(selected) != {"python", "repository"}:
            raise ValueError("Each runtime needs only its explicitly selected Python and repository")
        python = _absolute(selected["python"])
        repository = _absolute(selected["repository"], directory=True).resolve()
        result[stage] = dict(stage=stage, python=str(python), repository=str(repository))
    return result


def _git(repository, *args, input=None):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_OPTIONAL_LOCKS="0", GIT_NO_REPLACE_OBJECTS="1")
    result = subprocess.run(["git", "--no-optional-locks", "-C", str(repository),
        "-c", "core.fsmonitor=false", *args], input=input, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=environment, timeout=60, check=False)
    if result.returncode:
        raise ValueError("The selected runtime Git checkout could not be verified: "
                         + result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def _checkout(repository, expected, lock_raw):
    """Check actual source against Git blobs; retain original EOL-only tolerance."""
    commit = expected.get("commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("The original stage job needs its exact code commit")
    top = Path(_git(repository, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if top != repository or _git(repository, "rev-parse", "HEAD").decode().strip() != commit:
        raise ValueError("The selected runtime checkout differs from the original stage commit")
    if _git(repository, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Each staged runtime needs a clean Git checkout")
    lock = expected["environment_lock"]
    lock_path = _relative(repository, lock["path"])
    original = _git(repository, "show", commit + ":" + lock["path"])
    if _sha(original) != lock["sha256"] or original != lock_raw:
        raise ValueError("The selected checkout's tracked lock differs from the original stage lock")
    if _read(lock_path).replace(b"\r\n", b"\n") != original.replace(b"\r\n", b"\n"):
        raise ValueError("The selected checkout's working lock differs from its Git bytes")
    entries = []
    for entry in _git(repository, "ls-tree", "-r", "-z", commit, "--", "src").split(b"\x00"):
        if not entry:
            continue
        header, raw_name = entry.split(b"\t", 1)
        mode, kind, oid = header.decode("ascii").split()
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError("The admitted source tree cannot contain unbound symlinks or submodules")
        relative = raw_name.decode("utf-8")
        entries.append((relative, oid))
    if not entries or "src/darsia/__init__.py" not in {name for name, _ in entries}:
        raise ValueError("The selected checkout has no original DarSIA source package")
    packed = _git(repository, "cat-file", "--batch", input="".join(oid + "\n" for _, oid in entries).encode())
    cursor, snapshot, rows = 0, {}, []
    for relative, oid in entries:
        end = packed.index(b"\n", cursor)
        header = packed[cursor:end].decode("ascii").split()
        if len(header) != 3 or header[:2] != [oid, "blob"]:
            raise ValueError("Unreadable tracked source object")
        size = int(header[2])
        original = packed[end + 1:end + 1 + size]
        cursor = end + 2 + size
        path = _relative(repository, relative)
        actual = _read(path)
        if actual.replace(b"\r\n", b"\n") != original.replace(b"\r\n", b"\n"):
            raise ValueError("Source bytes differ from the original stage commit: " + relative)
        snapshot[os.path.normcase(str(path.resolve()))] = _sha(actual)
        rows.append(dict(path=relative, git_blob=oid, working_sha256=_sha(actual), size_bytes=len(actual)))
    tracked = {relative for relative, _ in entries}
    for path in (repository / "src").rglob("*"):
        if path.is_file() and path.relative_to(repository).as_posix() not in tracked:
            if "__pycache__" in path.parts and path.suffix == ".pyc":
                continue  # The guarded loader below compiles source and never reads these.
            raise ValueError("An untracked or ignored file overlays the admitted source tree: " + str(path))
    tree = _git(repository, "rev-parse", commit + "^{tree}").decode().strip()
    return dict(commit=commit, tree=tree, clean=True, source_tree_sha256=_sha(_bytes(rows)),
                source_files=len(rows), source_identity="Git blobs with verified EOL-only working representation"), snapshot


def _software(expected, dataset_root):
    software = expected.get("software")
    if not isinstance(software, dict) or software.get("mode") not in {"references", "include"}:
        raise ValueError("Each stage needs its exact packaged software closure")
    if software.get("code", {}).get("commit") != expected.get("commit"):
        raise ValueError("The stage software commit differs from its original job")
    _, versions = _bound(software["inventory"], root=dataset_root)
    if _json(versions) != {key: value for key, value in software.items() if key != "inventory"}:
        raise ValueError("The original software inventory document differs from the stage manifest")
    _, raw = _bound(software["environment_lock"], root=dataset_root)
    if _sha(raw) != expected.get("environment_lock", {}).get("sha256"):
        raise ValueError("Packaged software lock differs from the original stage job")
    pins = _pins(raw)
    libraries = software.get("libraries")
    if not isinstance(libraries, list):
        raise ValueError("Stage software must contain every exact locked library")
    recorded = {_normal_name(row["name"]): row["version"] for row in libraries}
    if len(recorded) != len(libraries) or recorded != pins:
        raise ValueError("The stage software library set differs from its original lock")
    python = software.get("python")
    if not isinstance(python, str) or re.fullmatch(r"\d+\.\d+\.\d+", python) is None:
        raise ValueError("The stage software needs its exact Python patch version")
    family = re.search(r"Python\s+(\d+\.\d+)", raw.decode("utf-8-sig"))
    if family is None or not python.startswith(family[1] + "."):
        raise ValueError("The exact Python patch version differs from the original lock's Python family")
    return software, raw, pins


def _site_paths():
    """Read this selected environment's metadata without executing site or .pth."""
    executable = Path(sys.executable).absolute()
    configs = [path for path in (executable.parent / "pyvenv.cfg",
               executable.parent.parent / "pyvenv.cfg") if path.is_file()]
    if len(configs) > 1:
        raise ValueError("The selected Python has ambiguous virtual-environment metadata")
    prefix, system = Path(sys.prefix), False
    if configs:
        prefix = configs[0].parent
        values = {}
        for line in configs[0].read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key.strip().lower()] = value.strip()
        system = values.get("include-system-site-packages", "false").lower() == "true"
    prefixes = [prefix] + ([Path(sys.base_prefix)] if system else [])
    paths = []
    for item in prefixes:
        scheme = sysconfig.get_paths(vars={"base": str(item), "platbase": str(item)})
        for name in ("purelib", "platlib"):
            path = Path(scheme[name]).resolve()
            if path.is_dir() and path not in paths:
                paths.append(path)
    return prefix, paths


def _probe(runtime, expected, dataset_root):
    python = _absolute(runtime["python"])
    if os.path.normcase(str(python)) != os.path.normcase(str(Path(sys.executable).absolute())):
        raise ValueError("The runtime probe is not running in the explicitly selected Python")
    repository = _absolute(runtime["repository"], directory=True).resolve()
    software, lock, pins = _software(expected, dataset_root)
    if platform.python_version() != software["python"]:
        raise ValueError("The selected interpreter differs from the original exact Python patch version")
    code, snapshot = _checkout(repository, expected, lock)
    prefix, paths = _site_paths()
    installed = {}
    for distribution in importlib.metadata.distributions(path=[str(path) for path in paths]):
        name = _normal_name(distribution.metadata.get("Name", ""))
        if not name or name in installed:
            raise ValueError("The selected runtime has missing or duplicate distribution identities")
        installed[name] = distribution.version
    if any(installed.get(name) != version for name, version in pins.items()):
        raise ValueError("The selected runtime's installed libraries differ from the exact original lock")
    receipt = dict(format=PROBE_FORMAT, stage=runtime.get("stage"), python=platform.python_version(),
        executable=str(python), repository=str(repository), code=code,
        environment_lock_sha256=_sha(lock), libraries=dict(sorted(pins.items())),
        extra_distributions={name: version for name, version in sorted(installed.items()) if name not in pins},
        site_startup_executed=False, scientific_imports_started=False)
    return receipt, snapshot, prefix, paths


def _install_source_guard(repository, snapshot, prefix, sites):
    """Only import tracked selected source; compile .py bytes without cached pyc."""
    import importlib.abc
    import importlib.machinery
    import importlib.util

    source_root = repository / "src"
    package_names = {Path(path).relative_to(source_root).parts[0].removesuffix(".py")
                     for path in snapshot}

    def checked(path):
        actual = Path(path).resolve()
        raw = _read(actual)
        if snapshot.get(os.path.normcase(str(actual))) != _sha(raw):
            raise ImportError("Source import differs from the admitted original stage: " + str(actual))
        return raw

    class Source(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            return compile(checked(self.path), self.path, "exec", dont_inherit=True)

    class Guard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".", 1)[0] not in package_names:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.origin is None:
                raise ImportError("The selected stage source has no tracked module: " + fullname)
            origin = Path(spec.origin).resolve()
            if not origin.is_relative_to(source_root):
                raise ImportError("The runtime selected a package outside its exact source checkout")
            checked(origin)
            if origin.suffix != ".py":
                raise ImportError("Repository code imports require their original tracked Python source")
            spec.loader = Source(fullname, str(origin))
            spec.cached = None
            return spec

    if any(name.split(".", 1)[0] in package_names for name in sys.modules):
        raise ValueError("Stage repository code was imported before exact runtime validation")
    sys.prefix = sys.exec_prefix = str(prefix)
    sys.path[:] = [str(source_root), *sys.path, *[str(path) for path in sites]]
    sys.meta_path.insert(0, Guard())


def _request(runtime, expected_job, dataset_root, **extra):
    return dict(runtime=deepcopy(runtime), expected_job=deepcopy(expected_job),
                dataset_root=str(Path(dataset_root).resolve()), driver=_asset(__file__), **extra)


def validate_runtime(runtime, expected_job, *, dataset_root):
    """Probe the explicit interpreter/checkout in a child without scientific imports."""
    spec = _request(runtime, expected_job, dataset_root, action="probe")
    process = subprocess.run([runtime["python"], "-I", "-S", "-B", str(Path(__file__).resolve())],
        input=_bytes(spec), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
    if process.returncode:
        raise ValueError("Stage runtime probe refused: " + process.stderr.decode("utf-8", errors="replace").strip())
    result = _json(process.stdout)
    if result.get("format") != PROBE_FORMAT:
        raise ValueError("The selected runtime returned no verified probe receipt")
    return result


def launch_stage(runtime, expected_job, execution, *, dataset_root, receipt_path, log_path):
    """Start a stage; return Popen. The child revalidates and writes a new receipt.

    execution requires task={absolute path,sha256,size_bytes},
    stores (absolute store-map path), series_id, job_id, attempt_id and backend.
    Optional threads is an explicit positive override; control is an absolute
    replay control JSON path. Pauses return state=paused without coupling.
    P1 coupling accepts helper={absolute path,sha256,size_bytes} for this
    exporter's staged_replay.py, phase2={runtime,expected_job,execution}, and
    original_aliases (explicit same-byte historical-to-transport descriptors).
    P2 coupled_inputs binds the complete successful P1 receipt by absolute path.
    """
    receipt = Path(receipt_path).absolute()
    log = Path(log_path).absolute()
    if receipt.exists() or log.exists():
        raise FileExistsError("A staged launch needs fresh receipt and log paths")
    for path in (receipt, log):
        if path.resolve().is_relative_to(Path(runtime["repository"]).resolve()):
            raise ValueError("Staged runtime receipts must remain outside the selected source checkout")
        path.parent.mkdir(parents=True, exist_ok=True)
    spec = _request(runtime, expected_job, dataset_root, action="run",
                    execution=deepcopy(execution), receipt_path=str(receipt))
    with log.open("xb") as stream:
        process = subprocess.Popen([runtime["python"], "-I", "-S", "-B", str(Path(__file__).resolve())],
            stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        process.stdin.write(_bytes(spec))
        process.stdin.close()
    except BaseException:
        process.terminate()
        process.wait()
        raise
    return process


class _Paused(Exception):
    pass


def _pause(control):
    if control is None:
        return
    path = Path(control)
    if not path.is_absolute():
        raise ValueError("The replay control path must be explicitly absolute")
    if path.is_file() and _json(_read(path)).get("action") == "pause":
        raise _Paused()


def _execution_digest(execution):
    return _sha(_bytes({key: value for key, value in execution.items()
                        if key not in {"coupled_inputs", "coupling"}}))


def _admit(execution, expected, dataset_root, repository, *, metadata_only=False):
    from darsia.fluidflower.pipeline.job_queue import LocalOutputBinding
    from darsia.fluidflower.pipeline.run_io import RunIOBinding
    from darsia.fluidflower.pipeline.store_runtime import local_attempt_output, resolve_catalog, verify_inputs
    from darsia.fluidflower.store_map import read_store_map

    _, raw = _bound(execution["task"])
    original = _json(raw)
    if (original.get("commit") != expected["commit"] or original.get("pipeline") != expected.get("pipeline")
            or original.get("precision") != expected.get("precision")
            or original.get("run") not in expected.get("expected_runs", ())
            or execution["series_id"] != expected.get("series_id")):
        raise ValueError("The replay task differs from its original stage job identity")
    if original["pipeline"] not in {"native-phase-statistics", "n1"}:
        raise ValueError("Only the original phase-one and N1 stage entry points are supported")
    if re.fullmatch(r"[0-9a-f]{32}", str(execution.get("attempt_id", ""))) is None:
        raise ValueError("A fresh explicit stage attempt identity is required")
    task = dict(original, job_id=execution["job_id"], attempt_id=execution["attempt_id"])
    if "threads" in execution:
        if type(execution["threads"]) is not int or execution["threads"] < 1:
            raise ValueError("An explicit positive stage thread count is required")
        task["threads"] = execution["threads"]
    stores_path = _absolute(execution["stores"])
    stores = read_store_map(stores_path, forbidden_roots=(repository,))
    rows = resolve_catalog(task, stores, execution["series_id"])
    if not metadata_only and execution.get("coupled_inputs") is not None:
        rows = _coupled_rows(execution, task, rows, dataset_root)
    verified = verify_inputs(task, rows)
    if metadata_only:
        # The P1 checkout can predate additive P2 metadata. Only original byte
        # admission runs here; actual P2 RunIO replay remains mandatory in P2.
        return task, rows, None, verified
    output, evidence = local_attempt_output(stores, execution["series_id"], task["job_id"],
                                            task["run"], task["attempt_id"])
    if output.exists():
        raise FileExistsError("Each staged reproduction attempt must use a fresh output path")
    local = LocalOutputBinding(stores.machine_id, Path(stores.roots(execution["series_id"])["results"]),
                               output, evidence)
    binding = RunIOBinding.replay(task, verified, local, repository=repository, admitted_commit=expected["commit"])
    return task, rows, output, binding


def _coupled_rows(execution, task, originals, dataset_root):
    from darsia.fluidflower.pipeline.store_runtime import ResolvedInput

    _, raw = _bound(execution["coupled_inputs"])
    result = _json(raw)
    coupling = result.get("coupling") or {}
    proof = coupling.get("receipt") or {}
    configurations = [row for row in originals if "run.configuration" in row.uses]
    if task.get("pipeline") != "n1" or len(configurations) != 1:
        raise ValueError("Coupled phase-one inputs require the original N1 configuration consumer")
    configuration_row = configurations[0]
    _, configuration_raw = _bound(dict(path=str(configuration_row.path.absolute()),
        sha256=configuration_row.reference["sha256"], size_bytes=configuration_row.reference["bytes"]))
    configuration = _json(configuration_raw)
    phase1 = configuration.get("phase1") or {}
    if (result.get("format") != RECEIPT_FORMAT or result.get("state") != "done"
            or coupling.get("phase2_execution_sha256") != _execution_digest(execution)
            or coupling.get("phase2_task") != execution["task"]
            or proof.get("format") != "darsia-staged-reproduction-coupling-v1"
            or proof.get("complete") is not True or proof.get("run") != task["run"]
            or proof.get("precision") != task["precision"]
            or proof.get("ordered_case_ids") != task["ordered_case_ids"]
            or proof.get("phase2_configuration", {}).get("sha256") != task["configuration"]["sha256"]
            or proof.get("phase2_configuration", {}).get("size_bytes") != task["configuration"]["bytes"]
            or proof.get("phase1_code_commit") != phase1.get("code_commit")
            or proof.get("original_phase1_completion") != phase1.get("completion")
            or proof.get("original_phase1_configuration") != phase1.get("configuration")
            or result.get("runtime", {}).get("code", {}).get("commit") != proof.get("phase1_code_commit")):
        raise ValueError("The coupled phase-one proof differs from this exact phase-two execution")
    rows = coupling.get("rows")
    if not isinstance(rows, list) or len(rows) != len(originals):
        raise ValueError("Coupled rows must preserve the complete original phase-two catalogue")
    replacements = {}
    for item in proof.get("substitutions", ()):
        key = _bytes(dict(reference=item["reference"], uses=item["uses"]))
        if key in replacements:
            raise ValueError("The coupling proof has duplicate input substitutions")
        replacements[key] = item["replayed"]
    rebuilt, consumed = [], set()
    for old, row in zip(originals, rows):
        if row.get("reference") != old.reference or row.get("uses") != list(old.uses):
            raise ValueError("Coupled rows change original input references, roles, or order")
        path = _absolute(row.get("path"))
        key = _bytes(dict(reference=old.reference, uses=list(old.uses)))
        replacement = replacements.get(key)
        if replacement is None:
            if path.resolve() != old.path.resolve():
                raise ValueError("An input location changed without original phase-one coupling proof")
        else:
            if (path != Path(replacement["path"]).absolute()
                    or replacement["sha256"] != old.reference["sha256"]
                    or replacement["size_bytes"] != old.reference["bytes"]):
                raise ValueError("The new input location differs from its exact regenerated payload")
            if not path.resolve().is_relative_to(Path(result["output"]).resolve()):
                raise ValueError("Regenerated input escapes the successful phase-one owned attempt")
            consumed.add(key)
        rebuilt.append(ResolvedInput(deepcopy(old.reference), path, tuple(old.uses)))
    if consumed != set(replacements):
        raise ValueError("The phase-one coupling proof has unconsumed input substitutions")
    return tuple(rebuilt)


def _couple(execution, result, binding, output, dataset_root):
    import importlib.util
    from darsia.fluidflower.mass.n1_pipeline import DataReader, asset
    from darsia.fluidflower.pipeline.run_io import current_binding

    request = execution["coupling"]
    if current_binding() is not None:
        raise ValueError("Historical coupling verification must start outside active scientific I/O")
    aliases = request.get("original_aliases")
    if not isinstance(aliases, list):
        raise ValueError("Explicit verified historical phase-one transport aliases are required")
    original_reader = DataReader(aliases=aliases, unc_only=False)
    helper_path, _ = _bound(request["helper"])
    if helper_path.resolve() != Path(__file__).with_name("staged_replay.py").resolve():
        raise ValueError("Coupling must use this exporter's explicitly verified replay helper")
    phase2 = request["phase2"]
    repository = _absolute(phase2["runtime"]["repository"], directory=True).resolve()
    _, lock, _ = _software(phase2["expected_job"], dataset_root)
    _checkout(repository, phase2["expected_job"], lock)
    task, rows, _, original_verified = _admit(phase2["execution"], phase2["expected_job"],
                                             dataset_root, repository, metadata_only=True)
    if task["pipeline"] != "n1":
        raise ValueError("Phase-one coupling requires the original N1 consumer task")
    with binding.activate():
        replay_reader = DataReader()
    module_name = "darsia.fluidflower.published_dataset._b6_staged_replay"
    specification = importlib.util.spec_from_file_location(module_name, helper_path)
    helper = importlib.util.module_from_spec(specification)
    # Execute the exact bytes just verified, avoiding cached bytecode entirely.
    _, helper_raw = _bound(request["helper"])
    exec(compile(helper_raw, str(helper_path), "exec", dont_inherit=True), helper.__dict__)
    configuration = original_verified.metadata("run.configuration")
    completion = asset(output / "phase1_completed.json")
    replacements, receipt = helper.verify_coupling(configuration, configuration["phase1"]["completion"],
        completion, original_reader, replay_reader, rows)
    return dict(phase1_task=deepcopy(execution["task"]), phase2_task=deepcopy(phase2["execution"]["task"]),
        phase2_execution_sha256=_execution_digest(phase2["execution"]), helper=deepcopy(request["helper"]),
        receipt=receipt, rows=[dict(reference=deepcopy(row.reference), path=str(row.path.absolute()),
                                   uses=list(row.uses)) for row in replacements])


def _run(spec, probe):
    execution, expected = spec["execution"], spec["expected_job"]
    dataset_root = Path(spec["dataset_root"])
    repository = Path(spec["runtime"]["repository"]).resolve()
    receipt = dict(format=RECEIPT_FORMAT, state="failed", runtime=probe,
                   task=deepcopy(execution["task"]), execution_sha256=_execution_digest(execution),
                   reproduction=True, qualified=False, production=False)
    try:
        _pause(execution.get("control"))
        task, _, output, binding = _admit(execution, expected, dataset_root, repository)
        receipt["output"] = str(output.absolute())
        boundaries = {"frame_started", "phase1_frame_started", "phase1_frame_completed"} if task["pipeline"] == "native-phase-statistics" else {
            "frame_started", "frame_completed"}

        def progress(event):
            if not isinstance(event, dict):
                raise TypeError("Original stage progress must be an event dictionary")
            print(json.dumps(dict(stage_progress=event), allow_nan=False), flush=True)
            if event.get("state") in boundaries:
                _pause(execution.get("control"))

        if task["pipeline"] == "native-phase-statistics":
            from darsia.fluidflower.pipeline.phase_statistics import run_phase1 as function
        else:
            from darsia.fluidflower.n1_pipeline import run_n1 as function
        science_task = dict(task, environment_lock=deepcopy(expected["environment_lock"]))
        with binding.activate():
            result = function(science_task, output, execution["backend"], task["precision"], progress)
            if (result.get("precision") != task["precision"] or result.get("backend") != execution["backend"]
                    or result.get("complete") is not True or result.get("whole_run") is not True
                    or result.get("ordered_case_ids") != task["ordered_case_ids"]):
                raise ValueError("Original stage result differs from its exact backend, precision, or complete selection")
            receipt["result"] = result
        if execution.get("coupling") is not None:
            if task["pipeline"] != "native-phase-statistics":
                raise ValueError("Only completed phase one can produce a coupling proof")
            receipt["coupling"] = _couple(execution, result, binding, output, dataset_root)
        receipt["state"] = "done"
    except _Paused:
        receipt["state"] = "paused"
    except Exception as error:
        receipt.update(error=type(error).__name__ + ": " + str(error), traceback=traceback.format_exc())
    return receipt


def main():
    """Read one driver request from stdin; never treat the selected data as code."""
    spec = _json(sys.stdin.buffer.read())
    try:
        path, _ = _bound(spec["driver"])
        if path.resolve() != Path(__file__).resolve():
            raise ValueError("The exact requested staged runtime driver is required")
        if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:
            raise ValueError("The staged runtime needs isolated startup without site hooks or bytecode writes")
        probe, snapshot, prefix, sites = _probe(spec["runtime"], spec["expected_job"], spec["dataset_root"])
        if spec["action"] == "probe":
            print(json.dumps(probe, sort_keys=True, allow_nan=False))
            return 0
        if spec["action"] != "run":
            raise ValueError("Unknown staged runtime operation")
        _install_source_guard(Path(spec["runtime"]["repository"]).resolve(), snapshot, prefix, sites)
        receipt = _run(spec, probe)
    except Exception as error:
        if spec.get("action") != "run":
            print(type(error).__name__ + ": " + str(error), file=sys.stderr)
            return 1
        receipt = dict(format=RECEIPT_FORMAT, state="failed", error=type(error).__name__ + ": " + str(error),
                       traceback=traceback.format_exc(), reproduction=True, qualified=False, production=False)
    path = Path(spec["receipt_path"])
    with path.open("xb") as stream:
        stream.write(_bytes(receipt))
    return 0 if receipt["state"] in {"done", "paused"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
