"""Transparent expected disk space and host-local whole-run reservations.

The model uses an explicit measured native profile and saved writer metadata.
It does not alter products, promise a codec bound, inspect arrays, or own a NAS
lease. All services on one machine/volume MUST share capacity.state_directory.
"""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, fields as dataclass_fields
from pathlib import Path, PureWindowsPath
import ctypes
import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess

from .job_inputs import _json_bytes, MAX_METADATA_BYTES
from .job_queue import name, write_immutable
from .store_runtime import RunUnavailable

DEFAULT_RESERVE_BYTES = 40 * 1024**3
PROFILE_FORMAT = "darsia-expected-space-profile-v1"
MAX_RESERVATIONS = 4096


def _integer(value, where, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(where + " must be an explicit nonnegative integer")
    return value


def _bound(value):
    if (not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}
            or not isinstance(value["path"], str) or not value["path"]
            or not isinstance(value["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None):
        raise ValueError("Exact capacity metadata path/SHA/size required")
    _integer(value["size_bytes"], "Capacity metadata bytes")
    return value


def read_bound(value):
    """Read only an explicitly supplied, bounded JSON metadata file."""
    _bound(value)
    if value["size_bytes"] > MAX_METADATA_BYTES:
        raise ValueError("Capacity metadata exceeds the existing metadata limit")
    path = Path(value["path"])
    before = path.stat()
    if before.st_size != value["size_bytes"]:
        raise ValueError("Capacity metadata SHA/size/version changed before read: " + str(path))
    with path.open("rb") as stream:
        raw = stream.read(value["size_bytes"] + 1)
    after = path.stat()
    version = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if (version(before) != version(after) or len(raw) != value["size_bytes"]
            or hashlib.sha256(raw).hexdigest() != value["sha256"]):
        raise ValueError("Capacity metadata SHA/size/version changed: " + str(path))
    return _json_bytes(raw), raw


def settings(stores, series_id):
    """The map is local; capacity metadata is separate from root aliases."""
    from ..store_map import capacity_settings
    from ..series_paths import series_key
    value = capacity_settings(stores.document)
    if value is None or series_key(series_id) not in value["profiles"]:
        raise RunUnavailable("Configure an explicit applicable capacity profile in the local store map; no claim made")
    return Path(value["state_directory"]), value["reserve_bytes"], value["profiles"][series_key(series_id)]


def _native_layout_shape(value, count, where):
    if (not isinstance(value, list) or len(value) != count
            or any(type(v) is not int or v < 1 for v in value)):
        raise ValueError("Explicit positive native layout dimensions required: " + where)
    return list(value)


def _native_layout_for_task(stores, series_id, task, config, rows=None):
    """Only already-declared JSON metadata; never array/image reads or a claim.

    Rig frames and RAW-only frames (raw_rig) are read the same way on every
    route: the released reference build, the bound first qualification and the
    ordinary measured profiles. A RAW-only layout takes its native grid from the
    frames' one RAW source binding and names that binding, so it never equals a
    rig layout, and a profile or reference of one kind never matches the other.

    A phase-one statistics task runs the same native and phase stages on the
    same frame DATA, so its layout is read the same way and is the same
    native-phase-n1 stage layout.
    """
    from .store_runtime import resolve_catalog
    if ((task.get("pipeline"), config.get("format")) not in {
            ("native-phase-n1", "darsia-native-phase-n1-run-v1"),
            ("native-phase-statistics", "darsia-native-phase-statistics-run-v1")}):
        raise ValueError("This measured layout is only for the genuine native-phase-n1 route")
    if rows is None:
        rows = resolve_catalog(task, stores, series_id)

    def document(use, descriptor):
        matches = [row for row in rows if use in row.uses]
        if len(matches) != 1:
            raise ValueError("One existing layout metadata role required: " + use)
        row = matches[0]
        _bound(descriptor)
        if (row.reference.get("sha256") != descriptor["sha256"]
                or type(row.reference.get("bytes")) is not int
                or row.reference["bytes"] != descriptor["size_bytes"]):
            raise ValueError("Layout role SHA/size differs from original run DATA: " + use)
        return read_bound(dict(path=str(row.path), sha256=row.reference["sha256"],
                               size_bytes=row.reference["bytes"]))[0]

    phase = document("phase.configuration", config["phase"]["configuration"])
    native = document("native.configuration", config["native"]["configuration"])
    params = phase["parameters"]
    native_hw = _native_layout_shape(params["geometry"]["native_shape_hw"], 2, "phase geometry")
    chw = _native_layout_shape(native["photometry"]["native_shape_chw"], 3, "native photometry")
    if chw[1:] != native_hw:
        raise ValueError("Native channels/image dimensions disagree with phase geometry")
    for where, value in [
            ("native source", native["source"]["native_shape_hw"]),
            ("native extension", native["extension"]["native_shape_hw"]),
            ("native support", native["support"]["native_shape_hw"]),
            ("native photometry", native["photometry"]["native_shape_hw"]),
            ("scientific native", params["scientific"]["native_shape_hw"])]:
        if _native_layout_shape(value, 2, where) != native_hw:
            raise ValueError("Explicit native layout fields disagree: " + where)
    source_box = params["geometry"]["source_box_xywh"]
    if (not isinstance(source_box, list) or len(source_box) != 4
            or any(type(v) is not int for v in source_box)
            or source_box[2:] != native_hw[::-1]):
        raise ValueError("Explicit source-box/native-shape binding required")
    phase_hw = None
    frames = config["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError("Actual nonempty frame layout required")
    kinds = {frame["native_task"]["kind"] for frame in frames}
    raw = kinds == {"raw_rig"}
    if kinds != {"rig"} and not raw:
        raise ValueError("Unmeasured native processor route requires its own profile")
    raw_bindings = set()
    for frame in frames:
        case = name(frame["case_id"])
        n = document("frame." + case + ".native_geometry", frame["native_task"]["geometry"])
        if raw:
            descriptor = frame["native_task"]["processor"]["raw_binding"]
            binding = document("frame." + case + ".native.raw_binding", descriptor)
            if (binding.get("contract") != "darsia-raw-rig-source-v1"
                    or n.get("contract") != "darsia-raw-native-frame-v1"
                    or n.get("run") != binding.get("run") or binding.get("run") != task.get("run")):
                raise ValueError("RAW-only frame requires its exact source binding and frame contract: " + case)
            raw_bindings.add(descriptor["sha256"])
            native_frame_hw = binding["native_shape_hw"]
        else:
            native_frame_hw = n["native_shape_hw"]
        g = document("frame." + case + ".rig_geometry", frame["geometry"])
        shapes = [native_frame_hw, g["source_native_shape_hw"],
                  frame["mass_spec"]["original_domain"]["native_shape_hw"]]
        if any(_native_layout_shape(v, 2, case) != native_hw for v in shapes):
            raise ValueError("Requested frame native shape differs: " + case)
        actual = _native_layout_shape(g["phase_shape_hw"], 2, case + " phase canvas")
        if _native_layout_shape(frame["mass_spec"]["photo_top"]["shape_hw"], 2, case + " top") != actual:
            raise ValueError("Literal phase-canvas fields disagree: " + case)
        if phase_hw is None:
            phase_hw = actual
        elif actual != phase_hw:
            raise ValueError("Mixed native canvas layouts require an explicitly measured profile")
    dtype = params["features"]["alignment"]["rgb_dtype"]
    if not isinstance(dtype, str) or dtype != task["precision"]:
        raise ValueError("Original task precision and explicit feature RGB dtype differ")
    layout = dict(format="darsia-native-phase-n1-measured-layout-v1",
                  pipeline="native-phase-n1", precision=task["precision"],
                  native_shape_hw=native_hw, phase_shape_hw=phase_hw,
                  native_shape_chw=chw, feature_rgb_dtype=dtype,
                  source_box_xywh=list(source_box),
                  native_configuration_sha256=config["native"]["configuration"]["sha256"],
                  phase_configuration_sha256=config["phase"]["configuration"]["sha256"])
    if raw:
        if len(raw_bindings) != 1:
            raise ValueError("A RAW-only layout requires one RAW source binding")
        # Never equal to a rig layout, so no rig profile or reference can match it.
        layout.update(native_task_kind="raw_rig", raw_source_binding_sha256=raw_bindings.pop())
    return layout


# The layout fields a run takes from its own run DATA: its configurations, its
# RAW binding and where its native box sits in the camera frame. The box size is
# the native grid, which every layout check below pins.
_LAYOUT_RUN_FIELDS = {"native_configuration_sha256", "phase_configuration_sha256",
                      "raw_source_binding_sha256", "source_box_xywh"}


def _check_measured_native_layout(profile, requested, *, other_run=False):
    """The measured and requested layouts must be the same closed layout exactly.

    A rig layout has the original ten fields. A RAW-only layout has those plus
    native_task_kind 'raw_rig' and its RAW source binding SHA-256. Each kind is
    measured on its own frames; neither kind's profile applies to the other.
    With other_run the task is another run than the measured one: its own
    _LAYOUT_RUN_FIELDS may differ, while its kind, grid, canvas, channels and
    precision must equal the measured ones.
    """
    fields = {"format", "pipeline", "precision", "native_shape_hw", "phase_shape_hw",
              "native_shape_chw", "feature_rgb_dtype", "source_box_xywh",
              "native_configuration_sha256", "phase_configuration_sha256"}
    raw_fields = fields | {"native_task_kind", "raw_source_binding_sha256"}
    measured = profile.get("measured_native_layout")
    for where, value in (("measured", measured), ("requested", requested)):
        raw = isinstance(value, dict) and "native_task_kind" in value
        if (not isinstance(value, dict) or set(value) != (raw_fields if raw else fields)
                or (raw and (value["native_task_kind"] != "raw_rig"
                    or not isinstance(value["raw_source_binding_sha256"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", value["raw_source_binding_sha256"]) is None))
                or value["format"] != "darsia-native-phase-n1-measured-layout-v1"
                or value["pipeline"] != "native-phase-n1"
                or value["precision"] not in {"float32", "float64"}
                or value["feature_rgb_dtype"] != value["precision"]):
            raise ValueError("Explicit closed measured/requested native layout required: " + where)
        hw = _native_layout_shape(value["native_shape_hw"], 2, where + " native")
        _native_layout_shape(value["phase_shape_hw"], 2, where + " phase")
        chw = _native_layout_shape(value["native_shape_chw"], 3, where + " channels")
        box = value["source_box_xywh"]
        if (chw[1:] != hw or not isinstance(box, list) or len(box) != 4
                or any(type(v) is not int for v in box) or box[2:] != hw[::-1]):
            raise ValueError("Explicit native channel/source-box dimensions disagree: " + where)
        for key in ("native_configuration_sha256", "phase_configuration_sha256"):
            if not isinstance(value[key], str) or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None:
                raise ValueError("Explicit measured/requested native/phase configuration SHA required")
    if measured.get("native_task_kind") != requested.get("native_task_kind"):
        raise ValueError("Requested native processor kind differs from the measured profile: rig and "
                         "RAW-only (raw_rig) frames each need a profile measured on their own kind")
    compared = lambda value: json.dumps({key: item for key, item in value.items()
                                         if not (other_run and key in _LAYOUT_RUN_FIELDS)},
                                        sort_keys=True, separators=(",", ":"))
    if compared(measured) != compared(requested):
        raise ValueError("Requested native layout/configuration differs from measured profile; no size/default fallback")


def _writer_identity(repository, commit, row):
    if (not isinstance(row, dict) or set(row) != {"relative_path", "sha256"}
            or not isinstance(row["relative_path"], str)
            or not isinstance(row["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None):
        raise ValueError("Explicit writer relative path/SHA required")
    relative = row["relative_path"]
    pure = PureWindowsPath(relative)
    if pure.drive or pure.root or ".." in pure.parts or not relative.startswith("src/darsia/fluidflower/"):
        raise ValueError("Writer identity must be repository-relative")
    original = subprocess.check_output(["git", "-C", str(repository), "show", commit + ":" + relative])
    actual = (Path(repository) / relative).resolve(strict=True)
    if not actual.is_relative_to(Path(repository).resolve()):
        raise ValueError("Capacity writer escaped repository")
    before, raw = actual.stat(), actual.read_bytes()
    after = actual.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or hashlib.sha256(original).hexdigest() != row["sha256"]
            or raw.replace(b"\r\n", b"\n") != original.replace(b"\r\n", b"\n")):
        raise ValueError("Capacity profile writer identity differs")


def _profile_writer_identity(repository, measured_commit, current_commit, row):
    """Whole writer identity, except the exact named metadata/JSON-read fixes."""
    if not isinstance(row, dict):
        raise ValueError("Explicit writer relative path/SHA required")
    measured_sha = row.get("measured_sha256")
    if set(row) == {"relative_path", "sha256"}:
        _writer_identity(repository, current_commit, row)
        original = subprocess.check_output(["git", "-C", str(repository), "show", measured_commit + ":" + row["relative_path"]])
        if hashlib.sha256(original).hexdigest() != row["sha256"]:
            raise ValueError("Measured/current actual writer differs; a new profile is required")
        return
    if (set(row) != {"relative_path", "sha256", "measured_sha256"}
            or row.get("relative_path") != "src/darsia/fluidflower/mass/n1_pipeline.py"
            or not isinstance(measured_sha, str) or re.fullmatch(r"[0-9a-f]{64}", measured_sha) is None):
        raise ValueError("Only the exact n1_pipeline output-location metadata fix accepts separate measured writer provenance")
    _writer_identity(repository, current_commit, {k: row[k] for k in ("relative_path", "sha256")})
    trees, parsed = [], []
    expected = ("a7c441225da1bee963dd411dacc28a29efe8f20b2e233bf2e3aea91293048c56",
                "289d8766f0dedad5b5bde3b46d17cc8870e79530fd2aaa6b75d7956285b64060")
    for commit, sha, function_sha in ((measured_commit, measured_sha, expected[0]), (current_commit, row["sha256"], expected[1])):
        raw = subprocess.check_output(["git", "-C", str(repository), "show", commit + ":" + row["relative_path"]])
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("Measured/current metadata-only writer provenance SHA differs")
        tree = ast.parse(raw)
        target = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_storage_facts"]
        if len(target) != 1 or hashlib.sha256(ast.dump(target[0], include_attributes=False).encode()).hexdigest() != function_sha:
            raise ValueError("The writer delta is not the exact reviewed output-location metadata fix")
        tree.body.remove(target[0])
        parsed.append(tree)
        trees.append(ast.dump(tree, include_attributes=False))
    if trees[0] != trees[1]:
        # Q61 is only the exact consumed-JSON hash recheck, never a general I/O
        # exception. The original storage-facts-only path above is unchanged.
        json_expected = ("22419d8d07474fe6ec1fced1324de3561d396f5913ab69831f1083a613ab255c",
                         "1f23d7ade688ea081e843bfa3408c467a348e9828a89fe1b35ed8c710a67e1a1")
        for tree, function_sha in zip(parsed, json_expected):
            classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DataReader"]
            methods = [] if len(classes) != 1 else [node for node in classes[0].body
                if isinstance(node, ast.FunctionDef) and node.name == "json"]
            if (len(methods) != 1 or
                    hashlib.sha256(ast.dump(methods[0], include_attributes=False).encode()).hexdigest() != function_sha):
                raise ValueError("An actual writer/executor body changed; the writer delta is not the exact reviewed DataReader.json hash recheck")
            classes[0].body.remove(methods[0])
        if ast.dump(parsed[0], include_attributes=False) != ast.dump(parsed[1], include_attributes=False):
            raise ValueError("An actual writer/executor body changed; a new measured profile is required")


def _orchestration_identity(repository, measured_commit, current_commit, row):
    """Known reporting-only runner delta: whole SHAs are explicit provenance.

    All seven unchanged execution/output functions are applicability evidence;
    prepare_job/main reporting changes never become a scientific equivalence.
    """
    relative = "src/darsia/fluidflower/pipeline/runner.py"
    functions = {"_phase_outputs", "_inventory", "validate_frame_artifacts", "_restore_predecessor",
                 "scalar_table", "run_series", "_run_series"}
    if (not isinstance(row, dict) or set(row) != {"relative_path", "measured_sha256", "current_sha256", "functions"}
            or row["relative_path"] != relative or not isinstance(row["functions"], dict)
            or set(row["functions"]) != functions):
        raise ValueError("Exact original runner execution-function identities required")
    for commit, field in [(measured_commit,"measured_sha256"),(current_commit,"current_sha256")]:
        if field == "current_sha256":
            _writer_identity(repository,commit,dict(relative_path=relative,sha256=row[field]))
        raw = subprocess.check_output(["git","-C",str(repository),"show",commit+":"+relative])
        if hashlib.sha256(raw).hexdigest() != row[field]:
            raise ValueError("Runner provenance file SHA differs")
        actual = {node.name:node for node in ast.parse(raw).body if isinstance(node,ast.FunctionDef)}
        for key in functions:
            expected = row["functions"][key]
            if (not isinstance(expected,str) or re.fullmatch(r"[0-9a-f]{64}",expected) is None
                    or hashlib.sha256(ast.dump(actual[key],include_attributes=False).encode()).hexdigest()!=expected):
                raise ValueError("An actual runner executor/output body changed; measured profile does not apply")


@dataclass(frozen=True)
class SpaceEstimate:
    run: str
    case_count: int
    measured_native_bytes_per_frame: int
    native_expected_bytes: int
    coarse_uncompressed_bytes: int
    header_metadata_copy_estimated_bytes: int
    transient_estimated_bytes: int
    products_per_frame_estimated_bytes: int
    profile: dict
    measured_commit: str
    model: str = "expected space; observed native profile plus uncompressed output additions and one in-flight frame allowance; no assumed compression gain"

    @property
    def products_estimated_bytes(self):
        return self.native_expected_bytes + self.coarse_uncompressed_bytes + self.header_metadata_copy_estimated_bytes

    def needed(self, *, other_remaining_bytes=0, reserve_bytes=DEFAULT_RESERVE_BYTES):
        return (self.products_estimated_bytes + self.transient_estimated_bytes
                + _integer(other_remaining_bytes, "Other active remaining bytes")
                + _integer(reserve_bytes, "Reserve bytes"))


@dataclass(frozen=True)
class OptionSpaceEstimate(SpaceEstimate):
    """A qualified core estimate plus the budgets of the task's declared optional outputs.

    products_per_frame_estimated_bytes and transient_estimated_bytes already
    include the option bytes of one frame (and the per-run option bytes), so
    the ledger reserves them as it reserves any estimate. Retained serialization
    bytes are included in the transient total for the whole task, never in the
    per-frame amount that committed-prefix progress releases.
    """
    option_frame_bytes: int = 0
    option_run_bytes: int = 0
    options: dict = None
    retained_serialization_estimated_bytes: int = 0

    @property
    def products_estimated_bytes(self):
        """The core's products plus the option bytes of every frame."""
        return super().products_estimated_bytes + self.option_frame_bytes * self.case_count


LEAN_ESTIMATE_MODEL = ("ESTIMATE, not a measurement: the chosen storage profile's documented "
                       "per-file sizes (storage_profile.estimate_job_bytes); no measured lean "
                       "profile exists yet and the full measured profile is never reused")


@dataclass(frozen=True)
class StorageProfileEstimate:
    """A lean task's own admission estimate (N55).

    The shared per-profile estimator, bound to the task's explicit storage
    profile and to the actual writer sources of its commit. It is never a
    reuse of the full measured profile, and never a measurement.
    """
    run: str
    case_count: int
    storage_profile: str
    colour_intermediates: str
    pipeline: str
    products_per_frame_estimated_bytes: int
    run_constant_estimated_bytes: int
    inflight_estimated_bytes: int
    estimate: dict
    writer_identity: list
    commit: str
    model: str = LEAN_ESTIMATE_MODEL

    @property
    def products_estimated_bytes(self):
        return self.products_per_frame_estimated_bytes * self.case_count

    @property
    def transient_estimated_bytes(self):
        # The run's one-off constant-input store stays reserved with the run.
        return self.inflight_estimated_bytes + self.run_constant_estimated_bytes

    def needed(self, *, other_remaining_bytes=0, reserve_bytes=DEFAULT_RESERVE_BYTES):
        return (self.products_estimated_bytes + self.transient_estimated_bytes
                + _integer(other_remaining_bytes, "Other active remaining bytes")
                + _integer(reserve_bytes, "Reserve bytes"))


def _require_full_profile(task, route):
    from ..storage_profile import declared_storage_profile
    if declared_storage_profile(task) != "full":
        raise ValueError(route + " binds the full storage profile of references and "
                         "qualification; a lean task is refused here")


def _lean_writers(pipeline, config):
    """The writer sources one lean estimate binds, by route and optional output."""
    rows = ["src/darsia/fluidflower/storage_profile.py", "src/darsia/fluidflower/mass/n1_pipeline.py",
            "src/darsia/fluidflower/mass/n1_statistics.py"]
    if pipeline == "native-phase-n1":
        rows += ["src/darsia/fluidflower/pipeline/series.py", "src/darsia/fluidflower/native/sidecar.py",
                 "src/darsia/fluidflower/pipeline/runner.py"]
    if config.get("pixel_domain") is not None:
        rows.append("src/darsia/fluidflower/mass/pixel_pipeline.py")
    if config.get("storage") is not None:
        rows += ["src/darsia/fluidflower/storage_resolution.py", "src/darsia/fluidflower/storage_outputs.py"]
        if pipeline == "n1":
            rows.append("src/darsia/fluidflower/storage_n1_outputs.py")
    if pipeline == "n1":
        # The staged lean contract binds its complete output/budget boundary,
        # including the actual metadata ceiling and unchanged N15/N16 writers.
        rows += ["src/darsia/fluidflower/pipeline/lean_capacity.py",
                 "src/darsia/fluidflower/pipeline/n1_capacity.py",
                 "src/darsia/fluidflower/pipeline/option_capacity.py",
                 "src/darsia/fluidflower/pipeline/io_ops.py",
                 "src/darsia_cache/publication_paths.py",
                 "src/darsia/fluidflower/pipeline/job_queue.py",
                 "src/darsia/fluidflower/pipeline/staged_data.py",
                 "src/darsia/fluidflower/pipeline/job_inputs.py"]
        if config.get("pixel_domain") is not None:
            from .n1_capacity import PIXEL_WRITERS
            rows += list(PIXEL_WRITERS)
        if config.get("storage") is not None and "n16" in (config.get("pixel_domain") or {}):
            from .n1_capacity import EXTENSIVE_WRITERS
            rows += list(EXTENSIVE_WRITERS)
    return list(dict.fromkeys(rows))


def _pinned_writer_rows(repository, commit, relatives):
    """Each writer source's Git-blob SHA-256 at the commit, verified unchanged on disk."""
    identity = []
    for relative in relatives:
        original = subprocess.check_output(["git", "-C", str(repository), "show", commit + ":" + relative])
        row = dict(relative_path=relative, sha256=hashlib.sha256(original).hexdigest())
        _writer_identity(repository, commit, row)
        identity.append(row)
    return identity


def _lean_writer_identity(repository, commit, pipeline, config):
    """The lean writer's own identity: its sources at the task commit, verified on disk.

    No measured lean profile pins them yet, so they are recorded with the
    estimate and its reservation, never compared with a full measurement.
    """
    return _pinned_writer_rows(repository, commit, _lean_writers(pipeline, config))


def lean_estimate(task, config, *, repository, stores=None, series_id=None, metadata_rows=None):
    """Admission estimate of a lean task: its own profile, never the full profile's."""
    from ..storage_profile import declared_storage_profile, run_estimate
    if declared_storage_profile(task) != "lean":
        raise ValueError("The lean estimate needs a task that chose storage_profile 'lean'")
    pipeline, cases = task.get("pipeline"), task.get("ordered_case_ids")
    if (pipeline not in {"native-phase-n1", "n1"} or not isinstance(cases, list) or not cases
            or len(set(cases)) != len(cases) or not isinstance(config.get("frames"), list)
            or [frame.get("case_id") for frame in config["frames"]] != cases
            or config.get("run") != task.get("run")
            or ("precision" in config and config["precision"] != task.get("precision"))
            or re.fullmatch(r"[0-9a-f]{40}", str(task.get("commit", ""))) is None):
        raise ValueError("Lean capacity estimate: task pipeline, precision, commit or exact case order differs")
    if pipeline == "n1":
        return _phase2_lean_estimate(stores, series_id, task, config,
                                    repository=repository, metadata_rows=metadata_rows)
    value = run_estimate("lean", config, pipeline=pipeline)
    writers = _lean_writer_identity(repository, task["commit"], pipeline, config)
    return StorageProfileEstimate(name(task["run"]), len(cases), "lean", value["colour_intermediates"],
        pipeline, value["per_frame_bytes"], value["per_run_bytes"], value["inflight_bytes"], value,
        writers, task["commit"])


def _phase2_lean_estimate(stores, series_id, task, config, *, repository, metadata_rows):
    """Exact-dtype fallback/store bound for an explicitly sealed lean stage.

    This computes result-space DATA only. The independent cache/provider
    admission must still prove the common staged run before an actual claim;
    neither a size projection nor a conversion sample grants that permission.
    """
    from . import lean_capacity
    from .staged_data import validate_phase2_run
    from .store_runtime import resolve_catalog
    from ..storage_profile import job_storage_profile
    if (task.get("format") != "darsia-alias-task-v1" or task.get("format_version") != 2
            or job_storage_profile(task) != "lean" or task.get("pipeline") != "n1"
            or task.get("backend") != "cpu" or task.get("precision") != "float32"):
        raise ValueError("The staged lean result bound requires its exact version-two CPU/float32 task")
    validate_phase2_run(config)
    frames = config["frames"]
    shape = (frames[0].get("current") or {}).get("phase_shape_hw")
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(size) is not int or size < 1 for size in shape)
            or any((frame.get("current") or {}).get("phase_shape_hw") != shape for frame in frames)):
        raise ValueError("Every staged lean frame must declare one exact label canvas")
    rows = metadata_rows
    if rows is None:
        if stores is None or series_id is None:
            raise ValueError("Staged lean budgeting needs the existing rooted input catalogue")
        rows = resolve_catalog(task, stores, series_id)

    def path_for_use(use, descriptor):
        matches = [row for row in rows if use in row.uses]
        if len(matches) != 1:
            raise ValueError("One explicit rooted run-store input role required: " + use)
        row = matches[0]
        sha, size = lean_capacity._pin(descriptor)
        if (row.reference.get("sha256") != sha or type(row.reference.get("bytes")) is not int
                or row.reference["bytes"] != size):
            raise ValueError("Run-store input role differs from its configuration SHA/size: " + use)
        return row.path

    store = lean_capacity.run_store_budget(config, path_for_use=path_for_use)
    value = lean_capacity.phase2_lean_budget(config, canvas_hw=shape, store_bound=store)
    if "opencl_acceleration" in config:
        from darsia_cache.common_qualification import _compute_control
        value["compute_control"] = _compute_control(config, "phase2")
    writers = _lean_writer_identity(repository, task["commit"], "n1", config)
    return StorageProfileEstimate(name(task["run"]), len(task["ordered_case_ids"]),
        "lean", "keep", "n1", value["admitted_frame_bytes"], value["admitted_run_bytes"],
        value["inflight_bytes"], value, writers, task["commit"], model=lean_capacity.MODEL)


@dataclass(frozen=True)
class PredeclaredN43ProducerEntitlement:
    run: str
    case_count: int
    maximum_aggregate_bytes: int
    request: dict
    administration: dict
    shared_service_reservation: dict
    model: str = "PREDECLARED all-stage aggregate; shared cache/output entitlement counted once; NOT measured profile"


@dataclass(frozen=True)
class PredeclaredFirstOutputEntitlement:
    run: str
    case_count: int
    maximum_output_bytes: int
    request: dict
    administration: dict
    all_stage_reference: dict
    scope: dict
    model: str = "Q104 WHOLE retained output entitlement once, including preparation/ops/in-flight files; NOT measured bytes-per-frame"

    @property
    def products_estimated_bytes(self):
        return self.maximum_output_bytes

    @property
    def transient_estimated_bytes(self):
        # Not an observed zero: in-flight files are inside the same whole-tree
        # ceiling. No separate allowance can enlarge the authorized entitlement.
        return 0

    def needed(self, *, other_remaining_bytes=0, reserve_bytes=DEFAULT_RESERVE_BYTES):
        return (self.maximum_output_bytes + _integer(other_remaining_bytes, "Other remaining outputs")
                + _integer(reserve_bytes, "Original reserve"))


def first_output_entitlement(runtime, task, config, rows):
    """First-only Q104 administration; ordinary measured estimates stay intact."""
    _require_full_profile(task, "The first Q104 qualification")
    first = runtime.cache_adapter.lease.first_qualification
    from darsia_cache.first_qualification import BOUND_OUTPUT_ADMINISTRATION, FIRST_OUTPUT_BASE_ENTITLEMENT_BYTES, checked_all_stage_reference, _same_path
    if not first.bound_mode:
        raise ValueError("No existing bound-request first qualification")
    first.require_task(runtime.cache_adapter.lease, task)
    plan = first.output_entitlement
    _state, reserve, binding = settings(runtime.stores, runtime.job["series_id"])
    if (binding != first.provider_binding or reserve != DEFAULT_RESERVE_BYTES
            or plan.get("format") != BOUND_OUTPUT_ADMINISTRATION
            or type(plan["administrative_output_entitlement_bytes"]) is not int
            or plan["administrative_output_entitlement_bytes"] < FIRST_OUTPUT_BASE_ENTITLEMENT_BYTES
            or config.get("storage") is not None or config.get("pixel_domain") is not None
            or config.get("probability_storage") is not None or "gas_probability_storage" in config
            or [frame["case_id"] for frame in config["frames"]] != task["ordered_case_ids"]):
        raise ValueError("First Q104 requires bound all-stage evidence, original exact39 route and original reserve")
    if not _same_path(plan["watched_output_scope"]["path"], runtime.stores.roots(runtime.job["series_id"])["results"]):
        raise ValueError("Q104 watch must cover the actual current whole results root, not the old reference root")
    data = read_bound(plan["all_stage_reference"])[0]
    checked_all_stage_reference(task, data, read=lambda pin: read_bound(pin)[0],
        maximum_output_bytes=plan["administrative_output_entitlement_bytes"])
    actual = _native_layout_for_task(runtime.stores, runtime.job["series_id"], task, config, rows)
    if actual != data["requested_native_layout"]:
        raise ValueError("First Q104 current layout differs from the actual all-stage reference")
    writers = data["output_writers"]
    required = {"src/darsia/fluidflower/mass/n1_pipeline.py",
        "src/darsia/fluidflower/mass/n1_statistics.py", "src/darsia/fluidflower/pipeline/series.py",
        "src/darsia/fluidflower/native/sidecar.py", "src/darsia/fluidflower/pipeline/runner.py"}
    if not isinstance(writers, list) or {row.get("relative_path") for row in writers} != required or len(writers) != len(required):
        raise ValueError("First Q104 requires the exact five existing output writers")
    for writer in writers:
        _writer_identity(runtime.repository, data["reference_commit"], writer)
        _writer_identity(runtime.repository, task["commit"], writer)
    return PredeclaredFirstOutputEntitlement(task["run"], len(task["ordered_case_ids"]),
        plan["administrative_output_entitlement_bytes"], first.binding, first.request["output_entitlement_plan"],
        plan["all_stage_reference"], plan["scope"])


def n43_entitlement(runtime, task, config, rows):
    """Explicit released producer route; ordinary load_estimate is unchanged."""
    _require_full_profile(task, "The N43 reference build")
    first=runtime.cache_adapter.lease.first_qualification
    from darsia_cache.first_qualification import REFERENCE_ENTITLEMENT_BYTES, _same_path
    if not first.reference_build:
        raise ValueError("No explicit N43 reference-build administration")
    first.require_task(runtime.cache_adapter.lease,task)
    if (config.get("storage") is not None or config.get("pixel_domain") is not None
            or config.get("probability_storage") is not None
            or "gas_probability_storage" in config
            or [frame["case_id"] for frame in config["frames"]]!=task["ordered_case_ids"]):
        raise ValueError("N43 binds the exact classic native-phase39 route without new output options")
    _state,_reserve,binding=settings(runtime.stores,runtime.job["series_id"])
    if binding!=first.request["reference_administration"]:
        raise ValueError("Result-space administration differs from the exact Root-released N43 request")
    actual=_native_layout_for_task(runtime.stores,runtime.job["series_id"],task,config,rows)
    admin=first.reference_administration
    if actual!=admin["requested_native_layout"]:
        raise ValueError("N43 actual native layout differs from the predeclared producer")
    writers=admin["output_writers"]
    required={"src/darsia/fluidflower/mass/n1_pipeline.py",
        "src/darsia/fluidflower/mass/n1_statistics.py", "src/darsia/fluidflower/pipeline/series.py",
        "src/darsia/fluidflower/native/sidecar.py", "src/darsia/fluidflower/pipeline/runner.py"}
    if (not isinstance(writers,list) or len(writers)>32
            or len({row.get("relative_path") for row in writers})!=len(writers)
            or not required<={row.get("relative_path") for row in writers}):
        raise ValueError("N43 predeclared actual output-writer closure incomplete")
    for writer in writers:
        _writer_identity(runtime.repository,task["commit"],writer)
    roots={row["role"]:row for row in admin["owned_roots"]}
    if (not _same_path(roots["job"]["path"],runtime.root)
            or not _same_path(roots["results"]["path"],runtime.stores.roots(runtime.job["series_id"])["results"])):
        raise ValueError("N43 actual job/results roots escape the predeclared all-stage scope")
    return PredeclaredN43ProducerEntitlement(task["run"],len(task["ordered_case_ids"]),
        REFERENCE_ENTITLEMENT_BYTES,first.binding,first.request["reference_administration"],
        runtime.cache_adapter.lease.reservation)


def estimate_run(task, config, profile, *, measurement, completion, layout=None,
                 layout_bytes=0, profile_identity=None, requested_native_layout=None, other_run=False):
    """Pure metadata estimate; missing applicability is refusal, not a default.

    other_run: the task is another run than the measured one (see
    _check_measured_native_layout); the arithmetic is the same per frame.
    """
    if any(not isinstance(row,dict) for row in (task,config,profile,measurement,completion)):
        raise ValueError("Explicit typed capacity metadata objects required")
    if task.get("pipeline") == "native-phase-statistics":
        raise ValueError("Phase-one output layout has no separately measured capacity contract; no full-pipeline profile reuse")
    if task.get("pipeline") == "native-phase-n1":
        _check_measured_native_layout(profile, requested_native_layout, other_run=other_run)
        if profile["measured_native_layout"]["precision"] != task.get("precision"):
            raise ValueError("Measured native layout and original task precision differ")
    cases = task.get("ordered_case_ids")
    if (not isinstance(cases, list) or not cases or len(set(cases)) != len(cases)
            or [s.get("case_id") for s in config.get("frames", [])] != cases
            or profile.get("format") != PROFILE_FORMAT
            or profile.get("pipeline") != task.get("pipeline")
            or profile.get("precision") != task.get("precision")
            or (task.get("pipeline") != "native-phase-n1" and config.get("precision") != task.get("precision"))
            or (task.get("pipeline") == "native-phase-n1" and "precision" in config and config["precision"] != task.get("precision"))
            or config.get("run") != task.get("run")
            or re.fullmatch(r"[0-9a-f]{40}", str(profile.get("measured_commit", ""))) is None):
        raise ValueError("Capacity profile/task pipeline, precision or exact case order differs")
    ids = profile.get("measured_case_ids")
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ValueError("Measured native profile requires its exact distinct case IDs")
    ids = [name(case) for case in ids]
    count = completion.get("actual_case_count", completion.get("case_count"))
    measured_commit = completion.get("actual_commit", completion.get("commit"))
    if (completion.get("complete") is not True or completion.get("passed") is not True
            or type(count) is not int or count != len(ids)
            or measured_commit != profile["measured_commit"]
            or completion.get("precision") != task["precision"]):
        raise ValueError("A complete applicable measured Main gate is required; no historical/current relabel")
    inventory = measurement.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Measured native file inventory required")
    total, seen, by_case = 0, set(), dict.fromkeys(ids, 0)
    for row in inventory:
        if not isinstance(row,dict):
            raise ValueError("Explicit measured file metadata required")
        relative = row.get("relative_path")
        if (not isinstance(relative, str) or not relative or relative in seen
                or PureWindowsPath(relative).drive or PureWindowsPath(relative).root
                or ".." in PureWindowsPath(relative).parts):
            raise ValueError("Measured file inventory requires unique owned relative names")
        seen.add(relative)
        size = _integer(row.get("bytes"), "Observed file bytes")
        total += size
        matches = set(PureWindowsPath(relative).parts) & set(ids)
        if len(matches) > 1:
            raise ValueError("Measured file belongs to multiple cases")
        if matches:
            by_case[next(iter(matches))] += size
    if (total != measurement.get("total_logical_bytes")
            or any(value <= 0 for value in by_case.values())):
        raise ValueError("Measured native total/case coverage differs")
    native_bpf = (total + len(ids) - 1) // len(ids)
    extra_raw, extra_meta = 0, 0
    request = config.get("storage")
    if request is not None:
        if (not isinstance(request, dict) or request.get("format") != "darsia-storage-output-request-v1"
                or request.get("gas_score_encoding") != "uint8-la-q255-v1"
                or not isinstance(layout, dict) or layout.get("precision") != task["precision"]
                or layout.get("complete") is not True
                or layout.get("format") != ("darsia-n1-storage-frame-v1" if task.get("pipeline") == "n1" else "darsia-storage-frame-v1")):
            raise ValueError("Explicit compatible saved writer layout required for storage estimate")
        native_shape = (layout.get("coarse") or {}).get("grid", {}).get("native_shape")
        native_shape = native_shape or layout.get("before_rules", {}).get("target_shape_hw")
        if (not isinstance(native_shape, list) or len(native_shape) != 2
                or any(type(v) is not int or v < 1 for v in native_shape)):
            raise ValueError("Writer metadata omits the original native image shape")
        h, w = native_shape
        # A full uncompressed LA image plus row-filter bytes; encoded bytes are
        # estimated without compression gain. Per-row chunk and PNG headers are
        # a transparent allocation allowance, not an observed file measurement.
        extra_raw += h * (2 * w + 1)
        extra_meta += 8 + 25 + 12 + 12 * (h + 1)
        if request.get("resolution_mm") != "native":
            from ..storage_resolution import metric_grid_shape
            height, width = metric_grid_shape(request.get("extent_m"), request.get("resolution_mm"))
            coarse = layout.get("coarse")
            identities = coarse.get("arrays") if isinstance(coarse, dict) else None
            if not isinstance(identities, dict) or not identities:
                raise ValueError("Original writer coarse array schema required")
            for key, value in identities.items():
                if not isinstance(key, str) or not key or not isinstance(value, dict):
                    raise ValueError("Explicit writer array identities required")
                match = re.fullmatch(r"[<>=|]([biuf])(1|2|4|8)", str(value.get("dtype", "")))
                if match is None:
                    raise ValueError("Unsupported actual writer array dtype")
                extra_raw += height * width * int(match[2])
                # Header allocation is explicitly estimated, not a codec proof.
                # Current simple .npy headers fit here; larger/changed schemas
                # need a new writer/profile rather than reusing this model.
                header = repr(dict(descr=value["dtype"], fortran_order=False, shape=(height, width)))
                extra_meta += 12 + ((len(header) + 64) // 64) * 64
                filename = (key + ".npy").encode("utf8")
                extra_meta += 30 + 46 + 2 * len(filename) + 20 + 28
            extra_meta += 22 + 76  # ZIP end records, including ZIP64 allowance.
        extra_meta += _integer(layout_bytes, "Observed layout metadata bytes")
        for row in layout.get("files", []):
            filename = PureWindowsPath(_bound({k: row[k] for k in ("path", "sha256", "size_bytes")} )["path"]).name
            if filename not in {"storage_stored_frame.npz", "storage_gas_probability_before_rules.png"}:
                extra_meta += row["size_bytes"]  # Literal already-saved copy sizes.
    n = len(cases)
    transient = max(by_case.values()) + extra_raw + extra_meta
    return SpaceEstimate(name(task["run"]), n, native_bpf, native_bpf * n,
        extra_raw * n, extra_meta * n, transient, native_bpf + extra_raw + extra_meta,
        dict(profile_identity or {}), profile["measured_commit"])


def load_estimate(stores, series_id, task, config, *, repository, metadata_rows=None):
    from darsia_cache.folders import folder_mode
    if folder_mode(stores, series_id):
        if task.get("pipeline") == "series-setup":
            from .series_setup_capacity import capped_setup_output
            return capped_setup_output(task, config)
        from .folder_capacity import folder_estimate
        return folder_estimate(task, config)
    if task.get("pipeline") == "preset-wasserstein":
        from ..preset_wasserstein_capacity import preset_estimate
        return preset_estimate(stores, series_id, task, config, repository=repository)
    if task.get("pipeline") == "n1-calibration-refit":
        from .refit_capacity import refit_estimate
        return refit_estimate(stores, series_id, task, config, repository=repository,
                              metadata_rows=metadata_rows)
    if task.get("pipeline") == "lattice-phase-n1":
        from .lattice_capacity import lattice_estimate
        return lattice_estimate(task, config)
    from ..storage_profile import declared_storage_profile
    if declared_storage_profile(task) == "lean":
        # The task's explicit choice binds its own estimate and writer identity;
        # the full measured profile below is never reused for a lean writer.
        return lean_estimate(task, config, repository=repository, stores=stores,
                             series_id=series_id, metadata_rows=metadata_rows)
    _state, _reserve, binding = settings(stores, series_id)
    profile, _raw = read_bound(binding)
    from darsia_cache.first_qualification import PROVIDER, BOUND_JOB_PROFILE, resolve_profile
    if profile.get("format") == "darsia-common-phase-qualified-provider-v1":
        from darsia_cache.bootstrap import _exact_authority_list
        profile = resolve_profile(binding, "jobs", lambda pin: read_bound(pin)[0],
            stores_source={key: stores.source[key] for key in ("path", "sha256", "size_bytes")},
            read_authorities=lambda pin: _exact_authority_list(pin, __file__))
        return _common_result_estimate(stores, series_id, task, config, profile, binding,
            repository=repository, metadata_rows=metadata_rows)
    if profile.get("format") == PROVIDER:
        from darsia_cache.bootstrap import _exact_authority_list
        profile = resolve_profile(binding, "jobs", lambda pin:read_bound(pin)[0],
            stores_source={key:stores.source[key] for key in ("path", "sha256", "size_bytes")},
            read_authorities=lambda pin:_exact_authority_list(pin, __file__))
        if profile.get("format") != BOUND_JOB_PROFILE:
            raise ValueError("Only an actually completed typed bound provider supplies this result profile")
        return _bound_result_estimate(stores, series_id, task, config, profile, binding,
            repository=repository, metadata_rows=metadata_rows)
    fields = {"format", "pipeline", "precision", "measured_commit", "measured_case_ids",
              "measurement", "completion", "writers", "orchestration", "layout_receipt"}
    if task.get("pipeline") == "native-phase-n1":
        fields.add("measured_native_layout")
    if "gas_probability_storage" in config:
        from ..phase.probability_storage import parse_request
        request = parse_request(config)
        fields.add("gas_probability_storage")
        if profile.get("gas_probability_storage") != request:
            raise ValueError("Probability output requires its own actually measured writer/layout profile")
    if set(profile) != fields:
        raise ValueError("Exact expected-space profile schema required")
    writers = profile["writers"]
    required = {"src/darsia/fluidflower/mass/n1_pipeline.py",
                "src/darsia/fluidflower/mass/n1_statistics.py"}
    if task.get("pipeline") in {"native-phase-n1","native-phase-statistics"}:
        required|={"src/darsia/fluidflower/pipeline/series.py","src/darsia/fluidflower/native/sidecar.py"}
    if task.get("pipeline") == "native-phase-statistics":
        required.add("src/darsia/fluidflower/pipeline/phase_statistics.py")
    if "gas_probability_storage" in config:
        required.add("src/darsia/fluidflower/phase/probability_storage.py")
    if config.get("storage") is not None:
        required |= {"src/darsia/fluidflower/storage_resolution.py", "src/darsia/fluidflower/storage_outputs.py"}
        if task.get("pipeline") == "n1":
            required.add("src/darsia/fluidflower/storage_n1_outputs.py")
    if (not isinstance(writers, list) or len(writers) > 32
            or any(not isinstance(row,dict) for row in writers)
            or len({r.get("relative_path") for r in writers}) != len(writers)
            or not required <= {r.get("relative_path") for r in writers}):
        raise ValueError("Profile omits an actual applicable output writer")
    for row in writers:
        _profile_writer_identity(repository, profile["measured_commit"], task["commit"], row)
    orchestration = profile["orchestration"]
    if task.get("pipeline") == "native-phase-n1":
        if not isinstance(orchestration,list) or len(orchestration)!=1:
            raise ValueError("Bind measured/current runner provenance and unchanged execution AST")
        _orchestration_identity(repository,profile["measured_commit"],task["commit"],orchestration[0])
    elif orchestration != []:
        raise ValueError("Unused orchestration identities cannot substitute for actual stage writers")
    measurement, _ = read_bound(profile["measurement"])
    completion, _ = read_bound(profile["completion"])
    layout, raw = (None, b"") if profile["layout_receipt"] is None else read_bound(profile["layout_receipt"])
    requested = (_native_layout_for_task(stores, series_id, task, config, metadata_rows)
                 if task.get("pipeline") == "native-phase-n1" else None)
    return estimate_run(task, config, profile, measurement=measurement, completion=completion,
                        layout=layout, layout_bytes=len(raw), profile_identity=binding,
                        requested_native_layout=requested)


def _common_result_estimate(stores, series_id, task, config, profile, binding, *, repository, metadata_rows):
    """Use only a completed common stage's own result allowance and exact route."""
    from darsia_cache.common_qualification import require_applicable, CommonStageEntitlement
    from .option_capacity import retained_serialization_bytes
    from .phase1_capacity import phase1_additions, validate_requests
    stage = require_applicable(profile, task, config=config)
    if stage == "phase2":
        return _phase2_lean_estimate(stores, series_id, task, config,
            repository=repository, metadata_rows=metadata_rows)
    space = profile["result_spaces"][stage]
    count, measured_count = len(task["ordered_case_ids"]), _integer(space["case_count"], "Qualified case count", positive=True)
    maximum = _integer(space["maximum_output_bytes"], "Qualified whole-stage output allowance", positive=True)
    if stage == "calibration":
        original = space["configuration"]
        current = task["configuration"]
        if (current.get("sha256") != original["sha256"]
                or current.get("bytes", current.get("size_bytes")) != original["size_bytes"]):
            raise ValueError("The common refit output allowance covers only its actual qualified plan")
    else:
        declared = validate_requests(config)
        requested = _native_layout_for_task(stores, series_id, task, config, metadata_rows)
        _check_measured_native_layout(space, requested, other_run=True)
        # Keep at least the whole proved stage, including its fixed files.
        maximum *= (count + measured_count - 1) // measured_count
        additions = phase1_additions(config, canvas_hw=requested["phase_shape_hw"],
            precision=task["precision"], declared=declared, include_core=False)
        extra, run_bytes = additions["frame_bytes"], additions["run_bytes"]
        maximum += extra * (count + 1) + run_bytes
        maximum += retained_serialization_bytes(count, extra, run_bytes=run_bytes)
    scope = dict(stage=stage, configuration=deepcopy(task["configuration"]))
    source = config.get("statistics", {}) if stage == "phase1" else config
    if stage != "calibration" and "opencl_acceleration" in source:
        scope["compute_control"] = deepcopy(profile["compute_profiles"][stage]["control"])
    return CommonStageEntitlement(task["run"], count, maximum, deepcopy(binding), deepcopy(binding), scope,
        model="Qualified common whole-stage allowance, scaled in complete stage blocks; declared optional bytes added")


def _bound_result_estimate(stores, series_id, task, config, profile, binding, *, repository, metadata_rows):
    """Original estimate arithmetic on genuine closed whole-allstage inventory.

    A qualified profile estimates every full task on its qualified head and
    route (commit, native-phase-n1, CPU, float32), not only the task it
    qualified (N188 D2): the qualified bytes per frame times the task's frame
    count, plus its largest frame in flight. Another run's layout must have the
    measured kind, grid and canvas; it names its own configurations and binding.

    A task may declare the gas probability (option_capacity.OPTIONS). Its
    request is checked before any profile read; it adds its own upper-bound
    budget per frame to the core, and its writer is pinned at the task commit.
    Every other optional output is refused, with the qualified-route refusal
    and its reason, and so is the colour-path coordinate unless the qualified
    profile measured it. A task without optional outputs takes the unchanged
    path.

    A full phase-one statistics task on the qualified head (commit, CPU,
    float32) is admitted by phase1_capacity's rule as another run: the
    measured preparation bytes per frame and the phase-one additions. A full
    phase-two n1 task is admitted by n1_capacity's rule: the measured
    mass-frame bytes per frame and the declared n1 options.
    """
    route = ("commit", "pipeline", "backend", "precision")
    refusal = "Qualified output profile does not cover this full/allstage task on its qualified head and route"
    qualified = profile.get("scope")
    # A phase-one statistics task and a phase-two n1 task run the qualified
    # stages on another route.
    staged = (task.get("pipeline") in {"native-phase-statistics", "n1"} and isinstance(qualified, dict)
              and qualified.get("pipeline") == "native-phase-n1")
    phase1 = staged and task.get("pipeline") == "native-phase-statistics"
    phase2 = staged and task.get("pipeline") == "n1"
    keys = ("commit", "backend", "precision") if staged else route
    if (not isinstance(qualified, dict) or any(task.get(key) != qualified.get(key) for key in keys)
            or task.get("pipeline") != (task.get("pipeline") if staged else "native-phase-n1")
            or task.get("backend") != "cpu" or task.get("precision") != "float32"):
        raise ValueError(refusal)
    optional = (config.get("storage") is not None or config.get("pixel_domain") is not None
                or "gas_probability_storage" in config or config.get("probability_storage") is not None)
    if optional:
        from .option_capacity import option_budget, validate_requests
        if phase1:
            from .phase1_capacity import validate_requests
        elif phase2:
            from .n1_capacity import validate_requests
        try:
            declared = validate_requests(config)
        except ValueError as error:
            raise ValueError(refusal + ": " + str(error)) from error
    colour = config.get("storage_profile")
    if isinstance(colour, dict) and colour.get("store_colour_coordinate") is True:
        from ..storage_profile import COLOUR_COORDINATE_ARCHIVE
        result = profile.get("result_profile")
        measurement = result.get("measurement") if isinstance(result, dict) else None
        rows = measurement.get("inventory") if isinstance(measurement, dict) else None
        if not any(isinstance(row, dict) and PureWindowsPath(str(row.get("relative_path", ""))).name
                   == COLOUR_COORDINATE_ARCHIVE for row in (rows if isinstance(rows, list) else ())):
            raise ValueError(refusal + ": the colour-path coordinate was not measured by the qualified profile")
    other_run = ({key: task.get(key) for key in (*route, "run", "ordered_case_ids")} != qualified
                 or profile.get("configuration") != task.get("configuration"))
    actual = profile["result_profile"]
    required = {"src/darsia/fluidflower/mass/n1_pipeline.py", "src/darsia/fluidflower/mass/n1_statistics.py",
        "src/darsia/fluidflower/pipeline/series.py", "src/darsia/fluidflower/native/sidecar.py",
        "src/darsia/fluidflower/pipeline/runner.py"}
    if (actual.get("format") != PROFILE_FORMAT or actual.get("measured_commit") != task["commit"]
            or actual.get("measured_case_ids") != qualified.get("ordered_case_ids")
            or not isinstance(actual.get("writers"), list)
            or len(actual["writers"]) != len(required)
            or {row.get("relative_path") for row in actual["writers"]} != required):
        raise ValueError("Actual closed output writer/case/source profile is incomplete")
    for writer in actual["writers"]:
        _profile_writer_identity(repository, actual["measured_commit"], task["commit"], writer)
    if phase2:
        # n1 has no native stage: its frames' label grid is checked instead.
        from .n1_capacity import n1_estimate
        try:
            return n1_estimate(task, config, actual, binding, declared=declared if optional else (),
                               repository=repository)
        except ValueError as error:
            raise ValueError(refusal + ": " + str(error)) from error
    requested = _native_layout_for_task(stores, series_id, task, config, metadata_rows)
    if phase1:
        from .phase1_capacity import phase1_estimate
        try:
            return phase1_estimate(task, config, actual, binding, requested=requested,
                                   declared=declared if optional else (), repository=repository)
        except ValueError as error:
            raise ValueError(refusal + ": " + str(error)) from error
    if not optional:
        return estimate_run(task, config, actual, measurement=actual["measurement"],
            completion=actual["completion"], requested_native_layout=requested,
            profile_identity=dict(binding), other_run=other_run)
    try:
        budget = option_budget(config, layout=requested, precision=task["precision"])
    except ValueError as error:
        raise ValueError(refusal + ": " + str(error)) from error
    writers = _pinned_writer_rows(repository, task["commit"], budget.writers)
    core = estimate_run(task, {key: value for key, value in config.items() if key not in declared},
        actual, measurement=actual["measurement"], completion=actual["completion"],
        requested_native_layout=requested, profile_identity=dict(binding), other_run=other_run)
    values = {field.name: getattr(core, field.name) for field in dataclass_fields(SpaceEstimate)}
    from .option_capacity import retained_serialization_bytes, run_file_bytes
    retained = retained_serialization_bytes(core.case_count,
        core.transient_estimated_bytes + budget.frame_bytes,
        run_bytes=run_file_bytes(actual["measurement"]["inventory"], actual["measured_case_ids"]) + budget.run_bytes)
    values.update(
        products_per_frame_estimated_bytes=core.products_per_frame_estimated_bytes + budget.frame_bytes,
        transient_estimated_bytes=core.transient_estimated_bytes + budget.frame_bytes + budget.run_bytes + retained,
        model=core.model + "; " + budget.model)
    return OptionSpaceEstimate(**values, option_frame_bytes=budget.frame_bytes,
        option_run_bytes=budget.run_bytes, retained_serialization_estimated_bytes=retained,
        options=dict(declared=list(budget.options), parts=budget.parts, writers=writers))


def volume_info(root):
    """Fresh space and actual containing-volume metadata; no write probe."""
    path = Path(root).resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    path = path.resolve(strict=True)
    stat = path.stat()
    anchor = str(path.anchor)
    if os.name == "nt":
        buffer = ctypes.create_unicode_buffer(32768)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        from ctypes import wintypes
        kernel.GetVolumePathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        kernel.GetVolumePathNameW.restype = wintypes.BOOL
        if not kernel.GetVolumePathNameW(str(path), buffer, len(buffer)):
            raise ctypes.WinError(ctypes.get_last_error())
        anchor = buffer.value
    if not stat.st_dev:
        raise RunUnavailable("OS did not identify the actual results volume")
    usage = shutil.disk_usage(path)
    return dict(volume_id=str(stat.st_dev), volume_anchor=anchor,
                free_bytes=int(usage.free), checked_container=str(path))


class _LocalVolumeLock:
    """Same OS mutex/flock primitive as ServiceLock, never a NAS lock."""
    def __init__(self, directory, key):
        self.directory, self.key = Path(directory), key
        self.handle = self.file = None

    def __enter__(self):
        if os.name == "nt":
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
            kernel.CreateMutexW.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle.restype = wintypes.BOOL
            self.handle = kernel.CreateMutexW(None, False, "Global\\DarSIACapacity-" + self.key)
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
            status = kernel.WaitForSingleObject(self.handle, 30000)
            if status not in (0, 0x80):
                kernel.CloseHandle(self.handle)
                self.handle = None
                raise RunUnavailable("Local capacity mutex is busy/unavailable; no claim made")
        else:
            import fcntl
            self.file = (self.directory / ("mutex-" + self.key)).open("a")
            fcntl.flock(self.file, fcntl.LOCK_EX)
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.ReleaseMutex(self.handle)
            kernel.CloseHandle(self.handle)
        if self.file is not None:
            self.file.close()


def _owner(value):
    if (not isinstance(value, dict) or set(value) != {"machine", "pid", "process_create_time"}
            or type(value["pid"]) is not int or value["pid"] <= 0
            or not isinstance(value["process_create_time"], str)
            or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value["process_create_time"]) is None):
        raise ValueError("Actual original PID/string process epoch required")
    name(value["machine"])
    return value


def _reservation_id(job_root, task):
    return hashlib.sha256((str(Path(job_root).resolve()).casefold() + "\0" +
        name(task["run"]) + "\0" + task["attempt_id"]).encode()).hexdigest()


def _read_local(path):
    path = Path(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Capacity record exceeds1MiB")
    return _json_bytes(path.read_bytes())


def _capacity_host_context():
    """Actual OS machine-common folder/account; never process environment paths."""
    if os.name != "nt":
        raise RunUnavailable("Host-wide capacity binding is not implemented on this OS; no per-user fallback")
    from ctypes import wintypes
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    shell=ctypes.WinDLL("shell32",use_last_error=True)
    advapi=ctypes.WinDLL("advapi32",use_last_error=True)
    common=ctypes.create_unicode_buffer(32768)
    # CSIDL_COMMON_APPDATA, SHGFP_TYPE_CURRENT: OS KnownFolder compatibility API.
    shell.SHGetFolderPathW.argtypes=[wintypes.HWND,ctypes.c_int,wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR]
    shell.SHGetFolderPathW.restype=ctypes.c_long
    if shell.SHGetFolderPathW(None,0x23,None,0,common)!=0:
        raise RunUnavailable("OS CommonAppData is unavailable; no binding or claim")
    host=ctypes.create_unicode_buffer(256);length=wintypes.DWORD(len(host))
    kernel.GetComputerNameW.argtypes=[wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
    kernel.GetComputerNameW.restype=wintypes.BOOL
    if not kernel.GetComputerNameW(host,ctypes.byref(length)):
        raise ctypes.WinError(ctypes.get_last_error())
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    kernel.LocalFree.argtypes=[ctypes.c_void_p]
    kernel.LocalFree.restype=ctypes.c_void_p
    advapi.OpenProcessToken.argtypes=[wintypes.HANDLE,wintypes.DWORD,ctypes.POINTER(wintypes.HANDLE)]
    advapi.OpenProcessToken.restype=wintypes.BOOL
    advapi.GetTokenInformation.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD)]
    advapi.GetTokenInformation.restype=wintypes.BOOL
    advapi.ConvertSidToStringSidW.argtypes=[ctypes.c_void_p,ctypes.POINTER(wintypes.LPWSTR)]
    advapi.ConvertSidToStringSidW.restype=wintypes.BOOL
    token=wintypes.HANDLE();sid_text=wintypes.LPWSTR()
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(),0x0008,ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        size=wintypes.DWORD()
        advapi.GetTokenInformation(token,1,None,0,ctypes.byref(size))  # TokenUser
        if ctypes.get_last_error()!=122 or not 0<size.value<=65536:
            raise RunUnavailable("Actual service-account token metadata is unavailable")
        data=ctypes.create_string_buffer(size.value)
        if not advapi.GetTokenInformation(token,1,data,size.value,ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        # TOKEN_USER begins with SID_AND_ATTRIBUTES; its first field is PSID.
        sid=ctypes.cast(data,ctypes.POINTER(ctypes.c_void_p)).contents.value
        if not advapi.ConvertSidToStringSidW(sid,ctypes.byref(sid_text)):
            raise ctypes.WinError(ctypes.get_last_error())
        account=sid_text.value
    finally:
        if sid_text:kernel.LocalFree(ctypes.cast(sid_text,ctypes.c_void_p))
        kernel.CloseHandle(token)
    return dict(host=host.value.casefold(),service_account_sid=account,
                common_root=str(Path(common.value)/"DarSIA"/"capacity-bindings"))


def _authority_alias(value):
    """An explicitly listed namespace root, never a physical-volume identity."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Explicit NAS server/share alias required")
    pure = PureWindowsPath(value)
    if (not pure.is_absolute() or not pure.drive.startswith("\\\\")
            or pure.parts != (pure.anchor,) or pure.drive.startswith(("\\\\?", "\\\\."))):
        raise ValueError("NAS alias must name one whole normalized UNC server/share root")
    return str(pure).casefold()


def _owner_scope(document, binding, results_root, context):
    """Validate optional legacy alias provenance; no physical/host exclusivity."""
    _bound(binding)
    fields = {"format", "canonical_owner_path", "aliases", "owner", "single_computing_host"}
    if (not isinstance(document, dict) or set(document) != fields
            or document.get("format") != "darsia-single-host-nas-authority-v1"
            or document.get("single_computing_host") is not True):
        raise RunUnavailable("NAS results need the exact immutable single-host authority config; no task claimed")
    aliases = document["aliases"]
    if not isinstance(aliases, list) or not 0 < len(aliases) <= 128:
        raise ValueError("Bounded explicit NAS alias list required")
    keys, drives = [], []
    for value in aliases:
        if isinstance(value, str):
            if drives:
                raise ValueError("UNC aliases must precede explicit drive aliases")
            keys.append(_authority_alias(value))
        elif (isinstance(value, dict) and set(value) == {"drive", "target_unc"}
                and isinstance(value["drive"], str)
                and re.fullmatch(r"[A-Z]:", value["drive"]) is not None):
            drives.append(dict(drive=value["drive"], target_unc=_authority_alias(value["target_unc"])))
        else:
            raise ValueError("NAS aliases must be UNC roots or exact drive/target_unc bindings")
    if (keys != sorted(set(keys))
            or [value["drive"] for value in drives] != sorted({value["drive"] for value in drives})
            or any(value["target_unc"] not in keys for value in drives)):
        raise ValueError("NAS authority aliases must be unique and canonically ordered")
    canonical = document["canonical_owner_path"]
    if (not isinstance(canonical, str) or not canonical or "\x00" in canonical
            or not PureWindowsPath(canonical).is_absolute()
            or ".." in PureWindowsPath(canonical).parts
            or PureWindowsPath(canonical).name.casefold() != "nas_owner.json"
            or str(PureWindowsPath(canonical)).casefold() != str(PureWindowsPath(binding["path"])).casefold()
            or _authority_alias(PureWindowsPath(canonical).anchor) not in keys):
        raise ValueError("NAS owner descriptor must name its canonical shared NAS_OWNER.json")
    actual = PureWindowsPath(str(results_root))
    in_scope = (_authority_alias(actual.anchor) in keys if actual.drive.startswith("\\\\")
                else any(value["drive"] == actual.drive.upper() for value in drives))
    if not actual.is_absolute() or ".." in actual.parts or not in_scope:
        raise RunUnavailable("NAS results root is outside the complete operator-listed authority scope; use local outputs; no task claimed")
    owner = document["owner"]
    if (not isinstance(owner, dict) or set(owner) != {"machine_id", "host", "service_account_sid"}
            or not isinstance(owner["host"], str) or not owner["host"]
            or owner["host"] != owner["host"].strip().casefold()
            or not isinstance(owner["service_account_sid"], str)
            or re.fullmatch(r"S-1-[0-9]+(?:-[0-9]+)+", owner["service_account_sid"]) is None):
        raise ValueError("Exact actual computing host/account and machine-id provenance required")
    name(owner["machine_id"])
    # N51 withdraws Q48's single computing-host rule for RESULTS. This legacy
    # descriptor remains optional alias/path provenance, not permission for one
    # host/account. The ledger binds the actual local host/SID independently.
    return dict(format="darsia-single-host-nas-authority-scope-v1",
        owner_document=dict(path=canonical, sha256=binding["sha256"], size_bytes=binding["size_bytes"]),
        aliases=keys + drives, owner=dict(owner),
        accounting="one conservative host/authority pool across every listed namespace; no physical-volume equivalence claimed")


def _mapped_results_alias(scope, results_root):
    if scope is None:
        return None
    actual = PureWindowsPath(str(results_root))
    return next((value for value in scope["aliases"] if isinstance(value, dict)
                 and value["drive"] == actual.drive.upper()), None)


def _verify_mapped_results(scope, results_root, mapping_provider):
    selected = _mapped_results_alias(scope, results_root)
    if selected is None:
        return
    from ..job_filesystem import windows_mapped_drive_target
    provider = windows_mapped_drive_target if mapping_provider is None else mapping_provider
    try:
        target = provider(selected["drive"])
        observed = _authority_alias(target)
    except (OSError, ValueError, TypeError) as error:
        raise RunUnavailable("Current mapped results target is unavailable; no task claimed: " + str(error)) from error
    if observed != selected["target_unc"]:
        raise RunUnavailable("Mapped results drive changed its exact approved UNC target; no task claimed")


def remote_results_owner(stores, series_id, results_root, *, admission=None, context=None,
                         mapping_provider=None):
    """Admit RESULTS on either host; optional legacy alias metadata stays exact."""
    from ..job_filesystem import inspect_results_root, valid_results_admission, LocalFileAdmission
    proof = inspect_results_root(results_root) if admission is None else admission
    if not valid_results_admission(proof, results_root):
        raise RunUnavailable("Configured results filesystem has no positive output admission; no task claimed")
    if isinstance(proof, LocalFileAdmission) or proof.local is True:
        return None
    from ..store_map import capacity_settings
    configured = capacity_settings(stores.document)
    binding = None if configured is None else configured.get("remote_owner")
    if binding is None:
        # Positive filesystem admission is still mandatory. N51 requires no
        # NAS owner election, hard results quota or cross-host reservation.
        return None
    _bound(binding)
    if not 0 < binding["size_bytes"] <= 16 * 1024:
        raise ValueError("NAS owner metadata must fit 16 KiB")
    try:
        document, _ = read_bound(binding)
    except (OSError, ValueError) as error:
        raise RunUnavailable("The pinned NAS owner document is missing or changed; no task claimed: " + str(error)) from error
    scope = _owner_scope(document, binding, results_root,
                         _capacity_host_context() if context is None else context)
    _verify_mapped_results(scope, results_root, mapping_provider)
    return scope


class CapacityLedger:
    """Immutable own reservation/progress/release files, shared across jobs."""
    def __new__(cls, stores, series_id, results_root, **kwargs):
        from darsia_cache.folders import folder_mode
        if cls is CapacityLedger and folder_mode(stores, series_id):
            from .folder_capacity import FolderCapacityLedger
            return FolderCapacityLedger(stores, series_id, results_root, **kwargs)
        return super().__new__(cls)

    def __init__(self, stores, series_id, results_root, *, volume_provider=volume_info,
                 context_provider=_capacity_host_context, mapping_provider=None, job_root=None):
        from ..job_filesystem import inspect_local_file
        from ..store_map import capacity_settings
        configured=capacity_settings(stores.document)
        if configured is None:
            raise RunUnavailable("Configure local capacity state before claiming; no measured-profile fallback")
        self.directory,self.reserve=Path(configured["state_directory"]),configured["reserve_bytes"]
        evidence = inspect_local_file(self.directory / ".capacity-probe")
        if not evidence.allowed:
            raise RunUnavailable("Capacity state must be on an admitted LOCAL filesystem")
        from ..store_map import _external_file
        _external_file(self.directory / "identity.json", tuple(stores.roots(series_id).values()))
        self.machine_id, self.results_root = stores.machine_id, Path(results_root)
        self.directory=self.directory.expanduser().resolve()
        if self.directory.exists() and not self.directory.is_dir():
            raise RunUnavailable("Configured capacity state_directory is not a directory; no host binding or claim")
        self.volume_provider,self.context_provider=volume_provider,context_provider
        self.stores,self.series_id,self.mapping_provider=stores,series_id,mapping_provider
        self.mapping_observation=None
        context=context_provider()
        self.remote_scope=remote_results_owner(stores,series_id,self.results_root,context=context,
                                               mapping_provider=self._mapped_target)
        self.volume=volume_provider(self.results_root)
        if (not isinstance(context,dict) or set(context)!={"host","service_account_sid","common_root"}
                or not isinstance(context["host"],str) or not context["host"]
                or not isinstance(context["service_account_sid"],str)
                or re.fullmatch(r"S-1-[0-9]+(?:-[0-9]+)+",context["service_account_sid"]) is None):
            raise RunUnavailable("Actual OS host, service-account SID and common root required")
        self.host=context["host"].casefold()
        self.account=context["service_account_sid"]
        self.common_root=_external_file(Path(context["common_root"])/"binding-probe.json",tuple(stores.roots(series_id).values())).parent
        common_evidence=inspect_local_file(self.common_root/"binding-probe.json")
        if not common_evidence.allowed:
            raise RunUnavailable("Machine-common capacity binding must be positively LOCAL")
        # Logical map/service aliases never create a second physical namespace.
        pool = (str(self.volume["volume_id"]) if self.remote_scope is None else
                "nas-authority:" + self.remote_scope["owner_document"]["sha256"])
        self.key=hashlib.sha256((self.host+"\0"+pool).encode()).hexdigest()
        self.root=self.directory/self.key
        self.binding_path=self.common_root/(self.key+".json")

    def _common_binding(self, *, create=False):
        expected=dict(format="darsia-host-volume-capacity-binding-v1",physical_host=self.host,
            volume_id=self.volume["volume_id"],service_account_sid=self.account,
            state_directory=str(self.directory),binding_file=str(self.binding_path))
        if self.remote_scope is not None:
            expected=dict(format="darsia-host-nas-authority-capacity-binding-v1",physical_host=self.host,
                service_account_sid=self.account,remote_scope=self.remote_scope,
                state_directory=str(self.directory),binding_file=str(self.binding_path))
        if self.binding_path.exists():
            if _read_local(self.binding_path)!=expected:
                raise RunUnavailable("Host-volume capacity binding names another configured directory or service-account SID; no claim")
            return True
        if create:
            self.common_root.mkdir(parents=True,exist_ok=True)
            write_immutable(self.binding_path,expected)
            return True
        if self.root.exists():
            raise RunUnavailable("Preserved ledger has no authoritative host-volume binding; inspect it, no fallback")
        return False

    @contextmanager
    def locked(self, *, create=True):
        # Existing admission keeps its original create path. Read-only J6 never
        # initializes a root, binding, identity or POSIX lock file.
        if not create:
            if not all(path.is_dir() for path in (self.common_root, self.directory, self.root)):
                raise RunUnavailable("Read-only projection needs the existing result ledger")
            if os.name != "nt" and not (self.common_root / ("mutex-" + self.key)).is_file():
                raise RunUnavailable("Read-only projection has no existing local mutex file")
        if os.name!="nt" and create:self.common_root.mkdir(parents=True,exist_ok=True)
        with _LocalVolumeLock(self.common_root,self.key):
            if not self._common_binding(create=create):
                raise RunUnavailable("Read-only projection needs the existing result binding")
            if create:
                self.directory.mkdir(parents=True,exist_ok=True)
                self.root.mkdir(exist_ok=True)
            marker=dict(format="darsia-local-capacity-volume-v2",physical_host=self.host,
                service_account_sid=self.account,volume_id=self.volume["volume_id"],
                state_directory=str(self.directory),host_binding=str(self.binding_path))
            if self.remote_scope is not None:
                marker=dict(format="darsia-host-local-capacity-authority-v1",physical_host=self.host,
                    service_account_sid=self.account,remote_scope=self.remote_scope,
                    state_directory=str(self.directory),host_binding=str(self.binding_path))
            path=self.root/"identity.json"
            if path.exists():
                if _read_local(path)!=marker:
                    raise RunUnavailable("Capacity directory/host/account/volume identity differs")
            elif create:
                write_immutable(path,marker)
            else:
                raise RunUnavailable("Read-only projection needs the existing result identity")
            yield self

    def _mapped_target(self, drive):
        from ..job_filesystem import windows_mapped_drive_target
        provider = windows_mapped_drive_target if self.mapping_provider is None else self.mapping_provider
        target = provider(drive)
        self.mapping_observation = dict(drive=drive, target_unc=target,
            provider="WNetGetConnectionW" if self.mapping_provider is None else "injected mapping provider")
        return target

    def fresh_volume(self):
        if _mapped_results_alias(self.remote_scope, self.results_root) is not None:
            self.mapping_observation = None
            current = remote_results_owner(self.stores, self.series_id, self.results_root,
                context=self.context_provider(), mapping_provider=self._mapped_target)
            if current != self.remote_scope:
                raise RunUnavailable("Mapped results owner authority changed; no task claimed")
        value = self.volume_provider(self.results_root)
        if value["volume_id"] != self.volume["volume_id"]:
            raise RunUnavailable("Results volume changed during capacity admission")
        if self.mapping_observation is not None:
            value = dict(value, mapped_drive_observation=dict(self.mapping_observation))
        return value

    def remaining(self, *, exclude_job=None):
        if not self._common_binding():
            return 0  # Prospective readonly report; no v2 claim can exist unbound.
        total = 0
        from itertools import islice
        paths = list(islice(self.root.glob("r-*/reserved.json"), MAX_RESERVATIONS + 1))
        if len(paths) > MAX_RESERVATIONS:
            raise RunUnavailable("Preserved capacity history needs explicit maintenance; no claim")
        for path in paths:
            row = _read_local(path)
            if (row.get("physical_host")!=self.host or row.get("service_account_sid")!=self.account
                    or (self.remote_scope is None and row.get("volume_id")!=self.volume["volume_id"])
                    or (self.remote_scope is not None and row.get("remote_scope")!=self.remote_scope)):
                raise ValueError("Reservation belongs to another actual host/account/capacity scope")
            if (row.get("format") not in {"darsia-local-capacity-reservation-v1",
                    "darsia-n43-predeclared-capacity-mirror-v1", "darsia-first-whole-output-reservation-v1"}
                    or path.parent.name != "r-" + _reservation_id(row["job_root"], row)
                    or type(row.get("case_count")) is not int or row["case_count"] < 1
                    or len(row.get("ordered_case_ids", [])) != row["case_count"]
                    or len(set(row["ordered_case_ids"])) != row["case_count"]):
                raise ValueError("Original reservation identity/case coverage differs")
            _owner(row["owner"])
            if row.get("reservation_model")=="N43-shared-aggregate-service-mirror":
                _integer(row["maximum_aggregate_bytes"],"Predeclared N43 aggregate",positive=True)
                if (row["format"]!="darsia-n43-predeclared-capacity-mirror-v1"
                        or row["maximum_aggregate_bytes"]!=65728995330
                        or row["case_count"]!=39 or row["run"]!="ac52"
                        or not isinstance(row.get("shared_service_reservation"),dict)):
                    raise ValueError("N43 mirror differs from the actual one service entitlement")
                _bound(row["request"]);_bound(row["administration"])
                if row["shared_service_reservation"]!=row["estimate"]["shared_service_reservation"]:
                    raise ValueError("N43 mirror differs from the actual one service entitlement")
            elif row.get("reservation_model") == "first-Q104-whole-output-once":
                from darsia_cache.first_qualification import BOUND_OUTPUT_ADMINISTRATION, FIRST_OUTPUT_BASE_ENTITLEMENT_BYTES
                if (row["format"] != "darsia-first-whole-output-reservation-v1"
                        or type(row.get("maximum_output_bytes")) is not int
                        or row["maximum_output_bytes"] < FIRST_OUTPUT_BASE_ENTITLEMENT_BYTES
                        or row["case_count"] != 39 or row["run"] != "ac52"
                        or row.get("not_a_measured_profile") is not True
                        or row.get("estimate", {}).get("maximum_output_bytes") != row["maximum_output_bytes"]):
                    raise ValueError("First Q104 whole-claim entitlement differs")
                for role in ("request", "administration", "all_stage_reference"):
                    _bound(row[role])
                    if row[role] != row["estimate"].get(role):
                        raise ValueError("First Q104 reservation binding differs: " + role)
                admin = read_bound(row["administration"])[0]
                if (admin.get("format") != BOUND_OUTPUT_ADMINISTRATION
                        or admin.get("administrative_output_entitlement_bytes") != row["maximum_output_bytes"]
                        or admin.get("scope") != row.get("scope")
                        or row["scope"] != row["estimate"].get("scope")
                        or row["scope"].get("ordered_case_ids") != row["ordered_case_ids"]):
                    raise ValueError("First Q119 reservation differs from actual predeclared whole-output plan")
            elif row.get("reservation_model") == "setup-explicit-whole-operation":
                from .series_setup_admission import validate_retained_reservation
                validate_retained_reservation(row, capacity_key=self.key,
                                              volume_id=self.volume["volume_id"])
            elif row.get("reservation_model") == "preset-wasserstein-whole-time":
                from ..preset_wasserstein_capacity import validate_retained_reservation
                validate_retained_reservation(row)
            else:
                if row["format"]!="darsia-local-capacity-reservation-v1":
                    raise ValueError("Unknown typed capacity reservation model")
                _integer(row["products_per_frame_estimated_bytes"], "Reserved products")
                _integer(row["transient_estimated_bytes"], "Reserved transient")
            if exclude_job is not None and Path(row["job_root"]).resolve() == Path(exclude_job).resolve():
                continue
            release = path.parent / "released.json"
            if release.exists():
                ended = _read_local(release)
                if (ended.get("reservation_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                        or ended.get("owner") != row["owner"] or ended.get("attempt_id") != row["attempt_id"]
                        or ended.get("format") != "darsia-local-capacity-release-v1"):
                    raise ValueError("Capacity release belongs to another reservation")
                continue
            committed = 0
            progress_paths = list(islice(path.parent.glob("p-*.json"), row["case_count"] + 1))
            if (row.get("reservation_model") in {"setup-explicit-whole-operation", "preset-wasserstein-whole-time"}
                    and progress_paths):
                raise ValueError("A whole-operation reservation cannot have frame progress")
            if len(progress_paths) > row["case_count"]:
                raise ValueError("Capacity progress exceeds original case count")
            for progress in progress_paths:
                update = _read_local(progress)
                if (update.get("owner") != row["owner"] or update.get("attempt_id") != row["attempt_id"]
                        or update.get("case_id") not in row["ordered_case_ids"]
                        or update.get("committed_prefix_count") != row["ordered_case_ids"].index(update["case_id"]) + 1):
                    raise ValueError("Capacity progress ownership differs")
                committed = max(committed, _integer(update.get("committed_prefix_count"), "Committed prefix"))
            if committed > row["case_count"]:
                raise ValueError("Capacity progress exceeds original case count")
            if row.get("reservation_model")=="N43-shared-aggregate-service-mirror":
                total += row["maximum_aggregate_bytes"]
            elif row.get("reservation_model") == "first-Q104-whole-output-once":
                # Never divide by 39 or shrink a whole-tree ceiling with prefix
                # progress. Actual committed/preparation files remain retained.
                total += row["maximum_output_bytes"]
            elif row.get("reservation_model") == "setup-explicit-whole-operation":
                total += row["maximum_output_bytes"]
            elif row.get("reservation_model") == "preset-wasserstein-whole-time":
                total += row["maximum_output_bytes"] + row["transient_estimated_bytes"]
            else:
                total += ((row["case_count"] - committed) * row["products_per_frame_estimated_bytes"]
                          + row["transient_estimated_bytes"])
        return total

    def bind(self, job_root, claim, owner, estimate):
        _owner(owner)
        from .series_setup_admission import SetupVolumeReservation
        if not self._common_binding():
            raise RunUnavailable("Bind the authoritative host-volume directory before any reservation")
        if claim.token != claim.task["attempt_id"] or estimate.case_count != len(claim.task["ordered_case_ids"]):
            raise ValueError("Capacity reservation differs from the actual claim")
        path = self.root / ("r-" + _reservation_id(job_root, claim.task))
        if path.exists():
            raise RunUnavailable("An original capacity reservation already exists for this claim token")
        path.mkdir()
        common = dict(machine_id=self.machine_id, physical_host=self.host,service_account_sid=self.account,
            volume_id=self.volume["volume_id"], owner=dict(owner), job_root=str(Path(job_root).resolve()),
            run=claim.task["run"], attempt_id=claim.token, case_count=estimate.case_count,
            ordered_case_ids=list(claim.task["ordered_case_ids"]))
        if isinstance(estimate,PredeclaredN43ProducerEntitlement):
            value = dict(common,format="darsia-n43-predeclared-capacity-mirror-v1",
                reservation_model="N43-shared-aggregate-service-mirror",
                maximum_aggregate_bytes=estimate.maximum_aggregate_bytes,
                shared_service_reservation=estimate.shared_service_reservation,
                request=estimate.request,administration=estimate.administration,
                not_a_measured_profile=True, estimate=asdict(estimate))
        elif isinstance(estimate, PredeclaredFirstOutputEntitlement):
            value = dict(common, format="darsia-first-whole-output-reservation-v1",
                reservation_model="first-Q104-whole-output-once", maximum_output_bytes=estimate.maximum_output_bytes,
                request=estimate.request, administration=estimate.administration,
                all_stage_reference=estimate.all_stage_reference, scope=estimate.scope,
                not_a_measured_profile=True, estimate=asdict(estimate))
        elif isinstance(estimate, SetupVolumeReservation):
            if (estimate.capacity_key != self.key or estimate.volume_id != self.volume["volume_id"]
                    or claim.task.get("pipeline") != "series-setup"
                    or estimate.operation != claim.task.get("operation")):
                raise ValueError("Setup output reservation differs from its actual volume or operation")
            value = dict(common, format="darsia-local-capacity-reservation-v1",
                reservation_model="setup-explicit-whole-operation",
                maximum_output_bytes=estimate.maximum_output_bytes,
                products_per_frame_estimated_bytes=estimate.maximum_output_bytes,
                transient_estimated_bytes=0,
                not_a_measured_profile=True, estimate=asdict(estimate))
            from .series_setup_admission import validate_retained_reservation
            validate_retained_reservation(value, capacity_key=self.key,
                                          volume_id=self.volume["volume_id"])
        elif claim.task.get("pipeline") == "preset-wasserstein":
            from ..preset_wasserstein_capacity import require_estimate, validate_retained_reservation
            require_estimate(claim.task, estimate, allow_administration=True)
            value = dict(common, format="darsia-local-capacity-reservation-v1",
                reservation_model="preset-wasserstein-whole-time",
                maximum_output_bytes=estimate.products_estimated_bytes,
                transient_estimated_bytes=estimate.transient_estimated_bytes,
                comparison_count=estimate.comparison_count, scope=estimate.scope,
                estimate=asdict(estimate))
            validate_retained_reservation(value)
        else:
            value = dict(common,format="darsia-local-capacity-reservation-v1",
                products_per_frame_estimated_bytes=estimate.products_per_frame_estimated_bytes,
                transient_estimated_bytes=estimate.transient_estimated_bytes, estimate=asdict(estimate))
        if self.remote_scope is not None:
            value.update(remote_scope=self.remote_scope, results_root=str(self.results_root),
                observed_client_volume=dict(self.volume),
                volume_identity_claim="none; scope accounting spans every operator-listed alias")
        write_immutable(path / "reserved.json", value)
        return path

    def progress(self, reservation, owner, event):
        self._common_binding()
        if event.get("state") not in {"frame_completed", "phase1_frame_completed"}:
            return
        row = _read_local(Path(reservation) / "reserved.json")
        if row.get("reservation_model") in {"setup-explicit-whole-operation", "preset-wasserstein-whole-time"}:
            raise ValueError("Whole-operation capacity remains reserved until actual release")
        if row["owner"] != owner or event.get("complete") is not True or event.get("run") != row["run"]:
            raise ValueError("Capacity progress must follow the original owned committed event")
        case = event.get("case_id")
        if case not in row["ordered_case_ids"]:
            raise ValueError("Capacity progress case is outside the original task")
        position = row["ordered_case_ids"].index(case) + 1
        # Both original executors commit a chronological prefix, including any
        # prefix admitted by their unchanged resume validators before this event.
        value = dict(owner=owner, attempt_id=row["attempt_id"], case_id=case,
                     committed_prefix_count=position, original_event=event)
        path = Path(reservation) / ("p-" + str(position).zfill(8) + ".json")
        if path.exists():
            if _read_local(path) != value:
                raise ValueError("Different original capacity progress already exists")
        else:
            write_immutable(path, value)

    def release(self, reservation, actor, *, retirement=None):
        self._common_binding()
        _owner(actor)
        path = Path(reservation) / "reserved.json"
        row = _read_local(path)
        if retirement is None:
            if actor != row["owner"]:
                raise ValueError("Only the original owner may close its own reservation")
            basis = "original_executor_left_and_IO_scope_closed"
        else:
            identities = retirement.get("identities", [])
            if (retirement.get("complete") is not True or retirement.get("all_absent") is not True
                    or retirement.get("worker_os_references_closed") is not True
                    or retirement.get("supervisor") != actor or retirement.get("run") != row["run"]
                    or retirement.get("attempt_id") != row["attempt_id"]
                    or not any(i.get("pid") == row["owner"].get("pid")
                        and i.get("process_create_time") == row["owner"].get("process_create_time") for i in identities)):
                raise ValueError("Original supervisor retirement does not bind this reservation owner")
            basis = "original_owned_child_retirement"
        value = dict(format="darsia-local-capacity-release-v1", owner=row["owner"], actor=actor,
            attempt_id=row["attempt_id"], basis=basis,
            reservation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), retirement=retirement)
        destination = Path(reservation) / "released.json"
        if not destination.exists():
            write_immutable(destination, value)
        # A prior own release is retained, never overwritten by later retirement.
        elif _read_local(destination).get("reservation_sha256") != value["reservation_sha256"]:
            raise ValueError("Capacity release changed reservation identity")


def report(estimate, volume, other_remaining, reserve):
    # N51 RESULTS admission is only this job estimate plus the reserve, as seen
    # by this host. Retained local records are informative, not a NAS quota.
    other = _integer(other_remaining, "Other active remaining bytes")
    needed = estimate.needed(reserve_bytes=reserve)
    free = _integer(volume["free_bytes"], "Actual free bytes")
    message = f"{estimate.run}: estimated needed {needed} bytes / free {free} bytes (this job, estimated transient space and reserve; estimate, not coordinated across hosts)"
    if isinstance(estimate, StorageProfileEstimate):
        message += "; storage profile " + estimate.storage_profile + ": " + estimate.model
    return dict(**asdict(estimate), needed_bytes=needed, free_bytes=free,
                other_active_remaining_bytes=other, reserve_bytes=reserve,
                other_active_remaining_included_in_results_estimate=False,
                results_space_policy="estimate, not coordinated across hosts",
                sufficient=free >= needed, volume=volume, message=message)


def configuration_for_task(stores, series_id, task, rows=None):
    """Read the one already-declared run configuration, never array payloads."""
    from ..store_map import require_input_containment
    bound = task["configuration"]
    if rows is None:
        path = require_input_containment(bound, stores, series_id)
    else:
        matches = [row.path for row in rows if "run.configuration" in row.uses
                   and row.reference == bound]
        if len(matches) != 1:
            raise ValueError("One original run.configuration role is required")
        path = matches[0]
    return read_bound(dict(path=str(path), sha256=bound["sha256"], size_bytes=bound["bytes"]))[0]


class ClaimCapacity:
    """One host-local critical section spans fresh estimate, claim and reservation."""
    def __init__(self, runtime, owner, on_refusal=None):
        self.runtime, self.owner, self.on_refusal = runtime, _owner(owner), on_refusal
        self.ledger = self.lock = self.error = None
        self.cache = None
        self.folder_mode = False
        self.estimates = {}
        self.setup_candidate = None

    def __enter__(self):
        try:
            root = self.runtime.stores.roots(self.runtime.job["series_id"])["results"]
            from darsia_cache.folders import folder_mode
            self.folder_mode = folder_mode(self.runtime.stores, self.runtime.job["series_id"])
            self.ledger = CapacityLedger(self.runtime.stores, self.runtime.job["series_id"], root,
                **({"job_root": self.runtime.root} if self.folder_mode else {}))
            if self.runtime.job.get("pipeline") != "series-setup" and not self.folder_mode:
                from darsia_cache.control import ClaimCacheBudget
                self.cache = ClaimCacheBudget(self.runtime, self.ledger, self.owner)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.error = "Capacity admission unavailable: " + str(error)
        return self

    def _unlock(self):
        try:
            if self.setup_candidate is not None:
                candidate, self.setup_candidate = self.setup_candidate, None
                candidate.unlock()
                if not candidate.bound and not candidate.admission.bound_reservations:
                    self.runtime.setup_verified_inputs.pop(candidate.task["attempt_id"], None)
            if self.cache is not None:
                self.cache.unlock()  # Reverse of result-volume -> cache acquisition.
        finally:
            if self.lock is not None:
                active,self.lock=self.lock,None
                active.__exit__(None,None,None)

    def _refuse(self, task, error):
        text = str(error)
        if self.runtime.capacity_refusals.get(task["run"]) != text:
            if self.on_refusal is not None:
                self.on_refusal(dict(kind="capacity_refusal", run=task["run"], message=text))
            self.runtime.capacity_refusals[task["run"]] = text
        raise RunUnavailable(text) from error

    def availability(self, task):
        self._unlock()  # A prior queue candidate may have lost its atomic rename.
        checkpoint = getattr(self, "checkpoint", lambda: None)
        checkpoint()
        if self.error is not None:
            self._refuse(task, RunUnavailable(self.error))
        # Original input/resume identity errors retain their original behavior.
        rows = self.runtime.availability(task)
        checkpoint()
        if rows is None and task.get("pipeline") == "series-setup":
            return None
        try:
            config = configuration_for_task(self.runtime.stores, self.runtime.job["series_id"], task, rows)
            checkpoint()
            if task.get("pipeline") == "series-setup":
                from .series_setup_worker_admission import setup_claim_admission
                self.setup_candidate = setup_claim_admission(self.runtime, task, config, self.owner,
                    input_rows=rows)
                checkpoint()
                self.setup_candidate.acquire()
                self.setup_candidate.check()
                checkpoint()
                self.runtime.capacity_refusals.pop(task["run"], None)
                return rows
            first=getattr(getattr(getattr(self.runtime,"cache_adapter",None),"lease",None),"first_qualification",None)
            lease = getattr(getattr(self.runtime, "cache_adapter", None), "lease", None)
            if self.folder_mode:
                estimate = load_estimate(self.runtime.stores, self.runtime.job["series_id"], task, config,
                    repository=Path(__file__).resolve().parents[4], metadata_rows=rows)
            elif task.get("pipeline") == "preset-wasserstein":
                from darsia_cache.preset_qualification import PresetQualification
                if isinstance(first, PresetQualification):
                    from ..preset_wasserstein_capacity import first_entitlement
                    estimate = first_entitlement(self.runtime, task, config)
                else:
                    if first is not None or getattr(lease, "profile", {}).get("format") not in {
                            "darsia-preset-wasserstein-bounded-profile-v1"}:
                        from ..preset_wasserstein_capacity import MEASURED_UNAVAILABLE
                        raise RunUnavailable(MEASURED_UNAVAILABLE)
                    estimate = load_estimate(self.runtime.stores, self.runtime.job["series_id"], task, config,
                                             repository=Path(__file__).resolve().parents[4], metadata_rows=rows)
            elif first is not None and getattr(first, "common_mode", False):
                from darsia_cache.common_qualification import common_output_entitlement
                estimate = common_output_entitlement(self.runtime, task, config, rows)
            elif first is not None and first.reference_build:
                estimate=n43_entitlement(self.runtime,task,config,rows)
            elif first is not None and first.bound_mode:
                estimate = first_output_entitlement(self.runtime, task, config, rows)
            else:
                if getattr(lease, "profile", {}).get("format") == "darsia-common-phase-import-profile-v1":
                    from darsia_cache.common_qualification import require_applicable, JOB_PROFILE
                    require_applicable(dict(lease.profile, format=JOB_PROFILE), task, config=config)
                estimate = load_estimate(self.runtime.stores, self.runtime.job["series_id"], task, config,
                                         repository=Path(__file__).resolve().parents[4], metadata_rows=rows)
            # Slow original input admission/profile reads run outside this local
            # transaction. Only fresh space -> claim -> reservation is serialized.
            if self.cache is not None:
                self.cache.prepare(task, estimate)
            self.lock=self.ledger.locked()
            self.lock.__enter__()
            if self.cache is not None:
                self.cache.acquire()  # Fixed result-volume -> cache lock order.
            other=self.ledger.remaining()
            cache_report = (self.cache.check(task, estimate, other) if self.cache is not None else
                            dict(mode="folder", disposable=True))
            if isinstance(estimate,PredeclaredN43ProducerEntitlement):
                # cache_report already checks the one WHOLE aggregate against
                # each affected volume. Never add another 65GB output budget.
                current=dict(cache_report,model=estimate.model,cache=cache_report)
            else:
                volume = self.ledger.fresh_volume()
                if self.folder_mode:
                    volume = dict(volume, free_bytes=max(0, volume["free_bytes"] - other))
                current = report(estimate,volume,other,self.ledger.reserve)
                current["cache"] = cache_report
            if not current["sufficient"]:
                raise RunUnavailable(current["message"] + "; leave this whole run pending")
            self.estimates[task["attempt_id"]] = estimate
            self.runtime.capacity_refusals.pop(task["run"],None)
            return rows
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._unlock()
            self._refuse(task,error)

    def bind(self, claim):
        if claim is not None:
            if claim.task.get("pipeline") == "series-setup":
                if self.setup_candidate is None:
                    raise ValueError("The actual Setup claim has no locked original candidate")
                if not hasattr(self.runtime, "setup_admissions"):
                    self.runtime.setup_admissions = {}
                self.runtime.setup_admissions[claim.token] = self.setup_candidate
                self.setup_candidate.bind(claim)
                return
            lease = getattr(getattr(self.runtime, "cache_adapter", None), "lease", None)
            first = getattr(lease, "first_qualification", None)
            if first is not None:
                first.remember_claim_locked(lease, claim)
            reservation = self.ledger.bind(self.runtime.root, claim, self.owner,
                                           self.estimates[claim.token])
            self.runtime.capacities[claim.token] = (self.ledger, reservation, self.owner)
            if self.cache is not None:
                self.cache.bind(claim)  # Managed mode retains its original writer admission.

    def claim_controls(self, task, machine):
        """Bound Setup's finite original claim controls; leave other IO intact."""
        if task.get("pipeline") != "series-setup":
            from contextlib import nullcontext
            return nullcontext()
        if self.setup_candidate is None:
            raise ValueError("Setup claim controls require their locked original candidate")
        return self.setup_candidate.preclaim_scope(task, machine)

    def __exit__(self, *args):
        self._unlock()


def release_retired_capacity(root, task, stores, retirement, *, wait_seconds=None):
    """Use the original supervisor's proven retirement; never inspect/kill PIDs."""
    from .job_queue import read_json
    from darsia_cache.folders import folder_mode
    job = read_json(Path(root) / "job.json")
    ordinary = folder_mode(stores, job["series_id"])
    if stores.document.get("capacity") is None and not ordinary:
        # Pre-J6 attempts cannot own a reservation in a never-configured ledger.
        return
    if task.get("pipeline") == "series-setup":
        from .series_setup_worker_admission import release_retired_setup_capacity
        return release_retired_setup_capacity(root, task, stores, retirement, wait_seconds=wait_seconds)
    result_root = stores.roots(job["series_id"])["results"]
    ledger = CapacityLedger(stores, job["series_id"], result_root,
        **({"job_root": root} if ordinary else {}))
    path = ledger.root / ("r-" + _reservation_id(root, task))
    if not (path / "reserved.json").exists():
        # Older pre-J6 attempts have no new reservation to release.
        return
    ledger.release(path, retirement["supervisor"], retirement=retirement)


def whole_job_report(stores, series_id, tasks, configs, *, repository, job_root=None, cache_provider=None):
    """Whole-job projection, using only an explicitly admitted cache provider.

    No service is bootstrapped, profile synthesized or ledger initialized here.
    Missing/invalid cache qualification refuses combined sufficiency. Original
    result estimates remain visible and actual claim repeats fresh checks.
    """
    if len(tasks) != len(configs) or not tasks or len({t["run"] for t in tasks}) != len(tasks):
        raise ValueError("Exact nonempty task/configuration correspondence required")
    root = stores.roots(series_id)["results"]
    from ..job_filesystem import inspect_results_root
    admission = inspect_results_root(root)
    if not admission.allowed:
        raise RunUnavailable("Configured results filesystem refused: " + admission.reason)
    from darsia_cache.folders import folder_mode
    ordinary = folder_mode(stores, series_id)
    ledger = CapacityLedger(stores, series_id, root, **({"job_root": job_root} if ordinary else {}))
    estimates = [load_estimate(stores, series_id, task, config, repository=repository)
                 for task, config in zip(tasks, configs)]
    cache = dict(complete=False, sufficient=False, state="MISSING_QUALIFIED_CACHE_PROVIDER",
        reason="Inject the actual accepted WorkerCacheAdapter; no default namespace/service/profile is constructed",
        required_cache_bytes=None, eligible_reclaimable_logical_bytes=None,
        quota_headroom_bytes=None, projection_only=True)
    if ordinary:
        cache = dict(complete=True, sufficient=True, state="DISPOSABLE_FOLDER_CACHE",
            reason="Regenerable per-computer cache; output-space admission is in the shared job",
            required_cache_bytes=None, eligible_reclaimable_logical_bytes=None,
            quota_headroom_bytes=None, projection_only=True,
            result_other_remaining_bytes=ledger.remaining(exclude_job=job_root), result_volume=ledger.fresh_volume())
    if cache_provider is not None:
        try:
            from darsia_cache.worker_adapter import WorkerCacheAdapter
            if not isinstance(cache_provider, WorkerCacheAdapter):
                raise RunUnavailable("Only the actual startup-admitted worker cache adapter may project cache")
            cache = cache_provider.project_expected_space(stores, series_id, tasks, estimates, ledger,
                                                          job_root=job_root)
        except (OSError, ValueError, KeyError, TypeError) as error:
            cache["state"] = "REFUSED_CACHE_PROFILE_OR_EXISTING_LEDGER"
            cache["reason"] = str(error)
    if cache["complete"]:
        other, volume = cache["result_other_remaining_bytes"], cache["result_volume"]
    else:
        other = ledger.remaining(exclude_job=job_root)
        volume = ledger.fresh_volume()
    products = sum(row.products_estimated_bytes for row in estimates)
    transient = sum(row.transient_estimated_bytes for row in estimates)
    needed = products + transient + ledger.reserve
    combined = volume["free_bytes"] >= needed and cache["complete"] and cache["sufficient"]
    cache_message = cache["state"] + (": " + cache["reason"] if "reason" in cache else
        f"; required total {cache['required_cache_bytes']} bytes, quota headroom {cache['quota_headroom_bytes']} bytes")
    lean_models = list(dict.fromkeys(row.model for row in estimates if isinstance(row, StorageProfileEstimate)))
    profile_note = "; lean runs: " + "; ".join(lean_models) if lean_models else ""
    return dict(format="darsia-whole-job-expected-space-v1", machine_id=stores.machine_id,
        results_root=str(root), results_protocol=admission.protocol, results_local=admission.local,
        runs=[asdict(row) for row in estimates], products_expected_bytes=products,
        transient_estimated_bytes=transient, other_active_remaining_bytes=other,
        other_active_remaining_included_in_results_estimate=False,
        results_space_policy="estimate, not coordinated across hosts",
        reserve_bytes=ledger.reserve, needed_bytes=needed, free_bytes=volume["free_bytes"],
        results_space_sufficient=volume["free_bytes"] >= needed,
        sufficient=combined, volume=volume, required_product_bytes=products,
        required_cache_bytes=cache["required_cache_bytes"],
        eligible_reclaimable_logical_bytes=cache["eligible_reclaimable_logical_bytes"],
        quota_headroom_bytes=cache["quota_headroom_bytes"], cache=cache,
        cache_policy_state=cache["state"], complete_admission_sufficient=combined,
        scope="read-only result/cache projection for explicitly admitted service; no reservation or claim; preserved products not subtracted; fresh checks repeat before actual claim",
        remote_scope=getattr(ledger,"remote_scope",None),
        message=f"Expected results needed {needed} bytes / free {volume['free_bytes']} bytes for this whole job on {stores.machine_id}; estimate, not coordinated across hosts; cache: {cache_message}" + profile_note)


def report_job_root(job_root, stores, *, cache_provider=None):
    """Explicit UI/CLI read of declared job/task/configuration metadata only."""
    from .job_queue import read_json
    root = Path(job_root)
    job = read_json(root / "job.json")
    if job.get("format") != "darsia-alias-job-v1":
        raise RunUnavailable("Expected-space report is for explicit alias jobs only")
    runs = job["expected_runs"]
    if not isinstance(runs, list) or not runs or len(runs) > 4096 or len(set(runs)) != len(runs):
        raise ValueError("Bounded distinct original whole-job runs required")
    tasks = [read_json(root / "results" / name(run) / "task.json") for run in runs]
    if any(task["run"] != run for task, run in zip(tasks, runs)):
        raise ValueError("Original run task differs from job order")
    configs = [configuration_for_task(stores, job["series_id"], task) for task in tasks]
    return whole_job_report(stores, job["series_id"], tasks, configs,
        repository=Path(__file__).resolve().parents[4], job_root=root, cache_provider=cache_provider)


def prepared_reports(stores_list, series_id, tasks, configs, job_root, *, cache_providers=None):
    """Optional per-machine projection returned to caller, never job DATA."""
    rows = []
    seen = set()
    for stores in stores_list:
        if stores.machine_id in seen:
            raise ValueError("Supply each machine's capacity map once")
        seen.add(stores.machine_id)
        try:
            row = whole_job_report(stores, series_id, tasks, configs,
                repository=Path(__file__).resolve().parents[4], job_root=job_root,
                cache_provider=None if cache_providers is None else cache_providers.get(stores.machine_id))
        except (OSError, ValueError) as error:
            row = dict(machine_id=stores.machine_id, available=False,
                       needed_bytes=None, free_bytes=None, reason=str(error))
        rows.append(row)
    return rows
