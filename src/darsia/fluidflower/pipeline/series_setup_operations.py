"""Original corrected Apply and RAW Build under admitted operation writers.

The existing worker owns claims, capacity, publication and retirement. This
module neither enqueues tasks nor adopts geometry or qualifies training data.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
from time import monotonic


def _read(bound):
    """Use the original bounded, version-checked reader for exact JSON controls."""
    from .capacity import read_bound
    return read_bound(bound)[0]


@contextmanager
def _native_parameters(bound):
    """Bind the original parameter JSON before loading numerical entry points."""
    from ..native.config import configure_parameters, reset_parameters
    document = _read(bound)
    values = dict(DARSIA_NATIVE_CONFIGURATION=str(Path(bound["path"]).resolve()),
                  DARSIA_NATIVE_CONFIGURATION_SHA256=bound["sha256"])
    previous = {key: os.environ.get(key) for key in values}
    token = configure_parameters(document)
    os.environ.update(values)
    try:
        yield
    finally:
        reset_parameters(token)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _sink():
    """Require the actual active bounded metadata writer and live reservation."""
    from .series_setup_sink import SetupMetadataSink, current_sink
    sink = current_sink()
    if not isinstance(sink, SetupMetadataSink):
        raise ValueError("Setup execution requires its active admitted metadata sink")
    return sink


def _write(path, document):
    """Keep immutable control bytes under the active whole-operation ceiling."""
    from ..rig_raw_build import _descriptor
    sink, path = _sink(), Path(path)
    raw = (json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    sink.admission.require_write(path)
    if path.exists():
        if path.stat().st_size != len(raw):
            raise ValueError("Existing Setup control differs; all artifacts were retained")
        with path.open("rb") as stream:
            existing = stream.read(len(raw) + 1)
        if existing != raw:
            raise ValueError("Existing Setup control differs; all artifacts were retained")
    else:
        sink.mkdir(path.parent, parents=True, exist_ok=True)
        sink.write_bytes(path, raw)
    return _descriptor(path)


class _OperationProgress:
    """Keep repeated original check notifications distinct within one operation."""

    def __init__(self, callback):
        self.callback, self.sequence = callback, 0
        self.last_notification = self.last_checkpoint = None

    def __call__(self, event):
        nested = "message" in event
        if nested:
            now = monotonic()
            if self.last_notification is not None and now - self.last_notification < 10.0:
                checkpoint = getattr(self.callback, "checkpoint", None)
                if (checkpoint is not None and (self.last_checkpoint is None
                        or now - self.last_checkpoint >= 1.0)):
                    checkpoint()
                    self.last_checkpoint = monotonic()
                return
            self.sequence += 1
            event = dict(event, check_progress_sequence=self.sequence)
        result = self.callback(event)
        self.last_checkpoint = monotonic()
        if nested:
            # Count real work time between samples, excluding the metadata
            # callback itself. Stage transitions still publish immediately.
            self.last_notification = self.last_checkpoint
        return result


def _operation_progress(progress):
    """Reuse one nested-check sequence through this operation's original helpers."""
    return (progress if not progress or isinstance(progress, _OperationProgress)
            else _OperationProgress(progress))


@contextmanager
def _reuse_source_reads():
    from ..setup_source_reads import reuse_setup_source_reads
    with reuse_setup_source_reads():
        yield


@contextmanager
def _source_read_group():
    from ..setup_source_reads import setup_source_read_group
    with setup_source_read_group():
        yield


@contextmanager
def _check_progress(progress, event, *, entered=True):
    """Forward original entered-check stages through the worker's actual callback."""
    if not progress:
        yield
        return
    from ..adoption_check_progress import observe_check_progress
    if entered:
        progress(dict(event))
    with observe_check_progress(lambda message: progress(dict(event, message=message)), strict=True):
        yield


@_source_read_group()
def _geometry_sources(project, current, runs, progress=None, *, operation="apply"):
    """Retain sources returned by original saved-geometry readers, without guessing."""
    from .. import corrected_setup as original
    from ..adoption_check_progress import report_check_progress
    progress = _operation_progress(progress)
    config, result = current["configuration"], []
    for index, run in enumerate(runs, start=1):
        event = dict(state="setup_progress", operation=operation, run=run,
                     step="verify_saved_geometry", run_index=index, run_count=len(runs))
        with _check_progress(progress, event):
            registered = original._registered_binding(project.root, current, config, run)
            if registered is None:
                original._require_unedited_adoption(current, config)
                fields = original.selected_fields(config, run)
                row = config["frame_binding"]["runs"][run]
                guarded = [fields[key] for key in ("shape_corrected_baseline", "labels", "facies", "depth")]
                guarded += [row["configured_depth_map"], original._revision_properties(config, row, None)]
            else:
                guarded = [*registered["guarded_sources"], *registered["files"].values()]
            for bound in guarded:
                report_check_progress("Verifying saved geometry source…")
                original._verified(bound)
            result.append(dict(run_id=run, registered=deepcopy(registered), guarded_sources=deepcopy(guarded)))
    return result


@_reuse_source_reads()
@_source_read_group()
def _reverify(records, progress=None, *, operation="apply"):
    """Recheck the exact retained scientific source descriptors at completion."""
    from .. import corrected_setup as original
    from ..adoption_check_progress import report_check_progress
    progress = _operation_progress(progress)
    for index, record in enumerate(records, start=1):
        event = dict(state="setup_progress", operation=operation, run=record["run_id"],
                     step="reverify_saved_geometry", run_index=index, run_count=len(records))
        with _check_progress(progress, event):
            for bound in record["guarded_sources"]:
                report_check_progress("Verifying saved geometry source…")
                original._verified(bound)


@_reuse_source_reads()
def run_apply(config, plan, revision, project, progress=None):
    """Apply the reference first, then capture requests only after all postimages."""
    from .. import corrected_setup as original
    from ..corrected_setup_all_runs import apply_order
    from ..rig_raw_build import _descriptor, capture_configuration_sources, read_build_request
    from ..series_geometry import SeriesRunInputs
    from .series_setup_protocol import APPLY_COMPLETE_FORMAT
    from .run_io import current_binding
    from .series_setup_run_io import original_apply_run, original_configuration_sources
    progress = _operation_progress(progress)
    _sink()
    frame = revision["configuration"]["frame_binding"]
    if frame["display_reference_run"] != plan["display_reference_run"]:
        raise ValueError("The Setup display reference differs from its saved geometry")
    from ..setup_completed_apply import read_completed_apply
    retained = read_completed_apply(config, plan, revision)
    if retained is not None:
        if progress:
            progress(dict(state="setup_progress", operation="apply", run=plan["display_reference_run"],
                          step="reuse_completed_apply"))
        completed = {row["run_id"]: row for row in retained["runs"]}
        children = _prepare_build_children(config, plan, completed, retained["bound"], progress)
        return dict(coordinator_completion=retained["bound"], children=children,
                    guarded_geometry=retained["guarded_geometry"], Build_run=False,
                    optical_calibration_run=False)
    completed, geometry = {}, {}
    for run in apply_order(plan["runs"], plan["display_reference_run"]):
        event = dict(state="setup_progress", operation="apply", run=run, step="apply")
        with _check_progress(progress, event):
            reference = SeriesRunInputs(run, Path(frame["runs"][run]["configuration"]["path"]))
            settings = next(build for build in plan["build_plans"] if build["run_id"] == run).get("raw_setup_settings")
            if current_binding() is not None:
                exported = original_apply_run(project, revision, reference, raw_setup_settings=settings)
            else:
                exported = original.export_corrected_setup(project, reference=reference)
                options = {} if settings is None else dict(raw_setup_settings=settings)
                original.apply_corrected_selected_run(project, exported, rebuild_facies_by_name=True, **options)
            applied = original._apply_completion(exported.destination, exported.receipt)
            if applied is None:
                raise ValueError("The original Apply did not produce a complete per-run record")
            # Retain the inputs the original selected-run transaction proved.
            binding = exported.receipt["binding"]
            registered = binding.get("registered_geometry")
            sources = [*binding["sources"].values(), binding["properties"],
                       binding["configured_depth_source"]]
            if registered is not None:
                sources += [*registered["guarded_sources"], *registered["files"].values()]
            guarded_sources = []
            for bound in sources:
                if bound not in guarded_sources:
                    guarded_sources.append(bound)
            geometry[run] = dict(run_id=run, registered=deepcopy(registered),
                                 guarded_sources=deepcopy(guarded_sources))
            completed[run] = dict(run_id=run, export=_descriptor(exported.destination / "export.json"),
                apply_completion=_descriptor(exported.destination / "apply-complete.json"))
    # The common TOML can be changed by the reference. No future request is
    # captured until every original selected-run transaction has completed.
    for index, build in enumerate(plan["build_plans"], start=1):
        event = dict(state="setup_progress", operation="apply", run=build["run_id"],
                     step="capture_configuration_sources", run_index=index,
                     run_count=len(plan["build_plans"]))
        with _check_progress(progress, event):
            paths = [Path(path) for path in build["configuration_paths"]]
            if current_binding() is not None:
                captured = original_configuration_sources(project, revision, paths)
            else:
                original.require_corrected_export_for_build(paths)
                captured = list(capture_configuration_sources(paths))
            completed[build["run_id"]]["configuration_files"] = captured
    guarded = [geometry[run] for run in plan["runs"]]
    _reverify(guarded, progress=progress)
    coordinator = dict(format=APPLY_COMPLETE_FORMAT, complete=True,
        **{key: deepcopy(config[key]) for key in
           ("setup_plan", "adopted_revision", "project_manifest", "series_manifest")},
        revision_id=revision["revision_id"], current_sha256=config["adopted_revision"]["sha256"],
        runs=[completed[run] for run in plan["runs"]], Build_run=False, optical_calibration_run=False)
    root = Path(plan["control_root"])
    coordinator_bound = _write(root / "apply-complete.json", coordinator)
    children = _prepare_build_children(config, plan, completed, coordinator_bound, progress)
    return dict(coordinator_completion=coordinator_bound, children=children,
                guarded_geometry=guarded, Build_run=False, optical_calibration_run=False)


def _prepare_build_children(config, plan, completed, coordinator_bound, progress=None):
    """Stage the original Build requests after genuine Apply, including retry."""
    children = []
    for index, build in enumerate(plan["build_plans"], 1):
        event = dict(state="setup_progress", operation="apply", run=build["run_id"],
                     step="prepare_build_requests", run_index=index, run_count=len(plan["build_plans"]))
        with _check_progress(progress, event):
            children.append(_prepare_build_child(config, plan, build, completed, coordinator_bound))
    return children


def _prepare_build_child(config, plan, build, completed, coordinator_bound):
    from ..rig_raw_build import read_build_request
    from .series_setup_protocol import validate_setup_operation
    root, run = Path(plan["control_root"]), build["run_id"]
    request = dict(deepcopy(build["request_template"]),
                   configuration_files=deepcopy(completed[run]["configuration_files"]))
    request_bound = _write(root / "build" / run / "raw-build-request.json", request)
    read_build_request(request_bound, artifact_kind="raw_rig_baseline")
    operation = dict(format=config["format"], run=run, operation="build",
        **{key: deepcopy(config[key]) for key in
           ("project_root", "project_manifest", "adopted_revision", "series_manifest", "setup_plan")},
        **{key: deepcopy(build[key]) for key in
           ("native_parameters", "target_machine", "metadata_output", "output_scope")},
        configuration_files=deepcopy(completed[run]["configuration_files"]),
        raw_build_request=request_bound, apply_completion=completed[run]["apply_completion"],
        coordinator_completion=coordinator_bound)
    validate_setup_operation(operation)
    operation_bound = _write(root / "build" / run / "operation.json", operation)
    return dict(run_id=run, configuration=operation_bound,
                raw_build_request=request_bound, target_machine=build["target_machine"])


@_reuse_source_reads()
def run_build(config, plan, revision, project, raw_request, progress=None):
    """Call the original full RAW setup with its exact post-Apply request."""
    from .. import corrected_setup as original
    from ..rig_raw_build import CERTIFICATE_NAME, _descriptor, read_build_request
    from .run_io import current_binding
    from .series_setup_run_io import original_geometry_sources, original_build
    progress = _operation_progress(progress)
    _sink()
    guarded = (original_geometry_sources(project, revision, [config["run"]],
                                         progress=progress, operation="build")
               if current_binding() is not None else _geometry_sources(
                   project, revision, [config["run"]], progress=progress, operation="build"))
    request, _pins = read_build_request(config["raw_build_request"], artifact_kind="raw_rig_baseline")
    if request != raw_request:
        raise ValueError("The original RAW request differs from its admitted controls")
    paths = [Path(row["path"]) for row in config["configuration_files"]]
    event = dict(state="setup_progress", operation="build", run=config["run"], step="build")
    with _check_progress(progress, event):
        if current_binding() is not None:
            original_build(project, revision, paths, config["raw_build_request"])
        else:
            original.require_corrected_export_for_build(paths)
            from ..setup_rig import setup_rig
            from darsia.presets.workflows.rig import Rig
            setup_rig(Rig, paths, show=False, raw_build_request=config["raw_build_request"])
    certificate = _descriptor(Path(request["output"]["destination"]) / CERTIFICATE_NAME)
    evidence = _read(certificate)
    _reverify(guarded, progress=progress, operation="build")
    read_build_request(config["raw_build_request"], artifact_kind="raw_rig_baseline")
    return dict(guarded_geometry=guarded, raw_build_request=deepcopy(config["raw_build_request"]),
        configuration_files=deepcopy(config["configuration_files"]), baseline_renderer=certificate,
        produced_files=deepcopy(evidence["outputs"]), certificate_scope=evidence["certificate_scope"],
        complete_rig_model=evidence["complete_rig_model"],
        production_qualification_claimed=evidence["production_qualification_claimed"],
        training_context_eligibility_claimed=False)


def execute_setup(task, output, backend, precision, progress=None, *, stage_workers=1):
    """Execute one original operation under the worker's exact RunIO and sink."""
    from .run_io import for_task, task_without_resume
    from .series_setup_inputs import _same
    from .alias_job import _manifest
    from .series_setup_protocol import validate_setup_operation, validate_setup_plan
    binding = for_task(task)
    if binding is None:
        raise ValueError("Setup requires its exact active per-run I/O binding")
    binding.require_output(output)
    sink = _sink()
    sink.admission.require_write(Path(output))
    config = validate_setup_operation(_read(binding.configuration_descriptor), task=task)
    if (not _same(task_without_resume(sink.admission.task), task_without_resume(task))
            or not _same(sink.admission.configuration, config)):
        raise ValueError("Setup sink and RunIO belong to different operation admissions")
    if backend != "cpu" or precision != task["precision"]:
        raise ValueError("Setup must use its explicit CPU worker and task precision")
    controls = binding.setup_controls
    if controls["scope"] != "series-setup-controls-only":
        raise ValueError("Setup requires its original named control catalogue")
    plan = validate_setup_plan(_read(config["setup_plan"]))
    revision = _read(config["adopted_revision"])
    if not _same(plan, controls["plan"]) or not _same(revision, controls["revision"]):
        raise ValueError("Setup controls changed after admitted replay")
    for key in ("project_manifest", "native_parameters"):
        _read(config[key])
    _manifest(config["series_manifest"])
    with _native_parameters(config["native_parameters"]):
        from ..project import FluidFlowerProject
        project = FluidFlowerProject.load(Path(config["project_root"]))
        if config["operation"] == "apply":
            operation = run_apply(config, plan, revision, project, progress)
        elif config["operation"] == "image-preparation":
            from .setup_image_preparation import run_image_preparation
            from .series_setup_run_io import _original_scope
            with _original_scope(project, revision):
                actual = dict(config, output_root=str(Path(config["output_root"]) / task["attempt_id"]))
                prepared = run_image_preparation(actual, workers=stage_workers, progress=progress)
            operation = dict(image_preparation=prepared, phase_inference_performed=False,
                             model_training_performed=False, optical_calibration_run=False)
        else:
            operation = run_build(config, plan, revision, project, controls["raw_request"], progress)
        for key in ("project_manifest", "adopted_revision", "native_parameters"):
            _read(config[key])
        _manifest(config["series_manifest"])
        result = dict(format="darsia-series-setup-result-v1", complete=True, pipeline="series-setup",
            run=task["run"], operation=config["operation"], backend=backend, precision=precision,
            setup_plan=deepcopy(config["setup_plan"]), worker_references_closed=True, **operation)
        receipt = _write(Path(output) / "operation_result.json", result)
    return dict(result, files={"operation_result.json": receipt})
