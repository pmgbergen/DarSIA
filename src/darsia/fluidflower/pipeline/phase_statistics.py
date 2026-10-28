"""Worker and ordinary local phase-one entries sharing original producers.

The existing worker dispatch supplies its genuine active alias RunIOBinding.
An optional current Build declaration retains its original readers and refuses
unavailable physical semantics before N1 preparation. Resume remains unsupported.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import socket
import time
import re
import uuid

from . import series
from .job_queue import descriptor, write_immutable
from .staged_data import PHASE1_FORMAT, FRAME_FORMAT, COMPLETION_FORMAT, BEFORE_FORMAT, validate_phase1_run

PIPELINE = "native-phase-statistics"
MANIFEST_FORMAT = "darsia-phase-statistics-run-manifest-v1"


def _task_configuration(task, output, backend, precision):
    """Require the actual existing worker's positive owned per-run I/O context."""
    from ..mass.n1_pipeline import DataReader
    from .run_io import for_task
    if (task.get("pipeline") != PIPELINE or task.get("format") != "darsia-alias-task-v1"
            or backend not in {"cpu", "cuda"} or precision != task.get("precision")):
        raise ValueError("Phase one requires its explicit pipeline and actual alias backend/precision")
    if task.get("resume_from") is not None:
        raise ValueError("Phase-one resume is not admitted; never replay an existing prefix")
    from .phase1_protocol import validate_phase1_task
    validate_phase1_task(task)
    binding = for_task(task)
    if binding is None:
        raise ValueError("Phase one requires a positive original owned RunIOBinding")
    binding.require_output(output)
    configuration = deepcopy(binding.configuration_descriptor)
    config = validate_phase1_run(DataReader(unc_only=True).json(configuration))
    if config["run"] != task["run"] or config["precision"] != precision:
        raise ValueError("Phase-one task/configuration run or precision differs")
    native_code_admission = deepcopy(binding.code_audit)
    if not native_code_admission or any(row.get("commit") != task["commit"] for row in native_code_admission):
        raise ValueError("Phase one lacks its actual original RunIO native-code admission")
    return config, configuration, native_code_admission


def _save_before(stage, spec, capture, shape):
    """Serialize the unchanged Q28 LA evidence before summary/mass work."""
    from PIL import Image
    from ..storage_outputs import encode_before
    from ..mass.n1_pipeline import _write_json, asset
    la, evidence = encode_before(capture, tuple(shape))
    document = dict(format=BEFORE_FORMAT, complete=True,
                    source=deepcopy(spec["source"]), before_rules=evidence)
    result = dict(state=evidence["state"])
    if la is not None:
        from .io_ops import publish_file
        path = Path(stage) / "storage_gas_probability_before_rules.png"
        publish_file(path, lambda stream: Image.fromarray(la, mode="LA").save(stream, format="PNG"))
        document["png"] = asset(path)
        result["png"] = document["png"]
    result["receipt"] = _write_json(Path(stage) / "phase_before.json", document)
    return result


def _save_original_photo(stage, task, spec, request, reader):
    """Use only an explicitly bound original pre-light photo producer or pending."""
    from ..mass.n1_pipeline import _write_json
    if request is None:
        receipt = _write_json(Path(stage) / "phase_original_photo_pending.json",
            dict(format="darsia-phase-original-photograph-pending-v1", state="pending",
                 reason="explicit_saved_pre_light_photograph_not_bound", source=deepcopy(spec["source"])))
        return dict(state="pending", receipt=receipt)
    # This separately qualified optional product must be present in the final
    # source closure before its explicit request can execute. It does not decode
    # camera RAW or run registration; only the bound saved geometry-only JPEG,
    # or the bound lossless original of a RAW photographic review.
    from ..storage_original_photo import publish_raw_original_png, publish_registered_original_png
    publish = publish_raw_original_png if "raw_review" in request else publish_registered_original_png
    result = publish(Path(stage) / "registered_original.png",
        task=task, frame_spec=spec, resolve_asset=reader.path, **request)
    return dict(state="complete", receipt=result["receipt"], png=result["photograph"])


def _save_statistics(stage, extracted):
    """Original N1 gzip/JSON serialization, retaining all summary values."""
    from darsia._opencv_runtime import require_ipp_off, validate_observation
    require_ipp_off("phase1-statistics:before-publication")
    if "opencv_ipp" in extracted:
        validate_observation(extracted["opencv_ipp"])
    from ..mass.n1_pipeline import _jsonable, asset, _write_json
    from .io_ops import publish_file
    path = Path(stage) / "native_statistics.json.gz"
    # Exact old JSON conversion/separators. The N111-retried publisher writes a
    # temporary first, so the gzip header names the final file, as before.
    data = json.dumps(_jsonable(extracted["statistics"]), allow_nan=False,
                      separators=(",", ":")).encode("utf-8")

    def write(raw):
        with gzip.GzipFile(filename=str(path), fileobj=raw, mode="wb", compresslevel=3) as compressed:
            compressed.write(data)
    publish_file(path, write)
    _write_json(Path(stage) / "statistics_geometry_report.json", extracted["sandmap"])
    bound = asset(path)
    if "opencv_ipp" in extracted:
        facts = ({"pixel_backend_facts": extracted["pixel_backend_facts"]}
                 if "pixel_backend_facts" in extracted else {})
        _write_json(Path(stage) / "native_statistics_runtime.json", dict(
            format="darsia-native-statistics-runtime-v1", statistics=bound,
            opencv_ipp=extracted["opencv_ipp"], **facts))
    return bound


def _commit_phase1(stage, spec, *, task, configuration, native, phase, statistics, before, photograph, probabilities=None):
    """One new immutable stage commit anchors every closed file before progress."""
    from darsia._opencv_runtime import require_ipp_off
    from .runner import _inventory
    stage = Path(stage)
    commit = stage / "phase1_commit.json"
    if commit.exists():
        raise FileExistsError("Phase-one frame already committed; no replay is allowed")
    record = dict(format=FRAME_FORMAT, complete=True, run=task["run"],
        case_id=spec["case_id"], frame_index=spec["frame_index"], elapsed_h=spec["elapsed_h"],
        source=deepcopy(spec["source"]), configuration=deepcopy(configuration), precision=task["precision"],
        native=native["record"], native_receipt=native["record_asset"], phase=phase,
        statistics=statistics, before_rules=before, photograph=photograph, files=_inventory(stage))
    if probabilities is not None:
        record["probabilities"] = probabilities
    # Reuse the already-qualified strict metadata/ownership/role validator.
    from .staged_data import _frame_record
    _frame_record(record, spec, configuration, task["precision"], dict(path=str(commit)))
    record["opencv_ipp"] = phase.get("opencv_ipp", require_ipp_off("phase1:commit"))
    write_immutable(commit, record)
    return descriptor(commit)


def run_phase1(task, output, backend, precision, on_progress=None):
    """Existing worker stage entrypoint, not a standalone or generated runner."""
    return _phase1_thread_runtime(task.get("threads", 1),
        lambda runtime: _run_phase1(task, output, backend, precision, on_progress, runtime))


def _phase1_thread_runtime(threads, run):
    """Keep the original OpenCV/BLAS thread lifetime for either caller."""
    from darsia._opencv_runtime import scientific_thread
    ipp_start = scientific_thread("run_phase1:start")
    import cv2
    import scipy.linalg
    from threadpoolctl import threadpool_info, threadpool_limits
    if type(threads) is not int or threads < 1:
        raise ValueError("Positive internal thread count required")
    old_threads = cv2.getNumThreads()
    try:
        cv2.setNumThreads(threads)
        with threadpool_limits(limits=threads):
            runtime = dict(opencv_ipp=ipp_start, requested_threads=threads, cv2_threads=cv2.getNumThreads(),
                           threadpools=threadpool_info())
            return run(runtime)
    finally:
        cv2.setNumThreads(old_threads)


def _run_phase1(task, output, backend, precision, on_progress, runtime):
    from ..mass.n1_pipeline import _write_json
    config, configuration, native_code_admission = _task_configuration(task, output, backend, precision)
    current_build = None
    if "current_build_binding" in config:
        from .current_build import acquire_current_build
        from .run_io import for_task
        current_build = acquire_current_build(config, task, for_task(task))
    probability_request = None
    if "gas_probability_storage" in config:
        from ..phase import probability_storage
        probability_request = probability_storage.parse_request(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Each phase-one owned attempt must be fresh")
    clock = time.perf_counter()
    settings = dict(host=socket.gethostname(), backend=backend, precision=precision,
                    threads=task["threads"], thread_runtime=runtime,
                    phase_precision="explicit-historical-stage-dtypes")
    manifest = dict(format=MANIFEST_FORMAT, job_id=task["job_id"], run=task["run"],
        configuration=configuration,
        code=dict(commit=task["commit"], identity_algorithm="sha256-git-commit-ascii"),
        commit_identity_digest=hashlib.sha256(task["commit"].encode("ascii")).hexdigest(),
        native_code_admission=dict(scope="Actual original RunIO native-source roles plus display provenance only; not a repository code-closure digest",
                                   entries=native_code_admission), settings=settings,
        stage_identity=dict(pipeline=PIPELINE, configuration=deepcopy(task["configuration"]),
                            run=task["run"], commit=task["commit"], precision=precision,
                            backend=backend, threads=task["threads"], attempt_id=task["attempt_id"],
                            phase1_configurations_sha256=task["phase1_configurations_sha256"]),
        frames={spec["case_id"]: spec for spec in config["frames"]})
    manifest_asset = _write_json(output / "manifest.json", manifest)
    return _phase1_frames(config, task, output, backend, precision, on_progress,
        configuration, manifest_asset, clock, probability_request,
        current_build=current_build)


def run_phase1_local(config, output, *, local_build, code_commit,
                     precision="float32", threads=1, on_progress=None, prepared_native=None):
    """Run ordinary local P1 with explicit Build inputs and the original loop.

    ``code_commit`` is source attribution, not a qualification or clean-tree
    requirement. Local descriptors retain the original content checks. This
    entry does not acquire an alias task, a provider, or a RunIOBinding.
    """
    from .local_build import LocalBuildInputs
    if type(local_build) is not LocalBuildInputs:
        raise TypeError("Ordinary local phase one requires explicit LocalBuildInputs")
    if not isinstance(code_commit, str) or re.fullmatch(r"[0-9a-f]{40}", code_commit) is None:
        raise ValueError("Actual local phase-one code commit attribution is required")
    if type(threads) is not int or threads < 1:
        raise ValueError("Positive internal thread count required")
    config = deepcopy(config)
    # The explicit local Build object supplies this route's physical inputs;
    # the old alias-task declaration is not a second local admission mechanism.
    config.pop("current_build_binding", None)
    config = local_build.bind_configuration(config)
    validate_phase1_run(config)
    if precision != config["precision"]:
        raise ValueError("Local phase-one configuration and precision differ")
    return _phase1_thread_runtime(threads,
        lambda runtime: _run_phase1_local(config, output, local_build, code_commit,
                                          precision, threads, on_progress, runtime, prepared_native))


def _run_phase1_local(config, output, local_build, code_commit,
                      precision, threads, on_progress, runtime, prepared_native=None, optical_recipe=None):
    from ..mass.n1_pipeline import _write_json
    probability_request = None
    if "gas_probability_storage" in config:
        from ..phase import probability_storage
        probability_request = probability_storage.parse_request(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Each local phase-one output directory must be fresh")
    clock = time.perf_counter()
    config_path = output / ("phase1_configuration.json" if optical_recipe is None else "optical_configuration.json")
    write_immutable(config_path, config)
    configuration = descriptor(config_path)
    task = dict(run=config["run"], precision=precision, threads=threads,
                commit=code_commit, execution_id=uuid.uuid4().hex)
    settings = dict(host=socket.gethostname(), backend="cpu", precision=precision,
                    threads=threads, thread_runtime=runtime,
                    phase_precision="explicit-historical-stage-dtypes")
    manifest = dict(format=MANIFEST_FORMAT, execution_mode="local",
        execution_id=task["execution_id"], run=config["run"], configuration=configuration,
        code=dict(commit=code_commit, identity_algorithm="sha256-git-commit-ascii"),
        commit_identity_digest=hashlib.sha256(code_commit.encode("ascii")).hexdigest(),
        settings=settings, frames={spec["case_id"]: spec for spec in config["frames"]})
    manifest_asset = _write_json(output / ("manifest.json" if optical_recipe is None else "optical_manifest.json"), manifest)
    return _phase1_frames(config, task, output, "cpu", precision, on_progress,
        configuration, manifest_asset, clock, probability_request,
        local_build=local_build, unc_only=False,
        **({} if prepared_native is None else {"prepared_native": prepared_native}),
        **({} if optical_recipe is None else {"optical_recipe": optical_recipe}))


def _phase1_frames(config, task, output, backend, precision, on_progress,
                   configuration, manifest_asset, clock, probability_request,
                   *, current_build=None, local_build=None, unc_only=True, prepared_native=None,
                   optical_recipe=None):
    """The single original frame loop and serializers, for worker and local P1."""
    from darsia._opencv_runtime import scientific_thread, completed_frame, require_ipp_off
    from ..mass.n1_pipeline import N1Pipeline
    from ..mass.production_precision import PixelMath
    from ..phase.infer import infer_labels
    from .runner import _phase_outputs
    if probability_request is not None:
        from ..phase import probability_storage
    if local_build is None and current_build is not None:
        local_build = current_build.physical_inputs()
    build_inputs = current_build if local_build is None else local_build
    if local_build is not None:
        config = local_build.bind_configuration(config)
    if backend == "cpu":
        math_provider = PixelMath(precision)
    else:
        from ..mass.mass_cuda import CudaHostPixelMath
        math_provider = CudaHostPixelMath(precision, device_id=int(task["device_id"]),
                                         pool_limit_bytes=int(task["pool_limit_bytes"]))
    inputs, statistics_context = None, None
    records = []
    try:
        inputs = (series.SeriesInputs(config, output) if unc_only
                  else series.SeriesInputs(config, output, unc_only=False, **(
                      {} if prepared_native is None else {"prepared_native": prepared_native})))
        statistics_config = dict(config["statistics"], run=config["run"],
                                 frames=config["frames"], data_aliases=config.get("data_aliases", ()))
        current_build_options = {} if current_build is None else {"current_build": current_build}
        if local_build is not None:
            current_build_options = {"local_build": local_build}
        statistics_context = N1Pipeline(statistics_config, precision=precision,
            pixel_math=math_provider, unc_only=unc_only, statistics_only=optical_recipe is None,
            frame_loading_only=optical_recipe is not None, **current_build_options)
        for spec in config["frames"]:
            ipp_before = scientific_thread("run_phase1:" + spec["case_id"] + ":before")
            if on_progress:
                on_progress(dict(state="frame_started", stage=PIPELINE, run=task["run"],
                    case_id=spec["case_id"], frame_index=spec["frame_index"]))
            start = time.perf_counter()
            stage = output / "preparation" / spec["case_id"]
            native = inputs.native(spec, stage)
            frame = inputs.frame(spec, native)
            if build_inputs is not None:
                frame = build_inputs.check_phase(spec, native, frame)
            phase_capture = {}
            if probability_request is not None:
                phase_capture["gas_probability_storage"] = {}
            inferred = infer_labels(frame, inputs.previous, inputs.model_bundle,
                                    inputs.parameters, output_capture=phase_capture)
            if build_inputs is not None:
                build_inputs.check_inferred(spec, frame, inferred)
            phase = _phase_outputs(stage, spec, frame, inferred)
            before = _save_before(stage, spec, phase_capture, native["record"]["full"]["phase_shape_hw"])
            probabilities = None
            if probability_request is not None:
                probabilities = probability_storage.write_product(stage, spec,
                    phase_capture["gas_probability_storage"], model_bindings=inputs.phase["model_bundles"],
                    phase_configuration=config["phase"]["configuration"], labels=inferred["labels"],
                    label_descriptor=phase["labels"])
            del phase_capture
            photograph = _save_original_photo(stage, task, spec,
                config.get("photographs", {}).get(spec["case_id"]), inputs.reader)
            statistics_spec = series.bind_mass_frame(spec, native["record"], phase,
                                                     baseline=inputs.mass_baseline)
            if optical_recipe is None:
                extracted = statistics_context.extract_statistics(statistics_spec)
                extracted["opencv_ipp"] = statistics_context.last_opencv_ipp
                if "opencl_acceleration" in statistics_config:
                    extracted["pixel_backend_facts"] = math_provider.facts()
                statistics = _save_statistics(stage, extracted)
                del extracted
                phase["opencv_ipp"] = completed_frame(ipp_before, "run_phase1:" + spec["case_id"] + ":after")
                committed = _commit_phase1(stage, spec, task=task, configuration=configuration,
                    native=native, phase=phase, statistics=statistics, before=before, photograph=photograph,
                    **({} if probabilities is None else dict(probabilities=probabilities)))
            else:
                from .current_optical_preparation import sample_local_frame, save_samples
                from ..mass.n1_pipeline import _write_json
                phase["opencv_ipp"] = completed_frame(ipp_before, "run_phase1:" + spec["case_id"] + ":after")
                selected = {row["case_id"] for row in optical_recipe.document["runs"].get(config["run"], {}).get("frames", ())}
                sample = (save_samples(stage, sample_local_frame(statistics_context, statistics_spec, optical_recipe))
                          if spec["case_id"] in selected else None)
                committed = _write_json(stage / "optical_phase_source.json", dict(
                    format="darsia-current-optical-phase-frame-v1", run=config["run"],
                    case_id=spec["case_id"], source=spec["source"], configuration=configuration,
                    native=native["record"], native_receipt=native["record_asset"], phase=phase,
                    before_rules=before, photograph=photograph, probabilities=probabilities,
                    samples=sample, phase_complete=True, statistics_complete=False))
            records.append(committed)
            # Acquisition history and actual progress follow the immutable
            # stage commit. A lost lease/control callback prevents the next frame.
            inputs.advance(frame)
            del frame, inferred, native, phase, before, photograph
            if on_progress:
                on_progress(dict(state="phase1_frame_completed" if optical_recipe is None else "optical_phase_frame_completed",
                    complete=optical_recipe is None, stage=PIPELINE,
                    run=task["run"], case_id=spec["case_id"], frame_index=spec["frame_index"],
                    compute_seconds=time.perf_counter() - start, phase1_commit=committed))
    finally:
        statistics_context = None
        if inputs is not None:
            inputs.close()
        close = getattr(math_provider, "close", None)
        if callable(close):
            close()
    if optical_recipe is not None:
        return dict(format="darsia-current-optical-phase-run-v1", complete=True,
            phase_sources_complete=True, statistics_complete=False, run=config["run"],
            configuration=configuration, frames=records, ordered_case_ids=config["ordered_case_ids"],
            recipe_sha256=optical_recipe.sha256, code_commit=task["commit"])
    return _complete_phase1_run(config, task, output, records, configuration,
                                manifest_asset, backend, precision, clock, unc_only)


def _complete_phase1_run(config, task, output, records, configuration, manifest_asset,
                         backend, precision, clock, unc_only):
    from darsia._opencv_runtime import require_ipp_off
    completion = dict(format=COMPLETION_FORMAT, complete=True, run=config["run"],
        configuration=configuration, precision=precision, code_commit=task["commit"],
        ordered_case_ids=config["ordered_case_ids"], frames=records, history_complete=True,
        opencv_ipp=require_ipp_off("run_phase1:completed"))
    core = output / "phase1_completed.json"
    write_immutable(core, completion)
    wall = time.perf_counter() - clock
    result = dict(format="darsia-phase-statistics-worker-result-v1", complete=True, whole_run=True,
        pipeline=PIPELINE, run=task["run"], backend=backend, precision=precision,
        working_pixel_dtype=precision, case_count=len(records), ordered_case_ids=config["ordered_case_ids"],
        phase1_completion=descriptor(core), phase1_frames=records, manifest=manifest_asset,
        configuration=configuration, physical_correction=config["statistics"]["physical_correction"],
        opencv_ipp=require_ipp_off("run_phase1:result"),
        worker_references_closed=True, fitted_mass_calibration_read=False,
        mass_evaluation_calls=0, wall_seconds=wall,
        public_dispatch_resume_and_publication_qualified=False)
    if not unc_only:
        result.pop("worker_references_closed")
        result.pop("public_dispatch_resume_and_publication_qualified")
        result.update(execution_mode="local", execution_id=task["execution_id"])
    return result


def run_optical_phase_sources(config, output, *, local_build, code_commit, optical_recipe,
                              on_progress=None, prepared_native=None):
    """Run inference once while deferring only the not-yet-fitted optical paths."""
    from .local_build import LocalBuildInputs
    if type(local_build) is not LocalBuildInputs or config["statistics"]["shared"]["optics"] is not None:
        raise ValueError("Fresh optical preparation needs its local Build and unresolved optics")
    config = local_build.bind_configuration(deepcopy(config))
    validate_phase1_run(config, require_optics=False)
    return _phase1_thread_runtime(1, lambda runtime: _run_phase1_local(
        config, output, local_build, code_commit, "float32", 1, on_progress, runtime,
        prepared_native, optical_recipe))


def finish_optical_phase_statistics(config, output, sources, *, local_build, code_commit,
                                     on_progress=None, reader=None):
    """Complete P1 from the exact closed phase/native files; never infer or decode RAW."""
    from .local_overflow_sources import SavedLocalBuildInputs
    from ..mass.n1_pipeline import DataReader, N1Pipeline, _write_json
    from ..mass.production_precision import PixelMath
    from darsia._opencv_runtime import scientific_thread
    if not isinstance(local_build, SavedLocalBuildInputs):
        raise TypeError("The existing saved local Build replay is required")
    reader = reader or DataReader()
    original = reader.json(sources["configuration"])
    original["statistics"]["shared"]["optics"] = deepcopy(config["statistics"]["shared"]["optics"])
    if (original != config or sources.get("phase_sources_complete") is not True
            or sources.get("statistics_complete") is not False or sources["code_commit"] != code_commit):
        raise ValueError("Prepared phase sources differ from the current optical statistics configuration")
    validate_phase1_run(config)
    output = Path(output)
    if (output / "phase1_completed.json").exists():
        raise FileExistsError("The prepared run already has complete P1 statistics")
    clock = time.perf_counter()
    configuration = _write_json(output / "phase1_configuration.json", config)
    task = dict(run=config["run"], precision="float32", commit=code_commit, execution_id=uuid.uuid4().hex)
    manifest_asset = _write_json(output / "manifest.json", dict(format=MANIFEST_FORMAT,
        execution_mode="local", execution_id=task["execution_id"], run=config["run"],
        configuration=configuration, code=dict(commit=code_commit, identity_algorithm="sha256-git-commit-ascii"),
        commit_identity_digest=hashlib.sha256(code_commit.encode("ascii")).hexdigest(),
        settings=dict(host=socket.gethostname(), backend="cpu", precision="float32", threads=1),
        frames={row["case_id"]: row for row in config["frames"]}))
    statistics_config = dict(config["statistics"], run=config["run"], frames=config["frames"],
                             data_aliases=config.get("data_aliases", ()))
    def execute(_runtime):
        pipeline = N1Pipeline(statistics_config, precision="float32", pixel_math=PixelMath("float32"),
                              statistics_only=True, local_build=local_build)
        records = []
        if len(sources["frames"]) != len(config["frames"]):
            raise ValueError("Prepared optical frames do not cover the selected run")
        for spec, binding in zip(config["frames"], sources["frames"]):
            record = reader.json(binding)
            if record["case_id"] != spec["case_id"] or record["source"] != spec["source"]:
                raise ValueError("Prepared optical phase frame names another source")
            mass = series.bind_mass_frame(spec, record["native"], record["phase"],
                                         baseline=config["phase"]["reference"]["full"])
            local_build.restore(spec, record, mass)
            scientific_thread("fresh-optics-statistics:" + spec["case_id"])
            extracted = pipeline.extract_statistics(mass)
            extracted["opencv_ipp"] = pipeline.last_opencv_ipp
            stage = output / "preparation" / spec["case_id"]
            statistics = _save_statistics(stage, extracted)
            committed = _commit_phase1(stage, spec, task=task, configuration=configuration,
                native=dict(record=record["native"], record_asset=record["native_receipt"]),
                phase=record["phase"], statistics=statistics, before=record["before_rules"],
                photograph=record["photograph"], probabilities=record["probabilities"])
            records.append(committed)
            if on_progress:
                on_progress(dict(state="phase1_frame_completed", complete=True, stage=PIPELINE,
                    run=config["run"], case_id=spec["case_id"], frame_index=spec["frame_index"], phase1_commit=committed))
        return _complete_phase1_run(config, task, output, records, configuration,
                                   manifest_asset, "cpu", "float32", clock, False)
    return _phase1_thread_runtime(1, execute)
