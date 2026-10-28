"""Phase-map computing on the metric lattice: pipeline ``lattice-phase-n1``.

A lattice run uses the native-phase-n1 run configuration. Its bound storage
request adds ``computing_resolution_mm`` (0.5 or 1 mm), and the request's
extent fixes the lattice. The native runner refuses such a storage request,
so a lattice configuration cannot run natively by mistake; job preparation
gives its task this pipeline id (:func:`lattice_task`).

Per frame, ``series.SeriesInputs`` prepares the native inputs unchanged in a
local transient folder, ``lattice_inputs.LatticeRun`` coarsens them, the
unchanged ``phase.infer.infer_labels`` classifies the lattice,
``lattice_mass.LatticeMass`` evaluates mass with the unchanged N1 code, and
``n1_pipeline._publish_frame`` publishes the frame. The manifest, every frame
and the run result record the computing resolution and that coarse results
are not validated against native computing.

The request stores on the computing lattice, or at 1 or 2 mm over 0.5 or 1 mm
computing. A coarser stored resolution keeps every lattice frame file and adds
the frame's stored products, reduced from its lattice products by exact
blocks (``lattice_storage``), to the frame's stage before the frame commit.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import socket
import tempfile
import time

PIPELINE = "lattice-phase-n1"
RESOLUTION_KEY = "computing_resolution_mm"
FRAME_FORMAT = "darsia-lattice-series-frame-v1"
VALIDATION = dict(state="not_validated",
                  reason="Coarse computing is not yet compared with native computing; "
                         "the acceptance limits follow that comparison.")


def lattice_request(config, precision):
    """Admit a run configuration's lattice request.

    The storage request must carry ``computing_resolution_mm`` and a coarse
    extent. Its other fields pass the unchanged storage request check. The
    stored resolution equals the computing resolution or is an integer
    multiple of it (1 or 2 mm over 0.5 or 1 mm); the extent lies on the stored
    lattice, so both origin-zero lattices nest exactly.
    """
    from ..computing_lattice import computing_resolution_mm
    from ..lattice_storage import block_factor
    from ..storage_outputs import parse_request
    storage = config.get("storage")
    if not isinstance(storage, dict) or RESOLUTION_KEY not in storage:
        raise ValueError("A lattice run needs computing_resolution_mm in its bound storage request")
    computing = computing_resolution_mm(storage[RESOLUTION_KEY])
    plain = {key: value for key, value in storage.items() if key != RESOLUTION_KEY}
    request = parse_request(dict(config, storage=plain), precision)
    stored = request["resolution_mm"]
    if stored == "native" or float(stored) < computing:
        raise ValueError("The stored resolution cannot be finer than the computing resolution")
    block_factor(float(stored), computing)
    return dict(request, **{RESOLUTION_KEY: computing})


def lattice_task(task, config):
    """Give a prepared task the lattice pipeline when its configuration asks for it.

    A configuration without a lattice request returns the task unchanged, so
    native job files stay byte-identical.
    """
    storage = config.get("storage")
    if not isinstance(storage, dict) or RESOLUTION_KEY not in storage:
        return task
    request = lattice_request(config, task["precision"])
    from .series import PIPELINE as NATIVE
    if task.get("pipeline") != NATIVE:
        raise ValueError("Lattice computing extends the native-phase-n1 route only")
    return dict(task, pipeline=PIPELINE, **{RESOLUTION_KEY: request[RESOLUTION_KEY]})


def load_lattice_run(task):
    """Read and admit the task's native-phase-n1 configuration for a lattice run."""
    from ..mass.n1_pipeline import DataReader
    from . import series
    from .run_io import for_task
    if task.get("pipeline") != PIPELINE:
        raise ValueError("This runner requires an explicit lattice-phase-n1 task")
    binding = for_task(task)
    reader = DataReader(unc_only=True)
    configuration = task["configuration"] if binding is None else binding.configuration_descriptor
    config = series.validate_run(reader.json(configuration))
    if config["run"] != task["run"] or config["mass"]["calibration"]["sha256"] != task["calibration_sha256"]:
        raise ValueError("Task scientific run/calibration binding differs")
    return config


def _phase_outputs(stage, spec, lattice_frame, inferred):
    """Write the lattice phase products and the lattice predecessor inputs."""
    import numpy as np
    from PIL import Image
    from ..mass.n1_pipeline import _write_json, _write_npz, asset
    from ..native.sidecar import write_log_linear
    from .io_ops import publish_file
    from .job_queue import _create_directory
    _create_directory(stage, parents=True)
    labels = inferred["labels"]
    if labels.dtype != np.uint8 or labels.shape != lattice_frame["pixel_area_mm2"].shape:
        raise ValueError("Lattice phase labels/grid differ")
    label_path = stage / "phase_labels.png"
    publish_file(label_path, lambda stream: Image.fromarray(labels).save(stream, format="PNG"))
    classes = _write_npz(stage / "phase_classes.npz", inferred["classes"])
    area = dict(_write_npz(stage / "phase_area.npz", {"pixel_area_mm2": inferred["pixel_area_mm2"]}),
                array_key="pixel_area_mm2")
    qa = _write_json(stage / "phase_qa.json", inferred["qa"])
    masks = _write_npz(stage / "phase_operator_masks.npz", dict(inferred["seam_masks"], **inferred["post_masks"]))
    temporal = lattice_frame["temporal_input"]
    paths = {name: stage / (name + ".npy") for name in ("temporal_log", "temporal_valid", "full_log", "full_valid")}
    write_log_linear(paths["temporal_log"], temporal.log)
    write_log_linear(paths["temporal_valid"], temporal.valid)
    write_log_linear(paths["full_log"], lattice_frame["full_log"])
    write_log_linear(paths["full_valid"], lattice_frame["full_valid"])
    history = dict(complete=True, source=spec["source"], current_h=np.eye(3).tolist(),
                   **{name: asset(path) for name, path in paths.items()},
                   temporal_origin_xy=list(temporal.origin_xy), temporal_provenance=temporal.provenance,
                   native_scientific_identity=_identity(lattice_frame["scientific"]))
    history_asset = _write_json(stage / "history_input.json", history)
    return dict(complete=True, source_sha256=spec["source"]["source_sha256"], labels=asset(label_path),
                classes=classes, area=area, qa=qa, operator_masks=masks, history_input=history_asset)


def _identity(record):
    """Content identities of a native record whose transient files are not kept."""
    if isinstance(record, dict):
        if {"path", "sha256"} <= set(record):
            return {key: value for key, value in record.items() if key != "path"}
        return {key: _identity(value) for key, value in record.items()}
    if isinstance(record, list):
        return [_identity(value) for value in record]
    return record


def _inventory(stage):
    """Describe every lattice stage file of one frame."""
    from ..mass.n1_pipeline import asset
    return {path.name: asset(path) for path in sorted(Path(stage).iterdir()) if path.is_file()}


def validate_lattice_frame(spec, result, final_directory, config):
    """Admit a committed lattice frame's stage payloads before a resume uses it."""
    from ..mass.n1_pipeline import DataReader
    from . import series
    product = result["prediction"]["pipeline"]
    request = lattice_request(config, result["precision"])
    if (product.get("complete") is not True or product.get("format") != FRAME_FORMAT
            or product.get("pipeline") != PIPELINE or product["source"] != spec["source"]
            or product["case_id"] != spec["case_id"]
            or product["configuration_sha256"] != config["_configuration_sha256"]
            or product.get(RESOLUTION_KEY) != request[RESOLUTION_KEY]
            or product["phase"]["source_sha256"] != spec["source"]["source_sha256"]):
        raise ValueError("Committed lattice frame source/configuration differs")
    reader = DataReader(unc_only=False)
    owned = Path(final_directory).resolve().parents[1] / "preparation" / spec["case_id"]
    inventory = product["files"]
    if not inventory:
        raise ValueError("Committed lattice stage inventory is empty")
    for relative, bound in inventory.items():
        if Path(relative).name != relative or Path(bound["path"]).resolve() != (owned / relative).resolve():
            raise ValueError("Committed lattice stage file escapes its producer attempt")
        reader.path(bound)
    for key in ("labels", "classes", "area", "qa", "operator_masks", "history_input"):
        bound = product["phase"][key]
        if bound["sha256"] != inventory[Path(bound["path"]).name]["sha256"]:
            raise ValueError("Lattice phase descriptor differs from the committed inventory")
    history = reader.json(product["phase"]["history_input"])
    series.restore_history(history, spec)
    for name in ("temporal_log", "temporal_valid", "full_log", "full_valid"):
        bound = history[name]
        if bound["sha256"] != inventory[Path(bound["path"]).name]["sha256"]:
            raise ValueError("Lattice history differs from the committed inventory")
    if result["source_binding"] != spec["source"]["source_binding"]:
        raise ValueError("Mass source binding differs from the selected acquisition")
    if result["bindings"]["phase_sha256"] != product["phase"]["labels"]["sha256"]:
        raise ValueError("N1 consumed a different lattice phase output")
    from ..computing_lattice import computing_lattice
    from ..lattice_storage import stored_layer, verify_stored_entry
    layer = stored_layer(request, computing_lattice(request["extent_m"], request[RESOLUTION_KEY]))
    if layer is None:
        if "stored" in product:
            raise ValueError("A frame stored on its computing lattice has no coarser stored products")
    else:
        verify_stored_entry(product.get("stored"), layer, spec=spec, inventory=inventory, result=result,
                            reader=reader, configuration_sha256=config["_configuration_sha256"])
    return product


def _restore_predecessor(inputs, lattice_run, result, spec):
    """Restore the lattice predecessor, and the native one's identity, from a commit."""
    import numpy as np
    from ..phase.inputs import TemporalNativeInput
    from . import series
    product = result["prediction"]["pipeline"]
    history = inputs.reader.json(product["phase"]["history_input"])
    series.restore_history(history, spec)
    arrays = {name: inputs.reader.array(history[name])
              for name in ("temporal_log", "temporal_valid", "full_log", "full_valid")}
    for array in arrays.values():
        array.setflags(write=False)
    temporal = TemporalNativeInput(arrays["temporal_log"], arrays["temporal_valid"],
                                   tuple(history["temporal_origin_xy"]), history["temporal_provenance"])
    previous = dict(source=history["source"], full_log=arrays["full_log"], full_valid=arrays["full_valid"],
                    current_h=np.eye(3), temporal_input=temporal)
    lattice_run.previous = previous
    # The native inputs read only the predecessor's identity and provenance.
    inputs.previous = dict(previous)


def run_lattice_series(task, output, backend, precision, on_progress=None):
    """Execute one lattice run with the native runner's thread limits.

    Working precision applies to N1 only. The caller owns queue claims,
    retirement, lease fences and watchdogs, as for ``runner.run_series``.
    """
    from darsia._opencv_runtime import scientific_thread
    ipp_start = scientific_thread("run_lattice_series:start")
    import cv2
    from threadpoolctl import threadpool_info, threadpool_limits

    threads = task.get("threads", 1)
    if type(threads) is not int or threads < 1:
        raise ValueError("Positive internal thread count required")
    old_threads = cv2.getNumThreads()
    try:
        cv2.setNumThreads(threads)
        with threadpool_limits(limits=threads):
            runtime = dict(requested_threads=threads, cv2_threads=cv2.getNumThreads(),
                           threadpools=threadpool_info(), opencv_ipp=ipp_start)
            return _run_lattice_series(task, output, backend, precision, on_progress, runtime)
    finally:
        cv2.setNumThreads(old_threads)


def _run_lattice_series(task, output, backend, precision, on_progress, runtime):
    """Admit, prepare, compute, publish and close one lattice run."""
    from darsia._opencv_runtime import scientific_thread, completed_frame, require_ipp_off
    from ..computing_lattice import computing_lattice
    from ..lattice_storage import stored_layer, write_stored_frame
    from ..mass.lattice_mass import LatticeMass, lattice_record
    from ..mass.n1_pipeline import N1Pipeline, DataReader, _write_json, _publish_frame
    from ..mass.production_precision import PixelMath
    from ..phase.infer import infer_labels
    from ..phase.physical_units import conversion_report
    from ..storage_profile import run_setting, require_template_colour_coordinate
    from . import series
    from .checkpoint import verify_frame_prefix, run_identity, frame_producer
    from .job_queue import retry_transient, require_unc
    from .lattice_inputs import LatticeRun
    from .run_io import for_task
    from .runner import scalar_table

    config = load_lattice_run(task)
    config["_configuration_sha256"] = task["configuration"]["sha256"]
    request = lattice_request(config, precision)
    if task.get(RESOLUTION_KEY) != request[RESOLUTION_KEY]:
        raise ValueError("Task and configuration computing resolutions differ")
    if "gas_probability_storage" in config:
        raise ValueError("The gas-probability storage request encodes native-centre pixels; lattice runs omit it")
    lattice = computing_lattice(request["extent_m"], request[RESOLUTION_KEY])
    layer = stored_layer(request, lattice)
    setting = run_setting(task, config)
    require_template_colour_coordinate(config, config["mass"]["template"])
    if backend not in {"cpu", "cuda"} or precision != task["precision"]:
        raise ValueError("Actual backend/precision differs from queue task")
    io_binding = for_task(task)
    if io_binding is None:
        require_unc(str(output))
    else:
        io_binding.require_output(output)
    output = Path(output)
    retry_transient(lambda: output.mkdir(parents=True, exist_ok=True))
    if retry_transient(lambda: any(output.iterdir())):
        raise FileExistsError("Each attempt must be fresh")
    clock = time.perf_counter()
    prefix = verify_frame_prefix(task, config, output, backend, precision, task.get("resume_from"),
                                 frame_validator=validate_lattice_frame,
                                 calibration_sha256=task["calibration_sha256"])
    if io_binding is not None:
        io_binding.admit_resume_root(prefix["source"])
    resume_asset = _write_json(output / "resume.json", dict(
        complete=True, source=prefix["source"], prefix_case_ids=prefix["prefix_case_ids"],
        frame_producers=prefix["frame_producers"], predecessor_input=prefix["predecessor_input"]))
    settings = dict(host=socket.gethostname(), backend=backend, precision=precision, threads=task["threads"],
                    thread_runtime=runtime, phase_precision="explicit-historical-stage-dtypes",
                    **{RESOLUTION_KEY: request[RESOLUTION_KEY]})
    manifest = dict(
        format="darsia-series-run-manifest-v1", pipeline=PIPELINE, job_id=task["job_id"], run=task["run"],
        configuration=task["configuration"], task_identity=run_identity(task, backend, precision,
                                                                        task["calibration_sha256"]),
        code=dict(commit=task["commit"], identity_algorithm="sha256-git-commit-ascii"),
        code_sha256=hashlib.sha256(task["commit"].encode("ascii")).hexdigest(), settings=settings,
        frames={spec["case_id"]: spec for spec in config["frames"]}, resume_checkpoint=resume_asset,
        lattice=dict(request={key: value for key, value in request.items() if key != "precision"},
                     lattice=lattice.metadata(), validation=VALIDATION,
                     storage_profile=dict(setting, applies="lattice runs keep no native intermediates; "
                                                           "every profile stores the lattice layout")))
    if layer is not None:
        manifest["lattice"]["stored"] = layer.metadata()
    manifest_asset = _write_json(output / "manifest.json", manifest)
    identity = dict(run=task["run"], job_id=task["job_id"], configuration=task["configuration"],
                    code_commit=task["commit"], calibration_sha256=task["calibration_sha256"])
    records = list(prefix["records"])
    producers = list(prefix["frame_producers"])
    if backend == "cpu":
        math_provider = PixelMath(precision)
    else:
        from ..mass.mass_cuda import CudaHostPixelMath
        math_provider = CudaHostPixelMath(precision, device_id=int(task["device_id"]),
                                          pool_limit_bytes=int(task["pool_limit_bytes"]))
    inputs = None
    try:
        with tempfile.TemporaryDirectory(prefix="darsia-lattice-") as scratch:
            inputs = series.SeriesInputs(config, output)
            mass_config = dict(config["mass"]["template"], run=config["run"], frames=config["frames"],
                               data_aliases=config.get("data_aliases", ()))
            native_mass = N1Pipeline(mass_config, precision=precision, pixel_math=math_provider, unc_only=True)
            lattice_run = LatticeRun(config, inputs, lattice, reference_matrix=native_mass.reference_matrix,
                                     fields=native_mass.config["geometry_fields"])
            directory = output / "lattice"
            reference = config["phase"]["reference"]["full"]
            mass = LatticeMass(native_mass, lattice, directory, canvas_shape_hw=reference["phase_shape_hw"],
                               canvas_matrix=lattice_run.reference_canvas_matrix, precision=precision,
                               pixel_math=math_provider, unc_only=True)
            del native_mass
            units = _write_json(directory / "physical_units.json", conversion_report(
                inputs.parameters, lattice_run.rules, pixels_per_cell=lattice_run.pixels_per_cell))
            run_record = _write_json(directory / "lattice_inputs.json", lattice_run.provenance())
            baseline = lattice_record(directory, "lattice_baseline", lattice_run.baseline.full_log,
                                      lattice_run.baseline.full_valid, lattice, fields=mass.fields,
                                      provenance=dict(role="baseline",
                                                      source_sha256=config["phase"]["reference"]["source_sha256"]))
            if records:
                index = prefix["predecessor_index"]
                old = records[-1]
                committed = DataReader().json(dict(old["files"]["result.json"],
                                                   path=str(Path(old["nas_directory"]) / "result.json")))
                _restore_predecessor(inputs, lattice_run, committed, config["frames"][index])
            for position in range(len(records), len(config["frames"])):
                spec = config["frames"][position]
                ipp_before = scientific_thread("run_lattice_series:" + spec["case_id"] + ":before")
                if on_progress:
                    on_progress(dict(state="frame_started", run=task["run"], case_id=spec["case_id"],
                                     frame_index=spec["frame_index"]))
                start = time.perf_counter()
                transient = Path(scratch) / spec["case_id"]
                native = inputs.native(spec, transient)
                frame = inputs.frame(spec, native)
                lattice_frame, assignment, matrix = lattice_run.frame(spec, native, frame)
                bundle = lattice_run.bundle(inputs, assignment)
                inferred = infer_labels(lattice_frame, lattice_run.previous, bundle, lattice_run.rules)
                stage = output / "preparation" / spec["case_id"]
                phase = _phase_outputs(stage, spec, lattice_frame, inferred)
                current = lattice_record(stage, "lattice_current", lattice_frame["full_log"],
                                         lattice_frame["full_valid"], lattice, fields=mass.fields,
                                         provenance=dict(role="current", source=spec["source"]))
                mass_spec = mass.frame_spec(spec, stage, current=current, baseline=baseline, phase=phase,
                                            assignment=assignment, canvas_matrix=matrix,
                                            native_record=native["record"]["full"])
                capture = None if layer is None else {}
                evaluated = mass.evaluate(mass_spec, mass.box_masks(assignment, assignment.source_shape),
                                          output_capture=capture)
                receipt = _write_json(stage / "lattice_frame.json", dict(
                    format=FRAME_FORMAT, case_id=spec["case_id"], pipeline=PIPELINE,
                    **{RESOLUTION_KEY: request[RESOLUTION_KEY]}, lattice=lattice.metadata(),
                    run_lattice_inputs=run_record, physical_units=units, mass_template=mass.config_asset,
                    native_record=_identity(native["record"]), native_intermediates="transient, not retained",
                    validation=VALIDATION))
                # Stored products join the stage before the frame commit binds it.
                stored = None if layer is None else write_stored_frame(
                    stage, layer, identity=identity, spec=spec, inferred=inferred, phase=phase,
                    capture=capture, evaluated=evaluated, validation=VALIDATION)
                stage_binding = dict(
                    format=FRAME_FORMAT, complete=True, pipeline=PIPELINE, case_id=spec["case_id"],
                    configuration_sha256=task["configuration"]["sha256"], source=spec["source"],
                    **{RESOLUTION_KEY: request[RESOLUTION_KEY]}, lattice=lattice.metadata(),
                    receipt=receipt, native=_identity(native["record"]), phase=phase,
                    files=_inventory(stage), numerical_phase_calls=1, validation=VALIDATION,
                    opencv_ipp=completed_frame(ipp_before, "run_lattice_series:" + spec["case_id"] + ":after"))
                if stored is not None:
                    stage_binding["stored"] = stored
                evaluated["prediction"]["pipeline"] = stage_binding
                stage_binding["opencv_ipp"]["mass_evaluation"] = mass.last_opencv_ipp
                evaluated["opencv_ipp"] = stage_binding["opencv_ipp"]
                elapsed = time.perf_counter() - start
                record = _publish_frame(output, manifest, manifest_asset, mass_spec, evaluated, mass.config,
                                        backend, position, elapsed, lean=None)
                records.append(record)
                producers.append(frame_producer(output, record, manifest_asset))
                # Full commit precedes advancement and actual progress.
                inputs.advance(frame)
                lattice_run.advance(lattice_frame)
                del frame, inferred, evaluated, native, lattice_frame, bundle, assignment, capture, stored
                shutil.rmtree(transient)
                if on_progress:
                    on_progress(dict(state="frame_completed", complete=True, run=task["run"],
                                     case_id=spec["case_id"], frame_index=spec["frame_index"],
                                     compute_seconds=elapsed, publication=record))
            table = scalar_table(records, config["table_fields"], output / "mass_table.csv")
            wall = time.perf_counter() - clock
            result = dict(
                complete=True, run=task["run"], whole_run=True, pipeline=PIPELINE, backend=backend,
                precision=precision, working_pixel_dtype=precision,
                ordered_case_ids=[spec["case_id"] for spec in config["frames"]], case_count=len(records),
                frames=len(records), records=records, table=table, configuration=task["configuration"],
                manifest=manifest_asset, calibration=config["mass"]["calibration"],
                physical_correction=config["mass"]["template"]["physical_correction"],
                anchor=config["mass"]["template"]["anchor"], wall_seconds=wall,
                newly_computed_frames=len(records) - len(prefix["records"]),
                images_per_hour=(len(records) - len(prefix["records"])) * 3600 / wall,
                reused_case_ids=list(prefix["prefix_case_ids"]), frame_producers=producers,
                elapsed_seconds=wall, lattice=lattice.metadata(), validation=VALIDATION,
                **{RESOLUTION_KEY: request[RESOLUTION_KEY]},
                opencv_ipp=require_ipp_off("run_lattice_series:completed"))
            if layer is not None:
                result["stored"] = layer.metadata()
            del mass
    finally:
        if inputs is not None:
            inputs.close()
        close = getattr(math_provider, "close", None)
        if callable(close):
            close()
    result["worker_references_closed"] = True
    return result
