"""Fixed native -> phase -> N1 execution for whole-run queue tasks.

Importing this module does not load models or numerical libraries. One worker
owns one run and the original queue publishes its terminal completion. All
inputs/parameters are DATA, and each stage calls the single library method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import socket
import time

from . import series
from .job_queue import JobQueue, descriptor, read_json, require_unc, write_immutable


def prepare_job(series_path, job_root, *, expected_sha256, series_manifest=None, capacity_stores=(),
                storage_profile=None):
    """Prepare new versioned alias metadata; legacy jobs remain readable/runnable.

    The registered schema-1 manifest is explicit because an execution-series
    plan is a different document. No missing-manifest fallback creates a new
    absolute-path job. J3 runtime remains unavailable for these new jobs.
    ``storage_profile`` is the job's explicit choice, 'full' or 'lean' (None:
    storage_profile.DEFAULT_STORAGE_PROFILE); the result reports the shared
    estimate for that profile under ``storage_estimate``.
    """
    from .alias_job import prepare_alias_job

    return prepare_alias_job(series_path, job_root, expected_sha256=expected_sha256,
                             series_manifest=series_manifest, capacity_stores=capacity_stores,
                             storage_profile=storage_profile)


def _phase_outputs(stage, spec, frame, inferred):
    """Write fresh categorical/physical stage products and acquisition inputs."""
    import numpy as np
    from PIL import Image
    from ..mass.n1_pipeline import _write_json, _write_npz, asset
    from ..native.sidecar import write_log_linear

    labels = inferred["labels"]
    if labels.dtype != np.uint8 or labels.shape != frame["pixel_area_mm2"].shape:
        raise ValueError("Fresh phase labels/grid differ")
    label_path = stage / "phase_labels.png"
    from .io_ops import publish_file
    def write_labels(stream):
        Image.fromarray(labels).save(stream, format="PNG")
    publish_file(label_path, write_labels)
    classes = _write_npz(stage / "phase_classes.npz", inferred["classes"])
    area = dict(
        _write_npz(
            stage / "phase_area.npz", {"pixel_area_mm2": inferred["pixel_area_mm2"]}
        ),
        array_key="pixel_area_mm2",
    )
    qa = _write_json(stage / "phase_qa.json", inferred["qa"])
    # Preserve input representations before history advancement. These are
    # source data, never classifiers' labels/classes/probabilities.
    temporal = frame["temporal_input"]
    temporal_log, temporal_valid = (
        stage / "temporal_log.npy",
        stage / "temporal_valid.npy",
    )
    write_log_linear(temporal_log, temporal.log)
    write_log_linear(temporal_valid, temporal.valid)
    history = dict(
        complete=True,
        source=spec["source"],
        current_h=frame["current_h"].tolist(),
        full=frame["scientific"]["full"],
        temporal_log=asset(temporal_log),
        temporal_valid=asset(temporal_valid),
        temporal_origin_xy=list(temporal.origin_xy),
        temporal_provenance=temporal.provenance,
    )
    history_asset = _write_json(stage / "history_input.json", history)
    masks = dict(inferred["seam_masks"], **inferred["post_masks"])
    mask_asset = _write_npz(stage / "phase_operator_masks.npz", masks)
    return dict(
        complete=True,
        source_sha256=spec["source"]["source_sha256"],
        labels=asset(label_path),
        classes=classes,
        area=area,
        qa=qa,
        operator_masks=mask_asset,
        history_input=history_asset,
    )


def _inventory(stage):
    """Describe every actual native/phase file, including generated windows."""
    from ..mass.n1_pipeline import asset

    return {
        path.name: asset(path)
        for path in sorted(Path(stage).iterdir())
        if path.is_file()
    }


def validate_frame_artifacts(spec, result, final_directory, config):
    """Strictly admit transitive native/phase/history payloads of a mass commit.

    The common checkpoint validator first admits original lease retirement,
    manifest/task identity and the flat mass commit inventory. This validator
    checks the extra full descriptors byte-bound through that result.json.
    """
    from ..mass.n1_pipeline import DataReader

    product = result["prediction"]["pipeline"]
    if (
        product.get("complete") is not True
        or product.get("format") != "darsia-series-frame-v1"
        or product["source"] != spec["source"]
        or product["case_id"] != spec["case_id"]
        or product["configuration_sha256"] != config["_configuration_sha256"]
    ):
        raise ValueError("Committed series-stage source/configuration differs")
    if (
        product["native"]["source_sha256"] != spec["source"]["source_sha256"]
        or product["phase"]["source_sha256"] != spec["source"]["source_sha256"]
    ):
        raise ValueError("Committed native/phase source identity differs")
    from ..storage_profile import dropped_after_commit
    reader = DataReader(unc_only=False)
    owned = Path(final_directory).resolve().parents[1] / "preparation" / spec["case_id"]
    inventory = product["files"]
    if not inventory:
        raise ValueError("Committed series-stage inventory is empty")
    # A lean 'drop' frame declared before its commit that these two colour
    # intermediates are removed afterwards; absent ones are not verified.
    dropped = dropped_after_commit(result)
    for relative, bound in inventory.items():
        if (
            Path(relative).name != relative
            or Path(bound["path"]).resolve() != (owned / relative).resolve()
        ):
            raise ValueError("Committed stage file escapes its actual producer attempt")
        if relative in dropped and not Path(bound["path"]).exists():
            continue
        reader.path(bound)
    for key in ("labels", "classes", "area", "qa", "operator_masks", "history_input"):
        bound = product["phase"][key]
        if bound["sha256"] != inventory[Path(bound["path"]).name]["sha256"]:
            raise ValueError(
                "Phase descriptor differs from committed transitive inventory"
            )
        reader.path(bound)
    requested = config.get("gas_probability_storage") is not None
    if ("probabilities" in product) != requested:
        raise ValueError("Committed optional probability request/product differs")
    if requested:
        from ..phase.probability_storage import parse_request, validate_receipt, same_metadata
        parse_request(config)
        roles = product["probabilities"]
        if type(roles) is not dict or set(roles) != {"archive", "receipt"}:
            raise ValueError("Explicit probability roles required")
        for bound in roles.values():
            if inventory.get(Path(bound["path"]).name) != bound:
                raise ValueError("Probability product differs from exact stage inventory")
        saved = validate_receipt(reader.json(roles["receipt"]), roles, source=spec["source"],
                                 case_id=spec["case_id"])
        if (not same_metadata(saved["phase_configuration"], config["phase"]["configuration"])
                or not same_metadata(saved["labels"], product["phase"]["labels"])):
            raise ValueError("Probability original configuration/label roles differ")
    native = product["native"]
    reader.json(native["audit"])
    history = reader.json(product["phase"]["history_input"])
    series.restore_history(history, spec)
    if history["full"] != native["full"]:
        raise ValueError("History is not the actual committed RAW acquisition")
    for bound in (history["temporal_log"], history["temporal_valid"]):
        if bound["sha256"] != inventory[Path(bound["path"]).name]["sha256"]:
            raise ValueError("Temporal history differs from committed stage inventory")
        if Path(bound["path"]).name in dropped and not Path(bound["path"]).exists():
            continue
        reader.path(bound)
    if result["source_binding"] != spec["source"]["source_binding"]:
        raise ValueError("Mass source binding differs from actual selected acquisition")
    if result["bindings"]["phase_sha256"] != product["phase"]["labels"]["sha256"]:
        raise ValueError("N1 consumed a different phase output")
    return product


def _restore_predecessor(inputs, result, spec):
    """Restore exact source history after all committed stage bytes were verified."""
    import numpy as np
    from ..phase.inputs import TemporalNativeInput
    from . import phase_inputs as loaders

    from ..storage_profile import dropped_after_commit
    product = result["prediction"]["pipeline"]
    history = inputs.reader.json(product["phase"]["history_input"])
    series.restore_history(history, spec)
    dropped = dropped_after_commit(result)
    if dropped and not all(
        (Path(product["files"][name]["path"])).exists() for name in dropped
    ):
        raise ValueError(
            "Resume needs the predecessor's colour intermediates, which the lean "
            "'drop' retention removed; start a fresh attempt"
        )
    full, valid = loaders.compose_extended_scientific(
        history["full"],
        parameters=inputs.parameters["scientific"],
        aliases=inputs.aliases,
    )
    temporal = TemporalNativeInput(
        inputs.reader.array(history["temporal_log"]),
        inputs.reader.array(history["temporal_valid"]),
        tuple(history["temporal_origin_xy"]),
        history["temporal_provenance"],
    )
    for array in (full, valid, temporal.log, temporal.valid):
        array.setflags(write=False)
    inputs.previous = dict(
        source=history["source"],
        full_log=full,
        full_valid=valid,
        current_h=np.asarray(history["current_h"], np.float64),
        temporal_input=temporal,
    )


def scalar_table(records, fields, destination):
    """Export ordered scalar result JSON only; never load a scientific NPZ."""
    from ..mass.n1_pipeline import DataReader, asset

    reader = DataReader()
    destination = Path(destination)
    rows = []
    for record in records:
        bound = dict(record["files"]["result.json"],
                     path=str(Path(record["nas_directory"]) / "result.json"))
        result = reader.json(bound)
        if result.get("complete") is not True or result["case_id"] != record["frame_id"]:
            raise ValueError("Table row is not its genuine committed frame")
        selected = {}
        for field in fields:
            value = result[field]
            if isinstance(value, (dict, list)):
                raise ValueError("Table fields must be scalar values")
            selected[field] = value
        if result.get("analysis_scope") is not None:
            selected.update(analysis_scope_json=json.dumps(result["analysis_scope"],
                ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")),
                unknown_mass_g=None)
            selected.update({"quantified_" + field: result["quantified_" + field]
                             for field in fields if "quantified_" + field in result})
        rows.append(selected)
    if any("analysis_scope_json" in row for row in rows):
        fields = list(dict.fromkeys([*fields, *(key for row in rows for key in row)]))
    from .io_ops import publish_file
    def write(binary):
        import io
        stream = io.TextIOWrapper(binary, encoding="utf-8", newline="")
        try:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
        finally:
            stream.detach()
    publish_file(destination, write)
    return asset(destination)


def run_series(task, output, backend, precision, on_progress=None):
    """Execute one fixed run with original phase stage dtypes and typed N1 math.

    Working precision applies to N1 only. Native storage, feature decoders,
    reductions and phase call order remain those of the configured methods.
    The caller owns queue claims, retirement, lease fences and watchdogs.
    """
    from darsia._opencv_runtime import scientific_thread
    ipp_start = scientific_thread("run_series:start")
    import cv2
    import scipy.linalg
    from threadpoolctl import threadpool_info, threadpool_limits

    threads = task.get("threads", 1)
    if type(threads) is not int or threads < 1:
        raise ValueError("Positive internal thread count required")
    old_threads = cv2.getNumThreads()
    try:
        cv2.setNumThreads(threads)
        with threadpool_limits(limits=threads):
            runtime = dict(
                requested_threads=threads,
                cv2_threads=cv2.getNumThreads(),
                threadpools=threadpool_info(),
                opencv_ipp=ipp_start,
            )
            return _run_series(task, output, backend, precision, on_progress, runtime)
    finally:
        cv2.setNumThreads(old_threads)


def _run_series(task, output, backend, precision, on_progress, runtime):
    from darsia._opencv_runtime import scientific_thread, completed_frame, require_ipp_off
    from ..mass.n1_pipeline import N1Pipeline, DataReader, _write_json, _publish_frame
    from ..mass.production_precision import PixelMath
    from ..phase.infer import infer_labels
    from .checkpoint import verify_frame_prefix, run_identity, frame_producer

    config = series.load_run(task)
    config["_configuration_sha256"] = task["configuration"]["sha256"]
    from ..storage_profile import resolve as storage_setting, run_setting
    setting = run_setting(task, config)  # The job's choice; legacy tasks write full.
    if ("storage_profile" in config["mass"]["template"]
            and storage_setting(config["mass"]["template"]) != setting):
        raise ValueError("The series storage_profile differs from its mass template")
    from ..storage_profile import require_template_colour_coordinate
    require_template_colour_coordinate(config, config["mass"]["template"])
    storage_request = None
    if "storage" in config:
        from .. import storage_outputs
        storage_request = storage_outputs.parse_request(config, precision)
    probability_request = None
    if "gas_probability_storage" in config:
        from ..phase import probability_storage
        probability_request = probability_storage.parse_request(config)
    if backend not in {"cpu", "cuda"} or precision != task["precision"]:
        raise ValueError("Actual backend/precision differs from queue task")
    from .run_io import for_task
    io_binding = for_task(task)
    current_build = None
    if "current_build_binding" in config:
        if task.get("resume_from") is not None:
            raise ValueError("Current Build requires fresh original frame consumers; resume is unsupported")
        from .current_build import acquire_current_build
        current_build = acquire_current_build(config, task, io_binding)
    if io_binding is None:
        require_unc(str(output))
    else:
        io_binding.require_output(output)
    output = Path(output)
    from .job_queue import retry_transient
    retry_transient(lambda: output.mkdir(parents=True, exist_ok=True))
    if retry_transient(lambda: any(output.iterdir())):
        raise FileExistsError("Each attempt must be fresh")
    clock = time.perf_counter()
    prefix = verify_frame_prefix(
        task,
        config,
        output,
        backend,
        precision,
        task.get("resume_from"),
        frame_validator=validate_frame_artifacts,
        calibration_sha256=task["calibration_sha256"],
    )
    if io_binding is not None:
        io_binding.admit_resume_root(prefix["source"])
    if storage_request is not None:
        storage_outputs.verify_prefix(prefix, task, config, storage_request)
    resume_asset = _write_json(
        output / "resume.json",
        dict(
            complete=True,
            source=prefix["source"],
            prefix_case_ids=prefix["prefix_case_ids"],
            frame_producers=prefix["frame_producers"],
            predecessor_input=prefix["predecessor_input"],
        ),
    )
    settings = dict(
        host=socket.gethostname(),
        backend=backend,
        precision=precision,
        threads=task["threads"],
        thread_runtime=runtime,
        phase_precision="explicit-historical-stage-dtypes",
    )
    manifest = dict(
        format="darsia-series-run-manifest-v1",
        job_id=task["job_id"],
        run=task["run"],
        configuration=task["configuration"],
        task_identity=run_identity(
            task, backend, precision, task["calibration_sha256"]
        ),
        code=dict(commit=task["commit"], identity_algorithm="sha256-git-commit-ascii"),
        code_sha256=hashlib.sha256(task["commit"].encode("ascii")).hexdigest(),
        settings=settings,
        frames={spec["case_id"]: spec for spec in config["frames"]},
        resume_checkpoint=resume_asset,
    )
    if setting["profile"] == "lean":
        manifest["storage_profile"] = dict(setting)
    manifest_asset = _write_json(output / "manifest.json", manifest)
    records = list(prefix["records"])
    producers = list(prefix["frame_producers"])
    if backend == "cpu":
        math_provider = PixelMath(precision)
    else:
        from ..mass.mass_cuda import CudaHostPixelMath

        math_provider = CudaHostPixelMath(
            precision,
            device_id=int(task["device_id"]),
            pool_limit_bytes=int(task["pool_limit_bytes"]),
        )
    inputs = None
    try:
        inputs = series.SeriesInputs(config, output)
        if records:
            index = prefix["predecessor_index"]
            old = records[-1]
            reader = DataReader()
            result = reader.json(
                dict(
                    old["files"]["result.json"],
                    path=str(Path(old["nas_directory"]) / "result.json"),
                )
            )
            _restore_predecessor(inputs, result, config["frames"][index])
        local_build = None if current_build is None else current_build.physical_inputs()
        physical_config = config if local_build is None else local_build.bind_configuration(config)
        build_inputs = current_build if local_build is None else local_build
        mass_config = dict(
            physical_config["mass"]["template"],
            run=config["run"],
            frames=config["frames"],
            data_aliases=config.get("data_aliases", ()),
        )
        current_build_options = {} if current_build is None else {"current_build": current_build}
        if local_build is not None:
            current_build_options = {"local_build": local_build}
        mass = N1Pipeline(
            mass_config, precision=precision, pixel_math=math_provider, unc_only=True,
            **current_build_options,
        )
        lean = None
        if setting["profile"] == "lean":
            from ..storage_profile import LeanRun
            lean = LeanRun(output, mass_config, mass, setting,
                           colour_intermediates_produced=True)
        for position in range(len(records), len(config["frames"])):
            spec = config["frames"][position]
            ipp_before = scientific_thread("run_series:" + spec["case_id"] + ":before")
            if on_progress:
                on_progress(
                    dict(
                        state="frame_started",
                        run=task["run"],
                        case_id=spec["case_id"],
                        frame_index=spec["frame_index"],
                    )
                )
            start = time.perf_counter()
            native = inputs.native(spec, output / "preparation" / spec["case_id"])
            frame = inputs.frame(spec, native)
            if build_inputs is not None:
                frame = build_inputs.check_phase(spec, native, frame)
            if storage_request is None and probability_request is None:
                inferred = infer_labels(
                    frame, inputs.previous, inputs.model_bundle, inputs.parameters
                )
            else:
                phase_capture = {}
                if probability_request is not None:
                    phase_capture["gas_probability_storage"] = {}
                inferred = infer_labels(frame, inputs.previous, inputs.model_bundle,
                                        inputs.parameters, output_capture=phase_capture)
            if build_inputs is not None:
                build_inputs.check_inferred(spec, frame, inferred)
            stage = output / "preparation" / spec["case_id"]
            phase = _phase_outputs(stage, spec, frame, inferred)
            mass_spec = series.bind_mass_frame(
                spec, native["record"], phase, baseline=inputs.mass_baseline
            )
            if storage_request is None:
                evaluated = mass.evaluate(mass_spec)
            else:
                mass_capture = {}
                evaluated = mass.evaluate(mass_spec, output_capture=mass_capture)
                storage_entry = storage_outputs.write_frame(
                    output, task, spec, phase, inferred, phase_capture,
                    mass_capture, evaluated, storage_request)
            probability_product = None
            if probability_request is not None:
                probability_product = probability_storage.write_product(stage, spec,
                    phase_capture["gas_probability_storage"], model_bindings=inputs.phase["model_bundles"],
                    phase_configuration=config["phase"]["configuration"], labels=inferred["labels"],
                    label_descriptor=phase["labels"])
            stage_binding = dict(
                format="darsia-series-frame-v1",
                complete=True,
                case_id=spec["case_id"],
                configuration_sha256=task["configuration"]["sha256"],
                source=spec["source"],
                native=native["record"],
                phase=phase,
                files=_inventory(stage),
                numerical_phase_calls=1,
                opencv_ipp=completed_frame(ipp_before, "run_series:" + spec["case_id"] + ":after"),
            )
            if probability_product is not None:
                stage_binding["probabilities"] = probability_product
            if storage_request is not None:
                stage_binding["storage"] = storage_entry
                for bound in [storage_entry["receipt"], *storage_entry["files"]]:
                    stage_binding["files"][Path(bound["path"]).name] = bound
                storage_outputs.validate_publication_binding(stage_binding)
            evaluated["prediction"]["pipeline"] = stage_binding
            stage_binding["opencv_ipp"]["mass_evaluation"] = mass.last_opencv_ipp
            evaluated["opencv_ipp"] = stage_binding["opencv_ipp"]
            elapsed = time.perf_counter() - start
            record = _publish_frame(
                output,
                manifest,
                manifest_asset,
                mass_spec,
                evaluated,
                mass_config,
                backend,
                position,
                elapsed,
                lean=lean,
            )
            records.append(record)
            producers.append(frame_producer(output, record, manifest_asset))
            # Full commit precedes advancement and actual progress. LeaseLost
            # from the original callback prevents any later queue publication.
            inputs.advance(frame)
            if lean is not None:
                lean.committed(stage)
            del frame, inferred, evaluated, native
            if storage_request is not None:
                del phase_capture, mass_capture, storage_entry
            if on_progress:
                on_progress(
                    dict(
                        state="frame_completed",
                        complete=True,
                        run=task["run"],
                        case_id=spec["case_id"],
                        frame_index=spec["frame_index"],
                        compute_seconds=elapsed,
                        publication=record,
                    )
                )
        table = scalar_table(records, config["table_fields"], output / "mass_table.csv")
        if lean is not None:
            lean.finish()
        wall = time.perf_counter() - clock
        result = dict(
            complete=True,
            run=task["run"],
            whole_run=True,
            pipeline=series.PIPELINE,
            backend=backend,
            precision=precision,
            working_pixel_dtype=precision,
            ordered_case_ids=[spec["case_id"] for spec in config["frames"]],
            case_count=len(records),
            frames=len(records),
            records=records,
            table=table,
            configuration=task["configuration"],
            manifest=manifest_asset,
            calibration=config["mass"]["calibration"],
            physical_correction=mass_config["physical_correction"],
            anchor=mass_config["anchor"],
            wall_seconds=wall,
            newly_computed_frames=len(records) - len(prefix["records"]),
            images_per_hour=(len(records) - len(prefix["records"])) * 3600 / wall,
            reused_case_ids=list(prefix["prefix_case_ids"]),
            frame_producers=producers,
            elapsed_seconds=wall,
            opencv_ipp=require_ipp_off("run_series:completed"),
        )
        if lean is not None:
            result["lean_storage"] = lean.outcome_record()
        del mass
    finally:
        if inputs is not None:
            inputs.close()
        close = getattr(math_provider, "close", None)
        if callable(close):
            close()
    result["worker_references_closed"] = True
    return result


def main(argv=None):
    """Prepare a fixed series job without running any numerical worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--series-sha256", required=True)
    parser.add_argument("--series-manifest", type=Path, required=True,
                        help="Registered schema-1 series.toml with declared [roots]")
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--stores", type=Path, action="append", default=[],
                        help="Optional local machine map(s) for whole-job expected needed/free reports")
    parser.add_argument("--storage-profile", choices=("full", "lean"), default=None,
                        help="The job's storage profile (default: DEFAULT_STORAGE_PROFILE)")
    args = parser.parse_args(argv)
    from ..store_map import read_store_map
    maps = [read_store_map(path, forbidden_roots=(Path(__file__).resolve().parents[4],
                                                args.series_manifest.parent)) for path in args.stores]
    result = prepare_job(args.series, args.job, expected_sha256=args.series_sha256,
                         series_manifest=descriptor(args.series_manifest), capacity_stores=maps,
                         storage_profile=args.storage_profile)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
