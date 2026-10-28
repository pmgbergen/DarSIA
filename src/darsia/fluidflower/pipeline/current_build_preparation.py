"""Prepare the real selected RAW/geometry inputs for a completed Build run."""
from copy import deepcopy
import json
from pathlib import Path

from ..mass.build_native_bridge import BuildNativeBridge, BuildNativeFrame
from ..mass.native_optical_source import load_build_physical_context
from ..mass.n1_pipeline import DataReader, _write_json
from .build_frame_geometry import prepare_build_frame_geometry
from .build_phase_inputs import BuildPhaseReplacement, assemble_current_build_phase_inputs
from .current_phase_recipe import retained_phase_recipe, build_photometry_binding
from .local_build import LocalBuildInputs
from .raw_acquisition import RawAcquisitionInputs, FORMAT as RAW_FORMAT


UNALIGNED_DRIFT = "ROIs cannot be aligned by translation"


def prepare_current_build_acquisitions(operation_result, metric_model, selection_record,
                                       output, *, map_product=None, on_progress=None, reader=None,
                                       prepare_native=True, assemble_phase=True, prepared_photometry=None,
                                       leave_out_unaligned=False):
    """Run original RAW producers only after Start, preserving selected history.

    The current selection supplies identities, the registered frozen method
    supplies optical/phase roles, and the actual Build supplies physical fields.
    No previous frame's geometry or labels are used as a new acquisition.

    With ``leave_out_unaligned``, a selected frame whose drift correction
    finds no translation is left out instead of stopping the run; the result
    lists it under ``left_out`` with the reason. The baseline is never left out.
    """
    reader = reader or DataReader()
    from .setup_image_preparation import retained_series_aliases, image_basis_reader
    reader = image_basis_reader(reader, dict(source_aliases=retained_series_aliases(
        {"current": operation_result}, reader)))
    context = load_build_physical_context(operation_result, reader=reader)
    selection = reader.json(selection_record)
    selected = [row for row in selection["runs"] if row["run"] == context.run]
    if len(selected) != 1:
        raise ValueError("The current selection must include this completed Build exactly once")
    selected = selected[0]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    recipe = retained_phase_recipe(context.provenance["native_configuration"],
        selection["input_authority"]["phase_plan"], context.run, reader=reader,
        selected_model=selection["input_authority"].get("selected_model"), source_only=not assemble_phase)
    runtime = (prepared_photometry.get("native_configuration") if prepared_photometry is not None else None)
    if runtime is None:
        runtime = (selection["input_authority"].get("selected_model") or {}).get("native_configuration")
    if runtime is not None:
        from .current_phase_recipe import require_runtime_configuration
        require_runtime_configuration(context.provenance["native_configuration"], runtime, reader=reader)
        recipe["native_configuration"] = deepcopy(runtime)
    phase_binding = _write_json(root / "phase_parameters.json", recipe["phase"])
    photometry = build_photometry_binding(context, recipe, root / "photometry.json", reader=reader,
        image_choice_authority=selection["input_authority"]["current_series"],
        **({"prepared_photometry": prepared_photometry} if prepared_photometry is not None else {}))
    bridge = BuildNativeBridge(operation_result, photometry, root / "bridge", reader=reader)
    temporary = dict(run=context.run, native=dict(configuration=recipe["native_configuration"]),
        phase=dict(configuration=phase_binding), frames=[],
        statistics=dict(physical_correction=operation_result))
    local = LocalBuildInputs(temporary, operation_result=operation_result, metric_model=metric_model,
                            frame_receipts={}, reader=reader)
    request = reader.json(context.provenance["raw_build_request"])
    product = request["native_product"]["source_labels"] if map_product is None else map_product
    factor = reader.json(metric_model)["area_factor"]
    rows = [selected["baseline"], *selected["selected_frames"]]
    produced, specifications = {}, []
    replacements, native_records, acquisitions = [], {}, {}
    reusable = _review_source_cache(selection["input_authority"].get("selected_model"),
        context.run, operation_result, metric_model, reader)
    owner = None
    left_out = []
    for position, row in enumerate(rows):
        raw = dict(path=row["raw_path"], sha256=row["raw_sha256"])
        raw["size_bytes"] = row.get("recorded_raw_size_bytes") or Path(raw["path"]).stat().st_size
        case = context.run + "-" + raw["sha256"][:20]
        if on_progress:
            on_progress(dict(state="raw_preparation", run=context.run, case_id=case,
                             position=position, total=len(rows)))
        cached = reusable.get(raw["sha256"])
        baseline_cache = (prepared_photometry.get("baseline") if prepared_photometry is not None
                          and row["baseline"] else None)
        try:
            frame = (BuildNativeFrame(**cached["frame"]) if cached is not None
                     else bridge.frame(raw, root / ("raw-" + case),
                         **({"prepared_baseline": baseline_cache} if baseline_cache is not None else {})))
        except ValueError as error:
            if not (leave_out_unaligned and not row["baseline"] and str(error).startswith(UNALIGNED_DRIFT)):
                raise
            left_out.append(dict(raw_path=raw["path"], raw_sha256=raw["sha256"],
                frame_index=row["current_frame_index"], elapsed_h=row["elapsed_h"], reason=str(error)))
            if on_progress:
                on_progress(dict(state="frame_left_out", run=context.run, case_id=case, reason=str(error)))
            continue
        produced[case] = frame
        source = dict(run=context.run, source_path=raw["path"], source_sha256=raw["sha256"],
            source_binding=dict(raw_acquisition=deepcopy(raw), source_sha256=raw["sha256"]),
            current_frame_index=row["current_frame_index"], elapsed_h=row["elapsed_h"],
            previous_source_sha256=row["previous_raw_sha256"], selected=not row["baseline"],
            baseline=row["baseline"])
        spec = dict(case_id=case, frame_index=row["current_frame_index"],
            elapsed_h=row["elapsed_h"], raw_acquisition=raw, source=source,
            native_task=deepcopy(frame.native_task))
        geometry = prepare_build_frame_geometry(local, frame, map_product=product,
                                                 output=root / ("geometry-" + case))
        native = reader.json(frame.native_task["geometry"])
        roles = deepcopy(recipe["phase_roles"])
        area = dict(role="selected", factor=factor, revision=dict(metric_model=deepcopy(metric_model)))
        roles.update(area=area, current_h=native["baseline_to_native_h"],
                     reference_to_current_h=native["reference_to_native_h"])
        spec.update(geometry=geometry, phase_roles=roles)
        specifications.append(spec)
        replacements.append(BuildPhaseReplacement(spec["raw_acquisition"], geometry, area,
            roles["static"], roles["current_h"], roles["reference_to_current_h"]))
        if prepare_native:
            if cached is not None:
                acquisitions[case] = dict(deepcopy(cached["acquisition"]), source=deepcopy(source))
                native_records[case] = deepcopy(cached["native_record"])
            elif owner is None:
                owner = RawAcquisitionInputs(dict(format=RAW_FORMAT, run=context.run,
                    native=temporary["native"], phase=temporary["phase"], acquisitions=[spec],
                    temporal_pad_px=recipe["temporal_pad_px"]), unc_only=False)
                owner.processors[json.dumps(frame.native_task["processor"], sort_keys=True)] = bridge.processor
            # Only this acquisition is retained in the producer. Completed
            # native/ring/history arrays stay on disk, never in a cohort cache.
            if cached is None:
                owner.prepared_raw = {case: frame}
                stage = root / ("native-" + case)
                acquisitions[case] = owner.prepare(spec, stage)
                from ..mass.n1_pipeline import asset
                native_records[case] = asset(stage / "native.json")
        spec.pop("native_task")  # RAW selection assembler owns only acquisition roles.
        if on_progress:
            on_progress(dict(state="raw_prepared", run=context.run, case_id=case,
                             position=position + 1, total=len(rows)))
    if owner is not None:
        owner.processors.clear()
        owner.prepared_raw.clear()
    # This base owns only the explicitly retained method/support fields used
    # by the existing assembler; it is not a relabelled historical run.
    base = _write_json(root / "run_method.json", dict(run=context.run,
        native=temporary["native"], phase=temporary["phase"], selection_index_field="current_frame_index",
        provenance=dict(original_recipe_receipt=recipe["original_recipe_receipt"])))
    if assemble_phase:
        data = assemble_current_build_phase_inputs(bridge, base, specifications[0], specifications[0],
            specifications[1:], replacements, phase_configuration=phase_binding,
            temporal_pad_px=recipe["temporal_pad_px"], selection_record=selection_record, reader=reader)
    else:
        data = dict(format=RAW_FORMAT, run=context.run, reference_spec=specifications[0],
            frames=specifications[1:], source_only=True, phase_inference_performed=False)
    return dict(configuration=data, configuration_binding=_write_json(root / "raw_phase_inputs.json", data),
        bridge=bridge, produced=produced, prepared_native=native_records, acquisitions=acquisitions,
        metric_model=deepcopy(metric_model), operation_result=deepcopy(operation_result),
        phase_recipe=recipe, left_out=left_out)


def materialize_current_build_review(operation_result, metric_model, selection_record,
                                     output, review_root, *, on_progress=None, reader=None,
                                     preparer=None, review_writer=None, prepared_photometry=None,
                                     leave_out_unaligned=False):
    """Publish real corrected/cropped RAW inputs for review and fresh training.

    The same original native producer feeds later P1. This stage performs no
    optical classification, creates no phase labels and changes no old review.
    Its saved context and every frame point to the actual completed Build.
    With ``leave_out_unaligned``, frames the drift correction cannot align are
    left out of the review and listed in the result as ``left_out_frames``.
    """
    from ..mass.n1_pipeline import asset
    from ..raw_photographic_review import write_raw_review_root
    reader = reader or DataReader()
    prepared = (preparer or prepare_current_build_acquisitions)(operation_result, metric_model,
        selection_record, output, on_progress=on_progress, reader=reader, prepare_native=True,
        assemble_phase=False,
        **({"prepared_photometry": prepared_photometry} if prepared_photometry is not None else {}),
        **({"leave_out_unaligned": True} if leave_out_unaligned else {}))
    configuration, recipe = prepared["configuration"], prepared["phase_recipe"]
    run = configuration["run"]
    selected = next(row for row in reader.json(selection_record)["runs"] if row["run"] == run)
    left_out = prepared.get("left_out") or []
    omitted = {row["raw_sha256"] for row in left_out}
    acquired = [selected["baseline"], *(row for row in selected["selected_frames"]
                                        if row["raw_sha256"] not in omitted)]
    records = []
    for row in acquired:
        case = run + "-" + row["raw_sha256"][:20]
        records.append(prepared["acquisitions"][case])
    manifest = _write_json(Path(output) / "raw_acquisitions.json", dict(
        format=RAW_FORMAT, run=run, records=records, adopted=False,
        current_build=deepcopy(operation_result), metric_model=deepcopy(metric_model)))
    fields = recipe["phase"]["parameters"]["inputs"]["fields"]["native"]
    def support(record):
        native = reader.json(record["native_row"])
        return reader.array(dict(path=native[fields[2]], sha256=native[fields[3]]))
    reference = dict(acquisition_manifest=manifest, record_index=0, photo_support=support(records[0]))
    frames = []
    for index, (record, row) in enumerate(zip(records[1:], acquired[1:])):
        case = run + "-" + row["raw_sha256"][:20]
        frame = prepared["produced"][case]
        spec = next(spec for spec in configuration["frames"] if spec["case_id"] == case)
        frames.append(dict(case_id=case, frame_index=index, elapsed_h=row["elapsed_h"],
            native_record=prepared["prepared_native"][case], native_geometry=frame.native_task["geometry"],
            rig_geometry=spec["geometry"], photo_support=support(record)))
    raw_binding = prepared["produced"][run + "-" + acquired[0]["raw_sha256"][:20]].native_task["processor"]["raw_binding"]
    payload = dict(format="darsia-current-build-source-preparation-v1", run=run,
        operation_result=deepcopy(operation_result), metric_model=deepcopy(metric_model),
        raw_binding=deepcopy(raw_binding), frames={row["raw_sha256"]: dict(
            frame=dict(native_task=deepcopy(prepared["produced"][case].native_task),
                products=deepcopy(prepared["produced"][case].products),
                receipt=deepcopy(prepared["produced"][case].receipt)),
            native_record=deepcopy(prepared["prepared_native"][case]),
            acquisition=deepcopy(prepared["acquisitions"][case]))
            for row in acquired for case in [run + "-" + row["raw_sha256"][:20]]})
    payload_pin = _write_json(Path(output) / "prepared_sources.json", payload)
    photo = recipe["original_photo"]
    context_keys = ("photo_certified_sand_mask_path", "photo_certified_sand_mask_sha256",
        "photo_sand_qa_pass", "photo_sand_qa_scope", "photo_sand_reference_mask_path",
        "photo_sand_reference_mask_sha256", "photo_sand_reference_receipt_sha256", "enclosed_hole_policy")
    context = deepcopy(recipe.get("training_context", {key: deepcopy(photo[key]) for key in context_keys if key in photo}))
    path = (review_writer or write_raw_review_root)(review_root, run=run,
        native_configuration=recipe["native_configuration"], raw_binding=raw_binding,
        reference=reference, frames=frames, training_context=context,
        modes=("raw_detail",), current_build_preparation=payload_pin, reader=reader)
    result = dict(format="darsia-current-build-training-review-v1", run=run,
        manifest=asset(path), operation_result=deepcopy(operation_result),
        selection=deepcopy(selection_record), sources=prepared["configuration_binding"],
        raw_photometry_binding=deepcopy(raw_binding), phase_inference_performed=False)
    if left_out:
        result["left_out_frames"] = deepcopy(left_out)
    result["receipt"] = _write_json(Path(output) / "training_review.json", result)
    return result


def _review_source_cache(selected_model, run, operation_result, metric_model, reader):
    """Reuse actual source products only for this same Build, photometry and RAW."""
    if selected_model is None:
        return {}
    basis = reader.json(selected_model["image_basis"])
    rows = [row for row in basis["runs"] if row["run"] == run]
    if len(rows) != 1:
        raise ValueError("Selected source preparation has no unique run basis")
    manifest = reader.json(rows[0]["source_manifest"])
    pin = manifest.get("current_build_preparation")
    if pin is None:
        return {}
    value = reader.json(pin)
    if (value.get("format") != "darsia-current-build-source-preparation-v1" or value.get("run") != run
            or value.get("operation_result") != operation_result
            or any(value.get("metric_model", {}).get(key) != metric_model.get(key) for key in ("sha256", "size_bytes"))
            or value.get("raw_binding") != rows[0]["raw_binding"]):
        raise ValueError("Retained corrected source products belong to another Build or image basis")
    reader.path(value["metric_model"])
    reader.path(metric_model)
    for digest, item in value["frames"].items():
        frame = item["frame"]
        geometry = reader.json(frame["native_task"]["geometry"])
        receipt = reader.json(frame["receipt"])
        native = reader.json(item["native_record"])
        if (geometry["run"] != run or geometry["acquisition"]["sha256"] != digest
                or frame["native_task"]["processor"]["raw_binding"] != value["raw_binding"]
                or receipt["native_frame"] != frame["native_task"]["geometry"]
                or receipt["products"] != frame["products"]
                or native.get("source_sha256") != digest or native.get("processor_kind") != "raw_rig"):
            raise ValueError("Retained corrected source products differ from their original producer")
    return value["frames"]
