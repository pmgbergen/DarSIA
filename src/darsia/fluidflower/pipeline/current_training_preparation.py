"""Prepare existing reviewed labels on the actual completed Build image basis."""
from copy import deepcopy
from pathlib import Path
import uuid

from ..mass.n1_pipeline import asset, _write_json
from .current_inputs import current_input_reader


def _save(path, value):
    """Reuse exact completed preparation metadata, never overwrite evidence."""
    path = Path(path)
    if path.is_file():
        import json
        if json.loads(path.read_text(encoding="utf-8-sig")) != value:
            raise ValueError("An existing training preparation belongs to other inputs: " + str(path))
        return asset(path)
    return _write_json(path, value)


def _operations(snapshot):
    return {row["run"]: row["result"]["files"]["operation_result.json"]
            for row in snapshot["runs"]
            if row.get("operation") == "build" and row.get("state") == "Complete"}


def _same_build_basis(root, snapshot, selection, reader):
    """Compare actual setup authority when a later model/light choice changed TOML.

    Protocol workbook updates are calibration inputs; they do not invalidate
    unchanged geometry and completed RAW Build products. Source files, current
    geometry and the original native camera configuration still have to match.
    Storage alias rows may only have been added since the Setup plan; a later
    model or lighting choice is accepted whether or not rows were added.
    """
    from ..project import FluidFlowerProject
    from ..geometry_facies import GeometryRevisionStore
    from ..series_native_context import read_series_native_binding
    facts = snapshot["job_facts"]
    if facts.get("series", {}).get("sha256") == selection["input_authority"]["current_series"]["sha256"]:
        return True
    plan_pin = facts.get("setup_plan")
    if not plan_pin:
        return False
    path = Path(snapshot["root"]) / "setup-controls" / "plan.json"
    pin = dict(path=str(path), sha256=plan_pin["sha256"],
               size_bytes=plan_pin.get("bytes", plan_pin.get("size_bytes")))
    plan = reader.json(pin)
    # Retained Setup bytes are the original authority, not a reconstructed TOML.
    # A relocation may add aliases but never change or remove an earlier row.
    # Other saved choices are judged as without added rows: by the Build's own
    # inputs below, and the lighting choice by the Setup image basis where used.
    if plan.get("series_manifest"):
        import tomllib
        original_series = plan["series_manifest"]
        if original_series["sha256"] != facts["series"]["sha256"]:
            return False
        retained = dict(original_series,
                        path=str(path.parent / "series-manifest.toml"))
        before = tomllib.loads(reader.path(retained).read_text(encoding="utf-8-sig"))
        after = tomllib.loads(reader.path(selection["input_authority"]["current_series"]).read_text(
            encoding="utf-8-sig"))
        from ..storage_paths import series_path_aliases
        old_aliases = series_path_aliases(before, root)
        new_aliases = series_path_aliases(after, root)
        if any(new_aliases.get(source) != destination for source, destination in old_aliases.items()):
            return False
    project = FluidFlowerProject.load(plan["project_root"])
    current = GeometryRevisionStore(project).current_path
    if reader.json(asset(current)) != reader.json(plan["adopted_revision"]):
        return False
    native = read_series_native_binding(root)
    if native is None or native.sha256 != plan["native_parameters"]["sha256"]:
        return False
    operations = _operations(snapshot)
    for row in selection["runs"]:
        result = reader.json(operations[row["run"]])
        request = reader.json(result["raw_build_request"])
        if (request["baseline"]["sha256"] != row["baseline"]["raw_sha256"]
                or request["profile"]["sha256"] != native.sha256):
            return False
        for bound in result["configuration_files"]:
            reader.path(bound)
    return True


def find_completed_build(root, series_id, snapshots, selection, *, reader=None):
    reader = current_input_reader(reader)
    for snapshot in reversed(snapshots):
        facts = snapshot.get("job_facts", {})
        if (not snapshot.get("complete") or snapshot.get("errors")
                or facts.get("pipeline") != "series-setup" or facts.get("series_id") != series_id):
            continue
        try:
            if _same_build_basis(root, snapshot, selection, reader):
                return snapshot
        except (OSError, ValueError, KeyError, TypeError):
            continue
    raise ValueError("Complete Apply and Build for the current saved setup before preparing calibration.")


def selected_training_template(root, *, reader=None, model_reference=None):
    """Follow only recorded request ancestry, retaining exact original evidence."""
    from ..series_method import read_series_method
    from .current_model_basis import _model_root
    reader = current_input_reader(reader)
    method = read_series_method(root)
    folder = _model_root(root, model_reference or method.model_version)
    from ..phase_training.model import DualPhaseSignalModel, TRANSFERABLE_MODEL_SCHEMA
    from ..phase_features import OpticalFeatureContract
    heads, models = {}, []
    for role in ("co2_presence", "gas_within_co2"):
        head = asset(folder / "models" / "generic" / (role + "_signal.json"))
        heads[role] = head
        models.append(DualPhaseSignalModel.from_dict(reader.json(head)))
    first = models[0]
    if (any(model.schema != TRANSFERABLE_MODEL_SCHEMA for model in models)
            or first.feature_contract != models[1].feature_contract
            or first.texture_sigma != models[1].texture_sigma):
        raise ValueError("The selected model needs one shared transferable optical-head architecture")
    contract = OpticalFeatureContract.from_dict(first.feature_contract)
    if (contract.chemistry_adapter != "none" or contract.reference_adapter != "none"
            or not contract.include_dynamic):
        raise ValueError("The current fresh trainer requires generic baseline and dynamic optical features")
    model_template = dict(model_root=str(folder), heads=heads,
        feature_contract=deepcopy(first.feature_contract), texture_sigma=first.texture_sigma)
    pin, seen = asset(folder / "training_request.json"), set()
    for _ in range(32):
        if pin["sha256"] in seen:
            raise ValueError("The selected model repeats its training-request ancestry")
        seen.add(pin["sha256"])
        request = reader.json(pin)
        if request.get("bundles") and isinstance(request.get("settings"), dict):
            pairs = []
            for bundle in request["bundles"]:
                pair = {}
                for role in ("decisions", "combined_manifest"):
                    value = bundle[role]
                    bound = dict(path=value.get("snapshot_path") or value["path"], sha256=value["sha256"])
                    pair[role] = dict(bound, path=str(reader.path(bound)))
                pairs.append(pair)
            settings = dict(request["settings"], template_model=model_template,
                feature_version=contract.generic_version, texture_sigma=first.texture_sigma,
                change_evidence_ablation=contract.change_detector == "none")
            return dict(root=str(folder), request=pin, settings=settings, bundles=pairs)
        pin = request.get("inherited_training_request")
        if not isinstance(pin, dict):
            raise ValueError("The selected model has no recorded original training evidence")
    raise ValueError("The selected model has too many training-request ancestors")


def _training_selection(base, run, baseline, acquisitions, *, baseline_elapsed_h):
    by_sha = {row["acquisition"]["sha256"]: row for row in acquisitions
              if row.get("frame_index") is not None}
    baseline_sha = baseline["sha256"]
    rows = sorted((row for key, row in by_sha.items() if key != baseline_sha),
                  key=lambda row: (row["elapsed_h"], row["frame_index"]))
    reference = dict(raw_path=baseline["path"], raw_sha256=baseline_sha,
        recorded_raw_size_bytes=baseline["size_bytes"], current_frame_index=0,
        elapsed_h=baseline_elapsed_h,
        baseline=True, previous_raw_sha256=None)
    previous, selected = baseline_sha, []
    for index, row in enumerate(rows, 1):
        raw = row["acquisition"]
        selected.append(dict(raw_path=raw["path"], raw_sha256=raw["sha256"],
            recorded_raw_size_bytes=raw.get("size_bytes"), current_frame_index=index,
            elapsed_h=row["elapsed_h"], baseline=False, previous_raw_sha256=previous))
        previous = raw["sha256"]
    return dict(format="ffac-local-p1-selection-data-v1", decision=deepcopy(base["decision"]),
        history_policy="actual source acquisitions with each annotated frame's recorded predecessor",
        input_authority=deepcopy(base["input_authority"]),
        runs=[dict(run=run, baseline=reference, selected_frames=selected)])


def _in_order(function, items, threads):
    """Yield ``function(*item)`` for each item, in order, using up to ``threads`` threads.

    The calls only read shared inputs and write their own files. The first
    failure is raised in order; calls not yet started are then cancelled.
    """
    if threads <= 1 or len(items) <= 1:
        for item in items:
            yield function(*item)
        return
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=min(threads, len(items)))
    try:
        futures = [pool.submit(function, *item) for item in items]
        for future in futures:
            yield future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _reading_threads(workers):
    """Threads for the source reading before image preparation (mostly storage waits)."""
    import os
    requested = int(workers) or os.cpu_count() or 1
    return max(1, min(16, requested))


# One preparation worker holds one RAW frame and its corrected arrays. FF_AC
# workers used 2-5 GB each and mostly one thread (October 2026).
PREPARATION_WORKER_BYTES = 6 * 1024**3


def preparation_workers(workers, resources, tasks):
    """Largest process count for the image preparation of ``tasks`` runs.

    ``workers`` 0 (Automatic) allows the logical processors less a reserve for
    the window and storage; the pool then measures which count up to this
    finishes the most frames. Every count is bounded by the available memory
    at PREPARATION_WORKER_BYTES per worker and by the number of tasks.
    """
    cpu = resources["cpu"]
    logical = max(1, int(cpu.get("logical_cores") or cpu.get("physical_cores") or 1))
    reserve = 2 if logical >= 8 else 1
    requested = int(workers) or max(1, logical - reserve)
    memory = max(1, resources["memory"]["available_bytes"] // PREPARATION_WORKER_BYTES)
    return max(1, min(requested, memory, tasks))


def _materialize_run(arguments):
    """Task process: prepare one run, reporting each frame to the runner.

    A selected frame the drift correction cannot align is left out of the
    training and listed in the run's review; it does not stop the run.
    """
    from .current_build_preparation import materialize_current_build_review
    from ..worker_processes import report
    args, extra = (arguments[:5], {"prepared_photometry": arguments[5]}) if len(arguments) == 6 else (arguments, {})
    return materialize_current_build_review(*args, reader=current_input_reader(), on_progress=report,
                                            leave_out_unaligned=True, **extra)


def memory_margins(total_bytes):
    """Memory kept free while preparing images: (admission reserve, stop threshold).

    No new run starts unless the reserve and the largest worker fit in the
    available memory; below the stop threshold the newest run is stopped and
    queued again. Both are a share of the host's memory, at most 8 and 4 GiB.
    """
    return min(8 * 1024**3, int(total_bytes * 0.15)), min(4 * 1024**3, int(total_bytes * 0.06))


class _FrameCycles:
    """Frame times of each run, from one frame start to the next, with their concurrency."""

    def __init__(self):
        self._last = {}

    def cycle(self, event, now, running):
        """Return (seconds, concurrency) of the run's previous frame, or None.

        A frame counts only when the same number of runs ran at its start and
        at its end.
        """
        if not isinstance(event, dict) or event.get("state") != "raw_preparation":
            return None
        run = event.get("run")
        previous = self._last.get(run)
        self._last[run] = (now, running)
        if previous is None or previous[1] != running:
            return None
        return now - previous[0], running


def _prepare_runs_in_processes(tasks, workers, root, reviews, total, on_progress):
    """Prepare runs, each in its own process, at the count that finishes the most frames.

    An explicit ``workers`` count is used as given. Automatic starts at the
    physical cores, and ThroughputTuner adds workers while the measured frame
    rate rises. MemoryGuard admits one more run only when the largest worker
    seen still fits; when memory runs short anyway, the newest run is stopped
    and queued again. Each finished run is saved as it completes, also when
    another run fails; the failed runs are reported together at the end.
    """
    import time
    import psutil
    from ..execution import detect_host_resources
    from ..worker_processes import MemoryGuard, TaskFailures, ThroughputTuner, run_task_processes
    resources = detect_host_resources()
    maximum = preparation_workers(workers, resources, len(tasks))
    physical = int(resources["cpu"].get("physical_cores") or maximum)
    tuner = None if int(workers) else ThroughputTuner(min(physical, maximum), maximum)
    reserve, critical = memory_margins(psutil.virtual_memory().total)
    guard, cycles = MemoryGuard(reserve), _FrameCycles()

    def admit(running):
        target = maximum if tuner is None else tuner.target
        if running >= target:
            return target
        return running + 1 if running == 0 or guard.allows_another() else running

    remaining, frames = {run for run, _ in tasks}, [0]

    def progress():
        if on_progress:
            on_progress(dict(state="training_source_preparation", position=len(reviews),
                total=total, workers=(maximum if tuner is None else tuner.target),
                frames_prepared=frames[0], remaining_runs=sorted(remaining)))

    def on_event(event, running):
        now = time.monotonic()
        measured = cycles.cycle(event, now, running)
        if measured is not None and tuner is not None:
            tuner.unit_done(*measured, kind="raw_preparation", now=now)
        if isinstance(event, dict) and event.get("state") == "raw_prepared":
            frames[0] += 1
            progress()

    try:
        for run, review in run_task_processes(_materialize_run, tasks, admit=admit, on_event=on_event,
                                              critical_bytes=critical, log_folder=root / "worker_logs"):
            reviews[run] = review
            remaining.discard(run)
            _save(root / "sources" / run / "source_completed.json", review)
            progress()
    except TaskFailures as failed:
        runs = ", ".join(sorted(failed.failures))
        raise RuntimeError(
            f"Image preparation failed for {runs}. The {len(reviews)} prepared runs are kept, "
            "and Train prepares only the failed runs again.\n" + str(failed)) from failed


def prepare_training_from_completed_build(snapshot, *, series_root, series_id, output,
        actor, on_progress=None, reader=None, model_output_parent=None, materializer=None,
        request_producer=None, source_collector=None, projector=None, workers=1,
        training_template=None):
    """Freeze the real current RAW basis and reproject old labels before training.

    Called only by the GUI Start chain. Numerical fitting remains the existing
    Studio training process. All source reviews and prior model files remain
    unchanged; the new immutable request keeps their explicit ancestry.
    """
    from ..phase_training.annotation_basis_transfer import required_annotation_sources, transfer_bundle
    from ..phase_workflow import prepare_phase_training_request
    from ..series_native_context import read_series_native_binding
    from ..phase_training.scope import series_id as normalize_series
    from ..phase_saved_assets import VerifiedAssetReader
    from .current_calibration_selection import current_calibration_selection
    from .current_build_job import _metric_for_build
    from .current_build_preparation import materialize_current_build_review
    supplied_materializer = materializer
    reader = current_input_reader(reader)
    annotation_reader = VerifiedAssetReader()
    collector, project = source_collector or required_annotation_sources, projector or transfer_bundle
    materializer = materializer or materialize_current_build_review
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    selection = current_calibration_selection(series_root, actor=actor,
        reason="Retrain the selected phase model on the current prepared image basis", reader=reader)
    find_completed_build(series_root, series_id, [snapshot], selection, reader=reader)
    template = training_template or selected_training_template(series_root, reader=reader)
    import tomllib
    manifest = tomllib.loads(reader.path(selection["input_authority"]["current_series"]).read_text(encoding="utf-8-sig"))
    from ..image_normalisation import saved_choice, selected_corrections
    choice = (saved_choice(manifest, manifest["series_id"], manifest["runs"]["ids"])
              if manifest.get("image_normalisation") is not None else None)
    setup_basis = None
    if choice is not None and "color_method" in choice and selected_corrections(choice)["color_method"] != "saved_native":
        from .setup_image_preparation import completed_image_preparation, image_basis_reader
        setup_basis = completed_image_preparation(snapshot,
            selection["input_authority"]["current_series"], reader=reader)
        reader = image_basis_reader(reader, setup_basis)
    operations = _operations(snapshot)
    native = read_series_native_binding(series_root)
    native_pin = dict(path=str(native.path), sha256=native.sha256, size_bytes=native.path.stat().st_size)
    from ..native.raw_rig_source import execution_identity
    from ..native.full_photometry import ORDER
    identity = dict(template=template["request"], native_configuration=native_pin,
        raw_producer_execution=execution_identity(), correction_order=ORDER,
        series=selection["input_authority"]["current_series"], operations=operations)
    if setup_basis is not None:
        identity["setup_image_preparation"] = setup_basis["receipt"]
    _save(root / "input_basis.json", identity)
    source_checkpoint = root / "prepared_evidence.json"
    if source_checkpoint.is_file():
        saved = reader.json(asset(source_checkpoint))
        bundles = [tuple(Path(path) for path in pair) for pair in saved["bundles"]]
        for pair in saved["bundle_pins"]:
            for pin in pair:
                reader.path(pin)
        request = None
        for path in reversed(sorted((root / "requests").glob("*.json"))):
            pin = reader.json(asset(path))
            candidate = reader.json(pin)
            model = Path(candidate["output_root"])
            complete = False
            if (model / "summary.json").is_file():
                from .current_model_basis import require_current_model_basis
                try:
                    require_current_model_basis(series_root, selection, reader=reader, model_reference=model)
                    complete = True
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            started = any((model / name).exists() for name in
                ("raw_training_preparation.json", "training_started.json", "training_execution.json"))
            if complete or not started:
                request = dict(candidate, request_path=pin["path"], frozen_pairs=[
                    tuple(Path(bundle[key]["snapshot_path"]) for key in ("decisions", "combined_manifest"))
                    for bundle in candidate["bundles"]])
                break
        if request is None:
            request = (request_producer or prepare_phase_training_request)(
                output_parent=Path(model_output_parent or Path(template["root"]).parent),
                cache_parent=root / "training_cache", bundles=bundles, actor=actor, settings=saved["settings"])
            requests = root / "requests"
            requests.mkdir(exist_ok=True)
            _save(requests / (str(len(list(requests.glob("*.json")))).zfill(5) + ".json"), asset(request["request_path"]))
        return dict(saved, request=request, bundles=bundles,
            source_bundles=[tuple(Path(path) for path in pair) for pair in saved["source_bundles"]],
            selection=selection, reused_source_preparation=True)
    threads = _reading_threads(workers)

    def bundle_sources(index, bundle):
        decisions, combined = (reader.json(bundle[key]) for key in ("decisions", "combined_manifest"))
        kept = {run: value for run, value in decisions.get("decisions", {}).items() if run in operations}
        if not kept:
            return None
        decisions = dict(decisions, decisions=kept)
        combined = dict(combined, jobs=[row for row in combined.get("jobs", []) if row["run"] in kept])
        folder = root / "source_evidence" / str(index)
        folder.mkdir(parents=True, exist_ok=True)
        pins = dict(decisions=_save(folder / "decisions.json", decisions),
            combined_manifest=_save(folder / "combined.json", combined))
        required = collector(pins["decisions"], pins["combined_manifest"], reader=annotation_reader,
            source_native_configuration=native_pin, on_unbound="exclude")
        return dict(original=deepcopy(bundle), pins=pins, required=required)

    source_pairs, needs = [], {}
    for item in _in_order(bundle_sources, list(enumerate(template["bundles"])), threads):
        if item is None:
            continue
        for run, row in item["required"]["runs"].items():
            for value in row["acquisitions"]:
                value = deepcopy(value)
                value["acquisition"]["path"] = str(reader.resolve_path(value["acquisition"]["path"]))
                needs.setdefault(run, {})[value["acquisition"]["sha256"]] = value
        source_pairs.append(item)
    if not source_pairs:
        raise ValueError("The selected model has no existing reviewed labels for the active runs")
    from .current_image_preparation import prepare_image_photometry
    if setup_basis is not None:
        photometry = setup_basis
    else:
        photometry = (prepare_image_photometry(operations, choice,
            selection["input_authority"]["current_series"], root / "image_preparation",
            reader=reader, on_progress=on_progress) if choice is not None else None)
    def run_sources(run, operation):
        result = reader.json(operation)
        request = reader.json(result["raw_build_request"])
        baseline = dict(request["baseline"])
        same_path = lambda path: reader.resolve_path(path).resolve() == reader.resolve_path(baseline["path"]).resolve()
        saved = [row for row in manifest["image_selection"]["runs"][run]["selected"] if same_path(row["source_path"])]
        if len(saved) != 1:
            raise ValueError("The actual Build baseline has no unique saved acquisition time: " + run)
        selected = _training_selection(selection, run, baseline, list(needs.get(run, {}).values()),
            baseline_elapsed_h=saved[0]["elapsed_h"])
        run_root = root / "sources" / run
        run_root.mkdir(parents=True, exist_ok=True)
        selected_pin = _save(run_root / "selection.json", selected)
        checkpoint = run_root / "source_completed.json"
        if checkpoint.is_file():
            value = reader.json(asset(checkpoint))
            if value["operation_result"] != operation or value["selection"] != selected_pin:
                raise ValueError("Prepared training photographs belong to another Build or source selection")
            reader.path(value["manifest"])
            return run, value, None
        metric_path = run_root / "metric.json"
        metric = asset(metric_path) if metric_path.is_file() else _metric_for_build(operation, metric_path, reader)
        attempt = uuid.uuid4().hex[:12]
        arguments = (operation, metric, selected_pin, run_root / ("native-" + attempt),
                     run_root / ("review-" + attempt))
        if photometry is not None:
            arguments += (photometry["runs"][run],)
        return run, None, arguments

    reviews, tasks = {}, []
    for index, (run, review, arguments) in enumerate(_in_order(run_sources, list(operations.items()), threads)):
        if on_progress:
            on_progress(dict(state="training_source_preparation", run=run, position=index, total=len(operations)))
        if review is not None:
            reviews[run] = review
        else:
            tasks.append((run, arguments))
    if supplied_materializer is None and int(workers) != 1 and tasks:
        _prepare_runs_in_processes(tasks, workers, root, reviews, len(operations), on_progress)
    else:
        for run, arguments in tasks:
            args, extra = (arguments[:5], {"prepared_photometry": arguments[5]}) if len(arguments) == 6 else (arguments, {})
            if supplied_materializer is None:
                extra["leave_out_unaligned"] = True
            reviews[run] = materializer(*args, reader=reader, on_progress=on_progress, **extra)
            _save(root / "sources" / run / "source_completed.json", reviews[run])
    left_out = {run: {row["raw_sha256"]: row["reason"] for row in value["left_out_frames"]}
                for run, value in reviews.items() if value.get("left_out_frames")}
    if left_out and on_progress:
        names = ", ".join(f"{run} {Path(row['raw_path']).name}" for run, value in sorted(reviews.items())
                          for row in value.get("left_out_frames", []))
        on_progress(dict(state=f"Left out of training because the drift correction could not align them: {names}"))
    bundles, transfers = [], []
    for index, item in enumerate(source_pairs):
        result = project(item["pins"]["decisions"], item["pins"]["combined_manifest"],
            {run: value["manifest"] for run, value in reviews.items()},
            source_native_configuration=native_pin, reader=annotation_reader, on_unbound="exclude",
            **({"left_out": left_out} if left_out else {}))
        projected = result["decisions"]
        transfers.append(result["provenance"])
        combined = deepcopy(result["combined_manifest"])
        for job in combined["jobs"]:
            value = reviews[job["run"]]
            review_root = str(Path(value["manifest"]["path"]).parents[2])
            job.update(review_output_dir=review_root, analysis_source_review_root=review_root,
                       native_manifest_sha256=value["manifest"]["sha256"])
            job.pop("v5_raw_photometry", None)
        represented = {job["run"] for job in combined["jobs"]}
        combined["jobs"].extend(dict(
            job_id="current-build-" + run, run=run, dataset=normalize_series(series_id),
            candidate_id="current-build-" + run, review_output_dir=str(Path(value["manifest"]["path"]).parents[2]),
            analysis_source_review_root=str(Path(value["manifest"]["path"]).parents[2]),
            native_manifest_sha256=value["manifest"]["sha256"])
            for run, value in reviews.items() if run not in represented)
        folder = root / "projected_evidence" / str(index)
        folder.mkdir(parents=True, exist_ok=True)
        decision_pin = _save(folder / "decisions.json", projected)
        combined_pin = _save(folder / "combined.json", combined)
        bundles.append((Path(decision_pin["path"]), Path(combined_pin["path"])))
    settings = dict(template["settings"], model_scope="single-series", series=[normalize_series(series_id)],
        series_filter=[normalize_series(series_id)], parent_model=template["root"],
        current_build_training_source=dict(template_request=template["request"],
            selected_series=selection["input_authority"]["current_series"]))
    if int(workers) == 0:
        from ..execution import detect_host_resources
        requested = detect_host_resources()["cpu"]["physical_cores"]
    else:
        requested = max(1, int(workers))
    settings.update(stage_workers=requested, extraction_workers=requested, opencv_threads=1)
    _save(root / "annotation_transfers.json", dict(format="darsia-annotation-basis-transfers-v1",
        template=template["request"], transfers=transfers))
    source_bundles = [tuple(item["original"][role]["path"]
        for role in ("decisions", "combined_manifest")) for item in source_pairs]
    saved = dict(bundles=[[str(path) for path in pair] for pair in bundles],
        bundle_pins=[[asset(path) for path in pair] for pair in bundles], settings=settings,
        source_bundles=source_bundles, reviews=reviews, template=template["request"],
        phase_inference_performed=False)
    _save(source_checkpoint, saved)
    prepared = (request_producer or prepare_phase_training_request)(
        output_parent=Path(model_output_parent or Path(template["root"]).parent),
        cache_parent=root / "training_cache", bundles=bundles, actor=actor, settings=settings)
    (root / "requests").mkdir(exist_ok=True)
    _save(root / "requests" / "00000.json", asset(prepared["request_path"]))
    return dict(request=prepared, bundles=bundles,
                source_bundles=[tuple(Path(path) for path in pair) for pair in source_bundles],
                reviews=reviews, selection=selection,
                template=template["request"], phase_inference_performed=False)
