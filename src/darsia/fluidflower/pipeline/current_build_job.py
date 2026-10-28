"""Use completed GUI Build products in the existing local P1 worker."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from contextlib import contextmanager, nullcontext
import multiprocessing as mp
import queue
import socket
import time
import tomllib
import traceback
import uuid

from ..mass.n1_pipeline import DataReader, asset, _write_json as _write_bound_json

MODE = "current_build"
#: Job key: every connected machine may claim and run this job's runs.
SHARED_RUNS = "any_connected_machine"
CLAIM_FORMAT = "darsia-current-build-run-claim-v1"
SHARED_LEDGER_REASON = ("Other machines can share this job only when output-space reservations are kept "
                        "in the shared job folder. Choose this series' Cache folder in Start > Series folders "
                        "on every machine that joins.")
COMMON_PATHS_REASON = ("This job fits common colour paths between its stages, so it runs on its Build "
                       "machine only.")


def _write_json(path, document):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return _write_bound_json(path, document)


def _executing_machine(job):
    """The machine this process runs the job on: the Build machine unless shared."""
    return job.get("executing_machine") or job["target_machine"]


def claimable_runs(root, job):
    """Runs that no machine has claimed, finished or failed yet.

    A failed run leaves the other runs claimable; its failure stays recorded.
    """
    root = Path(root)
    return [row["run"] for row in job["inputs"]
            if not (root / "claims" / (row["run"] + ".json")).exists()
            and not (root / "failed" / (row["run"] + ".json")).exists()
            and not (root / "results" / row["run"] / "completed.json").exists()]


def _claim_run(job, run):
    """Claim one run for this machine; False when it is not claimable.

    The claim is a new exclusive file in the shared job folder that names the
    machine and process. It is never moved or deleted, so it keeps recording
    which machine ran the run.
    """
    from .job_queue import write_immutable
    from .worker import _identity
    root = Path(job["root"])
    if run not in claimable_runs(root, job):
        return False
    claim = dict(format=CLAIM_FORMAT, job_id=job["job_id"], run=run,
                 owner=_identity(_executing_machine(job) + "-cpu"),
                 claimed_utc=datetime.now(timezone.utc).isoformat())
    try:
        write_immutable(root / "claims" / (run + ".json"), claim)
    except FileExistsError:
        return False
    return True


def same_folder(first, second):
    """Whether two spellings name one folder (a mapped drive, a host name or an address).

    Equal after the storage aliases, or the same folder by the operating
    system's file identity. A folder this machine cannot open is not the same.
    """
    import ntpath
    import os
    from ..storage_paths import resolve_storage_path
    key = lambda path: ntpath.normcase(ntpath.normpath(str(resolve_storage_path(str(path)))))
    if key(first) == key(second):
        return True
    try:
        return os.path.samefile(first, second)
    except (OSError, ValueError):
        return False


def require_shared_machine(job, stores):
    """Refuse a machine that would save elsewhere or keep its own space ledger."""
    from darsia_cache.folders import folder_mode
    expected = Path(stores.roots(job["series_id"])["results"])
    actual = Path(job["results_root"]).parent
    if not same_folder(expected, actual):
        raise ValueError(f"This machine saves {job['series_id']} results in {expected}, but this job's "
                         f"results are in {actual}. Choose the same Results folder in Start > Series folders.")
    if not folder_mode(stores, job["series_id"]):
        raise ValueError(SHARED_LEDGER_REASON)


def _space_estimate(job, row):
    """Retain the existing full-output estimate and RAW preparation allowance."""
    from ..storage_profile import estimate_job_bytes
    from .capacity import StorageProfileEstimate
    reader = DataReader()
    result = reader.json(row["operation_result"])
    certificate = reader.json(result["baseline_renderer"])
    height, width = certificate["renderer"]["native_shape_hw"]
    estimate = estimate_job_bytes("full", row["selected_frames"],
        extra_frame_bytes=height * width * 65, extra_run_bytes=height * width * 65)
    return StorageProfileEstimate(row["run"], row["selected_frames"], "full", "keep",
        job["pipeline"], estimate["per_frame_bytes"], estimate["per_run_bytes"],
        estimate["inflight_bytes"], estimate, [], job["commit"],
        model="Full output estimate plus uncompressed current RAW and geometry preparation")


def worker_recommendation(job, stores, *, rows=None, inventory=None, active_pids=()):
    """Fresh host resources with the existing conservative worker-memory policy."""
    from ..execution import detect_host_resources
    from ..resource_advisor_panel import resource_advice
    from .capacity import CapacityLedger
    rows = list(job["inputs"] if rows is None else rows)
    inventory = deepcopy(detect_host_resources() if inventory is None else inventory)
    # Honour process affinity as well as physical CPU count.
    import psutil
    try:
        affinity = len(psutil.Process().cpu_affinity())
    except (AttributeError, NotImplementedError, psutil.Error):
        affinity = inventory["cpu"].get("logical_cores")
    if affinity:
        for field in ("physical_cores", "logical_cores"):
            if inventory["cpu"].get(field):
                inventory["cpu"][field] = min(inventory["cpu"][field], affinity)
    # Keep the unused part of each live worker's 20 GiB policy allocation
    # reserved even during its initially small imports/metadata phase.
    headroom = 0
    for pid in active_pids:
        try:
            rss = psutil.Process(pid).memory_info().rss
        except psutil.Error:
            rss = 0
        headroom += max(0, 20 * 1024**3 - rss)
    available_ram = inventory.get("memory", {}).get("available_bytes")
    if isinstance(available_ram, int):
        inventory["memory"]["available_bytes"] = max(0, available_ram - headroom)
    advice = resource_advice(dict(machine=stores.machine_id, inventory=inventory), job, "cpu", 1,
                             policy_budget=True, run_limit=len(rows))
    ledger = CapacityLedger(stores, job["series_id"], Path(job["results_root"]), job_root=job["root"])
    volume = ledger.fresh_volume()
    available = max(0, volume["free_bytes"] - ledger.reserve)
    # The same ledger prevents independent services from spending this space.
    # A fresh ledger need not be created merely to display a recommendation.
    try:
        with ledger.locked(create=False):
            available = max(0, available - ledger.remaining())
    except (OSError, ValueError):
        pass  # Runtime admission initializes/checks the existing ledger.
    estimates = sorted((_space_estimate(job, row).needed(reserve_bytes=0) for row in rows), reverse=True)
    disk_limit, reserved = 0, 0
    for needed in estimates:
        if reserved + needed > available:
            break
        reserved += needed
        disk_limit += 1
    available_cores = max(0, advice.get("core_limit", 0) - len(active_pids))
    workers = min(advice["workers"] or 0, disk_limit, available_cores)
    message = (f"Recommended: {workers} parallel run worker(s), one numerical thread per run. "
               + advice["reason"] + f" Disk allows {disk_limit} simultaneous run estimate(s). "
               + (f"{len(active_pids)} active workers retain their CPU/RAM allocations. " if active_pids else "")
               + "The worker rechecks current resources before starting each run.")
    return dict(advice, workers=workers, recommended_workers=workers, disk_limit=disk_limit,
        available=True, sufficient=workers > 0, message=message, free_bytes=volume["free_bytes"],
        results_root=job["results_root"], reason=message)


@contextmanager
def _capacity(stores, job, row, output, owner):
    """Reserve concurrent runs under the existing host-volume lock."""
    from types import SimpleNamespace
    from .capacity import CapacityLedger, report
    saved = _space_estimate(job, row)
    ledger = CapacityLedger(stores, job["series_id"], output.parent, job_root=job["root"])
    token = uuid.uuid4().hex
    task = dict(run=row["run"], attempt_id=token, ordered_case_ids=list(row["ordered_case_ids"]))
    with ledger.locked():
        volume = ledger.fresh_volume()
        remaining = ledger.remaining()
        spendable = dict(volume, free_bytes=max(0, volume["free_bytes"] - remaining))
        facts = report(saved, spendable, remaining, ledger.reserve)
        if not facts["sufficient"]:
            raise ValueError(facts["message"])
        reservation = ledger.bind(job["root"], SimpleNamespace(task=task, token=token), owner, saved)
    try:
        yield
    finally:
        with ledger.locked():
            ledger.release(reservation, owner)


def _depth_measurements(operation, reader):
    from darsia.presets.workflows.config.utils import _get_section_from_toml
    from .current_inputs import current_input_reader
    reader = current_input_reader(reader)
    result = reader.json(operation)
    paths = [reader.path(bound) for bound in result["configuration_files"]]
    depth = _get_section_from_toml(paths, "depth")
    path = Path(depth["measurements"])
    if not path.is_absolute():
        raise ValueError("The selected Build must resolve its explicit depth measurements path")
    # The bound configuration keeps its original locator after a project move.
    # Resolve the user's declared storage alias before pinning the same source
    # bytes for the new P1 job, just as for the surrounding Build inputs.
    bound = asset(reader.resolve_path(path))
    for original in result["configuration_files"]:
        reader.path(original)
    return bound


def _metric_for_build(operation, destination, reader):
    """Keep the measured camera metric and bind the actual new Build grid."""
    return _write_json(destination, _build_metric(operation, reader))


def _build_metric(operation, reader):
    """The measured camera metric bound to the actual new Build grid, not yet saved."""
    result = reader.json(operation)
    request = reader.json(result["raw_build_request"])
    source = request["native_product"]["source_model"]
    model = reader.json(source)
    certificate = reader.json(result["baseline_renderer"])
    if (model["run"] != result["run"]
            or model["reference_image"]["sha256"] != request["baseline"]["sha256"]):
        raise ValueError("The measured model and completed Build identify different RAW baselines")
    model = deepcopy(model)
    import numpy as np
    h = certificate["H_rig_corrected_native_to_baseline_camera_u"]
    grid = model["baseline_native_grid"]
    grid.update(renderer=deepcopy(result["baseline_renderer"]),
        H_rig_corrected_native_to_baseline_camera_u=deepcopy(h),
        H_baseline_camera_u_to_rig_corrected_native=np.linalg.inv(h).tolist(),
        shape_hw=deepcopy(certificate["renderer"]["native_shape_hw"]))
    model["reference_image"] = deepcopy(request["baseline"])
    model["current_build_source"] = dict(measured_model=source, operation_result=operation)
    return model


#: Each purpose's job name: which saved photographs phase one is prepared for.
PURPOSES = {"calibration": "Generate phase maps from model",
            "production": "Generate production phase maps from model"}


def _build_baseline(operation, reader):
    """The RAW descriptor a completed Build was made from (its request's baseline)."""
    return reader.json(reader.json(operation)["raw_build_request"])["baseline"]


#: Production never fits fresh colour paths on production photographs.
PRODUCTION_PATHS_REASON = ("Production phase maps use the colour paths from Setup's image correction. "
                           "Complete image correction in Setup first.")


def _optical_bindings(snapshot, authority, reader):
    """The job's colour-path keys: Setup's colour paths, a fresh fit, or none."""
    current = (tomllib.loads(reader.path(authority).read_text(encoding="utf-8-sig")) if authority else {})
    from ..image_normalisation import selected_corrections
    choice = current.get("image_normalisation") or {}
    if "color_method" not in choice or selected_corrections(choice)["color_method"] == "saved_native":
        return {}
    from .setup_image_preparation import completed_image_preparation, completed_setup_colour_paths
    image_basis = (completed_image_preparation(snapshot, authority, reader=reader)
                   if any(row.get("operation") == "image-preparation" for row in snapshot.get("runs", [])) else {})
    if image_basis.get("colour_paths"):
        paths = completed_setup_colour_paths(snapshot, authority, reader=reader)
        return dict(setup_colour_paths=deepcopy(paths["receipt"]),
                    setup_image_basis=deepcopy(image_basis["receipt"]))
    # Historical completed Builds retain their original P1 producer.
    from .current_optical_preparation import DEFAULT_SAMPLING, FRAME_SELECTION_METHOD, colour_path_image_count
    return dict(fresh_optical_paths=dict(format="darsia-current-optical-preparation-v1",
        sampling=deepcopy(DEFAULT_SAMPLING), frame_selection=dict(method=FRAME_SELECTION_METHOD,
            max_images_per_run=colour_path_image_count(current))))


def _require_shared_inputs(selection, optical, reader):
    """Refuse a shared job whose runs would need one machine or its own disk.

    Each run reads its statistics inputs at its start; they are read here
    through the shared-only ``reader`` first.
    """
    if "fresh_optical_paths" in optical:
        raise ValueError(COMMON_PATHS_REASON)
    from .current_statistics_recipe import retained_statistics_recipe
    for row in selection["runs"]:
        retained_statistics_recipe(selection, row["run"], reader=reader,
                                   fresh_images="setup_colour_paths" in optical)


def prepare_current_build_job(snapshot, *, series_root, series_id, stores_path=None,
                              machine=None, reader=None, actor=None, model_reference=None,
                              purpose="calibration", on_progress=None, any_connected_machine=False):
    """Prepare metadata from the selected completed Build, never launch work.

    ``purpose`` chooses the photographs: "calibration" (the default) uses the
    saved calibration selection; "production" uses the saved production image
    grid after each run's Build baseline, reading each photograph's SHA-256
    once (``on_progress`` receives that count). Both use the same method.

    ``any_connected_machine`` (default False: only the Build machine runs the
    job) lets every connected machine claim runs. Every input must then be
    readable from the shared folders: preparation reads only shared inputs
    and checks each run's statistics inputs, so a run never needs a file on
    the Build machine's own disk.
    """
    from ..analysis_preparation import job_identity
    from ..store_map import read_store_map
    from .current_calibration_selection import save_current_calibration_selection
    from .current_inputs import current_input_reader
    if purpose not in PURPOSES:
        raise ValueError("Choose phase maps for calibration or for production")
    if type(any_connected_machine) is not bool:
        raise ValueError("Choose explicitly whether other connected machines may run this job")
    reader = current_input_reader(reader, shared_only=any_connected_machine)
    if (snapshot.get("job_facts", {}).get("pipeline") != "series-setup"
            or snapshot.get("complete") is not True):
        raise ValueError("Select the completed Apply and Build job for this series")
    from .current_model_basis import require_current_model_basis
    model_binding = require_current_model_basis(series_root, reader=reader,
        model_reference=model_reference, build_snapshot=snapshot)
    identity = job_identity(Path(__file__).resolve().parents[4])
    stores = read_store_map(stores_path)
    roots = stores.roots(series_id)
    results = Path(roots["results"])
    from ..job_filesystem import inspect_results_root
    admission = inspect_results_root(results)
    if not admission.allowed:
        raise ValueError("P1 cannot use the configured results store: " + admission.reason)
    if any_connected_machine:
        from darsia_cache.folders import folder_mode
        if not folder_mode(stores, series_id):
            raise ValueError(SHARED_LEDGER_REASON)
    completed = {row["run"]: row["result"] for row in snapshot["runs"]
                 if row.get("operation") == "build" and row.get("state") == "Complete"}
    operator = actor or machine or socket.gethostname()
    production, optical, metrics = None, None, {}
    if purpose == "production":
        from .current_production_selection import current_production_selection
        production = current_production_selection(series_root,
            {run: _build_baseline(result["files"]["operation_result.json"], reader)
             for run, result in completed.items()}, actor=operator,
            reason="Use the saved production image grid, completed Build and the selected method",
            reader=reader, on_progress=on_progress)
        production["input_authority"]["selected_model"] = deepcopy(model_binding)
        # Every production refusal comes before a job folder exists.
        require_current_model_basis(series_root, production, reader=reader,
            model_reference=model_reference, build_snapshot=snapshot)
        optical = _optical_bindings(snapshot, production["input_authority"].get("current_series"), reader)
        if "fresh_optical_paths" in optical:
            raise ValueError(PRODUCTION_PATHS_REASON)
        metrics = {row["run"]: _build_metric(completed[row["run"]]["files"]["operation_result.json"], reader)
                   for row in production["runs"]}
        if any_connected_machine:
            _require_shared_inputs(production, optical, reader)
    job_id = "build-p1-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    root = Path(snapshot["root"]).parent / job_id
    root.mkdir(parents=True, exist_ok=False)
    (root / "inputs").mkdir()
    if production is None:
        selection = save_current_calibration_selection(series_root, root / "inputs" / "selection.json",
            actor=operator,
            reason="Use the saved chart, completed Build and original calibration acquisition history",
            reader=reader, model_binding=model_binding)
    else:
        selection = _write_json(root / "inputs" / "selection.json", production)
    selected = reader.json(selection)
    if production is None:
        require_current_model_basis(series_root, selected, reader=reader,
            model_reference=model_reference, build_snapshot=snapshot)
    inputs = []
    for selected_run in selected["runs"]:
        run = selected_run["run"]
        if run not in completed:
            raise ValueError("The selected Build job has no completed operation for " + run)
        operation = completed[run]["files"]["operation_result.json"]
        destination = root / "inputs" / (run + "-metric.json")
        metric = (_write_json(destination, metrics[run]) if run in metrics
                  else _metric_for_build(operation, destination, reader))
        inputs.append(dict(run=run, operation_result=operation, metric_model=metric,
            selected_frames=selected_run["counts"]["selected_frames"],
            ordered_case_ids=[run + "-" + frame["raw_sha256"][:20]
                              for frame in selected_run["selected_frames"]]))
    job = dict(pipeline="native-phase-statistics", execution_mode=MODE, job_id=job_id,
        commit=identity["commit"], environment_lock=identity["environment_lock"], precision="float32",
        expected_runs=[row["run"] for row in inputs], inputs=inputs, selection=selection,
        series_id=series_id, series_root=str(series_root), results_root=str(results / job_id),
        target_machine=machine or socket.gethostname(), created_utc=datetime.now(timezone.utc).isoformat())
    if purpose != "calibration":
        job["purpose"] = purpose
    if any_connected_machine:
        job[SHARED_RUNS] = True
    if optical is None:
        optical = _optical_bindings(snapshot, selected.get("input_authority", {}).get("current_series"), reader)
        if any_connected_machine:
            _require_shared_inputs(selected, optical, reader)
    job.update(optical)
    _write_json(root / "job.json", job)
    _write_json(root / "ui" / "master.json", dict(format="darsia-job-ui-master-v1",
        job=asset(root / "job.json"), name=PURPOSES[purpose], series_label=snapshot.get("series", series_id),
        machine=job["target_machine"], created_utc=job["created_utc"]))
    return dict(root=str(root), name=PURPOSES[purpose], active=True,
                ui_series_root=str(series_root), ui_series_id=series_id)


def phase_configuration(prepared, selection, statistics):
    """Join real per-frame native/geometry outputs to retained statistics inputs."""
    import cv2
    import numpy as np
    from .staged_data import PHASE1_FORMAT
    from ..mass.n1_pipeline import _dose
    reader = DataReader()
    data, recipe = prepared["configuration"], prepared["phase_recipe"]
    phase = recipe["phase"]
    chosen = next(row for row in selection["runs"] if row["run"] == data["run"])
    reference = prepared["acquisitions"][data["reference_spec"]["case_id"]]
    template = deepcopy(statistics["statistics"])
    template["physical_correction"] = deepcopy(prepared["operation_result"])
    template["shared"].update(floor_curve=deepcopy(phase["chain_assets"]["floor"]),
        floor_origin_xy=deepcopy(phase["parameters"]["geometry"]["floor_source_origin_xy"]))
    from .local_build import LocalBuildInputs
    probe = LocalBuildInputs(dict(run=data["run"], native=data["native"], phase=data["phase"],
        frames=[], statistics=template), operation_result=prepared["operation_result"],
        metric_model=prepared["metric_model"], frame_receipts={})
    # These are the actual adopted source measurements, not old physical rasters.
    template["shared"]["porosity_by_facies"] = {str(k): v["porosity"] for k, v in probe.roles.items()}
    template["shared"]["unquantified_facies"] = [k for k, v in probe.roles.items()
        if v["analysis_role"] in {"excluded", "colorchecker"}]
    template["dimensions_m"] = deepcopy(probe.context.grid["dimensions"])
    template["shared"]["depth_points"] = _depth_measurements(prepared["operation_result"], reader)
    template["atmosphere_series"] = deepcopy(statistics["atmosphere"])
    template["dose"] = deepcopy(statistics["dose"])
    fields = dict(source_to_native=phase["geometry_fields"]["reference_to_frame"],
        original_domain_to_native="source_to_native_h", carried_homography_provenance="source_to_native_h")
    template["geometry_fields"] = fields
    frames = []
    for spec, original in zip(data["frames"], chosen["selected_frames"]):
        spec = deepcopy(spec)
        spec.pop("raw_acquisition", None)
        spec["native_task"] = deepcopy(prepared["produced"][spec["case_id"]].native_task)
        geometry = reader.json(spec["geometry"])
        mask = dict(geometry["build_geometry"]["fields"], array_key="top_keep")
        outer = reader.array(mask)
        contours, _ = cv2.findContours(outer.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1:
            raise ValueError("The saved physical frame must define one connected outer boundary")
        polygon = contours[0][:, 0, :].astype(float) + np.asarray(geometry["native_origin_xy"])
        if len(polygon) < 3:
            raise ValueError("The physical frame boundary has fewer than three vertices")
        h = np.asarray(geometry[fields["source_to_native"]], float).copy()
        shift = np.eye(3)
        shift[:2, 2] = -np.asarray(geometry["native_origin_xy"])
        membership = dict(geometry["build_geometry"]["fields"], array_key="rig_domain")
        spec["mass_spec"] = dict(atmosphere=None, inner_polygon_native=polygon.tolist(),
            original_domain=dict(membership=membership, native_shape_hw=geometry["phase_shape_hw"],
                frame_index=spec["frame_index"], source_to_native_h=(shift @ h).tolist()),
            photo_top=mask, source_receipt=deepcopy(spec["geometry"]),
            nominal_injected_mass_g=_dose(template, spec),
            selected_target_h=original["nominal_fit_target_h"], light_epoch=statistics["light_epoch"],
            provenance=dict(fit_candidate=original["fit_candidate"],
                            original_nominal_fit_target_h=original["nominal_fit_target_h"]))
        frames.append(spec)
    return dict(format=PHASE1_FORMAT, run=data["run"], precision="float32", native=data["native"],
        phase=dict(configuration=data["phase"]["configuration"],
            reference={key: deepcopy(reference[key]) for key in ("full", "native_row")}
                      | {"source_sha256": reference["source"]["source_sha256"]},
            initial_previous=deepcopy(reference)), statistics=template,
        frames=frames, ordered_case_ids=[row["case_id"] for row in frames],
        history_policy=data["history_policy"], selection_index_field=data["selection_index_field"],
        gas_probability_storage=dict(format="darsia-decision-probability-storage-request-v1",
            encoding="float32-exact-decision-values"),
        storage=dict(format="darsia-storage-output-request-v1", resolution_mm="native", gas_score_encoding="uint8-la-q255-v1"),
        provenance=deepcopy(data.get("provenance", {})))


def _run_current_build_run(job, row, commit, stores_path, progress, *, capacity_reserved=False):
    """One isolated run; its RAW preparation and P1 history stay sequential."""
    from .worker import _identity
    from .current_build_preparation import prepare_current_build_acquisitions
    from .current_statistics_recipe import retained_statistics_recipe
    from .phase_statistics import run_phase1_local, _phase1_thread_runtime
    from .local_build import LocalBuildInputs
    from ..store_map import read_store_map
    stores = read_store_map(stores_path)
    expected_root = Path(stores.roots(job["series_id"])["results"]) / job["job_id"]
    if job.get(SHARED_RUNS) is True:
        require_shared_machine(job, stores)  # Another machine may spell the same folder differently.
        expected_root = Path(job["results_root"])
    elif expected_root != Path(job["results_root"]):
        raise ValueError("The configured local results root changed after preparation")
    from .current_inputs import current_input_reader
    reader = current_input_reader(shared_only=job.get(SHARED_RUNS) is True)
    selection = reader.json(job["selection"])
    if asset(Path(job["series_root"]) / "series.toml") != selection["input_authority"]["current_series"]:
        raise ValueError("The saved series changed after P1 preparation; prepare P1 again")
    run = row["run"]
    owner = _identity(_executing_machine(job) + "-cpu")
    setup_paths = job.get("setup_colour_paths")
    statistics = retained_statistics_recipe(selection, run, reader=reader, fresh_images=bool(setup_paths))
    prepared_photometry = None
    if setup_paths:
        from .setup_colour_paths import FORMAT as COLOUR_PATHS_FORMAT, colour_path_packages
        image_basis = reader.json(job["setup_image_basis"])
        prepared_photometry = image_basis["runs"][run]
        paths = reader.json(setup_paths)
        package = deepcopy(colour_path_packages(paths, reader=reader, require_per_run=True, run=run)[run])
        metadata = reader.json(package["metadata"])
        if (paths.get("format") != COLOUR_PATHS_FORMAT or paths.get("complete") is not True
                or paths["operations"][run] != row["operation_result"]
                or paths["input"]["image_basis"][run] != prepared_photometry
                or metadata["flash_bounds"] != statistics["flash_bounds"]):
            raise ValueError("Setup colour paths differ from this P1 source, image basis or flash convention")
    with (nullcontext() if capacity_reserved else _capacity(stores, job, row, expected_root / run, owner)):
        prepared = _phase1_thread_runtime(1, lambda _runtime: prepare_current_build_acquisitions(
            row["operation_result"], row["metric_model"], job["selection"],
            expected_root / run / "sources", on_progress=progress, reader=reader,
            **({"prepared_photometry": prepared_photometry} if prepared_photometry is not None else {})))
        if setup_paths:
            from .current_clearwater import prepare_current_clearwater
            clear_source = paths.get("input", {}).get("optical_sources", {}).get(run, {}).get(
                "clearwater_source", statistics["clearwater_source"])
            clear = prepare_current_clearwater(prepared, clear_source,
                                              expected_root / run / "clearwater", reader=reader)
            statistics["statistics"]["shared"].update(clearwater=clear["clearwater"], optics=package)
        config = phase_configuration(prepared, selection, statistics)
        prepared.pop("bridge", None)
        if config.get("ordered_case_ids") is not None and config["ordered_case_ids"] != row["ordered_case_ids"]:
            raise ValueError("Prepared P1 run differs from its selected acquisition order")
        local = LocalBuildInputs(config, operation_result=row["operation_result"],
            metric_model=row["metric_model"],
            frame_receipts={spec["case_id"]: prepared["produced"][spec["case_id"]].receipt
                            for spec in config["frames"]}, reader=reader)
        result = run_phase1_local(config, expected_root / run / "phase1", local_build=local,
            code_commit=commit, threads=1, on_progress=progress, prepared_native=prepared["prepared_native"])
        del prepared, config, local
    return result


def _execute_run_child(job, row, commit, stores_path, messages, stop, runner=None):
    """Only the parent writes the service log and publishes closed results."""
    from .worker import _identity, _job_control_boundary, _JobControlBoundary
    owner = _identity(_executing_machine(job) + "-cpu")
    count = 0
    work_stage = {"_prepare_optical_run": "image-preparation", "_infer_optical_run": "phase-inference",
                  "_finish_optical_run": "optical-statistics"}.get(getattr(runner, "__name__", ""))
    def progress(event):
        nonlocal count
        if event.get("run") != row["run"]:
            raise ValueError("A run worker reported another acquisition history")
        if event.get("state") in {"phase1_frame_completed", "optical_phase_frame_completed"}:
            count += 1
        if work_stage is not None:
            event = dict(event, work_stage=work_stage)
        messages.put(dict(kind="progress", completed_count=count, actual_worker=owner, **event))
        _job_control_boundary(Path(job["root"]))
        if stop.is_set():
            raise _JobControlBoundary("control_stopped")
    try:
        _job_control_boundary(Path(job["root"]))
        if stop.is_set():
            raise _JobControlBoundary("control_stopped")
        progress(dict(run=row["run"], state="preparing_sources"))
        result = (runner(job, row, commit, stores_path, progress) if runner is not None else
            _run_current_build_run(job, row, commit, stores_path, progress, capacity_reserved=True))
        messages.put(dict(kind="run_result", run=row["run"], state="complete", result=result, actual_worker=owner))
    except _JobControlBoundary as error:
        messages.put(dict(kind="run_result", run=row["run"], state=error.state, actual_worker=owner))
    except BaseException:
        messages.put(dict(kind="run_result", run=row["run"], state="failed",
                          traceback=traceback.format_exc(), actual_worker=owner))


def _optical_stage(job, stage, run, reader):
    return reader.json(asset(Path(job["root"]) / stage / run / "completed.json"))


def _check_optical_job_inputs(job, stores_path, reader):
    """Retain the same current-series and output-store checks as ordinary P1."""
    from ..store_map import read_store_map
    stores = read_store_map(stores_path)
    expected = Path(stores.roots(job["series_id"])["results"]) / job["job_id"]
    if expected != Path(job["results_root"]):
        raise ValueError("The configured local results root changed after preparation")
    selection = reader.json(job["selection"])
    if asset(Path(job["series_root"]) / "series.toml") != selection["input_authority"]["current_series"]:
        raise ValueError("The saved series changed after P1 preparation; prepare P1 again")
    return selection


def _prepare_optical_run(job, row, commit, stores_path, progress):
    """Decode each RAW once and produce clear water with those same corrections."""
    from .current_build_preparation import prepare_current_build_acquisitions
    from .current_statistics_recipe import retained_statistics_recipe
    from .current_clearwater import prepare_current_clearwater
    from .current_inputs import current_input_reader
    from .phase_statistics import _phase1_thread_runtime
    from .local_build import LocalBuildInputs
    reader = current_input_reader()
    selection = _check_optical_job_inputs(job, stores_path, reader)
    output = Path(job["results_root"]) / row["run"]
    statistics = retained_statistics_recipe(selection, row["run"], reader=reader, fresh_images=True)
    prepared = _phase1_thread_runtime(1, lambda _runtime: prepare_current_build_acquisitions(
        row["operation_result"], row["metric_model"], job["selection"],
        output / "sources", on_progress=progress, reader=reader))
    clear = prepare_current_clearwater(prepared, statistics["clearwater_source"],
                                       output / "clearwater", reader=reader)
    statistics["statistics"]["shared"]["clearwater"] = clear["clearwater"]
    config = phase_configuration(prepared, selection, statistics)
    receipts = {spec["case_id"]: prepared["produced"][spec["case_id"]].receipt for spec in config["frames"]}
    local = LocalBuildInputs(config, operation_result=row["operation_result"],
                            metric_model=row["metric_model"], frame_receipts=receipts, reader=reader)
    baseline = prepared["configuration"]["reference_spec"]["case_id"]
    basis = dict(adopted_revision=local.context.provenance["adopted_revision"],
        build_context=row["operation_result"], phase_configuration=config["phase"]["configuration"],
        native_configuration=config["native"]["configuration"],
        baseline=prepared["acquisitions"][baseline]["source"]["source_binding"]["raw_acquisition"],
        clearwater=clear["raw"], profile=prepared["produced"][baseline].native_task["processor"]["raw_binding"])
    data = dict(configuration=config, prepared_native=prepared["prepared_native"],
                frame_receipts=receipts, operation_result=row["operation_result"], metric_model=row["metric_model"])
    source_rows = {spec["case_id"]: spec["source"]["source_binding"]["raw_acquisition"] for spec in config["frames"]}
    return dict(format="darsia-current-optical-input-run-v1", complete=True, run=row["run"],
        preparation=_write_json(output / "optical_inputs.json", data), basis=basis,
        roles=[dict(facies_id=key, name=value["name"], analysis_role=value["analysis_role"])
               for key, value in sorted(local.roles.items())], source_rows=source_rows,
        flash_bounds=statistics["flash_bounds"], clearwater_receipt=clear["receipt"])


def _infer_optical_run(job, row, commit, stores_path, progress):
    from .current_inputs import current_input_reader
    from .local_build import LocalBuildInputs
    from .phase_statistics import run_optical_phase_sources
    from ..mass.pooled_optical_paths import recipe_from_document
    reader = current_input_reader()
    _check_optical_job_inputs(job, stores_path, reader)
    source = _optical_stage(job, "optical-inputs", row["run"], reader)
    prepared = reader.json(source["preparation"])
    config = prepared["configuration"]
    local = LocalBuildInputs(config, operation_result=prepared["operation_result"],
        metric_model=prepared["metric_model"], frame_receipts=prepared["frame_receipts"], reader=reader)
    recipe = recipe_from_document(reader.json(asset(Path(job["root"]) / "inputs" / "optical_recipe.json")))
    return run_optical_phase_sources(config, Path(job["results_root"]) / row["run"] / "phase1",
        local_build=local, code_commit=commit, optical_recipe=recipe,
        prepared_native=prepared["prepared_native"], on_progress=progress)


def _finish_optical_run(job, row, commit, stores_path, progress):
    from .current_inputs import current_input_reader
    from .local_overflow_sources import SavedLocalBuildInputs
    from .phase_statistics import finish_optical_phase_statistics
    reader = current_input_reader()
    _check_optical_job_inputs(job, stores_path, reader)
    source = _optical_stage(job, "optical-inputs", row["run"], reader)
    prepared = reader.json(source["preparation"])
    config = prepared["configuration"]
    config["statistics"]["shared"]["optics"] = reader.json(
        asset(Path(job["root"]) / "inputs" / "current_optics.json"))
    local = SavedLocalBuildInputs(config, operation_result=prepared["operation_result"],
        metric_model=prepared["metric_model"], frame_receipts=prepared["frame_receipts"], reader=reader)
    phases = _optical_stage(job, "optical-phases", row["run"], reader)
    return finish_optical_phase_statistics(config, Path(job["results_root"]) / row["run"] / "phase1",
        phases, local_build=local, code_commit=commit, reader=reader, on_progress=progress)


def _run_fresh_optical_job(job, commit, stores_path, workers, log):
    """One ordinary job: input preparation, inference, pooled paths, statistics."""
    from .current_inputs import current_input_reader
    from .current_optical_preparation import DEFAULT_IMAGES_PER_RUN, recipe_for_runs, load_samples
    from ..mass.pooled_optical_paths import save_common_path_package
    from .worker import _job_control_boundary, emit
    reader = current_input_reader()
    for runner, stage in ((_prepare_optical_run, "optical-inputs"), (_infer_optical_run, "optical-phases")):
        if stage == "optical-phases":
            rows = [_optical_stage(job, "optical-inputs", row["run"], reader) for row in job["inputs"]]
            recipe = recipe_for_runs(rows, reader.json(job["selection"]), Path(job["root"]) / "inputs",
                sampling=job["fresh_optical_paths"]["sampling"],
                max_images_per_run=job["fresh_optical_paths"].get("frame_selection", {}).get(
                    "max_images_per_run", DEFAULT_IMAGES_PER_RUN))
            _write_json(Path(job["root"]) / "inputs" / "optical_recipe.json", recipe.document)
        code = _dispatch_runs(job, commit, stores_path, workers, log, runner=runner,
                              publication_folder=stage, completion_event=stage + "_completed")
        if code:
            return code
        _job_control_boundary(Path(job["root"]))
    def samples():
        for run in recipe.document["fit_runs"]:
            result = _optical_stage(job, "optical-phases", run, reader)
            if result["recipe_sha256"] != recipe.sha256:
                raise ValueError("Optical samples were produced for another image basis")
            for binding in result["frames"]:
                _job_control_boundary(Path(job["root"]))
                record = reader.json(binding)
                if record["samples"] is not None:
                    yield load_samples(record["samples"], reader)
    emit(log, dict(kind="fitting_current_optical_paths", fit_runs=recipe.document["fit_runs"]))
    package = save_common_path_package(Path(job["results_root"]) / "optical_paths", recipe, samples())
    _write_json(Path(job["root"]) / "inputs" / "current_optics.json", package)
    _job_control_boundary(Path(job["root"]))
    return _dispatch_runs(job, commit, stores_path, workers, log, runner=_finish_optical_run)


def _dispatch_runs(job, commit, stores_path, requested_workers, log, *, runner=None,
                   recommendation=None, capacity_factory=None, publication_folder="results",
                   completion_event="run_completed", claim=None):
    """Spawn each independent run once and wait for its OS process to close.

    A failed run leaves the other runs running; the result is 1 when any run
    failed. ``claim``, for a job that several machines share, is called
    before each run starts; a run it refuses (taken by another machine) is
    skipped.
    """
    from .worker import emit, _job_control_boundary, _JobControlBoundary
    from ..store_map import read_store_map
    context = mp.get_context("spawn")
    messages, stop = context.Queue(), context.Event()
    pending, active, terminal = list(job["inputs"]), {}, {}
    failed, halted, last_heartbeat = False, False, time.monotonic()
    recommend = recommendation or (lambda rows, active: worker_recommendation(job, read_store_map(stores_path),
        rows=rows, active_pids=[process.pid for process in active.values()]))
    def reserve(row):
        from .worker import _identity
        return _capacity(read_store_map(stores_path), job, row,
                         Path(job["results_root"]) / row["run"], _identity(_executing_machine(job) + "-cpu"))
    capacity_factory = capacity_factory or reserve
    reservations = {}
    def drain(block=False):
        while True:
            try:
                event = messages.get(timeout=.2) if block else messages.get_nowait()
            except queue.Empty:
                return
            block = False
            if event.get("run") not in active:
                raise ValueError("A worker reported a run outside its active assignment")
            if event["kind"] == "run_result":
                terminal[event["run"]] = event
            else:
                emit(log, event)
    try:
        while pending or active:
            try:
                _job_control_boundary(Path(job["root"]))
            except _JobControlBoundary as error:
                if not halted:
                    emit(log, dict(kind=error.state, message=str(error)))
                halted = True
                stop.set()
            if pending and not halted and len(active) < requested_workers:
                advice = recommend(pending, active)
                slots = min(requested_workers - len(active), advice["workers"] or 0, len(pending))
                if not slots and not active:
                    raise ValueError(advice["reason"])
                for _ in range(slots):
                    row = None
                    while pending and row is None:
                        candidate = pending.pop(0)
                        row = candidate if claim is None or claim(candidate) else None
                    if row is None:
                        break
                    run = row["run"]
                    reservation = capacity_factory(row)
                    reservation.__enter__()
                    reservations[run] = reservation
                    process = context.Process(target=_execute_run_child,
                        args=(job, row, commit, stores_path, messages, stop, runner), name="P1-" + run)
                    try:
                        process.start()
                    except BaseException:
                        reservations.pop(run).__exit__(None, None, None)
                        raise
                    active[run] = process
                if slots:
                    emit(log, dict(kind="parallel_workers", requested_workers=requested_workers,
                        active_workers=len(active), reason=advice["reason"]))
            if not active:
                break
            drain(block=True)
            finished = [run for run, process in active.items() if not process.is_alive()]
            for run in finished:
                process = active[run]
                process.join()
                drain()
                reservations.pop(run).__exit__(None, None, None)
                event = terminal.pop(run, None)
                if event is None or process.exitcode != 0:
                    event = dict(state="failed", traceback=f"Run worker exited with code {process.exitcode} before a closed result")
                # A shared job records which machine ran each run.
                machine = ({"executed_by": _executing_machine(job)}
                           if job.get(SHARED_RUNS) is True else {})
                if event["state"] == "complete":
                    result = event["result"]
                    if result.get("complete") is not True or result.get("run") != run:
                        raise ValueError("A closed run worker returned another run or incomplete result")
                    _write_json(Path(job["root"]) / publication_folder / run / "completed.json",
                                dict(result, **machine))
                    emit(log, dict(kind=completion_event, run=run, worker_pid=process.pid))
                elif event["state"] == "failed":
                    # A failed run leaves the other runs running; every
                    # failure is recorded and the job reports them at its end.
                    failed = True
                    _write_json(Path(job["root"]) / "failed" / (run + ".json"),
                                dict(run=run, traceback=event["traceback"], **machine))
                    emit(log, dict(kind="parent_failure", run=run, traceback=event["traceback"]))
                else:
                    halted = True
                    stop.set()
                    emit(log, dict(kind=event["state"], run=run))
                process.close()
                del active[run]
                emit(log, dict(kind="parallel_workers", requested_workers=requested_workers, active_workers=len(active)))
            if time.monotonic() - last_heartbeat >= 5:
                emit(log, dict(kind="watchdog_heartbeat", active_workers=len(active)))
                last_heartbeat = time.monotonic()
        return 1 if failed else 0
    finally:
        stop.set()
        # Cooperative frame boundaries preserve partial outputs. Never kill a
        # RAW/phase producer merely because a peer failed or control changed.
        while active:
            drain(block=True)
            for run, process in list(active.items()):
                if not process.is_alive():
                    process.join()
                    reservations.pop(run).__exit__(None, None, None)
                    process.close()
                    del active[run]
        messages.close()
        messages.join_thread()


def run_current_build_job(args, job, commit, log):
    """Run independent histories in separate processes under the service lock.

    A job that every connected machine may run (SHARED_RUNS) starts on any
    CPU machine that saves to the job's results folder with the shared space
    ledger. That machine runs only the runs it claims first; runs another
    machine claimed or finished are left to it.
    """
    from .worker import emit, _JobControlBoundary
    shared = job.get(SHARED_RUNS) is True
    if args.backend != "cpu" or (not shared and job["target_machine"].casefold() != args.machine.casefold()):
        raise ValueError("Current Build P1 uses local CPU workers on the selected Build machine")
    runs = [row["run"] for row in job["inputs"]]
    if runs != job["expected_runs"] or len(set(runs)) != len(runs):
        raise ValueError("P1 must assign each selected run exactly once")
    job = dict(job, root=str(args.job))
    claimed, options = [], {}
    if shared:
        from ..store_map import read_store_map
        if job.get("fresh_optical_paths"):
            raise ValueError(COMMON_PATHS_REASON)
        job["executing_machine"] = args.machine
        require_shared_machine(job, read_store_map(args.stores))
        def claim(row):
            taken = _claim_run(job, row["run"])
            if taken:
                claimed.append(row["run"])
            return taken
        options["claim"] = claim
    elif any((args.job / "results" / run / "completed.json").exists() for run in runs):
        raise FileExistsError("This P1 job already has completed work; prepare a new job")
    emit(log, dict(kind="started", pipeline=job["pipeline"], execution_mode=MODE,
                   machine=args.machine, requested_workers=args.workers, numerical_threads_per_run=1))
    try:
        if job.get("fresh_optical_paths"):
            return _run_fresh_optical_job(job, commit, args.stores, args.workers, log)
        return _dispatch_runs(job, commit, args.stores, args.workers, log, **options)
    except _JobControlBoundary as error:
        emit(log, dict(kind=error.state, message=str(error)))
        return 0
    except Exception:
        literal = traceback.format_exc()
        # A shared job's parent may only mark the runs this machine claimed.
        remaining = [run for run in (claimed if shared else runs)
                     if not (args.job / "results" / run / "completed.json").exists()
                     and not (args.job / "failed" / (run + ".json")).exists()]
        if remaining:
            _write_json(args.job / "failed" / (remaining[0] + ".json"),
                        dict(run=remaining[0], traceback=literal))
        emit(log, dict(kind="parent_failure", traceback=literal))
        return 1
    finally:
        emit(log, dict(kind="stopped", message="Worker service stopped."))


def current_build_snapshot(registration, job, ui, *, reader):
    """Show source preparation separately from committed P1 frame progress."""
    from ..job_launch_state import read_launch_requests
    from ..jobs_progress import epoch, retain_event, retain_execution
    root = Path(registration["root"])
    logs, progress, terminal = [], {}, None
    active_workers, requested_workers = 0, None
    starts, finishes = {}, {}
    for path in sorted((root / "logs").glob("*.log")):
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 512 * 1024))
            tail = stream.read().decode("utf-8", errors="replace")
        logs.append(dict(path=str(path), tail=tail))
        for line in tail.splitlines():
            try:
                timestamp, _, encoded = line.partition(" ")
                event = json.loads(encoded)
                if epoch(timestamp) is None:
                    continue
                if event.get("kind") == "progress":
                    progress[event["run"]] = dict(event, reported_utc=timestamp)
                    starts.setdefault(event["run"], timestamp)
                    terminal = None
                elif event.get("kind") == "run_completed":
                    finishes[event["run"]] = (timestamp, event.get("worker_pid"))
                elif event.get("kind") == "started":
                    requested_workers = event.get("requested_workers")
                elif event.get("kind") == "parallel_workers":
                    active_workers = event["active_workers"]
                    requested_workers = event["requested_workers"]
                elif event.get("kind") == "stopped":
                    active_workers = 0
                elif event.get("kind") in {"paused", "control_stopped", "parent_failure"}:
                    terminal = event["kind"]
            except (ValueError, KeyError):
                continue
    rows, errors = [], []
    for item in job["inputs"]:
        run, total = item["run"], item["selected_frames"]
        event = progress.get(run, {})
        row = dict(run=run, state="Waiting", committed=0, total=total, machine="Not reported",
                   worker="Not reported", error="", case_ids=[])
        if event.get("actual_worker"):
            row["machine"] = event["actual_worker"].get("machine", "Not reported")
            retain_execution(row, result=dict(actual_worker=event["actual_worker"], backend="cpu"))
            row["started_utc"] = starts[run]
            retain_event(row, event, event["reported_utc"])
            finish, pid = finishes.get(run, (None, None))
            if pid == event["actual_worker"].get("pid"):
                row["finished_utc"] = finish
        complete = root / "results" / run / "completed.json"
        failure = root / "failed" / (run + ".json")
        if complete.exists():
            value = reader(complete)
            if value.get("complete") is not True or value.get("run") != run:
                raise ValueError("P1 completion identifies another run")
            row.update(state="Complete", committed=total, case_ids=value["ordered_case_ids"])
        elif failure.exists():
            row.update(state="Failed", error=reader(failure)["traceback"])
        elif event.get("state") == "preparing_sources":
            row["state"] = "Preparing source inputs"
        elif event.get("state") in {"raw_preparation", "raw_prepared"}:
            row["state"] = "Preparing RAW " + str(event["position"]) + "/" + str(event["total"])
        elif event:
            row.update(state="Working P1", committed=(event.get("completed_count", 0)
                if event.get("work_stage") != "phase-inference" else 0))
        if job.get("fresh_optical_paths"):
            def completed_stage(folder, format):
                path = root / folder / run / "completed.json"
                if not path.is_file():
                    return False
                value = reader(path)
                return value.get("complete") is True and value.get("run") == run and value.get("format") == format
            inputs_ready = completed_stage("optical-inputs", "darsia-current-optical-input-run-v1")
            phases_ready = completed_stage("optical-phases", "darsia-current-optical-phase-run-v1")
            prepared = total + 1 if inputs_ready else (
                event.get("position", 0) if event.get("work_stage") == "image-preparation" else 0)
            inferred = total if phases_ready else (
                event.get("completed_count", 0) if event.get("work_stage") == "phase-inference" else 0)
            row.update(completed_operations=prepared + inferred + row["committed"],
                       total_operations=3 * total + 1)
            if event.get("work_stage") == "phase-inference" and not phases_ready:
                row["state"] = "Inferring phases for new colour paths"
        if terminal in {"paused", "control_stopped"} and not complete.exists() and not failure.exists():
            row["state"] = ("Paused" if terminal == "paused" else "Stopped") + " · " + row["state"]
        rows.append(row)
    total = sum(row["total"] for row in rows)
    committed = sum(row["committed"] for row in rows)
    complete = all(row["state"] == "Complete" for row in rows)
    failed = sum(row["state"] == "Failed" for row in rows)
    workers = [dict(machine=row["machine"], worker=row["worker"], run=row["run"], state=row["state"],
                    last_progress_utc="See service log", error=row["error"])
               for row in rows if row["state"].startswith(("Preparing", "Working", "Inferring"))]
    return dict(root=str(root), name=ui.get("name", "Generate phase maps from model"),
        series=ui.get("series_label", job["series_id"]), created_utc=job["created_utc"],
        master_machine=job["target_machine"], job_facts=deepcopy(job), runs=rows, workers=workers, machines=[],
        logs=logs, registered_machines=[], launch_requests=read_launch_requests(root, reader=reader),
        run_count=len(rows), completed_runs=sum(row["state"] == "Complete" for row in rows),
        failed_runs=failed, committed=committed, total_frames=total, remaining=total - committed,
        frames_per_minute=None, eta_seconds=None, complete=complete, errors=errors,
        active_workers=active_workers, requested_workers=requested_workers,
        state=("Complete" if complete else "Working" if workers else "Failed" if failed
               else "Working" if progress else "Waiting"),
        **(dict(work_unit="operations") if job.get("fresh_optical_paths") else {}),
        rate_scope="RAW preparation, phase inference and statistics are separate operation units; only final committed P1 frames count as images")
