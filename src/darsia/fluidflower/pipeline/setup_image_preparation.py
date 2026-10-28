"""A real final Setup operation for the selected image preparation basis."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import tomllib

from ..mass.n1_pipeline import asset, _write_json
from .current_inputs import current_input_reader

IMAGE_RUN = "setup-image-preparation"
OPERATION = "image-preparation"
FORMAT = "darsia-setup-image-preparation-v1"
BASELINE_FORMAT = "darsia-setup-corrected-baseline-v1"


def image_output_scope(plan):
    """Declare this typed stage's standard queue-attempt namespace only.

    Older geometric plans list the Apply/Build attempt roots explicitly. The
    final typed operation has its own run id and therefore needs its own
    sibling under that same selected results/job namespace. Broad original
    scopes already covering all attempts need no added authority.
    """
    from .series_setup_jobs import APPLY_RUN
    scope = deepcopy(plan["output_scope"])
    roots = [Path(path) for path in scope["metadata_roots"]]
    attempts = {path.parent.parent / IMAGE_RUN / ".attempts" for path in roots
                if path.name == ".attempts" and path.parent.name == APPLY_RUN}
    if len(attempts) > 1:
        raise ValueError("Image preparation needs one original Apply attempt namespace")
    for path in attempts:
        if not any(path == root or path.is_relative_to(root) for root in roots):
            scope["metadata_roots"].append(str(path))
    return scope


def image_output_caps(plan, native):
    """Enforced limits for actual RGB/mask grids, not inherited RAW budgets."""
    shape = native.get("source", {}).get("native_shape_hw")
    if shape is None:
        caps = [row["request_template"]["output"] for row in plan["build_plans"]]
        return dict(maximum_output_bytes=sum(row["maximum_output_bytes"] for row in caps),
                    maximum_file_bytes=max(row["maximum_file_bytes"] for row in caps))
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(value) is not int or value < 1 for value in shape)):
        raise ValueError("Image preparation requires the declared positive native image shape")
    pixels = shape[0] * shape[1]
    parameters = len(json.dumps(native, ensure_ascii=False, indent=2).encode("utf-8"))
    per_file = max(pixels * 12 + pixels // 100 + 1024**2,
                   parameters * 2 + 1024**2, 16 * 1024**2)
    whole = len(plan["runs"]) * (pixels * 26 + parameters * 2 + 20 * 1024**2)
    return dict(maximum_output_bytes=whole, maximum_file_bytes=per_file)


def independent_choice(authority, *, reader=None):
    from ..image_normalisation import saved_choice, selected_corrections
    reader = current_input_reader(reader)
    document = tomllib.loads(reader.path(authority).read_text(encoding="utf-8-sig"))
    if document.get("method", {}).get("active") != "phase_model":
        return document, None
    choice = saved_choice(document, document["series_id"], document["runs"]["ids"])
    required = (document.get("method", {}).get("active") == "phase_model" and choice is not None
                and "color_method" in choice and selected_corrections(choice)["color_method"] != "saved_native")
    return document, choice if required else None


def configure_setup_image_preparation_job(job, plan_pin, plan, root, choice_authority=None, *,
                                        reuse_image_basis=None):
    """Prepare this job's new typed task; do not queue it or replace job.json.

    The old job's series/plan remain geometric ancestry. Only the new task's
    current series pin supplies the independently selected image corrections.
    Called by initial preparation or the existing transactional GUI Update.
    """
    from .series_setup_jobs import _immutable, _task, _manifest, _roots
    from .series_setup_protocol import FORMAT as OP_FORMAT, validate_setup_job
    root = Path(root)
    authority = deepcopy(choice_authority or plan["series_manifest"])
    _document, choice = independent_choice(authority)
    if choice is None:
        raise ValueError("No current independent image-preparation choice is selected")
    native = current_input_reader().json(plan["native_parameters"])
    caps = image_output_caps(plan, native)
    scope = image_output_scope(plan)
    _roots(dict(plan, output_scope=scope), root)
    from ..job_filesystem import inspect_results_root, valid_results_admission
    new_roots = [Path(path) for path in scope["metadata_roots"]
                 if path not in plan["output_scope"]["metadata_roots"]]
    for path in new_roots:
        if not valid_results_admission(inspect_results_root(path), path):
            raise ValueError("The image attempt namespace lacks positive filesystem admission")
    for path in new_roots:
        path.mkdir(parents=True, exist_ok=True)
    from ..native.raw_rig_source import execution_identity
    from .setup_colour_paths import COLOUR_PATH_SCOPE
    identity = hashlib.sha256(json.dumps(dict(choice=authority, geometry=plan["adopted_revision"],
        native=plan["native_parameters"], producer=execution_identity(), colour_path_scope=COLOUR_PATH_SCOPE,
        colour_path_producer=asset(Path(__file__).with_name("setup_colour_paths.py")),
        reused_image_basis=reuse_image_basis), sort_keys=True).encode()).hexdigest()[:24]
    config = dict(format=OP_FORMAT, run=IMAGE_RUN, operation=OPERATION, setup_plan=deepcopy(plan_pin),
        **{key: deepcopy(plan[key]) for key in ("project_root", "project_manifest", "adopted_revision",
            "native_parameters", "target_machine")},
        output_scope=scope,
        series_manifest=authority, choice_authority=authority, runs=list(plan["runs"]),
        job_root=str(root), output_root=str(root / "image-preparation" / identity),
        colour_path_scope=COLOUR_PATH_SCOPE, metadata_output=caps)
    if reuse_image_basis is not None:
        config["reuse_image_basis"] = deepcopy(reuse_image_basis)
    bound = _immutable(root / "setup-controls" / "image-operation.json", config)
    _source, _manifest_value, roots = _manifest(authority)
    updated = deepcopy(job)
    operations = [row for row in updated["setup_operations"] if row["run"] != IMAGE_RUN]
    updated["setup_operations"] = [*operations, dict(run=IMAGE_RUN, operation=OPERATION)]
    updated["expected_runs"] = [row["run"] for row in updated["setup_operations"]]
    updated["image_preparation"] = dict(configuration=bound, choice_authority=authority)
    validate_setup_job(updated)
    from .job_inputs import InputCatalog
    task = _task(bound, config, updated, roots, InputCatalog)
    return dict(job=updated, configuration=config, configuration_binding=bound, task=task)


def build_dependencies_ready(root, job):
    """Scheduling hint only; execution authenticates original completions."""
    from ..setup_completed_builds import retained_operations
    builds = [row["run"] for row in job["setup_operations"] if row["operation"] == "build"]
    missing = [run for run in builds if not (Path(root) / "done" / (run + ".json")).is_file()]
    if not missing:
        return bool(builds)
    if not (Path(root) / "ui/retained-builds.json").is_file():
        return False
    kept = retained_operations(root, job)
    return all(run in kept for run in missing)


def completed_build_operations(root, runs):
    from ..jobs_metadata import JobSnapshotReader
    snapshot = JobSnapshotReader()(dict(root=str(root)))
    operations = {row["run"]: row["result"]["files"]["operation_result.json"]
        for row in snapshot["runs"] if row.get("operation") == "build" and row.get("state") == "Complete"}
    if set(operations) != set(runs) or snapshot.get("errors"):
        raise ValueError("Complete the same saved geometry Build for every selected run before image preparation")
    return operations


def retained_series_aliases(operations, reader):
    """Bind original Build manifest bytes to their actual immutable snapshot."""
    aliases, plans = [], set()
    for operation in operations.values():
        result = reader.json(operation)
        if "setup_plan" not in result:
            continue  # The full original context reader remains authoritative.
        identity = (result["setup_plan"]["path"], result["setup_plan"]["sha256"])
        if identity in plans:
            continue
        plans.add(identity)
        plan = reader.json(result["setup_plan"])
        original = plan["series_manifest"]
        snapshot = Path(result["setup_plan"]["path"]).parent / "series-manifest.toml"
        if not snapshot.is_file():
            # No historical copy is invented; a still-current source remains
            # the original reader's obligation when the full context loads.
            continue
        physical = dict(original, path=str(snapshot))
        reader.path(physical)
        pair = dict(logical=deepcopy(original), physical=physical)
        if pair not in aliases:
            aliases.append(pair)
    return aliases


def image_basis_reader(reader, preparation):
    """Relocate only exact old manifest bytes; current choices keep their SHA."""
    from .current_inputs import CurrentInputReader
    aliases = preparation.get("source_aliases", [])
    class Reader(CurrentInputReader):
        def physical_descriptor(self, bound):
            for row in aliases:
                original = row["logical"]
                if all(bound.get(key) == original.get(key) for key in ("path", "sha256")):
                    if (row["physical"]["sha256"] != original["sha256"]
                            or row["physical"].get("size_bytes") != original.get("size_bytes")):
                        raise ValueError("A Setup source alias changes the original manifest bytes")
                    return dict(bound, path=row["physical"]["path"])
            return self.reader.physical_descriptor(bound) if hasattr(self.reader, "physical_descriptor") else bound
        def archive(self, bound, names, *, optional=()):
            return self.reader.archive(self.physical_descriptor(bound), names, optional=optional)
    return Reader(current_input_reader(reader))


def run_image_preparation(config, *, workers=1, progress=None, reader=None, producer=None,
                          colour_path_producer=None):
    """Fit and materialize baselines once; no phase inference or calibration."""
    from .current_image_preparation import prepare_image_photometry
    from ..native.raw_rig_source import execution_identity
    from ..native.full_photometry import ORDER
    reader = current_input_reader(reader)
    _series, choice = independent_choice(config["choice_authority"], reader=reader)
    if choice is None:
        raise ValueError("This Setup image operation no longer has its exact saved choice")
    operations = completed_build_operations(config["job_root"], config["runs"])
    from ..image_formats import image_family
    for run, pin in operations.items():
        result = reader.json(pin)
        request = reader.json(result["raw_build_request"])
        if image_family(request["baseline"]["path"]) != "raw":
            raise ValueError("This selected Setup image producer currently requires RAW acquisitions: "
                + run + ". The original direct-colour image workflow remains available.")
    root = Path(config["output_root"])
    path = root / "prepared_image_basis.json"
    if path.is_file():
        saved = reader.json(asset(path))
        if (saved.get("format") != FORMAT or saved.get("complete") is not True
                or saved["operations"] != operations or saved["choice_authority"] != config["choice_authority"]
                or saved["raw_execution_identity"] != execution_identity()):
            raise ValueError("Existing Setup image preparation belongs to another source or choice")
        for value in saved["runs"].values():
            reader.path(value["source_binding"])
            reader.path(value["baseline"])
        if saved.get("colorpaths_performed") is not True or not saved.get("colour_paths"):
            raise ValueError("This old image-only completion has no Setup colour paths. Prepare a fresh Finish setup job.")
        from .setup_colour_paths import colour_path_packages
        colour_path_packages(reader.json(saved["colour_paths"]), reader=reader, require_per_run=True)
        return asset(path)
    if progress:
        progress(dict(state="setup_progress", operation=OPERATION, run=IMAGE_RUN,
                      step="prepare_selected_images", message="Preparing selected image basis"))
    def report(event):
        if progress:
            progress(dict(state="setup_progress", operation=OPERATION, run=IMAGE_RUN,
                step=event.get("state", "image_preparation"), source_run=event.get("run"),
                run_index=event.get("position", 0), run_count=event.get("total", len(operations)),
                message=event.get("message", "Preparing image basis for " + str(event.get("run", "")))))
    reused = None
    reused_partial = False
    if config.get("reuse_image_basis") is not None:
        reused = reader.json(config["reuse_image_basis"])
        from .setup_colour_paths import PARTIAL_REUSE_FORMAT
        reused_partial = reused.get("format") == PARTIAL_REUSE_FORMAT
        if (reused.get("choice") != choice or reused.get("operations") != operations
                or reused.get("choice_authority") != config["choice_authority"]
                or reused.get("raw_execution_identity") != execution_identity()
                or reused.get("correction_order") != ORDER):
            raise ValueError("The saved image basis cannot be reused for these saved sources and corrections")
        if reused_partial:
            # This is a failed, retired attempt, never a completed Setup. Its
            # genuine finished photometry is independent of unfinished paths.
            # The optical producer owns failure/retirement and byte-version
            # verification; do not repeat its sample/payload scans here.
            if (reused.get("complete") is not False
                    or not isinstance(reused.get("source_root"), str) or not reused["source_root"]
                    or not isinstance(reused.get("file_versions"), dict)
                    or not isinstance(reused.get("image_basis"), dict)
                    or not isinstance(reused.get("image_basis", {}).get("runs"), dict)
                    or not isinstance(reused.get("optical_sources"), dict)
                    or not isinstance(reused.get("frame_selection"), dict)
                    or not isinstance(reused.get("failed_attempt"), dict)
                    or not isinstance(reused.get("retirement"), dict)
                    or any(not isinstance(reused["retirement"].get(key), dict)
                           for key in ("claim", "owner", "retirement", "stop"))):
                raise ValueError("The failed image basis lacks its typed retired partial source")
            runs = reused["image_basis"]["runs"]
        elif (reused.get("format") != FORMAT or reused.get("complete") is not True
                or not reused.get("colour_paths") or reused.get("colorpaths_performed") is not True):
            raise ValueError("The completed image basis cannot be reused for these saved sources and corrections")
        else:
            runs = reused.get("runs")
        record = reader.json(reused["photometry_record"])
        if record.get("runs") != runs or set(record.get("runs", {})) != set(operations):
            raise ValueError("The saved photometry record differs from the retained image basis")
        prepared = dict(record, record=deepcopy(reused["photometry_record"]))
        report(dict(state=("reusing_retired_image_corrections" if reused_partial else
                           "reusing_completed_image_corrections"), total=len(operations),
                    message="Reusing completed image corrections; rebuilding colour paths separately for each run"))
    else:
        prepared = (producer or prepare_image_photometry)(operations, choice,
            config["choice_authority"], root / "photometry", reader=reader,
            on_progress=report, workers=workers, materialize_baselines=True)
    if prepared is None or set(prepared["runs"]) != set(operations):
        raise ValueError("Setup did not produce every selected run's independent image basis")
    if colour_path_producer is None:
        from .setup_colour_paths import prepare_setup_colour_paths
        colour_path_producer = prepare_setup_colour_paths
    options = dict(workers=workers, reader=reader, on_progress=report)
    if reused is not None:
        if reused_partial:
            options["reuse_partial"] = deepcopy(config["reuse_image_basis"])
        else:
            options["reuse_receipt"] = deepcopy(reused["colour_paths"])
    colour_paths = colour_path_producer(operations, prepared,
        config["choice_authority"], root / "colour-paths", **options)
    document = dict(format=FORMAT, complete=True, choice=choice,
        choice_authority=deepcopy(config["choice_authority"]), operations=operations,
        adopted_revision=deepcopy(config["adopted_revision"]),
        native_configuration=deepcopy(config["native_parameters"]),
        raw_execution_identity=execution_identity(), correction_order=ORDER,
        runs=deepcopy(prepared["runs"]), photometry_record=deepcopy(prepared["record"]),
        source_aliases=retained_series_aliases(operations, reader),
        baseline_encoding="chw-float32-log-camera-linear-rgb-with-nan-invalid",
        phase_inference_performed=False, model_training_performed=False,
        colorpaths_performed=True, colour_paths=colour_paths,
        colorpaths_stage="after selected Setup crop, lighting and colour corrections; before Calibration")
    from .setup_colour_paths import COLOUR_PATH_SCOPE
    document["colour_path_scope"] = COLOUR_PATH_SCOPE
    if reused is not None:
        document["reused_image_basis"] = deepcopy(config["reuse_image_basis"])
    return _write_json(path, document)


def completed_image_preparation(snapshot, authority, *, reader=None):
    """Read the Setup output selected by its real typed completion only."""
    reader = current_input_reader(reader)
    rows = [row for row in snapshot.get("runs", []) if row.get("operation") == OPERATION
            and row.get("state") == "Complete"]
    if len(rows) != 1:
        raise ValueError("Run Image preparation in the same Setup job before training the new image basis")
    result = rows[0]["result"]
    pin = result.get("image_preparation")
    if pin is None and result.get("files", {}).get("operation_result.json"):
        pin = reader.json(result["files"]["operation_result.json"])["image_preparation"]
    value = reader.json(pin)
    _series, current_choice = independent_choice(authority, reader=reader)
    if (value.get("format") != FORMAT or value.get("complete") is not True
            or value.get("choice") != current_choice):
        raise ValueError("The completed Setup image basis differs from the currently saved image choices")
    expected = {row["run"]: row["result"]["files"]["operation_result.json"]
        for row in snapshot.get("runs", []) if row.get("operation") == "build" and row.get("state") == "Complete"}
    if value.get("operations") != expected:
        raise ValueError("The completed image basis belongs to different geometric Builds")
    return dict(value, receipt=deepcopy(pin))


def completed_setup_colour_paths(snapshot, authority, *, reader=None, require_per_run=True):
    """The real final Setup receipt, without a model or any phase/mass output."""
    from .current_optical_preparation import colour_path_image_count
    reader = current_input_reader(reader)
    basis = completed_image_preparation(snapshot, authority, reader=reader)
    if not basis.get("colorpaths_performed") or not basis.get("colour_paths"):
        raise ValueError("Finish setup has not generated colour paths on this image basis.")
    receipt = reader.json(basis["colour_paths"])
    document, _choice = independent_choice(authority, reader=reader)
    from .setup_colour_paths import FORMAT as PATH_FORMAT, LEGACY_FORMAT, colour_path_packages
    if (receipt.get("format") not in {PATH_FORMAT, LEGACY_FORMAT} or receipt.get("complete") is not True
            or receipt.get("operations") != basis["operations"]
            or receipt.get("choice_authority") != basis["choice_authority"]
            or set(receipt.get("runs", {})) != set(basis["operations"])
            or receipt.get("image_count_per_run") != colour_path_image_count(document)
            or any(receipt.get(key) is not False for key in
                ("phase_inference_performed", "model_training_performed", "mass_fit_performed"))):
        raise ValueError("The completed Setup colour paths differ from the current runs, image selection or corrections.")
    colour_path_packages(receipt, reader=reader, require_per_run=require_per_run)
    selected = receipt.get("input", {}).get("frame_selection")
    if selected is not None:
        from .setup_colour_paths import _selection
        if _selection(document, basis["operations"], Path(authority["path"]).parent, reader) != selected:
            raise ValueError("The saved Setup image selection changed after its colour paths were generated.")
    return dict(receipt, receipt=deepcopy(basis["colour_paths"]), image_basis=deepcopy(basis["receipt"]))
