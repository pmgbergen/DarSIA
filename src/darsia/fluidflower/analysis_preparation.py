"""Compile complete registered run inputs into a fresh, inert analysis plan."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess

from .analysis_plan import phase1_configuration, write_phase1_plan, write_phase2_plan
from .jobs_metadata import JobSnapshotReader
from .pipeline.job_queue import descriptor


def job_identity(repository):
    """Bind the current tracked source and lock; keep unrelated user files."""
    repository = Path(repository)
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"], text=True)
    untracked = subprocess.check_output(["git", "-C", str(repository), "ls-files", "--others", "--exclude-standard", "--", "src"], text=True)
    if dirty.strip() or untracked.strip():
        raise ValueError("Commit the analysis source before preparing a plan")
    relative = "requirements-worker.lock"
    lock = subprocess.check_output(["git", "-C", str(repository), "show", head + ":" + relative])
    if (repository / relative).read_bytes().replace(b"\r\n", b"\n") != lock.replace(b"\r\n", b"\n"):
        raise ValueError("The current environment lock differs from the tracked lock")
    return dict(commit=head, precision="float32", created_utc=datetime.now(timezone.utc).isoformat(),
                environment_lock=dict(path=relative, sha256=hashlib.sha256(lock).hexdigest()))


def _phase1_input_configurations(inputs, *, series_root, geometry_index, pressure_index,
                                 gas_probability, job, threads=1, document=None):
    """Read only supplied bound roles; refuse every missing run before writing.

    The input manifest binds each complete run configuration and the registered
    series manifest. It is produced by source preparation, not guessed from GUI
    strings. Geometry and pressure dataset indices are explicit selected inputs.
    """
    from .pipeline.alias_job import _manifest
    from .pipeline.job_inputs import InputCatalog
    reader = JobSnapshotReader()
    read = lambda bound: reader.read(Path(bound["path"]), bound)
    if document is None:
        document = read(inputs)
    geometry, pressure = read(geometry_index), read(pressure_index)
    manifest = descriptor(Path(series_root) / "series.toml")
    if (document.get("format") != "darsia-analysis-plan-inputs-v1"
            or document.get("series_manifest") != manifest):
        raise ValueError("Choose the source-preparation inputs for this exact registered series")
    _, registered, roots = _manifest(manifest)
    rows = document.get("runs")
    if not isinstance(rows, list) or not rows or len({row["run"] for row in rows}) != len(rows):
        raise ValueError("Source preparation must bind the complete distinct run order")
    if (geometry.get("format") != "darsia-measured-run-geometry-v1"
            or geometry.get("reference_models_complete") is not True):
        raise ValueError("The selected measured geometry index is incomplete")
    configurations, resources, errors = [], {}, []
    for row in rows:
        run = row["run"]
        try:
            original = InputCatalog(roots).doc(row["configuration"], "configuration", "analysis_inputs." + run)
            if original["run"] != run:
                raise ValueError("Run input belongs to another run")
            frame_index = read(geometry["runs"][run]["frame_data"])
            if (frame_index.get("run") != run or frame_index.get("complete_for_this_run") is not True
                    or set(frame_index["frames"]) != set(original["ordered_case_ids"])):
                raise ValueError("Measured frame geometry does not cover the complete selected order")
            frames = {case: read(value["frame_model"]) for case, value in frame_index["frames"].items()}
            atmosphere = read(pressure["runs"][registered["series_id"]][run]["atmosphere_series"])
            from .accepted_pressure_policy import verify_accepted_pressure
            verify_accepted_pressure(atmosphere)
            if atmosphere.get("provisional_gap_default") is True:
                raise ValueError("Provisional pressure gaps await an explicit decision")
            calibration = read(original["mass"]["calibration"])
            method = {key: calibration[key] for key in ("round4_spec_sha256", "amendment_sha256")}
            candidate = phase1_configuration(original, precision=job["precision"],
                atmosphere=atmosphere, frame_data=frames, gas_probability=gas_probability,
                method_bindings=method, role_catalog=InputCatalog(roots, original.get("data_aliases", ())))
            if "current_build_binding" in candidate:
                # The selected geometry index supplies the actual measured
                # baseline model used by the role-aware Build consumer.
                candidate["current_build_binding"]["metric_model"] = dict(
                    geometry["runs"][run]["reference_model"])
            provenance = candidate.setdefault("provenance", {})
            if "analysis_plan_inputs" in provenance:
                raise ValueError("The input run already binds another plan-preparation record")
            provenance["analysis_plan_inputs"] = dict(source_manifest=inputs, original_configuration=row["configuration"],
                geometry_index=geometry_index, pressure_index=pressure_index,
                original_method_source=original["mass"]["calibration"])
            configurations.append(candidate)
            resources[run] = dict(backend="cpu", threads=threads)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(run + ": " + str(error))
    if errors:
        raise ValueError("Analysis inputs are incomplete; no plan was written:\n" + "\n".join(errors))
    if descriptor(Path(series_root) / "series.toml") != manifest:
        raise ValueError("The registered series changed during preparation")
    return configurations, resources


def prepare_phase1_inputs(inputs, folder, *, series_root, geometry_index, pressure_index,
                          gas_probability, job_id, job, threads=1):
    """Prepare the original inert plan from complete, bound source inputs."""
    configurations, resources = _phase1_input_configurations(inputs,
        series_root=series_root, geometry_index=geometry_index, pressure_index=pressure_index,
        gas_probability=gas_probability, job=job, threads=threads)
    return write_phase1_plan(folder, configurations, job_id=job_id, job=job, resources=resources)


def source_metadata_descriptor(path):
    """Bind only an explicitly selected, bounded metadata file, never a RAW."""
    from .pipeline.job_inputs import MAX_METADATA_BYTES
    path = Path(path).resolve(strict=True)
    if (path.suffix.casefold() != ".json" or not path.is_file()
            or path.stat().st_size > MAX_METADATA_BYTES):
        raise ValueError("Choose an existing bounded source-metadata JSON file")
    return descriptor(path)


def collect_phase1_source_inputs(configurations, destination, *, series_root,
                                 geometry_index, pressure_index, gas_probability):
    """Write a manifest for existing complete products through original checks.

    This is metadata assembly, not a producer or an acceptance certificate for
    geometry, pressure, optics, or physical context. Missing products refuse;
    no field, eligibility flag, plan, approval, queue, or worker is created.
    Original Prepare reads these exact source identities again afterwards.
    """
    from .pipeline.alias_job import _manifest
    from .pipeline.job_queue import _immutable_bytes, write_immutable
    from .pipeline.job_inputs import MAX_METADATA_BYTES
    from .jobs_metadata import _bound
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Choose a fresh source-input manifest file")
    destination = destination.resolve()
    manifest = descriptor(Path(series_root) / "series.toml")
    _, registered, _ = _manifest(manifest)
    declared = (registered.get("runs") or {}).get("ids")
    if (not isinstance(declared, list) or not declared
            or any(not isinstance(run, str) or not run for run in declared)
            or len(set(declared)) != len(declared)):
        raise ValueError("The registered series needs its exact distinct run order")
    if not isinstance(configurations, (list, tuple)) or not configurations:
        raise ValueError("Choose the existing complete run configurations in registered run order")
    for bound in [geometry_index, pressure_index, *configurations]:
        _bound(bound)
        if bound["size_bytes"] > MAX_METADATA_BYTES:
            raise ValueError("Source collection reads bounded metadata only")
    reader, rows, selected = JobSnapshotReader(), [], []
    for bound in configurations:
        value = reader.read(Path(bound["path"]), bound)
        run = value.get("run")
        if run not in declared or run in selected:
            raise ValueError("Source configurations must name distinct registered runs")
        selected.append(run)
        rows.append(dict(run=run, configuration=dict(bound)))
    if selected != [run for run in declared if run in selected]:
        raise ValueError("Keep the registered run order; no configuration or frame is reordered")
    document = dict(format="darsia-analysis-plan-inputs-v1", series_manifest=manifest, runs=rows)
    raw = _immutable_bytes(document)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("Source-input manifests must remain bounded metadata")
    bound = dict(path=str(destination), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    _phase1_input_configurations(bound, series_root=series_root, geometry_index=geometry_index,
        pressure_index=pressure_index, gas_probability=gas_probability, job=dict(precision="float32"),
        document=document)
    write_immutable(destination, document)
    if descriptor(destination) != bound or descriptor(Path(series_root) / "series.toml") != manifest:
        raise ValueError("Source selection changed during manifest publication; no plan was prepared")
    return dict(inputs=bound, runs=selected, plan_prepared=False, worker_started=False)


def prepare_phase2_inputs(inputs, folder, *, series_root, fitted_calibration, job_id, job, threads=1,
                          storage_profile=None, stored_resolution="saved", phase1_jobs=None):
    """Use explicit genuine completion/anchor/storage roles from source preparation."""
    from .mass.n1_pipeline import DataReader
    reader = DataReader(unc_only=False)
    document = reader.json(inputs)
    manifest = descriptor(Path(series_root) / "series.toml")
    if (document.get("format") != "darsia-analysis-phase2-inputs-v1"
            or document.get("series_manifest") != manifest):
        raise ValueError("Choose the saved phase-one completion inputs")
    rows = document.get("runs")
    if not isinstance(rows, list) or not rows or len({row["run"] for row in rows}) != len(rows):
        raise ValueError("Explicit distinct phase-one completion order required")
    jobs = None
    if phase1_jobs is not None:
        jobs = reader.json(phase1_jobs)
        if (jobs.get("format") != "darsia-phase1-reproduction-jobs-v1"
                or set(jobs.get("runs") or {}) != {row["run"] for row in rows}):
            raise ValueError("Choose explicit original phase-one jobs for exactly these runs")
    sources = []
    for row in rows:
        source = {key: row[key] for key in (
            "completion_bound", "configuration_bound", "expected_phase1_commit", "fitted_anchor")}
        source["fitted_calibration"] = fitted_calibration
        for key in ("storage_data", "pixel_domain_data", "storage_profile", "frozen_refit", "phase1_origin",
                    "overflow_phase_split"):
            if key in row:
                source[key] = row[key]
        if storage_profile is not None or stored_resolution != "saved":
            from .production_plan_options import phase2_source_options
            source = phase2_source_options(source, storage_profile=storage_profile,
                stored_resolution=stored_resolution, precision=job["precision"])
        if jobs is not None:
            from .pipeline.reproduction_lineage import bind_phase1_origin
            origin = bind_phase1_origin(jobs["runs"][row["run"]], row["run"],
                configuration=source["configuration_bound"], completion=source["completion_bound"],
                expected_commit=source["expected_phase1_commit"])
            if "phase1_origin" in source and source["phase1_origin"] != origin:
                raise ValueError("The selected original job differs from the existing phase-one origin")
            source["phase1_origin"] = origin
        configuration = reader.json(source["configuration_bound"])
        if configuration.get("run") != row["run"]:
            raise ValueError("Phase-one completion input belongs to another run")
        sources.append(source)
    if descriptor(Path(series_root) / "series.toml") != manifest:
        raise ValueError("The registered series changed during preparation")
    selected_job = dict(job, calibration_sha256=fitted_calibration["sha256"])
    return write_phase2_plan(folder, sources, job_id=job_id, job=selected_job,
        resources={row["run"]: dict(backend="cpu", threads=threads) for row in rows}, reader=reader)
