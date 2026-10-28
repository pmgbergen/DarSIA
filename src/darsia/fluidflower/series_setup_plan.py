"""Prepare a finite Setup job from the current saved series metadata.

The original scientific readers run after the actual worker claim. Planning
reads bound JSON/TOML/protocol metadata and stats the already bound baseline;
it never decodes or rehashes RAW, adopts geometry or qualifies a rig model.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path, PureWindowsPath
import re
import stat
from types import SimpleNamespace

from .pipeline.capacity import read_bound
from .pipeline.job_inputs import InputCatalog
from .pipeline.job_queue import _immutable_bytes, name
from .pipeline.series_setup_inputs import _same
from .pipeline.series_setup_jobs import (
    APPLY_RUN, _control, _immutable, _manifest, _roots, _task,
    prepare_setup_job, shared_output_admission,
)
from .pipeline.series_setup_protocol import FORMAT, PLAN_FORMAT, validate_setup_plan


OUTPUT_CAPS = dict(maximum_output_bytes=8 * 1024**3, maximum_file_bytes=4 * 1024**3)


def _metadata(path):
    """Capture one bounded original metadata file without visiting its payloads."""
    from .corrected_setup import _descriptor, _small
    _small(path)
    return _descriptor(path)


def _paths(series, manifest, run, saved, *, project_root=None, revision=None):
    """Project only the original declared bundle and its bound current TOML."""
    from .rig_raw_write import _canonical
    table, settings = manifest["runs"], manifest.get("paths", {})
    declared = [
        (table.get("config_files") or {}).get(run)
        or str(Path(settings.get("run_configs", "config/runs")) / (run + ".toml")),
        (table.get("merged_files") or {}).get(run)
        or str(Path(settings.get("merged_configs", "config/merged")) / (run + ".toml")),
    ]
    selected = [_canonical(Path(value) if Path(value).is_absolute() else series / value)
                for value in declared]
    if any(not path.is_relative_to(series) for path in selected) or len(set(selected)) != 2:
        raise ValueError("The declared selected-run configuration bundle is not distinct inside this series")
    dual = (table.get("geometry_configuration_bindings") or {}).get(run)
    from .corrected_setup import _raw_configuration_bundle, _identity
    raw = _raw_configuration_bundle(series, run, saved["configuration"], selected,
        project_root=project_root, revision=revision)
    if raw is not None:
        current = raw["current_configuration"]
    elif dual is None:
        current = saved["configuration"]
        if Path(current["path"]).resolve() not in selected:
            raise ValueError("The saved configuration is outside this run's declared bundle: " + run)
    else:
        if (dual.get("run_id") != run or not _same(dual.get("source_configuration"), _identity(saved["configuration"]))
                or [Path(row["path"]).resolve() for row in dual.get("apply_configurations", ())] != selected
                or not _same(dual.get("current_configuration"), dual["apply_configurations"][-1])):
            raise ValueError("The saved dual configuration differs from its declared current bundle: " + run)
        current = dual["current_configuration"]
    from .corrected_setup import _verified
    _verified(current)
    common_value = settings.get("common_config", "config/common.toml")
    common = _canonical(Path(common_value) if Path(common_value).is_absolute() else series / common_value)
    paths = ([common] if common.is_file() else []) + [Path(current["path"])]
    return paths, selected


def _basis(config, run):
    """Read exactly the adopted run's bound model and its renderer metadata."""
    rows = [row for row in (config.get("run_geometry") or {}).get("runs", ())
            if row.get("run_id") == run and row.get("enabled", True)]
    if len(rows) != 1 or not isinstance(rows[0].get("registration", {}).get("target_model"), dict):
        raise ValueError("The adopted run lacks its exact target_model descriptor: " + run)
    model = read_bound(rows[0]["registration"]["target_model"])[0]
    if model.get("run_id", model.get("run")) != run:
        raise ValueError("The adopted target model names another run: " + run)
    descriptor = (model.get("baseline_native_grid") or {}).get("renderer")
    if not isinstance(descriptor, dict):
        raise ValueError("The adopted target model lacks baseline_native_grid.renderer: " + run)
    certificate = read_bound(descriptor)[0]
    if (certificate.get("format") != "darsia-raw-rig-baseline-build-v1"
            or certificate.get("artifact_kind") != "raw_shape_basis"
            or certificate.get("run_id") != run):
        raise ValueError("The target model does not name this run's original RAW shape basis: " + run)
    return descriptor, certificate


def _current_baseline(project, run, baseline, actual):
    """Keep the basis hash when an already approved photograph was relocated.

    Planning checks only saved review metadata and file stats. The ordinary
    RAW request reader still verifies the actual acquisition bytes at Build.
    """
    actual = Path(actual).resolve()
    from .storage_paths import same_storage_path
    if same_storage_path(baseline["path"], actual):
        return dict(baseline, path=str(actual))
    from . import rig_point_review as review
    document = review.load_review(project)
    review._check_document(document)
    row = document["runs"][run]
    review._check_approval(document, row)
    relocation = row.get("proposal_binding", {}).get("photograph_relocation") or {}
    old, new = relocation.get("original") or {}, relocation.get("current") or {}
    if (Path(old.get("path", "")).resolve() != Path(baseline["path"]).resolve()
            or Path(new.get("path", "")).resolve() != actual
            or any(old.get(key) != baseline[key] or new.get(key) != baseline[key]
                   for key in ("sha256", "size_bytes"))):
        raise ValueError("The current baseline lacks this run's exact approved relocation: " + run)
    return dict(baseline, path=str(actual))


def _domains(paths):
    """Keep stable, distinct, literal physical writer roots without broadening."""
    from .rig_raw_write import _canonical
    result = []
    for path in paths:
        actual = _canonical(path)
        if actual not in result:
            result.append(actual)
    return result


def _preflight(plan, root, roots, stores, series_id, commit, environment_lock, *, check_existing=True):
    """Replay the real control catalogue against two not-yet-written byte buffers."""
    from .store_map import input_reference
    from .pipeline.series_setup_jobs import CONTROL_LIMIT
    validate_setup_plan(plan)
    _roots(plan, root)
    job_id = name(root.name)
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("The Setup source commit must be a full lowercase SHA")
    lock = environment_lock
    if (not isinstance(lock, dict) or not isinstance(lock.get("path"), str) or not lock["path"]
            or not isinstance(lock.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", lock["sha256"]) is None):
        raise ValueError("The original environment lock needs its repository-relative path and SHA")
    relative = PureWindowsPath(lock["path"])
    if relative.drive or relative.root or ".." in relative.parts:
        raise ValueError("The original environment lock needs its repository-relative path and SHA")
    buffers = {}

    def virtual(path, document):
        raw = _immutable_bytes(document)
        if len(raw) > CONTROL_LIMIT:
            raise ValueError("Setup control exceeds the shared job metadata limit")
        buffers[str(path)] = raw
        return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))

    plan_bound = virtual(Path(plan["control_root"]) / "plan.json", plan)
    config = dict(format=FORMAT, run=APPLY_RUN, operation="apply", setup_plan=plan_bound,
        **{key: deepcopy(plan[key]) for key in ("project_root", "project_manifest", "adopted_revision",
            "series_manifest", "native_parameters", "runs", "build_plans", "target_machine",
            "metadata_output", "output_scope")})
    operation = virtual(Path(plan["control_root"]) / "apply-operation.json", config)
    for bound, document in ((plan_bound, plan), (operation, config)):
        if check_existing and Path(bound["path"]).exists() and not _same(_control(Path(bound["path"]))[0], document):
            raise ValueError("An existing immutable Setup control differs; all artifacts were retained")

    def catalogue(selected):
        def metadata(path):
            return buffers[str(path)] if str(path) in buffers else InputCatalog._read_metadata(path)

        def file_stat(path):
            if str(path) in buffers:
                return SimpleNamespace(st_mode=stat.S_IFREG, st_size=len(buffers[str(path)]))
            return Path(path).stat()

        return InputCatalog(selected, metadata_bytes=metadata, file_stat=file_stat)

    task = _task(operation, config, dict(job_id=job_id, commit=commit, precision="float32", threads=1),
                 roots, catalogue)
    local_roots = stores.roots(series_id)
    for row in task["input_catalog"]["inputs"]:
        alias = row["reference"]["root_alias"]
        if alias not in local_roots or Path(local_roots[alias]).resolve() != Path(roots[alias]).resolve():
            raise ValueError("Setup controls require the original configured store root: " + alias)
    for build in plan["build_plans"]:
        request = build["request_template"]
        pins = [(request[key], role) for key, role in
                (("baseline", "image"), ("profile", "rig_model"), ("shape_basis_certificate", "rig_model"))]
        pins.extend((row["source"], "protocol") for row in request["protocol_inputs"])
        for bound, role in pins:
            reference = input_reference(bound["path"], roots, sha256=bound["sha256"],
                                        size_bytes=bound["size_bytes"], role=role)
            alias = reference["root_alias"]
            if alias not in local_roots or Path(local_roots[alias]).resolve() != Path(roots[alias]).resolve():
                raise ValueError("Setup scientific inputs require their original configured store root: " + alias)
    shared_output_admission(root)


def _raw_operation_root(local_roots, series_id, job_id, revision_id):
    """Use the explicit working store, or the series' shared Results location.

    Retained native output is operation-owned; no service-cache lease or cleanup
    entitlement is created. The writer admits either actual local or shared IO.
    """
    from .rig_raw_write import _canonical
    from .series_paths import series_key
    base = local_roots.get("working")
    if base:
        retained = Path(base) / "retained-outputs" / "setup" / series_key(series_id)
    else:
        retained = Path(local_roots["results"])
    return _canonical(retained / job_id / "raw-models" / revision_id)


def prepare_current_series_setup(project, series_root, *, target_machine, job_root, commit,
                                 environment_lock, stores_path=None, plan_only=False):
    """Register Apply from saved metadata with explicit original RAW writer caps."""
    from .geometry_facies import GeometryRevisionStore
    from .job_filesystem import inspect_results_root, valid_results_admission
    from .project import FluidFlowerProject
    from .rig_raw_basis import protocol_sources
    from .rig_raw_input import camera_binding
    from .rig_raw_write import _canonical
    from .series_geometry import native_series_link
    from .series_native_context import camera_u_profile, read_series_native_binding
    from .store_map import read_store_map
    from darsia.presets.workflows.config.data import DataConfig
    from darsia.presets.workflows.config.protocols import ProtocolsConfig
    from darsia.presets.workflows.config.rig import RigConfig
    from darsia.presets.workflows.config.utils import _get_section_from_toml

    series, root = _canonical(series_root), _canonical(job_root)
    job_id = name(root.name)
    saved_project = FluidFlowerProject.load(project.root)
    if not _same(saved_project.manifest, project.manifest):
        raise ValueError("The current project manifest changed before Setup preparation")
    link = native_series_link(saved_project)
    if not link.get("series_root") or _canonical(link["series_root"]) != series:
        raise ValueError("The selected project lacks this exact native series link")
    manifest_bound = _metadata(series / "series.toml")
    _, manifest, roots = _manifest(manifest_bound)
    stores = read_store_map(stores_path, forbidden_roots=[series, project.root])
    if name(stores.machine_id) != name(target_machine):
        raise ValueError("The selected Setup target differs from the actual local store target")
    local_roots = stores.roots(manifest["series_id"])
    if Path(local_roots["series"]).resolve() != series:
        raise ValueError("The configured target names another native series root")
    store = GeometryRevisionStore(saved_project)
    current, _ = _control(store.current_path)
    revision_id = current.get("revision_id", "")
    if name(revision_id) != revision_id:
        raise ValueError("The saved revision needs its original safe lowercase identity")
    revision, adopted = _control(store.artifact_root / revision_id / "revision.json")
    if not _same(revision, current):
        raise ValueError("The current geometry differs from its immutable saved revision")
    operation_root = _raw_operation_root(local_roots, manifest["series_id"], job_id, revision_id)
    config, frame = current["configuration"], current["configuration"].get("frame_binding", {})
    declared = manifest.get("runs", {}).get("ids", [])
    if not isinstance(declared, list) or len(set(declared)) != len(declared):
        raise ValueError("The series requires its distinct declared run order")
    adopted_runs = set()
    for row in config.get("run_geometry", {}).get("runs", ()):
        if not row.get("enabled", True) or not row.get("registration", {}).get("target_model"):
            continue
        run = row.get("run_id")
        if run not in declared:
            raise ValueError("An enabled adopted target model names an undeclared series run: " + str(run))
        if run not in frame.get("runs", {}):
            raise ValueError("An enabled adopted run lacks its exact frame_binding entry: " + str(run))
        adopted_runs.add(run)
    runs = [run for run in declared if run in adopted_runs]
    display = frame.get("display_reference_run")
    if not runs or display not in runs or APPLY_RUN in runs:
        raise ValueError("The saved adopted runs lack their declared display reference or unique coordinator")
    native = read_series_native_binding(series)
    if native is None:
        raise ValueError("The series lacks its original bound native parameter JSON")
    native_bound = dict(path=str(native.path), sha256=native.sha256,
                        size_bytes=native.path.stat().st_size)
    read_bound(native_bound)
    builds, writer_paths, raw_roots = [], [root, Path(project.root) / "exports"], []
    for run in runs:
        paths, selected = _paths(series, manifest, run, frame["runs"][run],
                                 project_root=project.root, revision=revision_id)
        if _get_section_from_toml(paths, "data").get("use_cache", False) is not False:
            raise ValueError("Setup requires data.use_cache=false before reading DataConfig: " + run)
        data = DataConfig().load(paths, require_data=False, require_results=True)
        protocols = protocol_sources(ProtocolsConfig().load(paths))
        if not all(any(row["role"] == role for row in protocols)
                   for role in ("imaging", "injection", "pressure_temperature")):
            raise ValueError("The run lacks its original imaging, injection or pressure protocol: " + run)
        RigConfig().load(paths, results=data.results)
        target = _canonical(operation_root / run)
        if target.exists() or not valid_results_admission(inspect_results_root(target), target):
            raise ValueError("The operation-owned RAW destination must be fresh and on an admitted local or shared filesystem: " + run)
        profile = camera_u_profile(series, run)
        if profile is None or profile.path != native.path or profile.sha256 != native.sha256:
            raise ValueError("The run lacks its exact current native camera-U profile: " + run)
        camera = camera_binding(dict(configuration=str(native.path), sha256=native.sha256,
                                     profile_id=profile.profile_id, run_id=run))
        basis, evidence = _basis(config, run)
        previous_profile = evidence.get("inputs", {}).get("profile", {})
        previous_camera = evidence.get("inputs", {}).get("camera_binding", {})
        if (any(previous_profile.get(key) != native_bound[key] for key in ("sha256", "size_bytes"))
                or previous_camera.get("profile_id") != profile.profile_id):
            raise ValueError("The current camera profile differs from the original RAW shape basis: " + run)
        baseline = deepcopy(evidence.get("inputs", {}).get("raw"))
        if not isinstance(baseline, dict):
            raise ValueError("The adopted RAW shape basis lacks its baseline binding: " + run)
        baseline = _current_baseline(saved_project, run, baseline, data.baseline)
        if data.baseline.stat().st_size != baseline.get("size_bytes"):
            raise ValueError("The current baseline differs from the adopted RAW shape-basis binding: " + run)
        protected = _domains([*data.folders, data.baseline.parent])
        if any(target == path or target.is_relative_to(path) or path.is_relative_to(target) for path in protected):
            raise ValueError("The declared RAW target overlaps an acquisition root: " + run)
        # The injection rows also carry the schedule digest, so a later corrected
        # injection position keeps this Setup valid (protocol_content).
        from .protocol_content import with_content
        request = dict(format="darsia-raw-rig-build-request-v1", run_id=run,
            series_context=dict(root=str(series), manifest=manifest_bound), baseline=baseline,
            profile=deepcopy(native_bound), operation_overrides={}, protocol_inputs=with_content(protocols),
            shape_basis_certificate=basis, output=dict(destination=str(target), intended_root=str(target.parent),
                forbidden_raw_roots=list(map(str, protected)), **OUTPUT_CAPS))
        settings = dict(run_id=run, raw_camera=camera, rig_path=str(target))
        if ("analysis_exclusions" in config or "target_corrections" in config
                or "local_map_reuse" in (config.get("provenance") or {})):
            from .painted_analysis_exclusions import adopted_run_native_product
            product = adopted_run_native_product(current, run, project_root=project.root)
            if product is not None:
                request["native_product"] = deepcopy(product)
                settings["native_product"] = deepcopy(product)
                if product["analysis_exclusions"] is not None:
                    request["analysis_exclusions"] = deepcopy(product["analysis_exclusions"])
                    settings["analysis_exclusions"] = deepcopy(product["analysis_exclusions"])
        builds.append(dict(run_id=run, target_machine=target_machine, native_parameters=deepcopy(native_bound),
            configuration_paths=list(map(str, paths)), request_template=request, metadata_output=deepcopy(OUTPUT_CAPS),
            raw_setup_settings=settings))
        writer_paths.extend(path.parent for path in [*paths, *selected])
        raw_roots.append(target.parent)
    if len({row["request_template"]["output"]["destination"] for row in builds}) != len(builds):
        raise ValueError("Declared RAW rig destinations are not distinct between runs")
    writer_paths.extend(Path(local_roots["results"]) / job_id / run / ".attempts" for run in [APPLY_RUN, *runs])
    metadata_roots = _domains(writer_paths)
    for build in builds:
        raw_target = Path(build["request_template"]["output"]["destination"])
        if any(raw_target == path or raw_target.is_relative_to(path) or path.is_relative_to(raw_target)
               for path in metadata_roots):
            raise ValueError("The original RAW target overlaps a finite Setup metadata domain")
        build["output_scope"] = dict(metadata_roots=list(map(str, metadata_roots)), scratch_roots=[])
    plan = dict(format=PLAN_FORMAT, project_root=str(project.root), project_manifest=_metadata(project.manifest_path),
        adopted_revision=adopted, series_manifest=manifest_bound, native_parameters=native_bound,
        runs=runs, display_reference_run=display, control_root=str(root / "setup-controls"), build_plans=builds,
        target_machine=target_machine, metadata_output=deepcopy(OUTPUT_CAPS),
        output_scope=dict(metadata_roots=list(map(str, metadata_roots)), scratch_roots=[]))
    _preflight(plan, root, roots, stores, manifest["series_id"], commit, environment_lock,
               check_existing=not plan_only)
    if plan_only:
        return dict(plan=plan, numerical_calls_started=False)
    for path in metadata_roots:
        if not valid_results_admission(inspect_results_root(path), path):
            raise ValueError("A finite Setup metadata domain lacks positive filesystem admission: " + str(path))
    # All metadata, namespace and transport checks precede these empty folders.
    for path in [*metadata_roots, *raw_roots]:
        path.mkdir(parents=True, exist_ok=True)
    bound = _immutable(Path(plan["control_root"]) / "plan.json", plan)
    from .corrected_setup import _small
    source_bytes = _small(manifest_bound["path"])
    if (len(source_bytes) != manifest_bound["size_bytes"]
            or hashlib.sha256(source_bytes).hexdigest() != manifest_bound["sha256"]):
        raise ValueError("The saved series changed during Setup preparation")
    snapshot = Path(plan["control_root"]) / "series-manifest.toml"
    if snapshot.exists():
        if snapshot.read_bytes() != source_bytes:
            raise ValueError("The retained Setup series snapshot changed")
    else:
        with snapshot.open("xb") as stream:
            stream.write(source_bytes)
    registered = prepare_setup_job(bound, root, job_id=job_id, commit=commit, environment_lock=environment_lock)
    return dict(registered, selected_runs=list(runs), remaining_runs=[run for run in declared if run not in adopted_runs])
