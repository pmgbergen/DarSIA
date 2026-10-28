"""Tiny real queue/control replay and original Apply with private capacity.

Filesystem/volume and existing service-budget observations are synthetic. The
catalogue, claim, volume locks/reservations, source-code Git checks, saved
geometry proof, export, Apply and child queue writes are real. No RAW decode,
Build execution, scientific qualification or remote data is exercised.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import toml

from darsia.fluidflower import corrected_setup as original
from darsia.fluidflower import job_filesystem
from darsia.fluidflower import rig_raw_build
from darsia.fluidflower.pipeline import capacity
from darsia.fluidflower.pipeline import series_setup_jobs as jobs
from darsia.fluidflower.pipeline import series_setup_operations as operations
from darsia.fluidflower.pipeline import series_setup_worker_admission as worker
from darsia.fluidflower.pipeline.alias_job import _manifest
from darsia.fluidflower.pipeline.job_queue import JobQueue, LocalOutputBinding
from darsia.fluidflower.pipeline.run_io import (
    RAW_CODE_ROLES, RunIOBinding, current_binding,
)
from darsia.fluidflower.pipeline.series_setup_protocol import PLAN_FORMAT
from darsia.fluidflower.pipeline.series_setup_run_io import original_geometry_sources
from darsia.fluidflower.pipeline.store_runtime import (
    InputHashMismatch, resolve_catalog, verify_inputs,
)
from darsia.fluidflower.project import PROJECT_FORMAT, FluidFlowerProject
from .test_painted_setup_apply import named_setup  # noqa: F401
from .test_painted_adoption_table import (  # noqa: F401
    imported, synthetic_payload_admission, prepared_painting, _saved,
)
from .test_series_setup_operations import _document, _file
from .test_series_setup_worker_admission import _ServiceAdapter, _ServiceLedger


class _PublicationBoundary:
    """Expose the original queue interface without claiming terminal publication."""

    def __call__(self, *args, **kwargs):
        """Refuse an unexpected terminal publication in this Apply-only proof."""
        raise AssertionError("This proof stops after the actual child queue writes")

    def publish_progress(self, *args, **kwargs):
        """Refuse an unexpected frame publication; Setup has no frame products."""
        raise AssertionError("Setup does not produce frame commits")


@pytest.fixture
def real_setup(named_setup, tmp_path, monkeypatch, request):
    """Declare roots before saving the genuine immutable tiny geometry proof."""
    data = named_setup
    job_parent, results = tmp_path / "jobs", tmp_path / "local-results"
    controls = (job_parent / "setup-job" / "setup-controls"
                if "job-owned" in str(getattr(request, "param", "")) else tmp_path / "controls")
    raw_inputs = tmp_path / "raw-controls"
    for folder in (controls, raw_inputs, job_parent, results):
        folder.mkdir(parents=True, exist_ok=True)
    series_path = data.series / "series.toml"
    document = toml.loads(series_path.read_text())
    document.update(schema_version=1, series_id="synthetic", roots=dict(
        series=".", workspace=str(data.project.root), controls=str(controls),
        raw_controls=str(raw_inputs), jobs=str(job_parent), results=str(results)))
    series_path.write_text(toml.dumps(document), encoding="utf-8")
    relative_results = data.series / "relative-results"
    relative_results.mkdir()
    relative_probe = data.series / "config/path-probe.toml"
    relative_probe.write_text('[data]\nfolder="."\nbaseline="not-decoded.ARW"\n'
        'results="relative-results"\nuse_cache=false\n', encoding="utf-8")
    data.project.manifest.update(schema=1, format=PROJECT_FORMAT, project_id="fixture")
    prepared, *_ = prepared_painting(data, tmp_path)
    _saved(data, prepared)
    original_configurations = {}
    if str(getattr(request, 'param', 'legacy')).startswith('raw-targets'):
        retained = tmp_path / 'retained-source'
        retained.mkdir()
        old_manifest = original._descriptor(series_path)
        (retained / 'series.toml').write_bytes(series_path.read_bytes())
        data.project.manifest['tools']['native_series']['retained_shape_basis_sources'] = dict(
            format='darsia-retained-shape-basis-sources-v1', folder=str(retained),
            aliases=[dict(original=old_manifest, copy=dict(old_manifest, path='series.toml'))])
        document = toml.loads(series_path.read_text())
        for run in ('one', 'two'):
            for key, role in (('config_files', 'runs'), ('merged_files', 'merged')):
                old = data.series / 'config' / role / (run + '.toml')
                original_configurations[old] = old.read_bytes()
                target = data.series / 'current-raw' / role / (run + '.toml')
                target.parent.mkdir(parents=True, exist_ok=True)
                target_bytes = old.read_bytes() + b'\n# current declared destination\n'
                if request.param == 'raw-targets-properties':
                    selected = toml.loads(target_bytes.decode())
                    properties = data.series / 'current-raw' / 'selected-properties.csv'
                    properties.write_bytes(Path(selected['facies']['props']).read_bytes())
                    selected['facies']['props'] = str(properties)
                    target_bytes = toml.dumps(selected).encode()
                target.write_bytes(target_bytes)
                document['runs'].setdefault(key, {})[run] = str(target.relative_to(data.series))
        series_path.write_text(toml.dumps(document), encoding='utf-8')
    project_bound = _document(data.project.root / "series.json", data.project.manifest)
    # The executor must load the real public project class, not the fixture's
    # earlier geometry-only project stand-in.
    assert FluidFlowerProject.load(data.project.root).project_id == "fixture"
    current = original.GeometryRevisionStore(data.project).current()
    revision = original._descriptor(data.project.root / "results/geometry" /
                                    current["revision_id"] / "revision.json")
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    identities = {role: hashlib.sha256(subprocess.check_output(
        ["git", "-C", str(repository), "show", commit + ":" + relative])).hexdigest()
        for role, relative in RAW_CODE_ROLES.items()}
    native_source = dict(raw_execution_identity=identities)
    if getattr(request, "param", "legacy") == "legacy-native":
        native_source = dict(execution_identity=dict(processor_module_sha256="0" * 64),
            required_execution_identity_keys=["processor_module_sha256"])
    native = _document(controls / "native-parameters.json", dict(source=native_source))
    series = original._descriptor(series_path)
    baseline = _file(raw_inputs / "acquisitions/baseline.ARW", b"synthetic RAW; no decode")
    profile = _document(raw_inputs / "profile.json", dict(fixture_profile="opaque"))
    basis = _document(raw_inputs / "basis.json", dict(fixture_basis="not a renderer qualification"))
    caps = dict(maximum_output_bytes=8 * 1024**2, maximum_file_bytes=8 * 1024**2)
    scope = dict(metadata_roots=list(map(str, (data.project.root, data.series / "config",
                                             controls, job_parent, results))), scratch_roots=[])
    if original_configurations:
        scope['metadata_roots'].extend(str(data.series / 'current-raw' / role)
                                       for role in ('runs', 'merged'))
    builds = []
    for run in ("two", "one"):
        request = dict(format=rig_raw_build.REQUEST_FORMAT, run_id=run,
            series_context=dict(root=str(data.series), manifest=series), baseline=baseline,
            profile=profile, operation_overrides={}, protocol_inputs=[], shape_basis_certificate=basis,
            output=dict(destination=str(tmp_path / "raw-output" / run),
                intended_root=str(tmp_path / "raw-output"),
                forbidden_raw_roots=[str(raw_inputs / "acquisitions")], **caps))
        builds.append(dict(run_id=run, target_machine="fixture", native_parameters=native,
            configuration_paths=[str(data.series / "config/common.toml"),
                                 str(data.series / ('current-raw' if original_configurations else 'config')
                                     / 'merged' / (run + '.toml'))],
            request_template=request, metadata_output=caps, output_scope=scope))
    plan = dict(format=PLAN_FORMAT, project_root=str(data.project.root), project_manifest=project_bound,
        adopted_revision=revision, series_manifest=series, native_parameters=native,
        runs=["two", "one"], display_reference_run="one", control_root=str(controls),
        build_plans=builds, target_machine="fixture", metadata_output=caps, output_scope=scope)
    plan_bound = _document(controls / "plan.json", plan)

    def local(path, **kwargs):
        container = Path(path).resolve().parent
        while not container.exists():
            container = container.parent
        return job_filesystem.LocalFileAdmission(True, "fixture-local", "synthetic OS observation",
                                                 "private test only", str(container))

    monkeypatch.setattr(job_filesystem, "inspect_local_file", local)
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda path:
        SimpleNamespace(allowed=True, reason="synthetic transport", evidence="private test only"))
    registration = jobs.prepare_setup_job(plan_bound, job_parent / "setup-job", job_id="setup-job",
        commit=commit, environment_lock=dict(path="requirements-worker.lock", sha256="2" * 64))
    job, _ = jobs._control(Path(registration["job"]["path"]))
    task, _ = jobs._control(Path(registration["tasks"][0]["task"]["path"]))
    config, _ = jobs._control(controls / "apply-operation.json")
    _, _, roots = _manifest(series)
    admin = tmp_path / "capacity-admin"
    admin.mkdir()
    unused = _document(admin / "unused-profile.json", dict(unused=True))
    stores = SimpleNamespace(machine_id="fixture", roots=lambda series_id: {key: str(value) for key, value in roots.items()},
        document=dict(capacity=dict(state_directory=str(admin / "state"),
            profiles={"synthetic": unused}, reserve_bytes=40 * 1024**3)))
    rows = resolve_catalog(task, stores, "synthetic")
    verified = verify_inputs(task, rows)

    def volume(path):
        return dict(volume_id="fixture-volume", volume_anchor=str(tmp_path),
                    free_bytes=10**12, checked_container=str(tmp_path))

    context = dict(host="fixture", service_account_sid="S-1-5-21-100",
                   common_root=str(admin / "common"))
    ledger_type = capacity.CapacityLedger

    def ledger_factory(selected, series_id, root):
        return ledger_type(selected, series_id, root, volume_provider=volume,
                           context_provider=lambda: dict(context))

    import darsia_cache.capacity as cache_capacity
    import darsia_cache.control as cache_control
    import darsia_cache.worker_adapter as cache_worker
    monkeypatch.setattr(capacity, "CapacityLedger", ledger_factory)
    monkeypatch.setattr(cache_capacity, "volume_info", volume)
    monkeypatch.setattr(cache_control, "CacheBudgetLedger", _ServiceLedger)
    monkeypatch.setattr(cache_worker, "WorkerCacheAdapter", _ServiceAdapter)
    owner = dict(machine="fixture-cpu", pid=os.getpid(), process_create_time="12345.125")
    events = []
    service = _ServiceLedger(SimpleNamespace(root=str(admin / "service-cache")), events)
    adapter = _ServiceAdapter(service, owner)
    result_admission = local(results / ".result-probe")

    def output_factory(selected, token):
        path = results / selected["job_id"] / selected["run"] / ".attempts" / token
        return LocalOutputBinding("fixture", results, path, result_admission)

    runtime = SimpleNamespace(root=job_parent / "setup-job", job=job, stores=stores,
        service_id="fixture-cpu", backend="cpu", commit=commit, cache_adapter=adapter,
        setup_verified_inputs={task["attempt_id"]: verified}, output_factory=output_factory)
    helper = worker.setup_claim_admission(runtime, task, config, owner)
    helper.acquire()
    try:
        queue = JobQueue(runtime.root, machine_id="fixture")
        claim = queue.claim("fixture-cpu", "cpu", owner, availability=lambda selected: rows,
            output_factory=output_factory, publication_factory=_PublicationBoundary(),
            claim_controls=helper.preclaim_scope)
        assert claim is not None
        helper.bind(claim)
    finally:
        helper.unlock()
    binding = RunIOBinding.replay(claim.task, verified, claim.local_output,
                                 repository=repository, admitted_commit=commit)
    return SimpleNamespace(data=data, plan=plan, revision=current, task=task, config=config,
        helper=helper, binding=binding, claim=claim, queue=queue, runtime=runtime, events=events,
        plan_bytes=Path(plan_bound["path"]).read_bytes(), native=native, relative_probe=relative_probe,
        original_configurations=original_configurations)


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
@pytest.mark.parametrize('real_setup', ['legacy', 'raw-targets', 'raw-targets-properties', 'legacy-native', 'raw-targets-job-owned'], indirect=True)
def test_actual_setup_claim_replay_apply_and_ordered_child_enqueue(real_setup, monkeypatch):
    """Original scientific proof runs after real admission without widening RunIO."""
    case = real_setup
    binding, sink = case.binding, case.helper.sink
    assert case.helper.admission.bound_reservations
    assert binding.catalogue_scope == "series-setup-controls-only"
    assert {"raw_only." + role for role in RAW_CODE_ROLES} <= {
        row["role"] for row in binding.code_audit}
    science = case.revision["configuration"]["frame_binding"]["runs"]["one"]["fields"]["labels"]
    locations = dict(binding.locations)
    legacy_files = {}
    if Path(case.plan["control_root"]).name == "setup-controls":
        # Existing canonical exports and even an unfinished old Apply do not
        # belong to this new job's transaction or RAW output generation.
        old = original.export_corrected_setup(case.data.project, reference=SimpleNamespace(run_id="one"))
        original.apply_corrected_selected_run(case.data.project, old, rebuild_facies_by_name=True)
        child = original.export_corrected_setup(case.data.project, reference=SimpleNamespace(run_id="two"))
        (child.destination / "apply-claim.json").write_text('{"older":"unfinished"}')
        legacy_files = {path: path.read_bytes() for path in old.destination.rglob("*") if path.is_file()}
    common = case.data.series / "config/common.toml"
    before = common.read_bytes()
    require_export = original.require_corrected_export_for_build
    export_run = original.export_corrected_setup
    apply_run = original.apply_corrected_selected_run
    path_checks, applied_runs, exported_bindings, progress = [], [], {}, []

    def retain_export(*args, **kwargs):
        exported = export_run(*args, **kwargs)
        exported_bindings[exported.run_id] = deepcopy(exported.receipt["binding"])
        return exported

    def apply_original(project, exported, **kwargs):
        applied_runs.append(exported.run_id)
        return apply_run(project, exported, **kwargs)

    def unexpected_cohort_proof(*args, **kwargs):
        raise AssertionError("Apply must use the original selected-run export binding")

    def check_original_paths(paths):
        from darsia.presets.workflows.config.data import DataConfig
        from darsia.presets.workflows.config.utils import _with_storage_aliases
        assert current_binding().series_setup_original_reader is True
        observed = DataConfig().load(case.relative_probe, require_data=False, require_results=True)
        assert observed.results == case.data.series / "relative-results"
        assert observed.folders == [case.data.series / "."]
        assert observed.use_cache is False and observed.cache is None
        section = dict(path="fixture-original-path")
        assert _with_storage_aliases(section) is section
        path_checks.append(observed.results)
        return require_export(paths)

    monkeypatch.setattr(original, "require_corrected_export_for_build", check_original_paths)
    monkeypatch.setattr(original, "export_corrected_setup", retain_export)
    monkeypatch.setattr(original, "apply_corrected_selected_run", apply_original)
    monkeypatch.setattr(operations, "_geometry_sources", unexpected_cohort_proof)
    with binding.activate():
        with pytest.raises(InputHashMismatch, match="operation admission"):
            original_geometry_sources(case.data.project, case.revision, case.plan["runs"])
        with pytest.raises(InputHashMismatch, match="outside|Unlisted"):
            binding.asset_path(science["path"])
        with sink.activate():
            result = operations.execute_setup(case.claim.task, case.claim.output, "cpu", "float32", progress.append)
            assert current_binding() is binding
            children = jobs.enqueue_setup_builds(case.runtime.root, case.claim, result)
            repeated = jobs.enqueue_setup_builds(case.runtime.root, case.claim, result)
            assert all(row["already_enqueued"] for row in repeated)
        with pytest.raises(InputHashMismatch, match="outside|Unlisted"):
            binding.asset_path(science["path"])
    assert current_binding() is None and binding.locations == locations
    assert applied_runs == ["one", "two"]
    assert [event["run"] for event in progress if event["step"] == "reverify_saved_geometry"
            and "message" not in event] == case.plan["runs"]
    assert [row["run"] for row in children] == case.plan["runs"] == ["two", "one"]
    assert len(path_checks) == 2
    assert sorted(path.name for path in (case.runtime.root / "queue").glob("*.json")) == ["one.json", "two.json"]
    for child in children:
        task, _ = jobs._control(Path(child["task"]["path"]))
        child_rows = resolve_catalog(task, case.runtime.stores, "synthetic")
        assert verify_inputs(task, child_rows).run == task["run"]
        assert task["operation"] == "build" and task["target_machine"] == "fixture"
    coordinator = operations._read(result["coordinator_completion"])
    assert [row["run_id"] for row in coordinator["runs"]] == case.plan["runs"]
    if legacy_files:
        for path, previous in legacy_files.items():
            assert path.read_bytes() == previous
        for row in coordinator["runs"]:
            export_path = Path(row["export"]["path"])
            assert export_path.is_relative_to(case.runtime.root / "exports")
            receipt = json.loads(export_path.read_text())
            assert receipt["setup_operation"]["job_id"] == case.runtime.root.name
            assert receipt["setup_operation"]["project_root"] == str(case.data.project.root)
            # A later Build can resolve the original source project from its
            # bound receipt without inheriting the Apply worker's context.
            require_export([Path(item["path"]) for item in row["configuration_files"]])
        assert common.read_bytes() == before  # Already applied identical values.
    else:
        assert common.read_bytes() != before
    assert all(path.read_bytes() == raw for path, raw in case.original_configurations.items())
    assert result["Build_run"] is False and result["optical_calibration_run"] is False
    assert result["worker_references_closed"] is True
    for record in result["guarded_geometry"]:
        exported = exported_bindings[record["run_id"]]
        assert record["registered"] == exported["registered_geometry"]
        assert record["registered"]["guarded_sources"]
        assert all(bound in record["guarded_sources"] for bound in
                   (*exported["sources"].values(), exported["properties"], exported["configured_depth_source"]))
        for bound in record["guarded_sources"]:
            original._verified(bound)
    receipt = result["files"]["operation_result.json"]
    assert Path(receipt["path"]).parent == case.claim.output
    assert operations._read(receipt) == {key: value for key, value in result.items() if key != "files"}
    assert Path(case.config["setup_plan"]["path"]).read_bytes() == case.plan_bytes
    assert not Path(case.plan["build_plans"][0]["request_template"]["output"]["intended_root"]).exists()
    assert sink.written_bytes > receipt["size_bytes"] and sink.open_writers == 0
    assert sink.close()["actual_created_bytes"] == sink.written_bytes and sink.closed
    case.queue.assert_owner(case.claim)
    assert "service-locked" in case.events and case.events[-1] == "service-unlocked"
