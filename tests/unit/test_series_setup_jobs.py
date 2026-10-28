"""Actual control catalogues and queue files; no scientific Setup execution."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import job_filesystem
from darsia.fluidflower.pipeline import series_setup_jobs as jobs
from darsia.fluidflower.pipeline.job_inputs import InputCatalog
from .test_series_setup_inputs import controls, _doc, _file  # noqa: F401


@pytest.fixture
def setup_job(controls, monkeypatch):
    """Declare real private roots and keep all future Build paths absent."""
    case = controls
    source = Path(case.plan["series_manifest"]["path"])
    raw = ('schema_version = 1\nseries_id = "synthetic"\n[roots]\nseries = "."\nprivate = '
           + json.dumps(str(case.root)) + '\n').encode()
    series = _file(source, raw)
    case.plan["series_manifest"] = series
    case.plan["output_scope"] = dict(metadata_roots=[case.plan["control_root"], str(case.root / "jobs")],
                                      scratch_roots=[])
    for build in case.plan["build_plans"]:
        build["request_template"]["series_context"]["manifest"] = deepcopy(series)
        build["configuration_paths"] = [str(case.root / "future" / "after-apply.toml")]
        build["output_scope"] = deepcopy(case.plan["output_scope"])
    case.bound_plan = _doc(case.plan_path, case.plan)
    case.job_root = case.root / "jobs" / "setup-job"
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda _path:
        SimpleNamespace(allowed=True, reason="synthetic transport", evidence="private test only"))
    return case


def _prepare(case, **kwargs):
    """Use explicit source identity and the same real registered series."""
    return jobs.prepare_setup_job(case.bound_plan, case.job_root, job_id="setup-job",
        commit="1" * 40, environment_lock=dict(path="requirements-worker.lock", sha256="2" * 64), **kwargs)


def test_large_setup_templates_preflight_queue_and_replay_without_scientific_reads(setup_job):
    """Control-sized opaque templates are not mistaken for a 1 MiB job limit."""
    from darsia.fluidflower.series_setup_plan import _preflight
    from darsia.fluidflower.pipeline.alias_job import _manifest
    from darsia.fluidflower.pipeline.capacity import read_bound
    case = setup_job
    payload = dict(opaque_control_metadata='x' * (1024 * 1024 + 128))
    build = case.plan['build_plans'][0]
    build['request_template']['native_product'] = deepcopy(payload)
    build['raw_setup_settings'] = dict(run_id=build['run_id'],
        raw_camera=dict(configuration=case.plan['native_parameters']['path'],
                        sha256=case.plan['native_parameters']['sha256'],
                        profile_id='fixture', run_id=build['run_id']),
        rig_path=build['request_template']['output']['destination'], native_product=deepcopy(payload))
    case.bound_plan = _doc(case.plan_path, case.plan)
    _, manifest, roots = _manifest(case.plan['series_manifest'])
    _preflight(case.plan, case.job_root, roots, SimpleNamespace(roots=lambda series_id: roots),
        manifest['series_id'], '1' * 40,
        dict(path='requirements-worker.lock', sha256='2' * 64))
    result = _prepare(case)
    task, _ = jobs._control(Path(result['tasks'][0]['task']['path']))
    reference = task['configuration']
    operation, _ = jobs._control(Path(roots[reference['root_alias']]) / reference['relative_path'])
    assert operation['build_plans'][0]['request_template']['native_product'] == payload
    plan, bound = jobs._control(case.plan_path)
    assert bound['size_bytes'] > 1024 * 1024
    assert read_bound(bound)[0] == plan
    assert result['numerical_calls_started'] is False
    assert [path.name for path in (case.job_root / 'queue').glob('*.json')] == ['setup-apply.json']
    assert not (case.job_root / 'results' / build['run_id']).exists()
    case.plan_path.write_bytes(case.plan_path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='changed'):
        read_bound(bound)


def test_setup_controls_still_enforce_their_shared_finite_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, 'CONTROL_LIMIT', 128)
    path = tmp_path / 'oversized.json'
    with pytest.raises(ValueError, match='shared job metadata limit'):
        jobs._immutable(path, dict(metadata='x' * 129))
    assert not path.exists()
    path.write_bytes(b' ' * 129)
    with pytest.raises(ValueError, match='shared job metadata limit'):
        jobs._control(path)


def test_prepare_queues_only_apply_without_future_configuration_or_raw_reads(setup_job):
    """The real catalogue binds both original series roles and no future Build input."""
    case, observed = setup_job, []
    original_plan = case.plan_path.read_bytes()

    def catalog_factory(roots):
        def file_stat(path):
            observed.append(str(path))
            return Path(path).stat()
        return InputCatalog(roots, file_stat=file_stat)

    result = _prepare(case, catalog_factory=catalog_factory)
    job, _ = jobs._control(Path(result["job"]["path"]))
    task, _ = jobs._control(Path(result["tasks"][0]["task"]["path"]))
    assert job["expected_runs"] == ["setup-apply", "sample"]
    assert job["setup_operations"] == [dict(run="setup-apply", operation="apply"),
                                       dict(run="sample", operation="build")]
    assert task["run"] == "setup-apply" and task["ordered_case_ids"] == ["apply"]
    assert job["pipeline"] == task["pipeline"] == "series-setup"
    assert job["format_version"] == task["format_version"] == 1
    assert job["storage_profile"] == task["storage_profile"] == "full"
    assert job["execution_ready"] is True and task["execution_ready"] is True
    assert "calibration_sha256" not in job and "calibration_sha256" not in task
    assert task["series"] == task["series_manifest"]
    series_roles = next(row["uses"] for row in task["input_catalog"]["inputs"]
                        if row["reference"] == task["series"])
    assert {"run.execution_series", "run.series_manifest"} <= set(series_roles)
    assert all("raw." not in use for row in task["input_catalog"]["inputs"] for use in row["uses"])
    forbidden = [case.plan["build_plans"][0]["configuration_paths"][0],
                 *[case.plan["build_plans"][0]["request_template"][key]["path"]
                   for key in ("baseline", "profile", "shape_basis_certificate")]]
    assert not set(forbidden) & set(observed)
    assert [path.name for path in (case.job_root / "queue").glob("*.json")] == ["setup-apply.json"]
    assert not (case.job_root / "results/sample").exists()
    assert case.plan_path.read_bytes() == original_plan
    repeated = _prepare(case, catalog_factory=catalog_factory)
    assert repeated["tasks"][0]["already_enqueued"] is True
    repeated_task, _ = jobs._control(Path(repeated["tasks"][0]["task"]["path"]))
    assert repeated_task == task


@pytest.mark.parametrize("invalid", ["outside_scope", "coordinator_collision", "protected_acquisition",
                                      "build_metadata_ancestor", "build_protected_scratch", "empty_protection"])
def test_invalid_scope_or_coordinator_refuses_before_metadata_writes(setup_job, invalid):
    """A typed but inadmissible output selection cannot create a pending task."""
    case = setup_job
    if invalid == "outside_scope":
        case.plan["output_scope"]["metadata_roots"] = [case.plan["control_root"]]
    elif invalid == "coordinator_collision":
        case.plan["runs"] = ["setup-apply"]
        case.plan["display_reference_run"] = "setup-apply"
        build = case.plan["build_plans"][0]
        build["run_id"] = build["request_template"]["run_id"] = "setup-apply"
    elif invalid == "protected_acquisition":
        case.plan["output_scope"]["metadata_roots"] = [str(case.root)]
    elif invalid == "build_metadata_ancestor":
        case.plan["build_plans"][0]["output_scope"]["metadata_roots"] = [str(case.root)]
    elif invalid == "build_protected_scratch":
        case.plan["build_plans"][0]["output_scope"]["scratch_roots"] = [str(case.root / "acquisitions")]
    else:
        case.plan["build_plans"][0]["request_template"]["output"]["forbidden_raw_roots"] = []
    case.bound_plan = _doc(case.plan_path, case.plan)
    with pytest.raises(ValueError, match="metadata outputs|collides|protected RAW"):
        _prepare(case)
    assert not case.job_root.exists()
    assert not (Path(case.plan["control_root"]) / "apply-operation.json").exists()


def test_shared_transport_refuses_before_initial_control_creation(setup_job, monkeypatch):
    """The original shared-folder observation precedes every producer mkdir."""
    case = setup_job
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda _path:
        SimpleNamespace(allowed=False, reason="synthetic refusal", evidence="private test only"))
    with pytest.raises(ValueError, match="Shared queue transport refused"):
        _prepare(case)
    assert not case.job_root.exists()
    assert not (Path(case.plan["control_root"]) / "apply-operation.json").exists()
