"""Automatic plans from genuine tiny saved geometry and original input loaders.

The shared shape-basis fixture uses synthetic sensor/registration suppliers;
its original saved-byte, request and model proof remains intact. These tests
exercise metadata planning and the real queue, never RAW decoding or Build.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import toml

from darsia.fluidflower import corrected_setup as original
from darsia.fluidflower import job_filesystem
from darsia.fluidflower import round5_geometry
from darsia.fluidflower import series_setup_plan as planner
from darsia.fluidflower.pipeline.series_setup_jobs import _control
from darsia.fluidflower.project import FluidFlowerProject, PROJECT_FORMAT
from .test_painted_setup_apply import named_setup
from .test_painted_adoption_table import (  # noqa: F401
    imported, synthetic_payload_admission, prepared_painting, _saved,
)
from .test_series_setup_operations import _document


@pytest.fixture
def current_series(imported, synthetic_payload_admission, tmp_path, monkeypatch, request):
    """Save actual tiny adopted models after declaring complete original TOMLs."""
    mode = getattr(request, "param", "valid")
    project_root = imported.project.root
    raw_root = tmp_path / "owned_basis/raw"
    job_parent, results = tmp_path / "jobs", tmp_path / "claim-results"
    job_parent.mkdir()
    results.mkdir()
    original_dumps = toml.dumps

    def complete_inputs(document, *args, **kwargs):
        if isinstance(document, dict) and document.get("calibration") == {"marker": "unchanged"}:
            document = deepcopy(document)
            run = Path(document["data"]["results"]).name
            output = project_root / "results/native_data" / run
            output.mkdir(parents=True, exist_ok=True)
            document["data"].update(folder=str(raw_root), baseline=run + ".ARW", results=str(output),
                                    use_cache=(mode == "cache-enabled"))
            document["rig"] = dict(width=.9, height=.49, dim=2,
                                    path=str(project_root / "results/rig_models" / run))
            document["protocols"] = dict(imaging={str(raw_root): str(tmp_path / "owned_basis/imaging.csv")},
                injection=str(tmp_path / "owned_basis/injection.csv"),
                pressure_temperature=str(tmp_path / "owned_basis/pressure_temperature.csv"))
        return original_dumps(document, *args, **kwargs)

    monkeypatch.setattr(toml, "dumps", complete_inputs)
    data = named_setup.__wrapped__(imported, monkeypatch, synthetic_payload_admission)
    manifest = toml.loads((data.series / "series.toml").read_text())
    manifest.update(schema_version=1, series_id="synthetic", roots=dict(series=".", workspace=str(project_root),
        fixtures=str(tmp_path / "owned_basis"), models=str(tmp_path / "models"),
        jobs=str(job_parent), results=str(results)))
    # An unadopted declaration must not become an invented Build run.
    manifest["runs"]["ids"] = ["two", "one", "unused"]
    (data.series / "series.toml").write_text(toml.dumps(manifest), encoding="utf-8")
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    from darsia.fluidflower.pipeline.run_io import _CODE_ROLES
    identities = {role: hashlib.sha256(subprocess.check_output(
        ["git", "-C", str(repository), "show", commit + ":" + relative])).hexdigest()
        for role, relative in _CODE_ROLES.items() if role != "pilot_math_module_sha256"}
    write_document = round5_geometry.write_new_json

    def native_identity(path, document):
        if Path(path).name == "camera.json":
            document = deepcopy(document)
            document["source"].update(execution_identity=identities,
                required_execution_identity_keys=list(identities))
            if mode == "native-large":
                document["source"]["metadata_padding"] = "x" * (1024**2 + 1)
        return write_document(path, document)

    monkeypatch.setattr(round5_geometry, "write_new_json", native_identity)
    prepared, *_ = prepared_painting(data, tmp_path)
    _saved(data, prepared)
    data.project.manifest.update(schema=1, format=PROJECT_FORMAT, project_id="fixture")
    _document(project_root / "series.json", data.project.manifest)
    project = FluidFlowerProject.load(project_root)
    current = original.GeometryRevisionStore(project).current()
    native = tmp_path / "owned_basis/camera.json"
    admin = tmp_path / "machine-settings"
    admin.mkdir()
    stores_path = admin / "stores.toml"
    roots = toml.loads((data.series / "series.toml").read_text())["roots"]
    roots = {key: str(data.series) if key == "series" else value for key, value in roots.items()}
    stores_path.write_text(toml.dumps(dict(machine=dict(id="fixture"),
        stores=dict(synthetic=roots))), encoding="utf-8")

    def local(path, **kwargs):
        container = Path(path).resolve().parent
        while not container.exists():
            container = container.parent
        return job_filesystem.LocalFileAdmission(True, "fixture-local", "synthetic OS observation",
                                                 "private test only", str(container))

    monkeypatch.setattr(job_filesystem, "inspect_local_file", local)
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda path, **kwargs:
        SimpleNamespace(allowed=True, protocol="fixture-shared", reason="synthetic transport",
                        evidence="private test only"))
    return SimpleNamespace(project=project, series=data.series, current=current, native=native,
        native_bytes=native.read_bytes(), roots=roots, stores_path=stores_path,
        job_root=job_parent / "setup-job", commit=commit, data=data)


def _prepare(case, **kwargs):
    """Use the exact selected local target and requested job directory."""
    return planner.prepare_current_series_setup(case.project, case.series, target_machine="fixture",
        job_root=case.job_root, commit=case.commit,
        environment_lock=dict(path="requirements-worker.lock", sha256="2" * 64),
        stores_path=case.stores_path, **kwargs)


@pytest.mark.parametrize('imported', ['agree'], indirect=True)
def test_current_raw_targets_are_separate_from_the_historical_geometry_source(current_series):
    case = current_series
    manifest_path = case.series / 'series.toml'
    manifest = toml.loads(manifest_path.read_text())
    declared = manifest['runs']
    declared['geometry_configuration_bindings'] = {}
    for run in ('one', 'two'):
        source = case.current['configuration']['frame_binding']['runs'][run]['configuration']
        prior = []
        for key, role in (('config_files', 'runs'), ('merged_files', 'merged')):
            default = manifest.get('paths', {}).get(
                'run_configs' if role == 'runs' else 'merged_configs', 'config/' + role)
            mapping = declared.setdefault(key, {})
            old = case.series / mapping.get(run, str(Path(default) / (run + '.toml')))
            prior.append(original._descriptor(old))
            target = case.series / 'current-raw' / role / (run + '.toml')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(old.read_bytes() + b'\n# current declared destination\n')
            mapping[run] = str(target.relative_to(case.series))
        declared['geometry_configuration_bindings'][run] = dict(
            run_id=run, source_configuration=original._identity(source),
            apply_configurations=prior, current_configuration=prior[-1])
    manifest_path.write_text(toml.dumps(manifest), encoding='utf-8')
    result = _prepare(case)
    assert result['selected_runs'] == ['two', 'one']
    plan, _ = _control(case.job_root / 'setup-controls/plan.json')
    for build in plan['build_plans']:
        run = build['run_id']
        saved = case.current['configuration']['frame_binding']['runs'][run]['configuration']
        bundle = original._configuration_bundle(case.series, run, original._identity(saved),
            project_root=case.project.root, revision=case.current['revision_id'])
        raw = bundle['dual_configuration_binding']
        assert raw['format'] == original.RAW_CONFIGURATION_FORMAT
        assert raw['source_configuration'] == original._identity(saved)
        assert build['configuration_paths'][-1] == raw['current_configuration']['path']
        assert raw['current_configuration']['path'] != saved['path']
        assert raw['shape_basis_certificate'] == build['request_template']['shape_basis_certificate']
        source_bytes = Path(saved['path']).read_bytes()
        assert hashlib.sha256(source_bytes).hexdigest() == saved['sha256']
        wrong_source = dict(saved, sha256='0' * 64)
        with pytest.raises(ValueError, match='historical configuration'):
            original._configuration_bundle(case.series, run, original._identity(wrong_source),
                project_root=case.project.root, revision=case.current['revision_id'])
        merged = Path(raw['current_configuration']['path'])
        merged.write_bytes(merged.read_bytes() + b'# changed after export\n')
        with pytest.raises(ValueError, match='identity mismatch'):
            original._raw_configuration_bundle(case.series, run, original._identity(saved),
                bundle['configuration_bundles'][run], project_root=case.project.root,
                revision=case.current['revision_id'], destination=case.job_root / run,
                receipt=dict(binding=dict(dual_configuration_binding=raw)))


def test_baseline_relocation_requires_exact_saved_approval_and_preserves_hash(tmp_path, monkeypatch):
    from darsia.fluidflower import rig_point_review as review
    old = dict(path=str(tmp_path / 'old' / 'baseline.ARW'), sha256='a' * 64, size_bytes=4)
    current = dict(old, path=str(tmp_path / 'new' / 'baseline.ARW'))
    row = dict(proposal_binding=dict(photograph_relocation=dict(original=old, current=current)))
    document = dict(runs=dict(sample=row))
    checked = []
    monkeypatch.setattr(review, 'load_review', lambda project: document)
    monkeypatch.setattr(review, '_check_document', lambda value: checked.append(value))
    monkeypatch.setattr(review, '_check_approval', lambda value, selected: checked.append(selected))
    assert planner._current_baseline(None, 'sample', old, current['path']) == current
    assert checked == [document, row]
    current['sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='exact approved relocation'):
        planner._current_baseline(None, 'sample', old, current['path'])


@pytest.mark.parametrize('imported', ['agree'], indirect=True)
def test_mixed_case_local_machine_identity_prepares_real_setup(current_series):
    case = current_series
    settings = toml.loads(case.stores_path.read_text(encoding='utf-8'))
    settings['machine']['id'] = 'FiXtUrE'
    case.stores_path.write_text(toml.dumps(settings), encoding='utf-8')
    saved_settings = case.stores_path.read_bytes()
    with pytest.raises(ValueError, match='differs from the actual local store target'):
        planner.prepare_current_series_setup(case.project, case.series,
            target_machine='another-target', job_root=case.job_root, commit=case.commit,
            environment_lock=dict(path='requirements-worker.lock', sha256='2' * 64),
            stores_path=case.stores_path)
    assert not case.job_root.exists()
    result = _prepare(case)
    plan, _ = _control(case.job_root / 'setup-controls/plan.json')
    assert result['selected_runs'] == ['two', 'one']
    assert plan['target_machine'] == 'fixture'
    assert all(row['target_machine'] == 'fixture' for row in plan['build_plans'])
    assert (case.job_root / 'queue/setup-apply.json').is_file()
    assert case.stores_path.read_bytes() == saved_settings


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_current_saved_series_automatically_registers_only_adopted_apply(current_series, monkeypatch):
    """Reuse bound RAW pins and original profiles without reading acquisitions."""
    case = current_series
    sources = {path: path.read_bytes() for path in case.project.root.parent.rglob("*")
               if path.is_file() and path.suffix.upper() != ".ARW"}
    open_path = Path.open

    def no_raw_reads(path, mode="r", *args, **kwargs):
        if path.suffix.upper() == ".ARW" and "r" in mode:
            raise AssertionError("Metadata planning must reuse the bound baseline without opening RAW")
        return open_path(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_raw_reads)
    result = _prepare(case)
    assert result["selected_runs"] == ["two", "one"] and result["remaining_runs"] == ["unused"]
    plan, _ = _control(case.job_root / "setup-controls/plan.json")
    assert plan["runs"] == ["two", "one"] and plan["display_reference_run"] == "one"
    assert [row["run_id"] for row in plan["build_plans"]] == ["two", "one"]
    assert plan["metadata_output"] == dict(maximum_output_bytes=8 * 1024**3,
                                          maximum_file_bytes=4 * 1024**3)
    assert plan["control_root"] == str(case.job_root / "setup-controls")
    assert case.native.read_bytes() == case.native_bytes
    parameters = json.loads(case.native_bytes)
    assert "raw_execution_identity" not in parameters["source"]
    assert len(parameters["source"]["execution_identity"]) == 5
    for build in plan["build_plans"]:
        run = build["run_id"]
        row = next(item for item in case.current["configuration"]["run_geometry"]["runs"] if item["run_id"] == run)
        model = planner.read_bound(row["registration"]["target_model"])[0]
        certificate = planner.read_bound(model["baseline_native_grid"]["renderer"])[0]
        request = build["request_template"]
        assert request["baseline"] == certificate["inputs"]["raw"]
        assert request["shape_basis_certificate"] == model["baseline_native_grid"]["renderer"]
        assert request["operation_overrides"] == {}
        output_root = Path(case.roots["results"]) / case.job_root.name / "raw-models" / case.current["revision_id"]
        assert request["output"]["destination"] == str(output_root / run)
        assert request["output"]["intended_root"] == str(output_root)
        assert {item["role"] for item in request["protocol_inputs"]} == {
            "imaging", "injection", "pressure_temperature"}
        assert build["raw_setup_settings"]["raw_camera"] == dict(configuration=str(case.native),
            sha256=build["native_parameters"]["sha256"], profile_id="camera", run_id=run)
        assert build["raw_setup_settings"]["rig_path"] == request["output"]["destination"]
        for metadata in plan["output_scope"]["metadata_roots"]:
            assert not Path(request["output"]["destination"]).is_relative_to(Path(metadata))
        assert not Path(request["output"]["destination"]).exists()
    assert all(path.read_bytes() == raw for path, raw in sources.items())
    assert [path.name for path in (case.job_root / "queue").glob("*.json")] == ["setup-apply.json"]
    assert [item["run"] for item in result["tasks"]] == ["setup-apply"]
    assert not list((case.project.root / "exports").iterdir())


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
@pytest.mark.parametrize("current_series", ["native-large"], indirect=True)
def test_large_original_native_document_is_retained_without_raw_reads_or_model_writes(current_series, monkeypatch):
    """Queue the exact original native JSON beyond the small control ceiling."""
    case = current_series
    assert len(case.native_bytes) > 1024**2
    sources = {path: path.read_bytes() for path in case.project.root.parent.rglob("*")
               if path.is_file() and path.suffix.upper() != ".ARW"}
    open_path = Path.open

    def no_raw_reads(path, mode="r", *args, **kwargs):
        if path.suffix.upper() == ".ARW" and "r" in mode:
            raise AssertionError("Metadata planning must not open RAW")
        return open_path(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_raw_reads)
    result = _prepare(case)
    plan, _ = _control(case.job_root / "setup-controls/plan.json")
    original_native = dict(path=str(case.native), sha256=hashlib.sha256(case.native_bytes).hexdigest(),
                           size_bytes=len(case.native_bytes))
    assert plan["native_parameters"] == original_native
    parameters = json.loads(case.native_bytes)
    assert "raw_execution_identity" not in parameters["source"]
    assert len(parameters["source"]["execution_identity"]) == 5
    for build in plan["build_plans"]:
        assert build["native_parameters"] == original_native
        assert build["request_template"]["profile"] == original_native
        certificate = planner.read_bound(build["request_template"]["shape_basis_certificate"])[0]
        assert certificate["inputs"]["profile"] == original_native
        assert not Path(build["request_template"]["output"]["destination"]).exists()
    assert all(path.read_bytes() == raw for path, raw in sources.items())
    assert case.native.read_bytes() == case.native_bytes
    assert [row["run"] for row in result["tasks"]] == ["setup-apply"]
    assert (case.job_root / "queue/setup-apply.json").is_file()
    assert not list((case.project.root / "exports").iterdir())


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
@pytest.mark.parametrize("current_series", ["cache-enabled"], indirect=True)
def test_enabled_original_data_cache_refuses_before_loader_or_any_writer(current_series, monkeypatch):
    """An inadmissible cache flag cannot let DataConfig create cache folders."""
    from darsia.presets.workflows.config.data import DataConfig

    def forbidden(*args, **kwargs):
        raise AssertionError("DataConfig must not load before the cache flag check")

    monkeypatch.setattr(DataConfig, "load", forbidden)
    case = current_series
    with pytest.raises(ValueError, match="data.use_cache=false"):
        _prepare(case)
    assert not case.job_root.exists()
    assert not (case.project.root / "exports").exists()
    assert case.native.read_bytes() == case.native_bytes


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_populated_original_rig_path_is_preserved_while_fresh_operation_is_prepared(current_series):
    """An old model stays intact while the selected operation gets its own target."""
    case = current_series
    output = case.project.root / "results/rig_models/two"
    output.mkdir(parents=True)
    retained = output / "retained-partial.bin"
    retained.write_bytes(b"retained private output")
    original_config = Path(case.current["configuration"]["frame_binding"]["runs"]["two"]["configuration"]["path"])
    original_bytes = original_config.read_bytes()
    result = _prepare(case)
    plan, _ = _control(case.job_root / "setup-controls/plan.json")
    build = next(row for row in plan["build_plans"] if row["run_id"] == "two")
    expected = Path(case.roots["results"]) / case.job_root.name / "raw-models" / case.current["revision_id"] / "two"
    assert build["request_template"]["output"]["destination"] == str(expected)
    assert build["raw_setup_settings"]["rig_path"] == str(expected)
    assert result["selected_runs"] == ["two", "one"] and not expected.exists()
    assert retained.read_bytes() == b"retained private output"
    assert original_config.read_bytes() == original_bytes
    assert (case.job_root / "queue/setup-apply.json").is_file()
    assert case.native.read_bytes() == case.native_bytes


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_adopted_run_missing_frame_binding_refuses_before_any_job_writer(current_series):
    """A declared adopted target cannot disappear through selection intersection."""
    case = current_series
    damaged = deepcopy(case.current)
    del damaged["configuration"]["frame_binding"]["runs"]["two"]
    store = original.GeometryRevisionStore(case.project)
    _document(store.current_path, damaged)
    _document(store.artifact_root / damaged["revision_id"] / "revision.json", damaged)
    with pytest.raises(ValueError, match="adopted run lacks its exact frame_binding.*two"):
        _prepare(case)
    assert not case.job_root.exists()
    assert not (case.project.root / "exports").exists()
    assert case.native.read_bytes() == case.native_bytes


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_populated_operation_destination_refuses_before_any_job_writer(current_series):
    """A retained target for this exact job and revision cannot be relaunched."""
    case = current_series
    output = Path(case.roots["results"]) / case.job_root.name / "raw-models" / case.current["revision_id"] / "two"
    output.mkdir(parents=True)
    retained = output / "retained-partial.bin"
    retained.write_bytes(b"retained operation output")
    with pytest.raises(ValueError, match="operation-owned RAW destination must be fresh"):
        _prepare(case)
    assert retained.read_bytes() == b"retained operation output"
    assert not case.job_root.exists()
    assert not (case.project.root / "exports").exists()
    assert case.native.read_bytes() == case.native_bytes


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_revision_path_cannot_escape_the_owned_operation_root(current_series):
    """A malformed saved identifier cannot select an outside artifact or target."""
    case = current_series
    damaged = deepcopy(case.current)
    damaged["revision_id"] = "../outside"
    _document(original.GeometryRevisionStore(case.project).current_path, damaged)
    with pytest.raises(ValueError, match="Unsafe run or machine name"):
        _prepare(case)
    assert not case.job_root.exists()
    assert not (case.project.root / "exports").exists()
    assert not (Path(case.roots["results"]) / case.job_root.name).exists()
    assert case.native.read_bytes() == case.native_bytes

@pytest.mark.parametrize("imported", ["agree"], indirect=True)
@pytest.mark.parametrize("working_choice", [False, True])
def test_shared_results_keep_native_outputs_in_explicit_machine_store(current_series, tmp_path, monkeypatch, working_choice):
    """A shared Results root never becomes the local native writer destination."""
    case = current_series
    retained_root = tmp_path / "machine-local-source-store"
    document = toml.loads(case.stores_path.read_text())
    if working_choice:
        document["stores"]["synthetic"]["working"] = str(retained_root)
    document["stores"]["synthetic"]["source_cache"] = str(tmp_path / "other-cache")
    case.stores_path.write_text(toml.dumps(document), encoding="utf-8")
    native_before = case.native.read_bytes()
    original_inspect = job_filesystem.inspect_local_file

    def shared_results_are_not_local(path, **kwargs):
        path = Path(path).resolve()
        if path.is_relative_to(Path(case.roots["results"]).resolve()):
            return job_filesystem.LocalFileAdmission(False, "remote", "shared output", "test", str(path.parent))
        return original_inspect(path, **kwargs)

    monkeypatch.setattr(job_filesystem, "inspect_local_file", shared_results_are_not_local)
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda *a, **k:
        job_filesystem.SharedFolderAdmission(True, "smb", "private remote output", "test"))
    original_open = Path.open

    def no_raw_reads(path, mode="r", *args, **kwargs):
        if path.suffix.upper() == ".ARW" and "r" in mode:
            raise AssertionError("Planning must not decode or hash RAW images")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_raw_reads)
    result = _prepare(case)
    plan, _ = _control(case.job_root / "setup-controls/plan.json")
    base = retained_root / "retained-outputs/setup/synthetic" if working_choice else Path(case.roots["results"])
    expected = base / case.job_root.name / "raw-models" / case.current["revision_id"]
    for build in plan["build_plans"]:
        target = expected / build["run_id"]
        assert build["request_template"]["output"]["destination"] == str(target)
        assert build["raw_setup_settings"]["rig_path"] == str(target)
        assert build["target_machine"] == "fixture"
        assert not target.exists()
    assert expected.is_dir()
    assert result["selected_runs"] == ["two", "one"]
    assert case.native.read_bytes() == native_before
    assert not list((case.project.root / "exports").iterdir())


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_explicit_unadmitted_native_store_refuses_without_falling_back(current_series, tmp_path, monkeypatch):
    """An invalid explicit working store never authorizes writing elsewhere."""
    case = current_series
    retained_root = tmp_path / "configured-remote-source-store"
    document = toml.loads(case.stores_path.read_text())
    document["stores"]["synthetic"]["working"] = str(retained_root)
    case.stores_path.write_text(toml.dumps(document), encoding="utf-8")
    original_inspect = job_filesystem.inspect_results_root

    def denied(path, **kwargs):
        if Path(path).resolve().is_relative_to(retained_root):
            return SimpleNamespace(allowed=False)
        return original_inspect(path, **kwargs)

    monkeypatch.setattr(job_filesystem, "inspect_results_root", denied)
    with pytest.raises(ValueError, match="operation-owned RAW destination must be fresh"):
        _prepare(case)
    assert not case.job_root.exists() and not retained_root.exists()
    assert not (Path(case.roots["results"]) / case.job_root.name).exists()


def test_baseline_alias_reuses_original_pin_without_new_review_or_payload_read(tmp_path, monkeypatch):
    """Explicit relocation changes only the locator, retaining source SHA and size."""
    from darsia.fluidflower import rig_point_review as review
    old = tmp_path / "old photographs"
    old.mkdir()
    (old / "baseline.ARW").write_bytes(b"same")
    baseline = dict(path=str(old / "baseline.ARW"), sha256="a" * 64, size_bytes=4)
    before = deepcopy(baseline)
    moved = tmp_path / "current photographs"
    old.rename(moved)
    registry = tmp_path / "registry.toml"
    registry.write_text(toml.dumps(dict(schema_version=1,
        storage=dict(path_aliases={str(old): str(moved)}))), encoding="utf-8")
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(registry))
    monkeypatch.setattr(review, "load_review", lambda *a: (_ for _ in ()).throw(
        AssertionError("An explicitly aliased photo needs no new visual approval")))
    original_open = Path.open
    def no_raw(path, mode="r", *args, **kwargs):
        if path.suffix.upper() == ".ARW":
            raise AssertionError("A locator comparison must not read the RAW payload")
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_raw)
    actual = moved / "baseline.ARW"
    result = planner._current_baseline(None, "sample", baseline, actual)
    assert result == dict(before, path=str(actual))
    assert baseline == before
