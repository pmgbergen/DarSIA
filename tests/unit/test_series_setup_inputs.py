"""Real byte/catalogue controls; synthetic scientific payloads are never decoded.

These fixtures are not genuine geometry adoption, successful corrected Apply,
RAW registration or complete scientific input admission. They exercise actual
InputCatalog and the original full-request reader. Original proof replay stays
postclaim, with its independent native/grid/source tests.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline.job_inputs import InputCatalog
from darsia.fluidflower.pipeline import series_setup_inputs as inputs
from darsia.fluidflower.pipeline.series_setup_protocol import (
    APPLY_COMPLETE_FORMAT, FORMAT, PLAN_FORMAT,
)
from darsia.fluidflower import rig_raw_build
from darsia.fluidflower.store_map import StoreMapError


def _file(path, raw):
    """Capture exact private input bytes before the read-only operation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _doc(path, value):
    """Make a real strict-JSON control descriptor, not an admission mock."""
    return _file(path, (json.dumps(value, allow_nan=False) + "\n").encode())


def _snapshot(root):
    return {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def controls(tmp_path):
    """Supply the actual current typed metadata schemas and tiny opaque leaves."""
    project = tmp_path / "project"
    manifest = _doc(project / "series.json", {"format": "private-project-metadata"})
    revision_id = "a" * 32
    revision = _doc(project / "results" / "geometry" / revision_id / "revision.json",
        dict(format="darsia-fluidflower-geometry-facies-v1", schema=1,
             revision_id=revision_id, configuration={}))
    series = _file(project / "native-series" / "series.toml", b"# exact original series context\n")
    native = _doc(project / "native-parameters.json", {"format": "private-native-parameter-metadata"})
    baseline = _file(tmp_path / "acquisitions" / "baseline.ARW", b"synthetic RAW; must never be decoded")
    profile = _doc(project / "profile.json", {"format": "synthetic-profile"})
    basis = _doc(project / "basis.json", {"format": "synthetic-basis-not-a-native-certificate"})
    original = _file(project / "retained" / "00-before.toml", b"# original private configuration\n")
    configuration = _file(project / "native-series" / "run.toml", b"# actual post-Apply private bytes\n")
    protocol_rows = []
    for role in ("imaging", "injection", "pressure_temperature", "blacklist"):
        source = _file(project / "protocols" / (role + ".csv"), b"time,value\n0,1\n")
        protocol_rows.append(dict(role=role, folder=None, sheet=None, source=source))
    limits = dict(maximum_output_bytes=100_000, maximum_file_bytes=10_000)
    output_scope = dict(metadata_roots=[str(tmp_path / "control")], scratch_roots=[])
    request = dict(format=rig_raw_build.REQUEST_FORMAT, run_id="sample",
        configuration_files=[configuration],
        series_context=dict(root=str(Path(series["path"]).parent), manifest=series),
        baseline=baseline, profile=profile, operation_overrides={}, protocol_inputs=protocol_rows,
        output=dict(destination=str(tmp_path / "outputs" / "fresh"), intended_root=str(tmp_path / "outputs"),
                    forbidden_raw_roots=[str(tmp_path / "acquisitions")], **limits),
        shape_basis_certificate=basis)
    template = {key: deepcopy(value) for key, value in request.items() if key != "configuration_files"}
    build_plan = dict(run_id="sample", target_machine="private-host", native_parameters=native,
        configuration_paths=[configuration["path"]], request_template=template,
        metadata_output=limits, output_scope=output_scope)
    plan = dict(format=PLAN_FORMAT, project_root=str(project), project_manifest=manifest,
        adopted_revision=revision, series_manifest=series, native_parameters=native,
        runs=["sample"], display_reference_run="sample", control_root=str(tmp_path / "control"),
        build_plans=[build_plan], target_machine="private-host", metadata_output=limits,
        output_scope=output_scope)
    plan_path = tmp_path / "control" / "plan.json"
    bound_plan = _doc(plan_path, plan)
    common = dict(format=FORMAT, project_root=str(project), project_manifest=manifest,
        adopted_revision=revision, series_manifest=series, native_parameters=native,
        setup_plan=bound_plan, target_machine="private-host", metadata_output=limits,
        output_scope=output_scope)
    apply = dict(common, run="series-apply", operation="apply", runs=plan["runs"],
                 build_plans=plan["build_plans"])
    destination = project / "exports" / "native_setup" / revision_id
    export_doc = dict(run_id="sample", revision_id=revision_id, current_sha256=revision["sha256"],
                      root_export=None)
    export = _doc(destination / "export.json", export_doc)
    _doc(destination / "apply-claim.json", dict(run_id="sample", revision_id=revision_id))
    attempt = _doc(destination / "apply-attempts" / "one" / "receipt.json",
                   dict(state="complete", run_id="sample", revision_id=revision_id))
    applied_doc = dict(format="darsia-corrected-setup-apply-complete-v1", complete=True,
        run_id="sample", revision_id=revision_id, export=export, attempt_receipt=attempt,
        configurations=[dict(before=original, after=configuration)])
    applied = _doc(destination / "apply-complete.json", applied_doc)
    coordinator_doc = dict(format=APPLY_COMPLETE_FORMAT, complete=True, setup_plan=bound_plan,
        adopted_revision=revision, project_manifest=manifest, series_manifest=series,
        revision_id=revision_id, current_sha256=revision["sha256"],
        runs=[dict(run_id="sample", export=export, apply_completion=applied,
                   configuration_files=[configuration])], Build_run=False, optical_calibration_run=False)
    coordinator_path = tmp_path / "control" / "apply-complete.json"
    coordinator = _doc(coordinator_path, coordinator_doc)
    request_path = tmp_path / "control" / "build-request.json"
    request_bound = _doc(request_path, request)
    build = dict(common, run="sample", operation="build", configuration_files=[configuration],
        raw_build_request=request_bound, apply_completion=applied, coordinator_completion=coordinator)
    return SimpleNamespace(apply=apply, build=build, plan=plan, plan_path=plan_path,
        request=request, request_path=request_path, coordinator=coordinator_doc,
        coordinator_path=coordinator_path, configuration=configuration, baseline=baseline,
        root=tmp_path, revision=revision, export=export, applied=applied)


def _catalog(fixture, **kwargs):
    return InputCatalog({"private": str(fixture.root)}, **kwargs)


def _uses(catalog):
    return {use for row in catalog.entries.values() for use in row["uses"]}


def test_apply_binds_controls_without_reading_mutable_targets_or_future_raw(controls, monkeypatch):
    """Absent future files remain templates, rather than immutable input claims."""
    future = str(controls.root / "not-created" / "after-apply.toml")
    controls.plan["build_plans"][0]["configuration_paths"] = [future]
    controls.apply["build_plans"] = deepcopy(controls.plan["build_plans"])
    controls.apply["setup_plan"] = _doc(controls.plan_path, controls.plan)
    seen = []
    def file_stat(path):
        seen.append(str(path))
        return Path(path).stat()
    def forbidden(*args, **kwargs):
        raise AssertionError("Apply cannot consume a future full RAW request")
    monkeypatch.setattr(rig_raw_build, "read_build_request", forbidden)
    catalog = _catalog(controls, file_stat=file_stat)
    before = _snapshot(controls.root)
    result = inputs.catalogue_setup_inputs(catalog, controls.apply)
    assert result["scope"] == inputs.SCOPE and result["raw_request"] is None
    assert future not in seen and controls.configuration["path"] not in seen
    assert controls.baseline["path"] not in seen
    assert _uses(catalog) == {"setup.project_manifest", "setup.adopted_revision",
                             "setup.series_manifest", "setup.native_parameters", "setup.plan"}
    assert before == _snapshot(controls.root)


def test_build_catalogues_real_original_request_pins_without_decoding_or_writing(controls):
    """All named roles come from actual descriptors; no output is fabricated."""
    catalog = _catalog(controls)
    before, config = _snapshot(controls.root), deepcopy(controls.build)
    result = inputs.catalogue_setup_inputs(catalog, controls.build)
    assert result["scope"] == "series-setup-controls-only"
    assert result["raw_request"] == controls.request
    assert controls.build == config and before == _snapshot(controls.root)
    assert not Path(controls.request["output"]["destination"]).exists()
    assert {"setup.raw.baseline", "setup.raw.profile", "setup.raw.shape_basis",
            "setup.raw_request", "setup.apply_claim", "setup.apply_attempt",
            "setup.applied.0.before", "setup.applied.0.after"} <= _uses(catalog)
    assert {"setup.raw.protocol." + str(index) + "." + row["role"]
            for index, row in enumerate(controls.request["protocol_inputs"])} <= _uses(catalog)
    assert all(row["reference"]["role"] == "protocol" for row in catalog.entries.values()
               if any(use.startswith("setup.raw.protocol.") for use in row["uses"]))


def test_worker_admission_detects_same_size_raw_drift_without_preparation_hashes(controls, monkeypatch):
    """Preparation declares bytes; the real worker admission verifies them."""
    from darsia.fluidflower.pipeline.store_runtime import ResolvedInput, verify_inputs, InputHashMismatch
    path = Path(controls.baseline["path"])
    original_open = Path.open
    def no_raw_open(selected, *args, **kwargs):
        if selected == path:
            pytest.fail("Setup catalogue opened its RAW payload")
        return original_open(selected, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_raw_open)
    catalog = _catalog(controls)
    config = _doc(controls.root / "build-operation.json", controls.build)
    frozen = catalog.collect(config, controls.build)
    monkeypatch.setattr(Path, "open", original_open)
    before = path.read_bytes()
    path.write_bytes(b"X" + before[1:])
    assert path.stat().st_size == len(before)
    rows = [ResolvedInput(row["reference"], controls.root / row["reference"]["relative_path"], tuple(row["uses"]))
            for row in frozen["inputs"]]
    with pytest.raises(InputHashMismatch, match="SHA/bytes"):
        verify_inputs(dict(run=controls.build["run"]), rows)
    with pytest.raises(ValueError, match="changed"):
        rig_raw_build.read_build_request(controls.build["raw_build_request"], artifact_kind="raw_rig_baseline")


def test_native_sources_use_declared_aliases_and_exact_files_in_verified_replay(controls, monkeypatch, tmp_path):
    """Real catalogue/SHA admission and replay, with no numerical claim."""
    import toml
    from darsia.fluidflower import storage_paths
    from darsia.fluidflower.pipeline.series_setup_jobs import _task
    from darsia.fluidflower.pipeline.store_runtime import resolve_catalog, verify_inputs, RunUnavailable
    from darsia.fluidflower.pipeline.run_io import _Replay, RunIOBinding
    from darsia.fluidflower.pipeline.series_setup_run_io import _OriginalSetupReader
    old = Path(controls.baseline["path"])
    moved = controls.root / "moved-acquisitions"
    old.parent.rename(moved)
    registry = tmp_path / "registry.toml"
    registry.write_text(toml.dumps(dict(schema_version=1, storage=dict(path_aliases={str(old.parent): str(moved)}))))
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(registry))
    storage_paths.refresh_storage_paths()
    external = _file(controls.root.parent / "explicit-native-source.npz", b"opaque selected native payload")
    def selected_dependency(request, reader=None):
        if reader is not None:
            reader.path(external)
        return [external]
    monkeypatch.setattr(rig_raw_build, "_native_request_dependencies", selected_dependency)
    config = _doc(controls.root / "build-operation.json", controls.build)
    task = _task(config, controls.build, dict(job_id="fixture", commit="a" * 40,
        precision="float32", threads=1), {"private": str(controls.root)}, InputCatalog)
    assert len(task["input_catalog"]["native_exact_inputs"]) == 1
    assert task["input_catalog"]["payload_hashes_verified"] is False
    stores = SimpleNamespace(document={}, roots=lambda _id: dict(private=str(controls.root),
        series=str(Path(controls.build["series_manifest"]["path"]).parent)))
    rows = resolve_catalog(task, stores, "fixture")
    verified = verify_inputs(task, rows)

    class ReplayInputs:
        # The original methods run on real verified file evidence. This is a
        # pure metadata replay fixture, never an activated execution admission.
        _bind_data_path = RunIOBinding._bind_data_path
        _input_digest = RunIOBinding._input_digest
        _verified_bytes = RunIOBinding._verified_bytes
        _metadata_bytes = RunIOBinding._metadata_bytes
        native_input_digest = RunIOBinding.native_input_digest
        native_metadata_bytes = RunIOBinding.native_metadata_bytes
        asset_path = RunIOBinding.asset_path
        physical_descriptor = RunIOBinding.physical_descriptor
        reader_verifications = RunIOBinding.reader_verifications
        def _own_output(self, _path):
            return False

    runtime = ReplayInputs()
    runtime.task, runtime.verified, runtime.locations, runtime.code_locations = task, verified, {}, {}
    runtime.common_publications = {}
    replay = _Replay(runtime, ())
    for use in ("run.execution_series", "run.series_manifest"):
        row = verified.for_use(use)
        replay.add(dict(path=str(row.path), sha256=row.reference["sha256"], size_bytes=row.reference["bytes"]),
                   "configuration", use)
    actual = replay.collect(config, controls.build)
    replay.bind_declared_aliases()
    assert actual["inputs"] == task["input_catalog"]["inputs"]
    assert actual["native_exact_inputs"] == task["input_catalog"]["native_exact_inputs"]
    runtime.setup_controls = replay.setup_controls
    reader = _OriginalSetupReader(runtime)
    assert reader.asset_path(old) == moved / old.name
    assert reader.native_input_digest(old) == controls.baseline["sha256"]
    assert reader.physical_descriptor(controls.baseline)["path"] == str(moved / old.name)
    assert reader.asset_path(external["path"]) == Path(external["path"])
    from darsia.fluidflower.pipeline.job_inputs import validate_catalog
    invalid = deepcopy(task["input_catalog"])
    invalid["native_exact_inputs"][0]["descriptor"]["path"] = str(controls.root.parent)
    with pytest.raises(ValueError, match="path binding"):
        validate_catalog(invalid)
    with pytest.raises(StoreMapError, match="No declared root"):
        _catalog(controls).add_native(external, "rig_model", "setup.plan")
    Path(external["path"]).unlink()
    with pytest.raises(RunUnavailable, match="Input unavailable"):
        resolve_catalog(task, stores, "fixture")


def test_missing_named_control_refuses_without_creating_a_substitute(controls):
    """A declared native parameter JSON is consumed, rather than guessed."""
    path = Path(controls.build["native_parameters"]["path"])
    moved = path.with_suffix(".retained")
    path.rename(moved)
    with pytest.raises(FileNotFoundError):
        inputs.catalogue_setup_inputs(_catalog(controls), controls.build)
    assert moved.is_file() and not path.exists()


def test_foreign_revision_cannot_borrow_matching_run_controls(controls):
    """The immutable plan's literal revision descriptor is a positive join."""
    other = _doc(controls.root / "foreign-revision.json", dict(format="darsia-fluidflower-geometry-facies-v1",
                 schema=1, revision_id="b" * 32, configuration={}))
    controls.build["adopted_revision"] = other
    with pytest.raises(ValueError, match="another project or adopted revision"):
        inputs.catalogue_setup_inputs(_catalog(controls), controls.build)


@pytest.mark.parametrize("change", ["postimage", "run", "template"])
def test_request_must_match_actual_postimages_run_and_original_template(controls, change):
    """A valid independent full request is insufficient for this operation."""
    if change == "postimage":
        controls.request["configuration_files"] = [_file(controls.root / "foreign.toml", b"other configuration")]
    elif change == "run":
        controls.request["run_id"] = "foreign"
    else:
        controls.request["output"]["maximum_output_bytes"] += 1
    controls.build["raw_build_request"] = _doc(controls.request_path, controls.request)
    with pytest.raises(ValueError, match="full RAW request differs"):
        inputs.catalogue_setup_inputs(_catalog(controls), controls.build)


def test_stale_request_descriptor_refuses_exact_changed_json(controls):
    """Rewriting a current file does not update its previously bound identity."""
    controls.request["output"]["maximum_output_bytes"] += 1
    _doc(controls.request_path, controls.request)
    with pytest.raises(ValueError, match="SHA/size changed|size changed"):
        inputs.catalogue_setup_inputs(_catalog(controls), controls.build)


def test_wrong_request_role_is_not_silently_promoted_to_full_build(controls):
    """The original full-build request role is required, with no basis fallback."""
    controls.request["format"] = rig_raw_build.BASIS_REQUEST_FORMAT
    controls.build["raw_build_request"] = _doc(controls.request_path, controls.request)
    with pytest.raises(ValueError, match="full RAW request differs"):
        inputs.catalogue_setup_inputs(_catalog(controls), controls.build)


def test_other_coordinator_postimages_are_not_selected_by_matching_run_name(controls):
    """The exact selected Apply descriptor list must join, not only the id."""
    controls.coordinator["runs"][0]["configuration_files"] = [
        _file(controls.root / "unrelated.toml", b"unrelated applied bytes")]
    controls.build["coordinator_completion"] = _doc(controls.coordinator_path, controls.coordinator)
    with pytest.raises(ValueError, match="actual selected Apply postimages"):
        inputs.catalogue_setup_inputs(_catalog(controls), controls.build)


def test_byte_identical_remapped_locator_refuses_without_fallback(controls):
    """Existing generic alias capability must not broaden this original route."""
    original = controls.build["native_parameters"]
    copied = _file(controls.root / "copy" / "native.json", Path(original["path"]).read_bytes())
    catalog = _catalog(controls, aliases=[dict(logical=original, physical=copied)])
    with pytest.raises(ValueError, match="original physical locators"):
        inputs.catalogue_setup_inputs(catalog, controls.build)


def test_request_payload_root_admission_precedes_original_hash_reader(controls, monkeypatch):
    """An unadmitted acquisition cannot be opened by the original verifier."""
    request = deepcopy(controls.request)
    request["baseline"] = dict(path=str(controls.root.parent / "outside.ARW"), sha256="b" * 64, size_bytes=9)
    controls.plan["build_plans"][0]["request_template"]["baseline"] = deepcopy(request["baseline"])
    plan = _doc(controls.plan_path, controls.plan)
    controls.build["setup_plan"] = plan
    controls.coordinator["setup_plan"] = plan
    controls.build["coordinator_completion"] = _doc(controls.coordinator_path, controls.coordinator)
    controls.build["raw_build_request"] = _doc(controls.request_path, request)
    def forbidden(*args, **kwargs):
        raise AssertionError("Original hash verifier must not run before root admission")
    monkeypatch.setattr(rig_raw_build, "read_build_request", forbidden)
    with pytest.raises(StoreMapError, match="No declared root"):
        inputs.catalogue_setup_inputs(_catalog(controls), controls.build)
