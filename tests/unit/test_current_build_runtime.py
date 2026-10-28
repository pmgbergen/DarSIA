"""UNRUN, unsealed finite reader/refusal fixtures; no qualification evidence.

These exercise original metadata publication and original read-only coordinate
checks. They grant no startup, capacity, RAW scientific or physical eligibility
authority and are not a successful current-Build execution fixture.
"""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import current_build as current
from darsia.fluidflower.pipeline.job_queue import descriptor
from darsia.fluidflower.mass.n1_pipeline import DataReader
from .test_series_setup_results import _published, _put
from .test_build_area_data import area_case  # original finite coordinate fixture
from .test_build_geometry_data import case


class _UnsealedMetadataReader:
    """Actual byte reader for publication-only fixtures; not RunIO admission."""

    def __init__(self):
        self.reader = DataReader()
        self.consumed = {}

    def path(self, bound, *, original=False):
        path = self.reader.path(bound)
        self.consumed[str(path)] = bound
        return path

    def json(self, bound):
        self.path(bound)
        return self.reader.json(bound)

    def recheck(self):
        for bound in self.consumed.values():
            self.reader.path(bound)


def _publication(case):
    root = case.root
    paths = [root / "job.json", root / "results" / case.task["run"] / "task.json",
             root / "done" / (case.task["run"] + ".json"),
             root / "results" / case.task["run"] / "completed.json",
             Path(case.index["path"]), case.lease / "owner.json",
             case.lease / "artifacts" / "completed.json",
             case.lease / "artifacts" / "metadata" / "operation_result.json"]
    rows = sorted((dict(relative_path=path.relative_to(root).as_posix(), descriptor=descriptor(path))
                   for path in paths), key=lambda row: row["relative_path"])
    return dict(root=str(root), job=descriptor(paths[0]), task=descriptor(paths[1]),
                done=descriptor(paths[2]), operation_result=descriptor(paths[-1]), readset=rows)


def test_original_setup_terminal_is_consumed_without_claiming_physical_eligibility(tmp_path):
    case = _published(tmp_path)
    declaration = _publication(case)
    completed, receipt = current._Publication(declaration, _UnsealedMetadataReader()).verify(declaration)
    assert completed == case.result
    assert receipt == case.receipt
    assert receipt["complete_rig_model"] is False
    assert receipt["production_qualification_claimed"] is False


@pytest.mark.parametrize("role", ["index", "indexed_receipt", "owner"])
def test_original_terminal_direct_reads_require_the_named_readset(tmp_path, role):
    case = _published(tmp_path)
    declaration = _publication(case)
    removed = {"index": Path(case.index["path"]), "owner": case.lease / "owner.json",
               "indexed_receipt": case.lease / "artifacts" / "metadata" / "operation_result.json"}[role]
    declaration["readset"] = [row for row in declaration["readset"]
                              if Path(row["descriptor"]["path"]) != removed]
    with pytest.raises(ValueError, match="outside its exact readset"):
        current._Publication(declaration, _UnsealedMetadataReader()).verify(declaration)


def test_original_terminal_rejects_consistently_rebound_foreign_owner(tmp_path):
    case = _published(tmp_path)
    _put(case.lease / "owner.json", dict(case.owner, actual_owner=dict(machine="foreign-cpu", pid=123)))
    declaration = _publication(case)  # updated bytes do not turn the wrong owner into authority
    with pytest.raises(ValueError, match="owner differs"):
        current._Publication(declaration, _UnsealedMetadataReader()).verify(declaration)


def test_publication_readset_cannot_contain_unused_extra_metadata(tmp_path):
    case = _published(tmp_path)
    extra = case.root / "unused.json"
    _put(extra, {"complete": True})
    declaration = _publication(case)
    declaration["readset"].append(dict(relative_path="unused.json", descriptor=descriptor(extra)))
    with pytest.raises(ValueError, match="unconsumed"):
        current._Publication(declaration, _UnsealedMetadataReader()).verify(declaration)


def test_unlisted_build_source_refuses_before_original_decoder_or_output_fallback(tmp_path):
    from darsia.fluidflower.pipeline.run_io import _key
    path = tmp_path / "input.json"
    path.write_text("{}", encoding="utf-8")
    bound = descriptor(path)
    row = SimpleNamespace(reference={"role": "configuration", "sha256": bound["sha256"]})
    reader = object.__new__(current._Reader)
    reader.binding = SimpleNamespace(locations={_key(path): row})
    reader.allowed = []  # a different ordinary consumer does not supply current_build.*
    reader.original = None
    reader.consumed = {}
    with pytest.raises(ValueError, match="named catalogue consumer"):
        reader.path(bound)


def test_missing_active_run_binding_cannot_acquire_a_build():
    with pytest.raises(ValueError, match="actual replayed configuration"):
        current.acquire_current_build({}, {}, None)


def test_arbitrary_checked_object_constructor_is_refused():
    with pytest.raises(ValueError, match="must be acquired"):
        current.CheckedCurrentBuild(object(), config={}, task={}, binding=None,
                                    reader=None, physical=None, bridge=None)


@pytest.mark.parametrize("configuration", [None, [], True])
def test_adopted_physical_revision_requires_original_configuration_mapping(configuration):
    with pytest.raises(ValueError, match="original mapping"):
        current._require_supported_physical_revision({"configuration": configuration})


@pytest.mark.parametrize("adopted", [{"revision_id": "r"}, {"revision_id": "r", "configuration": {}}])
def test_non_opted_original_physical_reader_remains_available(adopted):
    current._require_supported_physical_revision(adopted)


def test_original_area_read_checks_do_not_decode_or_write_a_second_area(area_case):
    from darsia.fluidflower.pipeline.build_area_data import _read_measured_area_context
    arguments = area_case.seal()
    output = arguments.pop("output")
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in area_case.case.root.rglob("*") if path.is_file()}
    context = _read_measured_area_context(**arguments)
    assert context["geometry"]["area_parameters"]["physical_scale_approved"] is False
    assert context["area_fixed"] == .1
    assert not output.exists()
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == value for path, value in before.items())


@pytest.mark.parametrize("field", ["mass_eligible", "bottom_mass_allowed", "vertical_scale_independently_verified"])
def test_rebound_area_declarations_cannot_promote_original_negative_qualification(area_case, field):
    from darsia.fluidflower.pipeline.build_area_data import _read_measured_area_context
    area_case.revision[field] = True
    arguments = area_case.seal()
    arguments.pop("output")
    with pytest.raises(ValueError):
        _read_measured_area_context(**arguments)


def test_mass_boundary_has_no_success_or_eligibility_document_branch():
    """The original eligibility producer exists; its Build role/grid join does not."""
    tree = ast.parse(Path(current.__file__).read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "CheckedCurrentBuild")
    boundary = next(node for node in owner.body if isinstance(node, ast.FunctionDef)
                    and node.name == "check_mass")
    assert isinstance(boundary.body[-1], ast.Raise)
    assert isinstance(boundary.body[-1].exc, ast.Call)
    assert boundary.body[-1].exc.args[0].id == "PHYSICAL_INPUTS_UNAVAILABLE"
    assert not any(isinstance(node, ast.Return) for node in ast.walk(boundary))
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in {"load_frame", "array", "archive", "warp_continuous"}
                   for node in ast.walk(boundary))


def _function(path, name, owner=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body
    if owner is not None:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == owner).body
    return next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)


def _calls(function, name):
    return [node for node in ast.walk(function) if isinstance(node, ast.Call)
            and ((isinstance(node.func, ast.Name) and node.func.id == name)
                 or (isinstance(node.func, ast.Attribute) and node.func.attr == name))]


def test_real_series_consumer_keeps_single_native_and_frame_calls_before_both_phase_branches():
    """AST boundary assertion, not a substitute for an executed RAW/phase fixture."""
    path = Path(current.__file__).with_name("runner.py")
    function = _function(path, "_run_series")
    native, frame, check = (_calls(function, name) for name in ("native", "frame", "check_phase"))
    native = [call for call in native if isinstance(call.func, ast.Attribute)
              and isinstance(call.func.value, ast.Name) and call.func.value.id == "inputs"]
    frame = [call for call in frame if isinstance(call.func, ast.Attribute)
             and isinstance(call.func.value, ast.Name) and call.func.value.id == "inputs"]
    inference = _calls(function, "infer_labels")
    assert len(native) == len(frame) == len(check) == 1
    assert len(inference) == 2
    assert native[0].lineno < frame[0].lineno < check[0].lineno < min(call.lineno for call in inference)
    assert [argument.id for argument in check[0].args] == ["spec", "native", "frame"]
    acquire = _calls(function, "acquire_current_build")
    assert len(acquire) == 1 and acquire[0].lineno < native[0].lineno
    guard = next(node for node in ast.walk(function) if isinstance(node, ast.If)
                 and acquire[0] in list(ast.walk(node)))
    assert "current_build_binding" in ast.unparse(guard.test)
    resume = next(node for node in guard.body if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "task.get('resume_from') is not None")
    assert isinstance(resume.body[0], ast.Raise)
    assert resume.lineno < acquire[0].lineno


def test_original_single_mass_load_is_checked_before_response_preparation():
    path = Path(current.__file__).parents[1] / "mass" / "n1_pipeline.py"
    function = _function(path, "_prepare_statistics", owner="N1Pipeline")
    load, check, response = (_calls(function, name) for name in ("load_frame", "check_mass", "prepare_response"))
    assert len(load) == len(check) == len(response) == 1
    assert load[0].lineno < check[0].lineno < response[0].lineno
    assert [argument.id for argument in check[0].args] == ["spec", "frame", "x", "y", "photo", "report"]
    assert [(keyword.arg, ast.unparse(keyword.value)) for keyword in check[0].keywords] == [
        ("mass_config", "self.config")]


def test_current_build_stops_before_original_physical_readers_or_runtime_setup():
    path = Path(current.__file__).parents[1] / "mass" / "n1_pipeline.py"
    constructor = _function(path, "__init__", owner="N1Pipeline")
    guard = next(node for node in constructor.body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "current_build is not None")
    checks = [node for node in guard.body if isinstance(node, ast.If)]
    for expression in ("type(current_build) is not CheckedCurrentBuild", "frame_loading_only",
                       "'pixel_domain' in config"):
        check = next(node for node in checks if ast.unparse(node.test) == expression)
        assert isinstance(check.body[0], ast.Raise)
    validate, stop = (_calls(guard, name)[0]
                      for name in ("validate_mass_config", "require_physical_inputs"))
    assert [(keyword.arg, ast.unparse(keyword.value)) for keyword in validate.keywords] == [
        ("statistics_only", "statistics_only")]
    assert validate.lineno < stop.lineno
    assert stop.lineno < min(call.lineno for name in ("scientific_thread", "DataReader")
                            for call in _calls(constructor, name))


def test_acquisition_calls_original_publication_and_physical_reader_inside_exact_scope():
    path = Path(current.__file__)
    acquire = _function(path, "acquire_current_build")
    physical = _calls(acquire, "load_build_physical_context")
    checked = _calls(acquire, "_require_supported_physical_revision")
    terminal = _calls(_function(path, "verify", owner="_Publication"), "setup_terminal")
    assert len(physical) == len(checked) == len(terminal) == 1
    assert checked[0].lineno < physical[0].lineno
    assert _calls(acquire, "verify")[0].lineno < physical[0].lineno
    assert [keyword.arg for keyword in physical[0].keywords] == ["reader", "local_reuse"]
    scope = next(node for node in ast.walk(acquire) if isinstance(node, ast.With)
                 and physical[0] in list(ast.walk(node)))
    assert ast.unparse(scope.items[0].context_expr) == "reader.direct()"


def _direct_reader(path, *, allowed=True):
    """Finite actual byte-reader fixture, never a RunIO/startup qualification."""
    from darsia.fluidflower.pipeline.run_io import _key
    bound = descriptor(path)
    reference = dict(sha256=bound["sha256"], bytes=bound["size_bytes"])
    row = SimpleNamespace(reference=reference)
    reader = object.__new__(current._Reader)
    reader.binding = SimpleNamespace(locations={_key(path): row}, physical_descriptor=lambda value: value)
    reader.allowed = [reference] if allowed else []
    reader.original = DataReader()
    reader.consumed = {}
    return reader, bound


def test_original_painted_metadata_reader_uses_named_scope_and_restores_context(tmp_path):
    from darsia.fluidflower.painted_adoption_table import _checked
    from darsia.fluidflower.preset_wasserstein_inputs import current_scope
    path = tmp_path / "named.json"
    path.write_text('{"complete":false}', encoding="utf-8")
    reader, bound = _direct_reader(path)
    with reader.direct():
        assert _checked(bound, metadata=True) == {"complete": False}
        assert current_scope() is not None
    assert current_scope() is None
    assert reader.consumed


@pytest.mark.parametrize("consumer", ["descriptor", "read", "stat", "exists"])
def test_original_direct_consumer_cannot_use_an_unnamed_existing_file(tmp_path, consumer):
    path = tmp_path / "not-current-build.json"
    path.write_text('{"complete":true}', encoding="utf-8")
    reader, _bound = _direct_reader(path, allowed=False)
    direct = current._DirectReads(reader)
    arguments = (path, "exists") if consumer == "exists" else (path,)
    with pytest.raises(ValueError, match="named catalogue consumer"):
        getattr(direct, consumer)(*arguments)
    assert not reader.consumed


def test_direct_scope_refuses_changed_bytes_and_unwinds(tmp_path):
    from darsia.fluidflower.painted_adoption_table import _checked
    from darsia.fluidflower.preset_wasserstein_inputs import current_scope
    path = tmp_path / "changed.json"
    path.write_text('{"complete":false}', encoding="utf-8")
    reader, bound = _direct_reader(path)
    path.write_text('{"complete":true}', encoding="utf-8")
    with pytest.raises(ValueError):
        with reader.direct():
            _checked(bound, metadata=True)
    assert current_scope() is None


def test_direct_reader_does_not_relocate_original_painted_proof_paths(tmp_path):
    path = tmp_path / "original.json"
    relocated = tmp_path / "relocated.json"
    path.write_text("{}", encoding="utf-8")
    relocated.write_bytes(path.read_bytes())
    reader, bound = _direct_reader(path)
    reader.binding.physical_descriptor = lambda value: dict(value, path=str(relocated))
    with pytest.raises(ValueError, match="cannot be relocated"):
        current._DirectReads(reader).descriptor(path, bound)


def test_direct_reader_rejects_consistent_existing_file_with_wrong_expected_pin(tmp_path):
    path = tmp_path / "original.json"
    path.write_text("{}", encoding="utf-8")
    reader, bound = _direct_reader(path)
    with pytest.raises(ValueError, match="descriptor changed"):
        current._DirectReads(reader).descriptor(path, dict(bound, sha256="0" * 64))


@pytest.mark.parametrize("phase1", [False, True])
def test_stage_configuration_is_exact_without_granting_physical_inputs(phase1):
    config = dict(format="darsia-native-phase-statistics-run-v1" if phase1
                  else "darsia-native-phase-n1-run-v1", run="run", frames=[{"case_id": "selected"}])
    template = {"geometry_fields": {"source_to_native": "H"}}
    config.update(statistics=template) if phase1 else config.update(mass={"template": template})
    owner = SimpleNamespace(_config=config)
    actual = dict(template, run="run", frames=config["frames"], data_aliases=())
    current.CheckedCurrentBuild.validate_mass_config(owner, actual, statistics_only=phase1)
    with pytest.raises(ValueError, match="consumer stage"):
        current.CheckedCurrentBuild.validate_mass_config(owner, actual, statistics_only=not phase1)
    changed = deepcopy(actual)
    changed["frames"] = [{"case_id": "another"}]
    with pytest.raises(ValueError, match="unchanged original"):
        current.CheckedCurrentBuild.validate_mass_config(owner, changed, statistics_only=phase1)


@pytest.mark.parametrize("submitted", [None, {"complete": True}, {"mass_eligible": True}])
def test_physical_entry_has_no_document_or_flag_success_branch(submitted):
    # Only the active-lifetime precheck is neutral here. The test exercises an
    # unconditional refusal, never constructs an admitted consumer or fake model.
    owner = SimpleNamespace(_active=lambda: None, submitted=submitted)
    with pytest.raises(ValueError, match="physical input adapter is unavailable"):
        current.CheckedCurrentBuild.require_physical_inputs(owner)


def test_pending_dense_phase_state_is_single_frame_and_released_before_mass_checks():
    path = Path(current.__file__)
    phase = _function(path, "check_phase", "CheckedCurrentBuild")
    mass = _function(path, "check_mass", "CheckedCurrentBuild")
    phase_clear = next(call for call in _calls(phase, "clear")
                       if ast.unparse(call.func.value) == "self._phase")
    assert phase_clear.lineno < _calls(phase, "_active")[0].lineno
    pop = next(call for call in _calls(mass, "pop") if ast.unparse(call.func.value) == "self._phase")
    clear = next(call for call in _calls(mass, "clear") if ast.unparse(call.func.value) == "self._phase")
    assert pop.lineno < clear.lineno < _calls(mass, "_active")[0].lineno


def test_p1_uses_one_original_native_frame_and_checked_gap_path_before_inference():
    path = Path(current.__file__).with_name("phase_statistics.py")
    function = _function(path, "_phase1_frames")
    names = ("native", "frame", "check_phase", "infer_labels", "check_inferred", "_phase_outputs")
    calls = [_calls(function, name) for name in names]
    assert all(len(group) == 1 for group in calls)
    assert [group[0].lineno for group in calls] == sorted(group[0].lineno for group in calls)
    assignment = next(node for node in ast.walk(function) if isinstance(node, ast.Assign)
                      and node.value is calls[2][0])
    assert [ast.unparse(target) for target in assignment.targets] == ["frame"]


def test_original_gap_transport_precedes_restriction_and_only_retains_original_outputs():
    path = Path(current.__file__)
    phase = _function(path, "check_phase", "CheckedCurrentBuild")
    transport = _calls(phase, "transport_build_exclusion")
    restrict = _calls(phase, "restrict_phase_domain")
    assert len(transport) == len(restrict) == 1 and transport[0].lineno < restrict[0].lineno
    assert [ast.unparse(arg) for arg in transport[0].args] == [
        "self._physical", "self._reader", "receipt['build_to_frame_native_h']",
        "frame['context'].origin_xy", "frame['context'].rig_domain.shape",
        "context['phase']['geometry_parameters']"]
    inferred = _function(path, "check_inferred", "CheckedCurrentBuild")
    text = ast.unparse(inferred)
    assert "inferred['labels'][gap] != 255" in text
    assert "inferred['classes']['approved_analysis_gap'] = gap" in text
    assert "inferred['classes']['physical_rig_domain'] = frame['physical_rig_domain']" in text
    assert "inferred['qa']['approved_analysis_gap'] = row['gap_provenance']" in text
