"""TESTONLY future-binding protocol tests; no real GO, Build, reference or profile.

The source slot remains closed. Synthetic in-memory records only exercise the
new validator through the original administration; no controls are written.
"""
from copy import deepcopy
import ast
import hashlib
import inspect
import json
import stat
from types import SimpleNamespace

import pytest

from darsia_cache import first_qualification as first
from darsia_cache.policy import CachePolicyError
from .test_n43_reference_administration import administration_TESTONLY, pin_TESTONLY
from .test_cache_first_qualification import bound_plan_TESTONLY
from darsia.fluidflower.pipeline.job_inputs import InputCatalog, CATALOG_FORMAT, catalog_order_key, validate_catalog
from darsia.fluidflower.pipeline.series_setup_results import validate_setup_result


def encoded_TESTONLY(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")


def put_TESTONLY(objects, name, value):
    path = "C:/TESTONLY/" + name
    objects[path] = value
    raw = encoded_TESTONLY(value)
    return dict(path=path, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def rebind_plan_TESTONLY(monkeypatch, administration, release, plan, objects):
    # Rebind changed TESTONLY DATA honestly, so rejection is the association
    # guard rather than a stale descriptor in the test reader.
    administration["accepted_plan"] = put_TESTONLY(objects, "new-current-v9-plan.json", plan)
    release["plan_sha256"] = administration["accepted_plan"]["sha256"]
    monkeypatch.setattr(first, "CURRENT_V9_REFERENCE_RELEASE", deepcopy(release))


def typed_build_TESTONLY(run="ac52"):
    # This is a genuine original typed result shape, NOT owner/publication or
    # physical-source verification. Even this record must hit execution STOP.
    value = dict(format="darsia-series-setup-result-v1", pipeline="series-setup",
        complete=True, run=run, operation="build", backend="cpu", precision="float32",
        setup_plan=pin_TESTONLY("setup-plan.json"), worker_references_closed=True,
        files={"operation_result.json": pin_TESTONLY("operation_result.json")})
    validate_setup_result(value)
    return value


def current_TESTONLY(monkeypatch):
    state, administration, _old_read = administration_TESTONLY()
    objects = {}
    native_pin = put_TESTONLY(objects, "new-current-v9-native.json",
        {"TESTONLY": "bounded native metadata; no scientific payload or acceptance"})
    configuration = dict(format="darsia-native-phase-n1-run-v1", run="ac52",
        native={"configuration": native_pin})
    configuration_source = put_TESTONLY(objects, "new-current-v9-run.json", configuration)
    # Real original InputCatalog creates named references from actual bytes and
    # stat lengths. This bounded role fixture is not a full scientific closure.
    catalog = InputCatalog({"testonly": "C:/TESTONLY"},
        metadata_bytes=lambda path: encoded_TESTONLY(objects[str(path)]),
        file_stat=lambda path: SimpleNamespace(st_mode=stat.S_IFREG,
            st_size=len(encoded_TESTONLY(objects[str(path)]))))
    assert catalog.doc(configuration_source, "configuration", "run.configuration") == configuration
    catalog.doc(native_pin, "configuration", "native.configuration")
    configuration_reference = catalog.add(configuration_source, "configuration", "run.configuration")
    state.task.update(run="ac52", configuration=configuration_reference)
    state.task["input_catalog"] = dict(format=CATALOG_FORMAT, complete=True, run="ac52",
        configuration=configuration_reference,
        inputs=sorted(deepcopy(list(catalog.entries.values())), key=lambda row: catalog_order_key(row["reference"])),
        payload_hashes_verified=False, numerical_calls_started=False,
        runtime_support="TESTONLY metadata roles; no full scientific execution support")
    validate_catalog(state.task["input_catalog"])
    human = "TESTONLY-Owner-new-current-V9-exact-plan-GO"
    release = dict(format=first.CURRENT_V9_REFERENCE_RELEASE_FORMAT,
        authority_ids=["Q432", "TESTONLY-new-current-V9-GO"],
        records={first.CURRENT_V9_PREPARATION_RECORD: "coordinator", human: "owner"},
        human_go_record_id=human, plan_sha256="f" * 64)
    administration.update(authority_ids=deepcopy(release["authority_ids"]),
        output_configuration=deepcopy(configuration_reference))
    administration["requested_native_layout"]["native_configuration_sha256"] = native_pin["sha256"]
    scope = {key: state.provider["scope"][key] for key in
        ("pipeline", "backend", "precision", "run", "ordered_case_ids")}
    plan = dict(format=first.CURRENT_V9_REFERENCE_PLAN,
        authority_ids=deepcopy(release["authority_ids"]), scope=deepcopy(scope),
        configuration=deepcopy(state.task["configuration"]), configuration_source=configuration_source,
        input_catalog_sha256=first._digest(state.task["input_catalog"]),
        current_v9_build=put_TESTONLY(objects, "new-current-v9-build.json", typed_build_TESTONLY()),
        native_parameters=deepcopy(native_pin),
        python_executable=deepcopy(state.provider["python_executable"]),
        capacity_bound_plan=put_TESTONLY(objects, "new-current-capacity.json", bound_plan_TESTONLY()),
        requested_native_layout=deepcopy(administration["requested_native_layout"]),
        output_writers=deepcopy(administration["output_writers"]))
    answers = [dict(id=record, decided_by=by) for record, by in release["records"].items()]
    administration["answers_source"] = put_TESTONLY(objects, "answers.json", answers)
    rebind_plan_TESTONLY(monkeypatch, administration, release, plan, objects)
    reads = []

    def read(binding):
        reads.append(deepcopy(binding))
        value = objects[binding["path"]]
        raw = encoded_TESTONLY(value)
        assert binding["sha256"] == hashlib.sha256(raw).hexdigest()
        assert binding["size_bytes"] == len(raw)
        return deepcopy(value)

    return state, administration, release, plan, answers, objects, reads, read


def test_production_current_slot_is_closed_and_preparation_is_not_execution():
    assert first.CURRENT_V9_REFERENCE_RELEASE is None
    assert first.released_reference(["Q432"]) is None
    assert first.released_reference(["Q432", "unbound-human-GO"]) is None


def test_original_named_roles_and_unsealed_build_require_explicit_current_binding(monkeypatch):
    state, administration, release, plan, _answers, _objects, reads, read = current_TESTONLY(monkeypatch)
    assert first.released_reference(release["authority_ids"]) == release
    with pytest.raises(CachePolicyError, match="Current Build metadata binding refused"):
        first.checked_reference_administration(administration, state, read=read)
    assert {plan[role]["path"] for role in
        ("configuration_source", "current_v9_build", "native_parameters", "capacity_bound_plan")} <= {
        binding["path"] for binding in reads}
    assert not {"passed", "measured_profile", "reference_accepted"} & set(administration)
    assert first.released_reference(list(reversed(release["authority_ids"]))) is None


@pytest.mark.parametrize("fault", ["missing_human", "wrong_decider", "old_human", "old_plan",
                                  "old_authorities", "duplicate_ids", "missing_plan", "boolean_release"])
def test_incomplete_or_historical_source_slot_cannot_be_released(monkeypatch, fault):
    _state, _administration, release, *_rest = current_TESTONLY(monkeypatch)
    if fault == "missing_human":
        release.pop("human_go_record_id")
    elif fault == "wrong_decider":
        release["records"][release["human_go_record_id"]] = "coordinator"
    elif fault == "old_human":
        old = next(iter(first.REFERENCE_RELEASES.values()))
        release["human_go_record_id"] = next(key for key, by in old["records"].items() if by == "owner")
        release["records"] = {first.CURRENT_V9_PREPARATION_RECORD: "coordinator", release["human_go_record_id"]: "owner"}
    elif fault == "old_plan":
        release["plan_sha256"] = next(iter(first.REFERENCE_RELEASES.values()))["plan_sha256"]
    elif fault == "old_authorities":
        release["authority_ids"] = ["Q432", "N43"]
    elif fault == "duplicate_ids":
        release["authority_ids"].append("Q432")
    elif fault == "missing_plan":
        release["plan_sha256"] = None
    else:
        release = {"execution_authorized": True}
    monkeypatch.setattr(first, "CURRENT_V9_REFERENCE_RELEASE", release)
    assert first.released_reference(["Q432", "TESTONLY-new-current-V9-GO"]) is None
    with pytest.raises(CachePolicyError):
        first.checked_current_reference_release(release)


@pytest.mark.parametrize("fault", ["format", "configuration", "input_catalog", "runtime", "layout",
                                  "writers", "case_order", "capacity"])
def test_new_plan_refuses_each_changed_actual_binding(monkeypatch, fault):
    state, administration, release, plan, _answers, objects, _reads, read = current_TESTONLY(monkeypatch)
    if fault == "format":
        plan["format"] = "historical-reference-plan"
    elif fault == "configuration":
        plan["configuration"] = {"TESTONLY": "historical configuration"}
    elif fault == "input_catalog":
        state.task["input_catalog"]["inputs"].append({"TESTONLY": "changed actual input"})
    elif fault == "runtime":
        plan["python_executable"]["sha256"] = "b" * 64
    elif fault == "layout":
        plan["requested_native_layout"] = {"TESTONLY": "different native layout"}
    elif fault == "writers":
        plan["output_writers"] = []
    elif fault == "case_order":
        plan["scope"]["ordered_case_ids"].reverse()
    else:
        changed = deepcopy(objects[plan["capacity_bound_plan"]["path"]])
        changed["reserve_watchdog_period_seconds"] = 30
        plan["capacity_bound_plan"] = put_TESTONLY(objects, "changed-capacity.json", changed)
    rebind_plan_TESTONLY(monkeypatch, administration, release, plan, objects)
    match = ({"input_catalog": "Actual original complete named",
        "capacity": "volume-only bound"}).get(fault, "Current exact39")
    with pytest.raises(CachePolicyError, match=match):
        first.checked_reference_administration(administration, state, read=read)


@pytest.mark.parametrize("fault", ["missing_GO", "duplicate_GO", "wrong_GO_decider", "wrong_preparation_decider"])
def test_actual_authority_records_remain_unique_and_human(monkeypatch, fault):
    state, administration, release, _plan, answers, objects, _reads, read = current_TESTONLY(monkeypatch)
    human = next(row for row in answers if row["id"] == release["human_go_record_id"])
    if fault == "missing_GO":
        answers.remove(human)
    elif fault == "duplicate_GO":
        answers.append(deepcopy(human))
    elif fault == "wrong_GO_decider":
        human["decided_by"] = "coordinator"
    else:
        next(row for row in answers if row["id"] == first.CURRENT_V9_PREPARATION_RECORD)["decided_by"] = "owner"
    administration["answers_source"] = put_TESTONLY(objects, "changed-answers.json", answers)
    with pytest.raises(CachePolicyError, match="humanGO|Unique actual|decider differs"):
        first.checked_reference_administration(administration, state, read=read)


@pytest.mark.parametrize("fault", ["entitlement", "reserve", "parallel", "watchdog", "roots"])
def test_future_route_keeps_original_administration_guards(monkeypatch, fault):
    state, administration, _release, _plan, _answers, _objects, _reads, read = current_TESTONLY(monkeypatch)
    if fault == "entitlement":
        administration["aggregate_entitlement_bytes"] += 1
    elif fault == "reserve":
        administration["operational_reserve_bytes"] = 0
    elif fault == "parallel":
        state.request["public_command"][state.request["public_command"].index("--workers") + 1] = "2"
    elif fault == "watchdog":
        state.request["public_command"].append("--watchdog")
    else:
        administration["owned_roots"][1]["path"] = administration["owned_roots"][0]["path"] + "/nested"
    with pytest.raises(CachePolicyError, match="aggregate entitlement|CPU worker|roots overlap"):
        first.checked_reference_administration(administration, state, read=read)


def test_historical_entries_and_default_lookup_are_preserved(monkeypatch):
    historical = {key: value for key, value in first.REFERENCE_RELEASES.items()}
    current_TESTONLY(monkeypatch)
    for ids, entry in historical.items():
        assert first.released_reference(list(ids)) is entry
    assert first.released_reference(["Q432", "unbound-human-GO"]) is None
    assert first.released_reference(None) is None
    assert first.released_reference([["Q432"]]) is None


@pytest.mark.parametrize("fault", ["plan", "named_sha", "named_bytes", "named_role",
    "layout", "missing_role", "duplicate_role", "queued_source"])
def test_rebound_valid_unrelated_native_refuses_before_build(monkeypatch, fault):
    state, administration, release, plan, _answers, objects, reads, read = current_TESTONLY(monkeypatch)
    rows = state.task["input_catalog"]["inputs"]
    native_row = next(row for row in rows if "native.configuration" in row["uses"])
    other = put_TESTONLY(objects, "unrelated-native.json", {"TESTONLY": "different genuine metadata bytes"})
    if fault == "plan":
        plan["native_parameters"] = other
        # Rebind requested layout too: matching plan/layout must not hide a
        # different actual queued config's native descriptor.
        plan["requested_native_layout"]["native_configuration_sha256"] = other["sha256"]
        administration["requested_native_layout"] = deepcopy(plan["requested_native_layout"])
    elif fault == "named_sha":
        native_row["reference"]["sha256"] = other["sha256"]
    elif fault == "named_bytes":
        native_row["reference"]["bytes"] += 1
    elif fault == "named_role":
        native_row["reference"]["role"] = "rig_model"
    elif fault == "layout":
        plan["requested_native_layout"]["native_configuration_sha256"] = other["sha256"]
        administration["requested_native_layout"] = deepcopy(plan["requested_native_layout"])
    elif fault == "missing_role":
        rows.remove(native_row)
    elif fault == "duplicate_role":
        rows.append(deepcopy(native_row))
    else:
        plan["configuration_source"] = put_TESTONLY(objects, "unrelated-run.json",
            dict(format="darsia-native-phase-n1-run-v1", run="ac52",
                native={"configuration": other}))
    plan["input_catalog_sha256"] = first._digest(state.task["input_catalog"])
    rebind_plan_TESTONLY(monkeypatch, administration, release, plan, objects)
    with pytest.raises(CachePolicyError, match="queued config|named queued configuration"):
        first.checked_reference_administration(administration, state, read=read)
    assert plan["current_v9_build"]["path"] not in {binding["path"] for binding in reads}


@pytest.mark.parametrize("document", [{}, {"accepted": True},
    {"format": "historical-Build", "complete": True}, None])
def test_rebound_build_document_cannot_replace_missing_current_binding(monkeypatch, document):
    state, administration, release, plan, _answers, objects, reads, read = current_TESTONLY(monkeypatch)
    # A different original typed run and arbitrary truthy/empty DATA all remain
    # non-authority; hashes and the TESTONLY source release are rebound exactly.
    document = typed_build_TESTONLY("ac18") if document is None else document
    plan["current_v9_build"] = put_TESTONLY(objects, "rebound-build.json", document)
    rebind_plan_TESTONLY(monkeypatch, administration, release, plan, objects)
    with pytest.raises(CachePolicyError, match="Current Build metadata binding refused"):
        first.checked_reference_administration(administration, state, read=read)
    assert plan["current_v9_build"] in reads


def test_prospective_validator_keeps_original_preimport_boundary():
    # Importing darsia from this bootstrap hook would initialize numerical
    # producers before original startup admission, even if tests already did.
    tree = ast.parse(inspect.getsource(first.checked_current_reference_plan))
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
    assert isinstance(tree.body[0].body[-1], ast.Raise)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)]
    assert any(node.func.id == "checked_current_build_plan_metadata" for node in calls)
    assert "execution binding STOP" in ast.unparse(tree.body[0].body[-1])
