"""UNRUN prospective protocol/source tests; no accepted reference or execution.

Authoring validation is AST parsing only. A future explicit pytest invocation
would load the stdlib-only candidate and exercise DATA rejection/STOP paths.
TESTONLY descriptors below name no real evidence; no acceptance, completion,
retirement, scientific result, or human release document is fabricated here.
"""
from __future__ import annotations

import ast
import builtins
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


CANDIDATE = Path(__file__).resolve().parents[2] / "tools" / "qualification" / "compare_current_plan_strict39_PROSPECTIVE.py"
HISTORICAL = Path(
    "C:/FFAC_cache/validation/postlock_batch_20261020_v1/qualification_preparation/"
    "compare_R3_finalhead_qualification_PREPARATION_ONLY_v5.py"
)
HISTORICAL_SHA256 = "06ac1a5cdef18f71de623b9fa83f2b89645b56d934189a8c8a28e247c0089aa2"
ORIGINAL_HELPERS = {
    "numeric_checker": ("9949a184fef7e3b55c75336544fa04f52a2062f1b37cfdad1018f2243280b5ce", 17239),
    "record_extractors": ("95dc41bf655c5d22a91dd5bf7cf55430f0f5a511d8d7cdabe8f8e68378b6aa13", 21587),
    "read_locator": ("ba09417f90f79b0d6ce4a96ac93d9a9f8969bc2b1daa37dbd219f3ed718d758e", 8067),
    "typed_ipp_validator": ("a7f2550125fb102d1e297c7d72d2de01594fe91640d78a3e72869314bcdedb15", 3901),
}
PRESERVED = (
    "Reader", "utc", "key", "version", "metadata", "load", "write", "member",
    "_same_json", "_pair", "_current_completion", "_current_frame", "_n56_array_sha",
    "_strict_diagnostics", "_strict_n56_products", "check_candidate_retirement",
)
STDLIB_IMPORTS = {
    "argparse", "datetime", "hashlib", "json", "os", "pathlib", "socket", "sys",
    "traceback", "types", "copy", "re", "subprocess", "tomllib", "math", "__future__",
}


def source_TESTONLY():
    text = CANDIDATE.read_text(encoding="utf-8-sig")
    return text, ast.parse(text, filename=str(CANDIDATE))


def definitions_TESTONLY(tree):
    return {node.name: node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


NEUTRAL_HOST = "reference-host"


def neutral_host_TESTONLY(new, old):
    """Return the original host value that this copy spells NEUTRAL_HOST, or None.

    Every string constant that reads NEUTRAL_HOST in the new helper must
    replace one and the same original value; all other nodes stay compared.
    """
    values = {before.value for after, before in zip(ast.walk(new), ast.walk(old))
              if isinstance(after, ast.Constant) and after.value == NEUTRAL_HOST
              and isinstance(before, ast.Constant) and isinstance(before.value, str)}
    assert len(values) <= 1, values
    return values.pop() if values else None


class NeutralHost_TESTONLY(ast.NodeTransformer):
    """Spell one original host constant as NEUTRAL_HOST."""

    def __init__(self, host):
        self.host = host

    def visit_Constant(self, node):
        if node.value == self.host:
            node.value = NEUTRAL_HOST
        return node


def body_TESTONLY(node):
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            return body[1:]
    return body


def is_stop_TESTONLY(statement):
    return (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "_require_original_current_verifiers")


def assert_safe_top_level_TESTONLY(tree, *, contract_constants=False):
    """Check the import boundary before any future test loads this module."""
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name in STDLIB_IMPORTS for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module in STDLIB_IMPORTS
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assert not node.decorator_list
            if isinstance(node, ast.FunctionDef):
                defaults = node.args.defaults + [item for item in node.args.kw_defaults if item]
                assert not any(isinstance(item, ast.Call)
                               for default in defaults for item in ast.walk(default))
        elif isinstance(node, ast.Assign):
            # CASES uses only str(i); constants must not run loaders or probes.
            for call in (item for item in ast.walk(node.value) if isinstance(item, ast.Call)):
                if isinstance(call.func, ast.Name) and call.func.id in (
                        {"str", "tuple"} if contract_constants else {"str"}):
                    continue
                assert (contract_constants and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "replace" and len(call.args) == 2
                        and all(isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                                for arg in call.args) and not call.keywords)
        elif isinstance(node, ast.Expr):
            assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        elif isinstance(node, ast.If):
            assert ast.dump(node.test) == ast.dump(ast.parse('__name__ == "__main__"', mode="eval").body)
        else:
            pytest.fail("Unexpected executable module-level statement: " + type(node).__name__)


@pytest.fixture(scope="module")
def candidate():
    # This fixture is authored, not executed during source preparation.
    _text, tree = source_TESTONLY()
    assert_safe_top_level_TESTONLY(tree)
    spec = importlib.util.spec_from_file_location("_TESTONLY_current_plan_strict39", CANDIDATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pin_TESTONLY(role):
    raw = ("TESTONLY opaque descriptor, no evidence: " + role).encode("utf-8")
    return {"path": "C:/TESTONLY/" + role + ".json",
            "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def request_TESTONLY():
    roles = (
        "current_plan", "registered_series_manifest", "reference_acceptance", "Root_release",
        "reference_completed", "candidate_completed", "reference_retirement", "candidate_retirement",
        "reference_native_completion", "candidate_native_completion", "reference_source_runtime",
        "candidate_source_runtime", "environment_identity_proof", "scientific_source_equivalence",
        "reference_closed_readset", "candidate_closed_readset", "frozen_calibration",
    )
    value = {role: pin_TESTONLY(role) for role in roles}
    value.update(
        format="darsia-prospective-current-plan-strict39-request-v1",
        source_pair={"relationship": "same_source_independent_reexecution",
                     "Reference": {"commit": "1" * 40, "tree": "2" * 40},
                     "Candidate": {"commit": "1" * 40, "tree": "2" * 40}},
        host_tasks={host: pin_TESTONLY(host + "_task") for host in ("Reference", "Candidate")},
        host_execution_series={host: pin_TESTONLY(host + "_series") for host in ("Reference", "Candidate")},
        source_pins={role: {"path": "C:/TESTONLY/" + role + ".py", "sha256": sha, "size_bytes": size}
                     for role, (sha, size) in ORIGINAL_HELPERS.items()},
        candidate_attempt_directory="C:/TESTONLY/candidate-attempt",
        candidate_copy_root="C:/TESTONLY/candidate-attempt",
        product_readset=[pin_TESTONLY("unread-product")],
        output="C:/TESTONLY/uncreated-comparison-output", accepted_prior_comparisons=[],
    )
    return value


@pytest.mark.parametrize("location", [(), ("source_pair",), ("source_pair", "Reference"),
    ("host_tasks",), ("host_execution_series",), ("source_pins",), ("reference_acceptance",)])
def test_unknown_request_fields_are_rejected(candidate, location):
    value = request_TESTONLY()
    target = value
    for key in location:
        target = target[key]
    target["TESTONLY_extra"] = None
    with pytest.raises(ValueError, match="Closed prospective schema"):
        candidate.checked_prospective_request(value)


@pytest.mark.parametrize("raw", [b'{"key":0,"key":1}', b'{"outer":{"key":0,"key":1}}'])
def test_duplicate_json_keys_are_rejected(candidate, raw):
    with pytest.raises(ValueError, match="Duplicate JSON control key"):
        candidate._strict_json(raw)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e309", "-1e309"])
def test_nonfinite_json_constants_are_rejected(candidate, constant):
    with pytest.raises(ValueError, match="Nonfinite JSON control value"):
        candidate._strict_json(('{"nested":[' + constant + ']}').encode("ascii"))


@pytest.mark.parametrize("bad_size", [True, -1, 1.5, float("nan"), float("inf")])
def test_descriptor_lengths_are_actual_nonnegative_integers(candidate, bad_size):
    value = request_TESTONLY()
    value["reference_completed"]["size_bytes"] = bad_size
    with pytest.raises(ValueError, match="Exact absolute descriptor"):
        candidate.checked_prospective_request(value)


def test_numeric_sha_cannot_be_coerced_to_a_digest(candidate):
    value = request_TESTONLY()
    value["reference_completed"]["sha256"] = int("1" * 64)
    with pytest.raises(ValueError, match="Exact absolute descriptor"):
        candidate.checked_prospective_request(value)


@pytest.mark.parametrize("field", ["commit", "tree"])
def test_numeric_source_identities_cannot_be_coerced_to_strings(candidate, field):
    value = request_TESTONLY()
    for host in ("Reference", "Candidate"):
        value["source_pair"][host][field] = int("1" * 40)
    with pytest.raises(ValueError, match="actual full source identities"):
        candidate.checked_prospective_request(value)


def partial_metadata_TESTONLY():
    """Early negative-gate DATA only; no Build, catalogue, or acceptance proof."""
    q = request_TESTONLY()
    scope = dict(pipeline="native-phase-n1", backend="cpu", precision="float32", run="ac52",
                 ordered_case_ids=["ac52_f" + str(i) for i in range(1, 44) if i not in {8, 15, 22, 24}])
    configuration = {"TESTONLY": "unverified configuration identity marker"}
    plan = dict(format="darsia-current-v9-reference-plan-v1", authority_ids=["Q432", "TESTONLY-data"],
                scope=scope, configuration=configuration, input_catalog_sha256="0" * 64,
                requested_native_layout={}, output_writers=[])
    for role in ("configuration_source", "current_v9_build", "native_parameters",
                 "python_executable", "capacity_bound_plan"):
        plan[role] = pin_TESTONLY("plan_" + role)
    documents = {q["current_plan"]["path"]: plan}
    for host in ("Reference", "Candidate"):
        documents[q["host_tasks"][host]["path"]] = dict(
            scope, threads=1, commit=q["source_pair"][host]["commit"], configuration=configuration,
            job_id="TESTONLY-" + host, attempt_id="TESTONLY-attempt-" + host)
    return q, plan, documents


@pytest.mark.parametrize("host", ["Reference", "Candidate"])
def test_true_is_not_an_actual_single_thread_count(candidate, host):
    q, _plan, documents = partial_metadata_TESTONLY()
    documents[q["host_tasks"][host]["path"]]["threads"] = True
    with pytest.raises(ValueError, match="actual task scope/source/configuration"):
        candidate.checked_current_plan_metadata(q, read=lambda pin: deepcopy(documents[pin["path"]]))


def test_numeric_catalogue_digest_is_rejected_before_task_reads(candidate):
    q, plan, _documents = partial_metadata_TESTONLY()
    plan["input_catalog_sha256"] = int("1" * 64)
    reads = []

    def read_plan_only(pin):
        reads.append(pin)
        assert pin == q["current_plan"]
        return deepcopy(plan)

    with pytest.raises(ValueError, match="Exact current-plan catalogue digest"):
        candidate.checked_current_plan_metadata(q, read=read_plan_only)
    assert reads == [q["current_plan"]]


@pytest.mark.parametrize("relationship", ["same_source_independent_reexecution", "reviewed_source_pair"])
def test_consistent_source_relationship_is_data_only_and_cannot_lift_stop(candidate, relationship):
    value = request_TESTONLY()
    value["source_pair"]["relationship"] = relationship
    if relationship == "reviewed_source_pair":
        value["source_pair"]["Candidate"]["commit"] = "3" * 40
    before = deepcopy(value)
    checked = candidate.checked_prospective_request(value)
    assert checked == before and value == before
    with pytest.raises(candidate.CurrentReferenceBindingStop, match="TYPED_ACCEPTED_REFERENCE"):
        candidate._require_original_current_verifiers(checked)


@pytest.mark.parametrize("fault", ["different_commit_claimed_same", "same_commit_claimed_pair",
                                   "same_commit_different_tree", "untyped_relationship", "short_commit"])
def test_source_relationship_cannot_relabel_actual_identities(candidate, fault):
    value = request_TESTONLY()
    pair = value["source_pair"]
    if fault == "different_commit_claimed_same":
        pair["Candidate"]["commit"] = "3" * 40
    elif fault == "same_commit_claimed_pair":
        pair["relationship"] = "reviewed_source_pair"
    elif fault == "same_commit_different_tree":
        pair["Candidate"]["tree"] = "4" * 40
    elif fault == "untyped_relationship":
        pair["relationship"] = "TESTONLY-unreviewed"
    else:
        pair["Reference"]["commit"] = "1234567"
    with pytest.raises(ValueError, match="source identities|source relationship|Source relationship"):
        candidate.checked_prospective_request(value)


@pytest.mark.parametrize("suffix", ["completed", "retirement", "native_completion"])
def test_same_evidence_path_is_rejected_even_with_case_and_slash_changes(candidate, suffix):
    value = request_TESTONLY()
    value["candidate_" + suffix]["path"] = value["reference_" + suffix]["path"].upper().replace("/", "\\")
    with pytest.raises(ValueError, match="Self-comparison evidence path"):
        candidate.checked_prospective_request(value)


@pytest.mark.parametrize("role", list(ORIGINAL_HELPERS))
@pytest.mark.parametrize("field", ["sha256", "size_bytes"])
def test_original_helper_pin_drift_is_rejected(candidate, role, field):
    value = request_TESTONLY()
    value["source_pins"][role][field] = "0" * 64 if field == "sha256" else ORIGINAL_HELPERS[role][1] + 1
    with pytest.raises(ValueError, match="Original checker/reader bytes changed"):
        candidate.checked_prospective_request(value)


def tripwire_TESTONLY(*_args, **_kwargs):
    pytest.fail("STOP was bypassed into evidence I/O, a helper, or output")


def guard_effects_TESTONLY(monkeypatch, candidate):
    for name in ("document", "metadata", "load", "write"):
        monkeypatch.setattr(candidate, name, tripwire_TESTONLY)
    monkeypatch.setattr(Path, "mkdir", tripwire_TESTONLY)
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if (name.split(".")[0] in {"numpy", "psutil", "darsia"}
                or name.startswith("darsia_cache") and name != "darsia_cache.current_strict39"):
            pytest.fail("Numerical/product import occurred before the original verifier STOP")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)


@pytest.mark.parametrize("entry", ["_require_original_current_verifiers", "_current_task_basis",
                                   "check_current_source_pair", "_compare_after_original_admission"])
def test_direct_entries_cannot_bypass_missing_original_verifier(candidate, monkeypatch, entry):
    with monkeypatch.context() as patch:
        guard_effects_TESTONLY(patch, candidate)
        args = (object(), object()) if entry == "_compare_after_original_admission" else (object(),)
        with pytest.raises(candidate.CurrentReferenceBindingStop, match="BUILD_PUBLICATION_TO_QUEUED_FRAME"):
            getattr(candidate, entry)(*args)


@pytest.mark.parametrize("role_mode", ["absent", "none", "empty_roles"])
def test_run_stops_after_metadata_diagnostics_before_comparison_or_output(candidate, monkeypatch, tmp_path, role_mode):
    request = request_TESTONLY()
    if role_mode == "none":
        request["strict39_roles"] = None
    elif role_mode == "empty_roles":
        request["strict39_roles"] = dict.fromkeys(
            ("accepted_reference", "predeclared_contract", "postproducer_release", "strict_acceptance"))
    path = tmp_path / "TESTONLY-request.json"
    path.write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
    diagnostics = []

    def metadata_diagnostic_ONLY(value):
        diagnostics.append(deepcopy(value))
        return {"TESTONLY": "metadata diagnostic only; no authority or verifier"}

    with monkeypatch.context() as patch:
        guard_effects_TESTONLY(patch, candidate)
        patch.setattr(candidate, "checked_current_plan_metadata", metadata_diagnostic_ONLY)
        patch.setattr(candidate, "_compare_after_original_admission", tripwire_TESTONLY)
        with pytest.raises(candidate.CurrentReferenceBindingStop, match="TYPED_ACCEPTED_REFERENCE"):
            candidate.run(path)
    assert diagnostics == [request]
    assert list(tmp_path.iterdir()) == [path]


def test_static_public_and_direct_entry_guards_are_unconditional():
    _text, tree = source_TESTONLY()
    assert_safe_top_level_TESTONLY(tree)
    definitions = definitions_TESTONLY(tree)
    stop_body = body_TESTONLY(definitions["_require_original_current_verifiers"])
    assert len(stop_body) == 1 and isinstance(stop_body[0], ast.Raise)
    assert isinstance(stop_body[0].exc, ast.Call)
    assert isinstance(stop_body[0].exc.func, ast.Name)
    assert stop_body[0].exc.func.id == "CurrentReferenceBindingStop"
    for name in ("_current_task_basis", "check_current_source_pair", "_compare_after_original_admission"):
        assert is_stop_TESTONLY(body_TESTONLY(definitions[name])[0]), name
    run_body = body_TESTONLY(definitions["run"])
    guards = [index for index, node in enumerate(run_body) if is_stop_TESTONLY(node)]
    assert len(guards) == 1
    guard = guards[0]
    before = ast.Module(body=run_body[:guard], type_ignores=[])
    imports = [node for node in ast.walk(before) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert len(imports) == 1 and isinstance(imports[0], ast.ImportFrom)
    assert imports[0].module == "darsia_cache.current_strict39" and imports[0].level == 0
    assert [(alias.name, alias.asname) for alias in imports[0].names] == [("checked_role_bindings", None)]
    role_source = CANDIDATE.parents[2] / "src" / "darsia_cache" / "current_strict39.py"
    role_tree = ast.parse(role_source.read_text(encoding="utf-8"))
    assert_safe_top_level_TESTONLY(role_tree, contract_constants=True)
    assert not any(isinstance(node, ast.Call) and (
        isinstance(node.func, ast.Name) and node.func.id in {"load", "write", "_compare_after_original_admission"}
        or isinstance(node.func, ast.Attribute) and node.func.attr in {"mkdir", "write_text", "write_bytes"})
        for node in ast.walk(before))
    tail = run_body[guard + 1:]
    assert len(tail) == 1 and isinstance(tail[0], ast.Return)
    assert isinstance(tail[0].value, ast.Call) and tail[0].value.func.id == "_compare_after_original_admission"
    main = [node for node in tree.body if isinstance(node, ast.If)]
    assert len(main) == 1
    raises = [node for node in main[0].body if isinstance(node, ast.Raise)]
    assert len(raises) == 1
    exit_call = raises[0].exc
    assert isinstance(exit_call, ast.Call) and exit_call.func.id == "SystemExit"
    assert len(exit_call.args) == 1 and isinstance(exit_call.args[0], ast.Call)
    assert exit_call.args[0].func.id == "run"


def test_original_scientific_helpers_are_preserved_as_ast_and_text():
    raw = HISTORICAL.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == HISTORICAL_SHA256
    original = raw.decode("utf-8-sig").replace("\r\n", "\n")
    current, tree = source_TESTONLY()
    old = definitions_TESTONLY(ast.parse(original, filename=str(HISTORICAL)))
    new = definitions_TESTONLY(tree)
    for name in PRESERVED:
        host = neutral_host_TESTONLY(new[name], old[name])
        before = old[name] if host is None else NeutralHost_TESTONLY(host).visit(deepcopy(old[name]))
        assert ast.dump(new[name], include_attributes=False) == ast.dump(before, include_attributes=False), name
        text = ast.get_source_segment(original, old[name])
        if host is not None:
            text = text.replace(f'"{host}"', f'"{NEUTRAL_HOST}"')
        assert ast.get_source_segment(current, new[name]) == text, name
    assignments = {node.targets[0].id: node.value for node in tree.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    old_assignments = {node.targets[0].id: node.value for node in ast.parse(original).body
                       if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    for name in ("CASES", "FLOATS", "GEOMETRY"):
        assert ast.dump(assignments[name]) == ast.dump(old_assignments[name]), name
    assert ast.literal_eval(assignments["HISTORICAL_COMPARATOR_SHA256"]) == HISTORICAL_SHA256
    assert ast.literal_eval(assignments["ORIGINAL_HELPER_BYTES"]) == ORIGINAL_HELPERS
    assert ast.literal_eval(assignments["COMMIT"]) is None
    assert ast.literal_eval(assignments["HOST_COMMITS"]) == {"Reference": None, "Candidate": None}
