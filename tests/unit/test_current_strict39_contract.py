"""UNRUN source-authored DATA/refusal tests; no publication or acceptance fixture.

Synthetic descriptor strings and in-memory JSON readers below are deliberately
unsealed. They establish schema consistency only, never authority, actual OS
retirement, scientific parity, native eligibility or executable startup admission.
"""
from copy import deepcopy
import hashlib
import inspect
import json

import pytest

from darsia_cache import current_strict39 as subject


def pin(name, *, sha=None, size=1):
    return dict(path="C:/unsealed-contract-data/" + name,
                sha256=sha or hashlib.sha256(name.encode()).hexdigest(), size_bytes=size)


def scope():
    return dict(pipeline="native-phase-n1", backend="cpu", precision="float32", run="ac52",
                ordered_case_ids=list(subject.CASE_IDS), threads=1)


def pair(*, same=False):
    return dict(relationship="same_source_independent_reexecution" if same else "reviewed_source_pair",
                Reference=dict(commit="1" * 40, tree="2" * 40),
                Candidate=dict(commit=("1" if same else "3") * 40, tree=("2" if same else "4") * 40))


def base(role, **fields):
    return dict(format=subject.FORMATS[role], state=subject.STATE, scope=scope(), **fields)


def producer(host):
    return {role: pin(host + "/" + role + ".json") for role in subject.PRODUCER_ROLES}


def unsealed_roles():
    """Three prospective role bodies, no actual strict acceptance/retirement body."""
    bindings = {role: pin(role + ".json") for role in subject.ROLES}
    bindings["strict_acceptance"] = None
    reference = base("accepted_reference", current_plan=pin("plan.json"),
        configuration=dict(root_alias="fixture", relative_path="config.json", sha256="5" * 64,
                           role="configuration", bytes=1), input_catalog_sha256="6" * 64,
        current_v9_build=pin("build.json"), producer=producer("reference"), authority_record=pin("unissued-authority.json"))
    sources = {name: pin(name + ".py", sha=identity[0], size=identity[1])
               for name, identity in subject.ORIGINAL_HELPER_BYTES.items()}
    sources["adapter"] = pin("prospective-adapter.py")
    contract = base("predeclared_contract", accepted_reference=bindings["accepted_reference"],
        current_plan=reference["current_plan"], candidate_task=pin("candidate/task.json"), source_pair=pair(),
        source_pair_review=pin("unissued-source-review.json"), sources=sources,
        public_command=["python.exe", sources["adapter"]["path"], "--request", "unsealed-request.json"],
        report_format=subject.REPORT_FORMAT)
    release = base("postproducer_release", predeclared_contract=bindings["predeclared_contract"],
        accepted_reference=bindings["accepted_reference"],
        producers=dict(Reference=reference["producer"], Candidate=producer("candidate")),
        request_payload_sha256="7" * 64, authority_record=pin("unissued-comparison-authority.json"))
    objects = {bindings[role]["path"]: body for role, body in zip(subject.ROLES, (reference, contract, release))}
    return bindings, objects, reference, contract, release


def never_read(_pin):
    raise AssertionError("Absent roles must not open any metadata")


@pytest.mark.parametrize("roles", [None, {role: None for role in subject.ROLES}])
def test_absent_roles_are_inert_and_do_not_read(roles):
    result = subject.checked_role_bindings(roles, read=never_read)
    assert result["present_roles"] == []
    assert result["absent_roles"] == list(subject.ROLES)
    assert result["authority_granted"] is False
    assert result["original_admission_available"] is False


def test_consistent_unsealed_role_data_never_grants_admission():
    roles, objects, *_ = unsealed_roles()
    seen = []
    def read(bound):
        seen.append(bound["path"])
        return deepcopy(objects[bound["path"]])
    diagnostics = subject.checked_role_bindings(roles, read=read)
    assert len(seen) == 3  # No invented Build, authority, retirement or eligibility body read.
    assert diagnostics["metadata_only"] is True
    assert diagnostics["authority_granted"] is False
    with pytest.raises(subject.CurrentAdmissionUnavailable):
        subject.require_original_current_admission(diagnostics, accepted=True,
            verify_retirement=lambda *_: True, original_verifiers_available=True)


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(accepted=True),
    lambda d: d.update(state="ACCEPTED"),
    lambda d: d["scope"].update(threads=True),
    lambda d: d["scope"]["ordered_case_ids"].reverse(),
    lambda d: d["current_plan"].update(sha256=int("1" * 64)),
    lambda d: d["configuration"].update(relative_path="../escape.json"),
])
def test_closed_reference_data_refuses_authority_and_malformed_fields(mutate):
    _, _, reference, _, _ = unsealed_roles()
    mutate(reference)
    with pytest.raises(subject.ContractError):
        subject.checked_accepted_reference_data(reference)


def test_later_role_without_predecessor_is_refused_before_read():
    roles = subject.absent_role_bindings()
    roles["postproducer_release"] = pin("orphan.json")
    with pytest.raises(subject.ContractError):
        subject.checked_role_bindings(roles, read=never_read)


def test_cross_role_candidate_task_and_current_plan_are_bound():
    for role, field in (("predeclared_contract", "current_plan"), ("postproducer_release", "predeclared_contract")):
        roles, objects, *_ = unsealed_roles()
        objects[roles[role]["path"]][field] = pin("another.json")
        with pytest.raises(subject.ContractError):
            subject.checked_role_bindings(roles, read=lambda bound: objects[bound["path"]])


def test_original_helpers_and_actual_source_relationship_are_fixed():
    _, _, _, contract, _ = unsealed_roles()
    contract["sources"]["numeric_checker"]["sha256"] = "f" * 64
    with pytest.raises(subject.ContractError):
        subject.checked_predeclared_contract_data(contract)
    _, _, _, contract, _ = unsealed_roles()
    contract["source_pair"]["relationship"] = "same_source_independent_reexecution"
    with pytest.raises(subject.ContractError):
        subject.checked_predeclared_contract_data(contract)
    contract["source_pair"] = pair(same=True)
    assert subject.checked_predeclared_contract_data(contract)["source_pair"] == pair(same=True)
    with pytest.raises(subject.CurrentAdmissionUnavailable):
        subject.require_original_current_admission(contract)


def array(key, dtype, shape=None):
    return dict(key=key, dtype=dtype, shape=[2, 2] if shape is None else shape, bit_exact=True)


def unsealed_case(case=None, *, same_source=False):
    """Complete emitted-shape DATA, not actual products or a solver result."""
    case = case or subject.CASE_IDS[0]
    sources = pair(same=same_source)
    arrays = [array(name, "<f4") for name in subject.FLOATS]
    arrays.insert(2, array("phase_labels", "|u1"))
    arrays += [array(name, "|b1") for name in ("main_mass_eligible_mask", "unquantified_facies_mask")]
    arrays += [array(name, "<f8", []) for name in ("E_lo_g", "E_hi_g", "U_rest_g")]
    arrays += [array("U_rest_defined", "|b1", [])]
    arrays += [array(name, "<U64", []) for name in ("U_rest_amendment_sha256", "source_sha256", "phase_sha256", "calibration_sha256")]
    identities = []
    for host in ("Reference", "Candidate"):
        commit = sources[host]["commit"]
        identity = hashlib.sha256(commit.encode("ascii")).hexdigest()
        identities.append(dict(host=host, actual_commit=commit, stored_code_sha256=identity,
            expected_code_sha256=identity, shape=[], dtype="<U64",
            raw_array_sha256=hashlib.sha256(identity.encode("utf-32-le")).hexdigest()))
    arrays.append(dict(key="code_sha256", shape=[], dtype="<U64", bit_exact=same_source,
        per_host_actual_commit_identity_valid=True, per_host_identities=identities))
    archive = dict(role="products.npz", arrays=arrays, bit_exact=same_source,
                   all_scientific_arrays_bit_exact=True, per_host_code_identity_validated=True)
    geometry = dict(role="native_geometry.npz", arrays=[array(name, "|u1" if name == "facies" else "<f8") for name in subject.GEOMETRY],
                    bit_exact=True, all_scientific_arrays_bit_exact=True, per_host_code_identity_validated=False)
    stages = {host: {"native_log.npy": pin(host + "/" + case + "/native_log.npy", sha="8" * 64)}
              for host in ("Reference", "Candidate")}
    products = dict.fromkeys(subject.PRODUCT_FLAGS, True)
    products.update(all_products_bit_exact=same_source, comparisons=[archive, geometry,
        dict(role="native_log.npy", bit_exact=True, reference=stages["Reference"]["native_log.npy"],
             candidate=stages["Candidate"]["native_log.npy"])])
    results = {host: dict(case_id=case, complete=True, frame_index=int(case.split("f")[-1]), backend="cpu",
                         precision="float32", prediction=dict(pipeline=dict(files=files)))
               for host, files in stages.items()}
    map_row = dict(reference_maximum=1.0, absolute_limit=.001, max_difference=0.0,
        failing_pixels=0, first_failing_flat_index=None, nonfinite_support_mismatch_pixels=0,
        bit_identical=True, finite_pixels_above_limit=0, allowed_exceeding_pixels=0, passed=True)
    labels = dict(reference_co2=0.0, reference_gas=0.0, changed_co2_related=0.0, changed_co2_support=0.0,
        changed_gas=0.0, changed_other=0.0, co2_limit_cm2=.05, gas_limit_cm2=.05, changed_pixels=0,
        reference_class_pixels={"0": 4}, candidate_class_pixels={"0": 4}, passed=True)
    value = dict(contract="darsia-production-addendum8v3-frame-comparison-v1", complete=True, limits_passed=True,
        **subject.ORIGINAL_SPEC_HASHES,
        frame_pixels=4, frame_finite_pixels_above_any_map_limit=0, frame_allowed_exceeding_pixels=0,
        max_exceeding_pixel_fraction=.00001, comparison_kind="same_precision",
        reference_settings=dict(host="Main", backend="cpu", precision="float32"),
        candidate_settings=dict(host="Main", backend="cpu", precision="float32"),
        run="ac52", frame_index=int(case.split("f")[-1]), labels=labels,
        maps={name: deepcopy(map_row) for name in subject.FLOATS},
        scalars={name: dict(reference=1., candidate=1., difference=0., limit=.001, passed=True) for name in subject.DOMAIN_SCALARS},
        flags={"test_flag": dict(reference=False, candidate=False, different=False, reported_only=True)},
        first_failure=None, chunk_pixels=262144, qualification_complete=False,
        whole_run_history_and_wasserstein_still_required=True, production_permission_granted=False,
        strict_FFAC_bit_identical=True, strict_exact_diagnostics=dict.fromkeys(subject.DIAGNOSTICS, True),
        strict_N56_exact_products=products, case_id=case,
        reference_result=pin("Reference/" + case + "/result.json"),
        candidate_result=pin("Candidate/" + case + "/result.json"),
        actual_host_provenance=[{}, {}], actual_separately_derived_scientific_bindings=[{}, {}],
        per_host_phase_output_sha256=["c" * 64, "c" * 64],
        geometry_array_identities=[{name: dict(dtype="|u1" if name == "facies" else "<f8", shape=[2, 2], sha256="d" * 64)
                                    for name in subject.GEOMETRY} for _ in range(2)],
        geometry_identical_required=True)
    return value, sources, results


@pytest.mark.parametrize("same_source", [False, True])
def test_complete_case_data_preserves_reported_code_inequality_without_authority(same_source):
    value, sources, results = unsealed_case(same_source=same_source)
    result = subject.checked_strict39_case(value, case_id=value["case_id"], source_pair=sources, results=results)
    assert result["strict_N56_exact_products"]["all_products_bit_exact"] is same_source
    with pytest.raises(subject.CurrentAdmissionUnavailable):
        subject.require_original_current_admission(result)


@pytest.mark.parametrize("mutate", [
    lambda d: d["strict_exact_diagnostics"].pop("all_five_maps_bit_identical"),
    lambda d: d["maps"]["mass_total"].update(max_difference=1.0),
    lambda d: d["labels"].update(changed_pixels=1),
    lambda d: d.update(scalars={}),
    lambda d: d["scalars"][subject.DOMAIN_SCALARS[0]].update(candidate=2.),
    lambda d: d["flags"]["test_flag"].update(different=True),
    lambda d: d["strict_N56_exact_products"]["comparisons"].pop(),
    lambda d: d["strict_N56_exact_products"]["comparisons"][0]["arrays"].pop(0),
    lambda d: d["strict_N56_exact_products"]["comparisons"][0]["arrays"][0].update(dtype="<f8"),
    lambda d: d["strict_N56_exact_products"]["comparisons"][0]["arrays"][0].update(bit_exact=False),
    lambda d: d["strict_N56_exact_products"]["comparisons"][0]["arrays"][-1].update(per_host_actual_commit_identity_valid=False),
    lambda d: d["strict_N56_exact_products"].update(all_products_bit_exact=True),
    lambda d: d.update(production_permission_granted=True),
    lambda d: d.update(frame_pixels=True),
    lambda d: d["maps"]["mass_total"].update(reference_maximum=float("inf")),
    lambda d: d["geometry_array_identities"][0].update(depth_m={}),
    lambda d: d.update(spec_sha256="0" * 64),
])
def test_nested_original_checks_cannot_be_replaced_by_top_level_pass(mutate):
    value, sources, results = unsealed_case()
    mutate(value)
    with pytest.raises(subject.ContractError):
        subject.checked_strict39_case(value, case_id=value["case_id"], source_pair=sources, results=results)


def test_stage_descriptors_must_match_original_result_data():
    value, sources, results = unsealed_case()
    results["Candidate"]["prediction"]["pipeline"]["files"]["native_log.npy"] = pin("changed.npy")
    with pytest.raises(subject.ContractError):
        subject.checked_strict39_case(value, case_id=value["case_id"], source_pair=sources, results=results)


def unsealed_report_data():
    """Synthetic complete report shape; no real comparison, acceptance or OS proof."""
    _, _, _, contract, release = unsealed_roles()
    objects = {}
    completions = {host: dict(complete=True, whole_run=True, worker_references_closed=True,
        newly_computed_frames=39, frames=39, case_count=39, records=[dict(frame_id=case) for case in subject.CASE_IDS],
        reused_case_ids=[], ordered_case_ids=list(subject.CASE_IDS),
        commit=contract["source_pair"][host]["commit"], pipeline="native-phase-n1", backend="cpu",
        precision="float32", frame_producers=[]) for host in ("Reference", "Candidate")}
    rows = []
    for case in subject.CASE_IDS:
        receipt, _, results = unsealed_case(case)
        receipt_pin = pin("cases/" + case + ".json")
        objects[receipt_pin["path"]] = receipt
        rows.append(dict(case_id=case, passed=True, first_failure=None, receipt=receipt_pin))
        for host, role in (("Reference", "reference_result"), ("Candidate", "candidate_result")):
            result_pin = receipt[role]
            objects[result_pin["path"]] = results[host]
            completions[host]["frame_producers"].append(dict(case_id=case, result=result_pin))
    for host in completions:
        objects[release["producers"][host]["completion"]["path"]] = completions[host]
    report = dict(complete=True, utc="2026-10-23T00:00:00+00:00", format=subject.REPORT_FORMAT,
        comparison_label="Synthetic unsealed DATA", profile_or_worker_qualified=False,
        actual_host_commits={host: contract["source_pair"][host]["commit"] for host in completions},
        scientific_source_equivalence=pin("unissued-source-review.json"), case_count=39,
        ordered_case_ids=list(subject.CASE_IDS), comparisons=rows, frame_add8_limits_passed=True,
        all39_strict_FFAC_bit_identical=True, strict_acceptance="Original emitted-shape DATA only",
        numeric_checker=contract["sources"]["numeric_checker"], comparison_kind="same_precision",
        reference_completed=release["producers"]["Reference"]["completion"],
        candidate_completed=release["producers"]["Candidate"]["completion"],
        environment_identity_proof=pin("unissued-environment.json"),
        candidate_closed_readset=release["producers"]["Candidate"]["closed_readset"],
        source_relationship=contract["source_pair"]["relationship"], independent_scientific_qualification=False,
        Wasserstein_performed=False, Wasserstein_qualification="NOT_AVAILABLE_FROM_THESE_SAVED_FRAME_RESULTS; NO_NEW_DISTANCE_COMPUTED",
        production_permission_granted=False, eighty_GO=False, new_scientific_arrays_computed=False,
        native_read_closure={}, copy_locator_closure={}, accepted_prior_comparisons=[], new_comparisons_count=39)
    return report, contract, release, objects


def test_all39_unsealed_receipts_are_read_in_original_order_but_admission_stays_closed():
    report, contract, release, objects = unsealed_report_data()
    seen = []
    def read(bound):
        seen.append(bound["path"])
        return deepcopy(objects[bound["path"]])
    assert subject.checked_strict39_report(report, contract=contract, release=release, read=read) == report
    assert [path for path in seen if "/cases/" in path] == [row["receipt"]["path"] for row in report["comparisons"]]
    with pytest.raises(subject.CurrentAdmissionUnavailable):
        subject.require_original_current_admission(report)


@pytest.mark.parametrize("mutate", [
    lambda d: d["comparisons"].reverse(),
    lambda d: d["comparisons"].pop(),
    lambda d: d.update(all39_strict_FFAC_bit_identical=False),
    lambda d: d.update(accepted_prior_comparisons=["ac52_f1"]),
    lambda d: d.update(new_comparisons_count=True),
    lambda d: d.update(reference_completed=pin("other-completion.json")),
    lambda d: d.update(independent_scientific_qualification=True),
])
def test_whole39_data_refuses_reuse_wrong_bindings_and_false_scope_claims(mutate):
    report, contract, release, objects = unsealed_report_data()
    mutate(report)
    with pytest.raises(subject.ContractError):
        subject.checked_strict39_report(report, contract=contract, release=release,
                                        read=lambda bound: objects[bound["path"]])


def test_missing_original_result_metadata_does_not_become_a_successful_report():
    report, contract, release, objects = unsealed_report_data()
    first = objects[report["comparisons"][0]["receipt"]["path"]]
    del objects[first["candidate_result"]["path"]]
    with pytest.raises(KeyError):
        subject.checked_strict39_report(report, contract=contract, release=release,
                                        read=lambda bound: objects[bound["path"]])


@pytest.mark.parametrize("field", ["newly_computed_frames", "frames", "case_count"])
@pytest.mark.parametrize("wrong", [True, 39.0])
def test_original_completion_counts_require_literal_integers(field, wrong):
    report, contract, release, objects = unsealed_report_data()
    objects[release["producers"]["Candidate"]["completion"]["path"]][field] = wrong
    with pytest.raises(subject.ContractError):
        subject.checked_strict39_report(report, contract=contract, release=release,
                                        read=lambda bound: objects[bound["path"]])


def test_canonical_digest_preserves_original_utf8_newline_encoding():
    value = {"unicode": "ø", "b": 1, "a": [None]}
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    assert subject.canonical_digest(value) == hashlib.sha256(raw).hexdigest()


def test_module_has_no_product_import_or_dynamic_loader():
    # Source inspection only: this is not a startup or runtime qualification.
    import ast
    tree = ast.parse(inspect.getsource(subject))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert all(name.split(".")[0] in {"__future__", "copy", "hashlib", "json", "math", "pathlib", "re"} for name in imports)
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in {"exec", "eval", "__import__"} for node in ast.walk(tree))
