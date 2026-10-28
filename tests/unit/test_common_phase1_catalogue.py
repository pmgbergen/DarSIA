"""Common closure inputs use named metadata dependencies and preserve the fit plan."""
import builtins
from copy import deepcopy
from pathlib import Path
import hashlib
import json
from types import SimpleNamespace

import pytest

from darsia.fluidflower.mass.n1_pipeline import DataReader
from darsia.fluidflower.pipeline import common_calibration as common
from darsia.fluidflower.pipeline import staged_data
from darsia.fluidflower.pipeline.job_inputs import InputCatalog
from .test_staged_data import phase1_data, _closure, _rewrite_frame, _write


def binding(closed):
    return dict(format=common.PHASE1_INPUTS_FORMAT, configuration=closed["configuration"],
        completion=closed["completion"], code_commit="1" * 40)


def test_common_plan_binding_preserves_all_original_fitting_inputs_and_rules(tmp_path):
    plan = dict(format="darsia-n1-refit-plan-v1", run="refit", rules=dict(original=[1, 2]),
        runs=dict(fit=["fit"], heldout=["held"]), statistics=dict(fit=[dict(frame_index=1)]))
    before = deepcopy(plan)
    path = tmp_path / "phase1.json"
    raw = b'{"format":"darsia-native-phase-statistics-run-v1"}'
    path.write_bytes(raw)
    pin = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    result = common.bind_common_phase1(plan, configuration=pin, completion=dict(pin, path=str(tmp_path / "completed.json")),
        code_commit="1" * 40, reader=DataReader())
    assert plan == before
    assert {key: value for key, value in result.items() if key != "common_phase1"} == before
    assert result["common_phase1"]["format"] == common.PHASE1_INPUTS_FORMAT


def test_c_binder_carries_exact_p1_aliases_but_unused_aliases_add_no_resource(tmp_path):
    physical = tmp_path / "real.json"
    physical.write_bytes(b"{}")
    bound = dict(path=str(physical), sha256=hashlib.sha256(b"{}").hexdigest(), size_bytes=2)
    alias = dict(logical=dict(bound, path=str(tmp_path / "old.json")), physical=bound)
    config = tmp_path / "phase1.json"
    raw = json.dumps(dict(format="darsia-native-phase-statistics-run-v1", data_aliases=[alias])).encode()
    config.write_bytes(raw)
    pin = dict(path=str(config), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    plan = dict(format="darsia-n1-refit-plan-v1", run="refit", statistics={})
    result = common.bind_common_phase1(plan, configuration=pin, completion=dict(pin, path=str(tmp_path / "closed.json")),
        code_commit="1" * 40, reader=DataReader())
    assert result["data_aliases"] == [alias] and "data_aliases" not in plan
    catalogue = InputCatalog(dict(series=str(tmp_path)), result["data_aliases"])
    assert catalogue.physical(alias["logical"]) == bound
    assert catalogue.entries == {}
    conflicting = dict(plan, data_aliases=[dict(alias, physical=dict(bound, path=str(tmp_path / "different.json")))])
    with pytest.raises(ValueError, match="conflicts"):
        common.bind_common_phase1(conflicting, configuration=pin, completion=pin, code_commit="1" * 40,
            reader=DataReader())


def test_c_catalogue_resolves_the_named_p1_phase_alias_used_by_probability_metadata(phase1_data, tmp_path):
    # Dependency-walk metadata only: this receipt is not a passing probability
    # product. The original admitted-worker verifier remains mandatory.
    phase = _write(tmp_path / "real-phase.json", dict(model_bundles={}))
    logical = dict(phase, path=str(tmp_path / "old-phase.json"))
    phase1_data["phase"]["configuration"] = logical
    phase1_data["data_aliases"] = [dict(logical=logical, physical=phase)]
    closed = _closure(tmp_path, phase1_data)
    record = deepcopy(closed["record"])
    receipt = _write(Path(closed["frame"]["path"]).parent / "probability-receipt.json",
                     dict(synthetic_dependency_only=True))
    record["files"]["probability-receipt.json"] = receipt
    record["probabilities"] = dict(receipt=receipt)
    completed = _rewrite_frame(closed, record)
    plan = common.bind_common_phase1(dict(format="darsia-n1-refit-plan-v1", run="refit", statistics={}),
        configuration=closed["configuration"], completion=completed, code_commit="1" * 40, reader=DataReader())
    catalogue = InputCatalog(dict(series=str(tmp_path)), plan["data_aliases"])
    common.catalogue_common_phase1(catalogue, plan["common_phase1"], use="refit.common_phase1")
    selected = [row for row in catalogue.entries.values()
                if "refit.common_phase1.phase_configuration" in row["uses"]]
    assert len(selected) == 1 and selected[0]["reference"]["relative_path"] == "real-phase.json"
    assert not Path(logical["path"]).exists()
    with pytest.raises(ValueError, match="explicitly carry"):
        common.catalogue_common_phase1(InputCatalog(dict(series=str(tmp_path))), plan["common_phase1"],
                                      use="refit.common_phase1")


def test_metadata_dependency_walk_declares_every_original_verifier_read_without_science_imports(
        phase1_data, tmp_path, monkeypatch):
    closed = _closure(tmp_path, phase1_data)
    catalogue = InputCatalog(dict(series=str(tmp_path)))
    original_import = builtins.__import__
    def no_science(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "scipy"} or name.rsplit(".", 1)[-1] in {
                "n1_refit", "n1_statistics", "refit_runner", "series"}:
            pytest.fail("Catalogue expansion imported scientific validation: " + name)
        return original_import(name, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", no_science)
        common.catalogue_common_phase1(catalogue, binding(closed), use="closure.phase1")
    declared = {str(Path(tmp_path / row["reference"]["relative_path"])) for row in catalogue.entries.values()}
    # Run the unchanged verifier with a reader that refuses every undeclared
    # locator. The fixture's products are small synthetic bytes, not arrays.
    def physical(bound):
        if str(Path(bound["path"])) not in declared:
            raise ValueError("Original verifier read an undeclared closure input")
        return bound
    reader = DataReader()
    reader.io_binding = SimpleNamespace(physical_descriptor=physical)
    verified = staged_data.verify_phase1_completion(closed["completion"], closed["configuration"],
        expected_commit="1" * 40, reader=reader)
    assert verified["payload_bytes_verified"] is True and verified["scientific_arrays_decoded"] is False


@pytest.mark.parametrize("mutation", ("commit", "source", "missing_input"))
def test_forged_or_missing_common_phase1_dependencies_are_refused(phase1_data, tmp_path, mutation):
    closed = _closure(tmp_path, phase1_data)
    bound = binding(closed)
    if mutation == "commit":
        bound["code_commit"] = "2" * 40
    else:
        record = deepcopy(closed["record"])
        if mutation == "source":
            record["source"]["source_sha256"] = "f" * 64
        else:
            leaf = next(iter(record["files"]))
            record["files"][leaf]["path"] = str(tmp_path / "not-produced.bin")
        bound["completion"] = _rewrite_frame(closed, record)
    with pytest.raises((ValueError, OSError)):
        common.catalogue_common_phase1(InputCatalog(dict(series=str(tmp_path))), bound, use="closure.phase1")
