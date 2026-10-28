"""Saved CSV export gates; tiny synthetic coarse products only."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import storage_outputs as output
from darsia.fluidflower import storage_saved_products as saved
from darsia.fluidflower import storage_saved_tables as tables
from .test_storage_saved_products import _fixture, _resolver, _alter_archive, _reseal, _json


def _read_rows(path):
    with Path(path).open(encoding="utf8", newline="") as stream:
        return list(csv.DictReader(stream))


def _two_acquisitions(tmp_path):
    first = _fixture(tmp_path / "first", case="ac52_f9")
    second = _fixture(tmp_path / "second", case="ac52_f2")
    # A distinct second acquisition has twice the saved independent masses;
    # no array is derived from another component or from a temporal sum.
    def double_mass(arrays):
        for component in ("total", "aqueous", "gas"):
            arrays["cell_mass_" + component + "_g"] *= 2
    _alter_archive(second, double_mass)
    for component, name in (("total", "total_mass_g"), ("aqueous", "aqueous_mass_g"), ("gas", "gas_mass_g")):
        second["result"][name] *= 2
        second["document"]["native_mass_totals_g"][component] *= 2
        check = second["document"]["coarse"]["native_mass_checks"][component]
        check["native_total_g"] *= 2
        check["stored_cell_sum_g"] *= 2
        check = second["document"]["native_pixel_mass_checks"][component]
        check["native_total_g"] *= 2
        check["pixel_sum_g"] *= 2
    manifest = deepcopy(first["manifest"])
    source = deepcopy(first["manifest"]["frames"][first["record"]["frame_id"]])
    manifest["frames"] = {case: dict(deepcopy(source), case_id=case) for case in ("ac52_f9", "ac52_f2")}
    manifest_path = tmp_path / "manifest.json"
    _json(manifest_path, manifest)
    for state in (first, second):
        state["record"]["manifest_sha256"] = output._asset(manifest_path)["sha256"]
        _reseal(state)
    completion = dict(complete=True, pipeline="native-phase-n1", run="ac52", precision="float32", frames=2,
        case_count=2, ordered_case_ids=["ac52_f9", "ac52_f2"], records=[first["record"], second["record"]],
        manifest=output._asset(manifest_path), configuration=manifest["configuration"], calibration={"sha256": "c" * 64})
    return first, second, completion


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_asymmetric_independent_cell_sums_native_targets_and_exact_source_are_per_frame(tmp_path, precision):
    state = _fixture(tmp_path, precision=precision)
    def change(arrays):
        for key, values in (("cell_mass_total_g", [1., 2., 6.]),
                            ("cell_mass_aqueous_g", [.25, .5, 2.25]),
                            ("cell_mass_gas_g", [2.5, .25, .25])):
            arrays[key][0, 0], arrays[key][0, 1], arrays[key][1, 0] = values
    _alter_archive(state, change)
    original = {f["path"]: Path(f["path"]).read_bytes() for f in [state["entry"]["receipt"], *state["entry"]["files"]]}
    result = tables.export_stored_frame_csv(tmp_path / "mass.csv", state["record"], resolve_asset=_resolver(state))
    rows = _read_rows(result["table"]["path"])
    assert len(rows) == 1 and rows[0]["precision"] == precision and rows[0]["row_index"] == "0"
    row = rows[0]
    assert [float(row["cell_mass_" + c + "_sum_g"]) for c in ("total", "aqueous", "gas")] == [9., 3., 3.]
    assert float(row["cell_mass_total_sum_g"]) != float(row["cell_mass_aqueous_sum_g"]) + float(row["cell_mass_gas_sum_g"])
    assert [float(row["native_" + c + "_mass_g"]) for c in ("total", "aqueous", "gas")] == [9., 3., 3.]
    assert all(float(row["stored_minus_native_" + c + "_g"]) == 0 for c in ("total", "aqueous", "gas"))
    assert json.loads(row["source_json"]) == state["document"]["source"]
    receipt = json.loads(Path(result["receipt"]["path"]).read_bytes())
    assert receipt["frames"][0]["source"] == state["document"]["source"]
    assert receipt["frames"][0]["verified_relative_limit"] == 1e-6
    assert not receipt["time_aggregation"] and not receipt["reconstructed_mass_components"]
    assert not any(p.endswith(".png") for p in state["calls"])
    assert all(Path(path).read_bytes() == before for path, before in original.items())


def test_signed_subtolerance_differences_are_exported_without_replacing_cell_sums_by_targets(tmp_path):
    state = _fixture(tmp_path)
    expected = {}
    def change(arrays):
        for component, delta, target in (("total", 1e-6, 9.), ("aqueous", 1e-6, 3.), ("gas", -1e-6, 3.)):
            cells = arrays["cell_mass_" + component + "_g"]
            cells[0, 0] += np.float32(delta)
            value = float(cells.sum(dtype=np.float64))
            expected[component] = value - target
            state["document"]["coarse"]["native_mass_checks"][component].update(
                stored_cell_sum_g=value, absolute_difference_g=abs(value - target))
    _alter_archive(state, change)
    result = tables.export_stored_frame_csv(tmp_path / "rounding.csv", state["record"], resolve_asset=_resolver(state))
    row = _read_rows(result["table"]["path"])[0]
    assert expected["total"] > 0 and expected["aqueous"] > 0 and expected["gas"] < 0
    for component, target in (("total", 9.), ("aqueous", 3.), ("gas", 3.)):
        assert abs(expected[component]) <= 1e-6 * target
        assert float(row["stored_minus_native_" + component + "_g"]) == expected[component]
        assert float(row["cell_mass_" + component + "_sum_g"]) == target + expected[component]
        assert float(row["native_" + component + "_mass_g"]) == target


def test_original_nonlexical_acquisition_order_and_distinct_masses_are_never_accumulated(tmp_path):
    first, second, completion = _two_acquisitions(tmp_path)
    result = tables.export_stored_run_csv(tmp_path / "run.csv", completion, resolve_asset=_resolver(first))
    rows = _read_rows(result["table"]["path"])
    assert [row["case_id"] for row in rows] == ["ac52_f9", "ac52_f2"]
    assert [row["row_index"] for row in rows] == ["0", "1"]
    assert [float(row["cell_mass_total_sum_g"]) for row in rows] == [9., 18.]
    assert [float(row["cell_mass_aqueous_sum_g"]) for row in rows] == [3., 6.]
    assert [float(row["cell_mass_gas_sum_g"]) for row in rows] == [3., 6.]
    receipt = json.loads(Path(result["receipt"]["path"]).read_bytes())
    assert receipt["ordered_case_ids"] == completion["ordered_case_ids"]
    assert "mass_totals_g" not in receipt and not receipt["time_aggregation"]
    assert len(receipt["loaded_manifest_assets"]) == 1 and len(receipt["frames"]) == 2


def test_pending_before_is_preserved_without_a_before_png_read(tmp_path):
    state = _fixture(tmp_path, pending=True)
    result = tables.export_stored_frame_csv(tmp_path / "pending.csv", state["record"], resolve_asset=_resolver(state))
    receipt = json.loads(Path(result["receipt"]["path"]).read_bytes())
    assert receipt["frames"][0]["before_rules"] == state["document"]["before_rules"]
    assert receipt["frames"][0]["before_rules"]["state"] == "pending"
    assert not receipt["load_before"] and not any(p.endswith(".png") for p in state["calls"])
    assert float(_read_rows(result["table"]["path"])[0]["cell_mass_total_sum_g"]) == 9.


def test_missing_exact_asset_preserves_collect_descriptor_and_creates_no_csv(tmp_path):
    state = _fixture(tmp_path)
    resolve = _resolver(state)
    def unavailable(bound):
        if bound["path"].endswith("storage_stored_frame.npz"):
            raise FileNotFoundError(bound["path"])
        return resolve(bound)
    destination = tmp_path / "not_created/mass.csv"
    with pytest.raises(saved.StoredProductCollectionRequired) as failure:
        tables.export_stored_frame_csv(destination, state["record"], resolve_asset=unavailable)
    exact = next(f for f in state["entry"]["files"] if f["path"].endswith("storage_stored_frame.npz"))
    assert failure.value.descriptor == exact
    assert not destination.parent.exists()


def test_late_run_asset_missing_cannot_publish_a_partial_time_table(tmp_path):
    first, second, completion = _two_acquisitions(tmp_path)
    resolve = _resolver(first)
    missing = next(f for f in second["entry"]["files"] if f["path"].endswith("storage_stored_frame.npz"))
    def unavailable(bound):
        if bound["path"] == missing["path"]:
            raise FileNotFoundError(bound["path"])
        return resolve(bound)
    destination = tmp_path / "not_created/run.csv"
    with pytest.raises(saved.StoredProductCollectionRequired) as failure:
        tables.export_stored_run_csv(destination, completion, resolve_asset=unavailable)
    assert failure.value.descriptor == missing and not destination.parent.exists()


@pytest.mark.parametrize("occupied", ["csv", "receipt"])
def test_exclusive_outputs_refuse_before_reader_or_original_replacement(tmp_path, occupied):
    state = _fixture(tmp_path)
    destination = tmp_path / "mass.csv"
    path = destination if occupied == "csv" else destination.with_name("mass.csv.receipt.json")
    path.write_bytes(b"keep exact original")
    with pytest.raises(FileExistsError, match="immutable"):
        tables.export_stored_frame_csv(destination, state["record"], resolve_asset=_resolver(state))
    assert path.read_bytes() == b"keep exact original" and not state["calls"]


@pytest.mark.parametrize("invalid", ["duplicate", "different_order"])
def test_invalid_completion_order_never_makes_a_csv(tmp_path, invalid):
    first, second, completion = _two_acquisitions(tmp_path)
    if invalid == "duplicate":
        completion["records"] = [first["record"], first["record"]]
        completion["ordered_case_ids"] = ["ac52_f9", "ac52_f9"]
    else:
        completion["ordered_case_ids"].reverse()
    destination = tmp_path / "not_created/run.csv"
    with pytest.raises(ValueError, match="run/order"):
        tables.export_stored_run_csv(destination, completion, resolve_asset=_resolver(first))
    assert not destination.parent.exists() and not first["calls"]


def test_frame_expected_identity_is_not_weakened_by_export(tmp_path):
    state = _fixture(tmp_path)
    destination = tmp_path / "not_created/mass.csv"
    with pytest.raises(ValueError, match="expected run"):
        tables.export_stored_frame_csv(destination, state["record"], resolve_asset=_resolver(state), expected={"run": "wrong"})
    assert not destination.parent.exists()
