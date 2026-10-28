"""Synthetic typed-metadata mapping tests, not a scientific role admission."""
from copy import deepcopy

import pytest

from darsia.fluidflower.series_metadata_assembler import assemble_run_metadata


def _binding(index):
    return {"path": f"//server/share/role{index}.json", "sha256": str(index)*64, "size_bytes": 123}


def _frame(index):
    source = {"run": "run1", "selected": True, "source_path": f"camera/{index}.JPG",
              "source_sha256": str(index)*64, "camera_datetime": f"literal-camera-{index}",
              "elapsed_h": float(index), "frame_index": index*100, "work_index": index,
              "previous_source_sha256": "0"*64, "source_binding": {"original": _binding(index)},
              "studio_anchor": False}
    return {"case_id": f"run1_f{index}", "frame_index": index, "elapsed_h": float(index),
            "source": source, "geometry": _binding(index),
            "native_task": {"kind": "source", "geometry": _binding(index), "parent_frame_index": index,
                            "parent_reference": _binding(0), "processor": {"native_manifest": _binding(0)}},
            "phase_roles": {"area": {"role": "selected", "factor": 1.0},
                            "current_h": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            "reference_to_current_h": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            "static": _binding(0), "retained_source_cache": {"contract": "supplied"},
                            "encoding_contract": "supplied", "light_epoch_id": "supplied",
                            "native_row": None, "photo_sand_reference": _binding(0), "photo_proof": {"original": True}},
            "mass_spec": {"atmosphere": {"pressure_pa": 100000}, "inner_polygon_native": [[0, 0], [1, 1]],
                          "original_domain": _binding(0), "photo_top": _binding(0), "source_receipt": _binding(index),
                          "nominal_injected_mass_g": 1.0, "selected_target_h": float(index)}}


def _data():
    first, last = _frame(1), _frame(3)
    last["source"]["previous_source_sha256"] = "1"*64
    calibration = _binding(9)
    previous = {"format": "darsia-native-phase-n1-run-v1", "run": "run1", "frames": [first, last],
                "ordered_case_ids": ["run1_f1", "run1_f3"], "selection_index_field": "work_index",
                "native": {"configuration": _binding(0)},
                "phase": {"initial_previous": {"source": {"source_sha256": "0"*64}}},
                "mass": {"calibration": calibration, "template": {"calibration": calibration}},
                "table_fields": ["mass_g"], "history_policy": "baseline plus selected only"}
    records = [deepcopy(first), _frame(2), deepcopy(last)]
    selected = [deepcopy(record["source"]) for record in records]
    return previous, selected, records


def _raw_data():
    previous, selected, records = _data()
    for frame in [*previous["frames"], *records]:
        frame["native_task"] = {"kind": "raw_rig", "geometry": _binding(1),
                                "processor": {"raw_binding": _binding(0)}}
        frame["phase_roles"]["retained_source_cache"] = None
        frame["phase_roles"]["encoding_contract"] = None
    return previous, selected, records


def test_real_raw_task_abi_has_no_jpeg_parent_fields_and_preserves_existing_roles():
    previous, selected, records = _raw_data()
    before = deepcopy((previous, selected, records))
    result, receipt = assemble_run_metadata(previous, selected, records)
    assert result["ordered_case_ids"] == ["run1_f1", "run1_f2", "run1_f3"]
    for generated, supplied in zip(result["frames"], records):
        assert generated["native_task"] == supplied["native_task"]
        assert set(generated["native_task"]) == {"kind", "geometry", "processor"}
    assert not receipt["roles_invented"]
    assert (previous, selected, records) == before


@pytest.mark.parametrize("change", ["parent_field", "missing_raw_binding"])
def test_raw_role_validation_does_not_accept_a_jpeg_parent_or_a_missing_binding(change):
    previous, selected, records = _raw_data()
    if change == "parent_field":
        records[1]["native_task"]["parent_frame_index"] = 1
    else:
        records[1]["native_task"]["processor"] = {"native_manifest": _binding(0)}
    with pytest.raises(ValueError, match="RAW native roles"):
        assemble_run_metadata(previous, selected, records)


def test_typed_mapper_inserts_supplied_roles_without_changing_original_roles_or_inputs():
    previous, selected, records = _data()
    snapshot = deepcopy((previous, selected, records))
    result, receipt = assemble_run_metadata(previous, selected, records)
    assert result["ordered_case_ids"] == ["run1_f1", "run1_f2", "run1_f3"]
    assert [frame["frame_index"] for frame in result["frames"]] == [1, 2, 3]
    assert [frame["source"]["previous_source_sha256"] for frame in result["frames"]] == ["0"*64, "1"*64, "2"*64]
    assert [frame["native_task"]["parent_frame_index"] for frame in result["frames"]] == [1, 2, 3]
    assert [frame["source"]["frame_index"] for frame in result["frames"]] == [100, 200, 300]
    for original, generated in zip(previous["frames"], [result["frames"][0], result["frames"][2]]):
        for field in ("native_task", "geometry", "phase_roles", "mass_spec"):
            assert original[field] == generated[field]
    assert receipt["old_frame_count"] == 2 and receipt["new_frame_count"] == 1
    assert not receipt["roles_invented"] and not receipt["adopted_for_execution"]
    assert (previous, selected, records) == snapshot


@pytest.mark.parametrize("stage,field", [("native_task", "geometry"), ("phase_roles", "current_h"),
                                        ("phase_roles", "photo_proof"), ("mass_spec", "atmosphere")])
def test_missing_new_role_is_not_inferred_or_copied_from_a_neighbor(stage, field):
    previous, selected, records = _data()
    del records[1][stage][field]
    with pytest.raises(ValueError, match="Missing supplied"):
        assemble_run_metadata(previous, selected, records)


@pytest.mark.parametrize("mutation", ["stage", "source_binding", "case_id"])
def test_changed_original_role_or_case_is_rejected(mutation):
    previous, selected, records = _data()
    if mutation == "stage":
        records[2]["phase_roles"]["current_h"][0][0] = 1.01
    elif mutation == "source_binding":
        records[2]["source"]["source_binding"]["original"]["sha256"] = "changed"
    else:
        records[2]["case_id"] = "another_case"
    with pytest.raises(ValueError, match="Original"):
        assemble_run_metadata(previous, selected, records)


def test_role_coverage_is_exact_and_no_unselected_record_is_ignored():
    previous, selected, records = _data()
    with pytest.raises(ValueError, match="one-to-one"):
        assemble_run_metadata(previous, selected, records[:-1])
    with pytest.raises(ValueError, match="one-to-one"):
        assemble_run_metadata(previous, selected, [*records, _frame(4)])


def test_selection_order_is_never_sorted_or_repaired():
    previous, selected, records = _data()
    with pytest.raises(ValueError, match="relative order"):
        assemble_run_metadata(previous, list(reversed(selected)), records)


def test_original_fixed_validator_still_rejects_expected_current_product_inputs():
    previous, selected, records = _data()
    records[1]["mass_spec"]["labels"] = _binding(8)
    with pytest.raises(ValueError, match="fresh stage outputs"):
        assemble_run_metadata(previous, selected, records)


def test_unknown_original_top_level_metadata_is_copied_exactly():
    previous, selected, records = _data()
    extra = {"historical_source_note": "retained", "nested": [1, 2, 3]}
    previous["frames"][1]["extra_original_metadata"] = deepcopy(extra)
    records[2]["extra_original_metadata"] = deepcopy(extra)
    result, _ = assemble_run_metadata(previous, selected, records)
    assert result["frames"][2]["extra_original_metadata"] == extra


@pytest.mark.parametrize("mutation", ["change", "omit", "add"])
def test_whole_original_record_cannot_drop_change_or_add_unknown_metadata(mutation):
    previous, selected, records = _data()
    previous["frames"][1]["extra_original_metadata"] = {"value": "retained"}
    records[2]["extra_original_metadata"] = {"value": "retained"}
    if mutation == "change":
        records[2]["extra_original_metadata"]["value"] = "changed"
    elif mutation == "omit":
        del records[2]["extra_original_metadata"]
    else:
        records[2]["another_unknown_field"] = 12
    with pytest.raises(ValueError, match="whole frame DATA"):
        assemble_run_metadata(previous, selected, records)


@pytest.mark.parametrize("sha", ["a"*63, "A"*64, "g"*64, None, 123])
def test_new_supplied_identity_requires_canonical_lowercase_SHA(sha):
    previous, selected, records = _data()
    records[1]["source"]["source_sha256"] = sha
    with pytest.raises(ValueError, match="canonical64"):
        assemble_run_metadata(previous, selected, records)


@pytest.mark.parametrize("path", [None, "", "   "])
def test_supplied_source_path_cannot_be_blank(path):
    previous, selected, records = _data()
    records[1]["source"]["source_path"] = path
    with pytest.raises(ValueError, match="nonblank literal source path"):
        assemble_run_metadata(previous, selected, records)


@pytest.mark.parametrize("run", [None, "", "   ", 123])
def test_original_run_needs_actual_literal_text(run):
    previous, selected, records = _data()
    previous["run"] = run
    with pytest.raises(ValueError, match="actual literal run text"):
        assemble_run_metadata(previous, selected, records)


@pytest.mark.parametrize("field", ["source", "native_task", "geometry", "phase_roles", "mass_spec"])
def test_none_stage_or_source_mapping_fails_clearly(field):
    previous, selected, records = _data()
    records[1][field] = None
    with pytest.raises(ValueError, match="nonempty mapping"):
        assemble_run_metadata(previous, selected, records)


def test_none_record_fails_clearly_without_a_scientific_fallback():
    previous, selected, records = _data()
    records[1] = None
    with pytest.raises(ValueError, match="frame DATA must be a mapping"):
        assemble_run_metadata(previous, selected, records)
