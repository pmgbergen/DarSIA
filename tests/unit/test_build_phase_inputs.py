"""Metadata-only assembly joins; no RAW file, correction or scientific QA."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower.mass.n1_pipeline import DataReader, _write_json
from darsia.fluidflower.pipeline import build_phase_inputs as subject


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    def bound(label, digit):
        return dict(path=str(tmp_path / label), sha256=digit * 64, size_bytes=17)

    raws = [bound(str(index) + ".ARW", str(index)) for index in range(4)]
    photo, floor = bound("photo.npz", "a"), bound("floor.npy", "b")
    support = dict(cell_to_reference_native=[[8, 0, -124], [0, 8, 4], [0, 0, 1]],
        grid_archive_key="sand", grid_shape_hw=[382, 710], guard_px=8,
        fill_policy="original-fill", hole_max_area_px=60000, vertical_gap_max_px=64)
    old_native = _write_json(tmp_path / "old_native.json", dict(support=support, source_grid="old"))
    native = _write_json(tmp_path / "native.json", dict(support=support, source_grid="current"))
    phase = dict(chain_assets=dict(floor=floor),
                 parameters=dict(geometry=dict(floor_source_origin_xy=[-416, -496])),
                 physical_scale_approved=False)
    old_phase = _write_json(tmp_path / "old_phase.json", phase)
    new_phase = _write_json(tmp_path / "phase.json", dict(phase, explicit_supplied_grid="current"))
    identity = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]

    def spec(index, selected):
        return dict(case_id="frame" + str(index), frame_index=index, elapsed_h=float(index),
            source=dict(run="sample", source_sha256=raws[index]["sha256"],
                source_binding=dict(source_sha256=raws[index]["sha256"], source_kind="raw",
                    run="sample", elapsed_h=float(index),
                    raw_acquisition={key: raws[index][key] for key in ("path", "sha256")}),
                frame_index=index, elapsed_h=float(index), selected=selected,
                previous_source_sha256=None if index == 0 else raws[index - 1]["sha256"],
                studio_anchor=False, optical_geometry_qa_pass=False),
            geometry=bound("old_geometry" + str(index) + ".json", "c"),
            native_task=dict(kind="raw_rig", original_task_only=True), mass_spec=dict(original_mass_only=True),
            provenance=dict(original=False),
            phase_roles=dict(area=dict(role="selected", factor=1., revision={"original": True}),
                current_h=identity, reference_to_current_h=identity,
                static=dict(components=bound("old_static.npz", "d"), source_ids=[1], unchanged_source=False),
                retained_source_cache=None, encoding_contract=None, native_row=None,
                probability_receipt=None, retained_frame_index=None, light_epoch_id="saved",
                inherited_spacing_m=[.001, .001], probability_spacing_m=[.001, .001],
                placement_metadata=dict(original_scope=False), photo_sand_reference=photo,
                photo_proof=dict(sha256=photo["sha256"], phase_labels_used=False, photo_sand_qa_pass=False)))

    reference, previous, frames = spec(0, False), spec(1, False), [spec(2, True), spec(3, True)]
    for prepared, index in ((reference, 0), (previous, 1)):
        prepared["raw_acquisition"] = {key: raws[index][key] for key in ("path", "sha256")}
    old = dict(run="sample", native=dict(configuration=old_native),
        phase=dict(configuration=old_phase,
            reference=dict(source_sha256=raws[0]["sha256"], full=dict(source_sha256=raws[0]["sha256"])),
            initial_previous=dict(source=deepcopy(previous["source"]), full=dict(source_sha256=raws[1]["sha256"]))),
        frames=frames, history_policy="exact original acquisition history", selection_index_field="frame_index")
    old_binding = _write_json(tmp_path / "old_run.json", old)
    replacements = [subject.BuildPhaseReplacement(raw, bound("geometry" + str(i) + ".json", "e"),
        dict(role="selected", factor=1., revision={"producer": "measured"}),
        dict(components=bound("static" + str(i) + ".npz", "f"), source_ids=[1], unchanged_source=False),
        deepcopy(identity), deepcopy(identity)) for i, raw in enumerate(raws)]

    class Bridge:
        context = SimpleNamespace(run="sample", provenance=dict(native_configuration=native, raw_acquisition=raws[0]))

    monkeypatch.setattr(subject, "BuildNativeBridge", Bridge)
    bridge = Bridge()
    call_args = dict(bridge=bridge, old_run_configuration=old_binding, reference_spec=reference,
        initial_previous_spec=previous, selected_sources=[deepcopy(row["source"]) for row in frames],
        replacements=replacements, phase_configuration=new_phase, temporal_pad_px=0)
    return SimpleNamespace(args=call_args, old=old, phase=phase, native=native, raws=raws, tmp=tmp_path)


def test_assembly_preserves_source_photo_scope_and_inputs_without_reading_raw(inputs):
    before = deepcopy({key: value for key, value in inputs.args.items() if key != "bridge"})
    result = subject.assemble_build_phase_inputs(**inputs.args)
    assert {key: value for key, value in inputs.args.items() if key != "bridge"} == before
    assert result["format"] == subject.FORMAT and result["native"]["configuration"] == inputs.native
    assert result["phase"] == dict(configuration=inputs.args["phase_configuration"])
    assert result["history_policy"] == inputs.old["history_policy"]
    for actual, original in zip(result["frames"], inputs.old["frames"]):
        assert actual["source"] == original["source"] and not actual["source"]["optical_geometry_qa_pass"]
        assert "source_path" not in original["source"] and "raw_acquisition" not in original
        assert set(actual["source"]["source_binding"]["raw_acquisition"]) == {"path", "sha256"}
        assert set(actual["raw_acquisition"]) == {"path", "sha256", "size_bytes"}
        assert actual["phase_roles"]["photo_sand_reference"] == original["phase_roles"]["photo_sand_reference"]
        assert actual["phase_roles"]["photo_proof"] == original["phase_roles"]["photo_proof"]
        assert "native_task" not in actual and "mass_spec" not in actual
        assert not {"native_row", "probability_receipt", "retained_frame_index"} & actual["phase_roles"].keys()
        assert actual["phase_roles"]["retained_source_cache"] is None
        assert actual["phase_roles"]["encoding_contract"] is None
    assert all(not Path(raw["path"]).exists() for raw in inputs.raws)
    result["frames"][0]["source"]["optical_geometry_qa_pass"] = True
    assert not inputs.old["frames"][0]["source"]["optical_geometry_qa_pass"]


@pytest.mark.parametrize("distinct_case", [False, True], ids=["same-spec", "distinct-case"])
def test_same_raw_reference_and_history_need_one_replacement_without_relabelling(inputs, distinct_case):
    args = inputs.args
    args["initial_previous_spec"] = deepcopy(args["reference_spec"])
    if distinct_case:
        args["initial_previous_spec"]["case_id"] = "initial-history"
    inputs.old["phase"]["initial_previous"] = dict(source=deepcopy(args["reference_spec"]["source"]),
        full=dict(source_sha256=inputs.raws[0]["sha256"]))
    inputs.old["frames"][0]["source"]["previous_source_sha256"] = inputs.raws[0]["sha256"]
    args["selected_sources"][0] = deepcopy(inputs.old["frames"][0]["source"])
    args["old_run_configuration"] = _write_json(inputs.tmp / "shared_reference_run.json", inputs.old)
    args["replacements"] = [item for i, item in enumerate(args["replacements"]) if i != 1]
    result = subject.assemble_build_phase_inputs(**args)
    assert result["reference_spec"]["raw_acquisition"] == result["initial_previous_spec"]["raw_acquisition"]
    assert result["initial_previous_spec"]["case_id"] == args["initial_previous_spec"]["case_id"]
    assert (result["reference_spec"] == result["initial_previous_spec"]) is not distinct_case
    assert len(result["frames"]) == 2


@pytest.mark.parametrize("change", ["order", "subset", "source", "raw_path", "missing", "duplicate",
                                    "baseline", "history", "reference_full", "declared_raw", "prep_raw",
                                    "floor", "floor_origin", "support"])
def test_assembly_refuses_changed_original_joins_without_inventing_roles(inputs, change):
    args = inputs.args
    if change == "order":
        args["selected_sources"].reverse()
    elif change == "subset":
        args["selected_sources"].pop()
    elif change == "source":
        args["selected_sources"][0]["optical_geometry_qa_pass"] = True
    elif change == "raw_path":
        raw = dict(args["replacements"][2].raw_acquisition, path=str(inputs.tmp / "another.ARW"))
        args["replacements"][2] = replace(args["replacements"][2], raw_acquisition=raw)
    elif change == "missing":
        args["replacements"].pop()
    elif change == "duplicate":
        args["replacements"].append(args["replacements"][0])
    elif change == "baseline":
        args["bridge"].context.provenance["raw_acquisition"] = inputs.raws[1]
    elif change == "history":
        args["initial_previous_spec"]["source"]["optical_geometry_qa_pass"] = True
    elif change == "reference_full":
        inputs.old["phase"]["reference"]["full"]["source_sha256"] = "f" * 64
        args["old_run_configuration"] = _write_json(inputs.tmp / "changed_run.json", inputs.old)
    elif change == "declared_raw":
        inputs.old["frames"][0]["source"]["source_binding"]["raw_acquisition"]["sha256"] = "f" * 64
        args["selected_sources"][0] = deepcopy(inputs.old["frames"][0]["source"])
        args["old_run_configuration"] = _write_json(inputs.tmp / "changed_run.json", inputs.old)
    elif change == "prep_raw":
        args["reference_spec"]["raw_acquisition"]["path"] = str(inputs.tmp / "another-reference.ARW")
    elif change in {"floor", "floor_origin"}:
        phase = deepcopy(inputs.phase)
        if change == "floor":
            phase["chain_assets"]["floor"]["sha256"] = "f" * 64
        else:
            phase["parameters"]["geometry"]["floor_source_origin_xy"] = [0, 0]
        args["phase_configuration"] = _write_json(inputs.tmp / "changed_phase.json", phase)
    else:
        reader = DataReader()
        native = reader.json(inputs.native)
        native["support"]["hole_max_area_px"] += 1
        args["bridge"].context.provenance["native_configuration"] = _write_json(inputs.tmp / "changed_native.json", native)
    with pytest.raises(ValueError):
        subject.assemble_build_phase_inputs(**args)
    assert all(not Path(raw["path"]).exists() for raw in inputs.raws)
