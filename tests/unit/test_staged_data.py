"""Finite synthetic metadata/byte-closure tests; no image or array decoding.

The phase-one receipts here are deliberately synthetic unit fixtures in their
own temporary directory. They are never reported as genuine worker products.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from darsia.fluidflower.pipeline import staged_data as staged
from darsia.fluidflower.mass.n1_pipeline import DataReader, asset


def _bound(label):
    return dict(path="//unit-only/share/" + label, sha256="a" * 64, size_bytes=1)


@pytest.fixture
def phase1_data():
    """Explicit DATA specimen; referenced source payloads are not read."""
    source = dict(run="run-one", source_sha256="c" * 64, previous_source_sha256="b" * 64,
                  selected=True, work_index=1, elapsed_h=1., studio_anchor=False,
                  source_binding=dict(source_sha256="c" * 64),
                  original_unmapped_metadata=dict(keep_exactly=[1, "retained"]))
    baseline = dict(complete=True, source_sha256="b" * 64, phase_shape_hw=[2, 2],
                    native_origin_xy=[0, 0], native_centre=dict(native_shape_hw=[2, 2],
                    log_linear_rgb=_bound("baseline.npy"), valid_mask=_bound("baseline_valid.npy")),
                    ring_windows=[], binding=dict(geometry={"H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}))
    shared = dict(physical_reference=_bound("physical.npz"), reference_matrix=[[1, 0, 0], [0, -1, 1], [0, 0, 1]],
                  floor_curve=_bound("floor.npy"), floor_origin_xy=[0, 0], depth_points=[[0, 0, .01], [1, 0, .01], [0, 1, .01]],
                  porosity_by_facies={"1": .3}, swi_by_facies={"1": .2}, unquantified_facies=[],
                  clearwater=dict(rgb=_bound("clear.npy"), valid=_bound("clear_valid.npy"),
                                  H_reference_to_grid=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
                  optics=dict(metadata=_bound("optics.json"), color_paths={"1": _bound("colour.json")},
                              signals={"1": _bound("signal.csv")}, flash=_bound("flash.json")))
    template = dict(format=staged.STATISTICS_FORMAT, shared=shared,
                    physical_correction=_bound("correction.json"), recipe=dict(alkalinity_M=.001, btb_M=.0001),
                    dimensions_m=[1., 1.], boxes=dict(box1=[[0, 0], [1, 1]], box2=[[0, 0], [.5, .5]]),
                    summary=dict(baseline_gain=1.1, anchor_target_h=2.25, intervals=32768,
                                 max_column_distance_m=.020, max_depth_m=.0155, max_porosity=.45),
                    bindings=dict(round4_spec_sha256="d" * 64, amendment_sha256="e" * 64,
                                  original_metadata=dict(keep=["all", "values"])),
                    geometry_fields=dict(source_to_native="H", original_domain_to_native="H",
                                         carried_homography_provenance="original_H"))
    spec = dict(case_id="case-one", frame_index=1, elapsed_h=1., source=source,
                native_task=dict(kind="rig", geometry=_bound("native_geometry.json"),
                                 processor=dict(native_manifest=_bound("native_manifest.json")),
                                 parent_frame_index=None, parent_reference=False),
                geometry=_bound("rig_geometry.json"),
                phase_roles=dict(area=dict(role="selected", factor=1., revision=dict(
                                     contract="synthetic-explicit-area-revision-unit-v1", area_factor=1., source="synthetic unit fixture")), current_h=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                 reference_to_current_h=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                 static=dict(components=_bound("static.npz"), source_ids=["original"], unchanged_source=True),
                                 retained_source_cache=None, encoding_contract=dict(original="explicit"), light_epoch_id="same",
                                 inherited_spacing_m=[.5, .5], probability_spacing_m=[.5, .5],
                                 placement_metadata=dict(contract="synthetic explicit placement metadata"),
                                 native_row=_bound("native_row.json"), probability_receipt=_bound("probability.json")),
                mass_spec=dict(atmosphere=dict(pressure_bar=1., temperature_c=20.), inner_polygon_native=[[0, 0], [1, 0], [1, 1]],
                               original_domain=dict(parent_labels_map=_bound("original.png"), native_shape_hw=[2, 2], H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                               photo_top=_bound("top.npz"), source_receipt=_bound("source.json"), nominal_injected_mass_g=1.,
                               selected_target_h=1., light_epoch="same"))
    return dict(format=staged.PHASE1_FORMAT, run="run-one", precision="float32", frames=[spec],
                native=dict(configuration=_bound("native_configuration.json")),
                phase=dict(configuration=_bound("phase_configuration.json"),
                           reference=dict(full=baseline, native_row=_bound("reference_row.json"), source_sha256="b" * 64),
                           initial_previous=dict(current_h=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], full=deepcopy(baseline),
                                                 native_row=_bound("previous_row.json"), source=dict(source_sha256="b" * 64))),
                statistics=template, history_policy="strict original acquisition order", selection_index_field="work_index",
                ordered_case_ids=["case-one"],
                storage=dict(format="darsia-storage-output-request-v1", resolution_mm="native", gas_score_encoding="uint8-la-q255-v1"),
                provenance=dict(original_unknown_metadata=["retained", 17]))


def _write(path, value=None, raw=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists()
    path.write_bytes(raw if raw is not None else (json.dumps(value, allow_nan=False) + "\n").encode())
    return asset(path)


def _closure(tmp_path, data, *, pending=False):
    """Create small bound bytes, not real phase/native scientific products."""
    configuration = _write(tmp_path / "phase1_configuration.json", data)
    spec = data["frames"][0]
    root = tmp_path / "preparation" / spec["case_id"]
    files = {}
    def binary(name):
        bound = _write(root / name, raw=("synthetic-unit-bytes:" + name).encode())
        files[name] = bound
        return bound
    native = dict(complete=True, source_sha256=spec["source"]["source_sha256"], processor_kind="rig",
                  full=dict(complete=True, source_sha256=spec["source"]["source_sha256"], contract="unit-metadata-only",
                            storage="original", phase_shape_hw=[2, 2], native_origin_xy=[0, 0],
                            native_centre=dict(native_shape_hw=[2, 2], log_linear_rgb=binary("native_log.npy"), valid_mask=binary("native_valid.npy")),
                            ring_windows=[], binding=dict(geometry={"H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})),
                  audit=_write(root / "native_audit.json", dict(synthetic_unit_audit=True)))
    files["native_audit.json"] = native["audit"]
    native_receipt = _write(root / "native.json", native); files["native.json"] = native_receipt
    history = _write(root / "history_input.json", dict(complete=True, source=spec["source"], full=native["full"],
        temporal_log=binary("temporal_log.npy"), temporal_valid=binary("temporal_valid.npy")))
    files["history_input.json"] = history
    phase = dict(complete=True, source_sha256=spec["source"]["source_sha256"], labels=binary("phase_labels.png"),
                 classes=binary("phase_classes.npz"), area=dict(binary("phase_area.npz"), array_key="pixel_area_mm2"),
                 qa=binary("phase_qa.json"), operator_masks=binary("phase_masks.npz"), history_input=history)
    before_metadata = (dict(state="pending", reason="native_centre_not_computed", source_array="NativeCentreResult.g3_pixel") if pending
                       else dict(state="complete", encoding="uint8-la-q255-v1", rescaled=False, clipped=False,
                            source_array="NativeCentreResult.g3_pixel", validity_array="NativeCentreResult.flatfield_valid",
                            source_box_xywh=[0, 0, 2, 2], source_shape_hw=[2, 2], target_shape_hw=[2, 2],
                            finite_valid_pixels=4, finite_min=0., finite_max=1.,
                            source_identity=dict(sha256="a" * 64, dtype="<f4", shape=[2, 2]),
                            validity_identity=dict(sha256="b" * 64, dtype="|b1", shape=[2, 2]),
                            existing_probability_encoding=None,
                            quantization="L=round-to-nearest-even(255*p); decode L/255 only where A=255",
                            missing="A=0; L=0 is padding, not evidence",
                            coarse_reduction="arithmetic mean L/255 over A=255 pixels; valid-pixel fraction of all cell pixels; no-valid mean NaN"))
    before_document = dict(format=staged.BEFORE_FORMAT, complete=True, source=spec["source"], before_rules=before_metadata)
    if not pending:
        before_document["png"] = binary("storage_gas_probability_before_rules.png")
    before_receipt = _write(root / "phase_before.json", before_document); files["phase_before.json"] = before_receipt
    before = dict(state=before_metadata["state"], receipt=before_receipt)
    if not pending: before["png"] = before_document["png"]
    record = dict(format=staged.FRAME_FORMAT, complete=True, run=data["run"], case_id=spec["case_id"],
                  frame_index=spec["frame_index"], elapsed_h=spec["elapsed_h"], source=spec["source"],
                  configuration=configuration, precision=data["precision"], native=native, native_receipt=native_receipt,
                  phase=phase, statistics=binary("native_statistics.json.gz"), before_rules=before, files=files)
    frame = _write(root / "phase1_commit.json", record)
    completion = dict(format=staged.COMPLETION_FORMAT, complete=True, run=data["run"], configuration=configuration,
                      precision=data["precision"], code_commit="1" * 40, ordered_case_ids=data["ordered_case_ids"],
                      frames=[frame], history_complete=True)
    bound = _write(tmp_path / "phase1_completed.json", completion)
    calibration = _write(tmp_path / "new_fitted_calibration.json", dict(parameters=dict(candidate="N1", gain=1.7, beta=.4)))
    return dict(configuration=configuration, completion=bound, document=completion,
                frame=frame, record=record, calibration=calibration, files=files)


def _rewrite_metadata(bound, value):
    path = Path(bound["path"])
    path.write_bytes((json.dumps(value, allow_nan=False) + "\n").encode())
    return asset(path)


def _rewrite_frame(closure, record):
    frame = _rewrite_metadata(closure["frame"], record)
    completion = deepcopy(closure["document"]); completion["frames"] = [frame]
    return _rewrite_metadata(closure["completion"], completion)


def test_phase1_validation_is_readonly_and_preserves_all_metadata(phase1_data):
    before = deepcopy(phase1_data)
    assert staged.validate_phase1_run(phase1_data) is phase1_data
    assert phase1_data == before


@pytest.mark.parametrize("bad", ["tuple", "non_string_key"])
def test_sealing_never_coerces_non_json_metadata(phase1_data, bad):
    if bad == "tuple": phase1_data["provenance"]["unsupported"] = (1, 2)
    else: phase1_data["provenance"][7] = "would-be-coerced"
    with pytest.raises(ValueError, match="JSON"):
        staged.validate_phase1_run(phase1_data)


@pytest.mark.parametrize("bad", ["duplicate", "logical_sha", "logical_size", "physical_conflict", "unknown_role", "missing_size"])
def test_alias_roles_remain_exact_and_unambiguous(phase1_data, bad):
    logical, physical = _bound("logical.json"), _bound("physical.json")
    aliases = [dict(logical=logical, physical=physical)]
    if bad == "duplicate": aliases.append(deepcopy(aliases[0]))
    elif bad == "logical_sha": logical["sha256"] = "b" * 64
    elif bad == "logical_size": logical["size_bytes"] = 2
    elif bad == "physical_conflict":
        second = deepcopy(aliases[0]); second["logical"]["path"] = "//unit-only/share/second.json"
        second["logical"]["sha256"] = second["physical"]["sha256"] = "b" * 64
        aliases.append(second)
    elif bad == "unknown_role": aliases[0]["fallback"] = physical
    else: physical.pop("size_bytes")
    phase1_data["data_aliases"] = aliases
    with pytest.raises(ValueError): staged.validate_phase1_run(phase1_data)


@pytest.mark.parametrize("path", [(), ("native",), ("phase",), ("statistics",), ("statistics", "summary"),
                                   ("frames", 0), ("frames", 0, "native_task"), ("frames", 0, "phase_roles"),
                                   ("frames", 0, "mass_spec")])
def test_unknown_stage_roles_fail(phase1_data, path):
    value = phase1_data
    for key in path: value = value[key]
    value["unknown_role"] = _bound("never-read.bin")
    with pytest.raises(ValueError, match="unknown"):
        staged.validate_phase1_run(phase1_data)


@pytest.mark.parametrize("field", ["calibration", "anchor"])
def test_phase1_template_has_no_new_fitted_mass_parameters(phase1_data, field):
    phase1_data["statistics"][field] = _bound("fitted-mass-must-not-be-opened.json")
    with pytest.raises(ValueError, match="unknown"):
        staged.validate_phase1_run(phase1_data)


@pytest.mark.parametrize("field", ["baseline_gain", "intervals", "max_column_distance_m", "max_depth_m", "max_porosity"])
def test_historical_summary_definitions_are_required(phase1_data, field):
    phase1_data["statistics"]["summary"].pop(field)
    with pytest.raises(ValueError, match="missing"):
        staged.validate_phase1_run(phase1_data)


@pytest.mark.parametrize("field", ["round4_spec_sha256", "amendment_sha256"])
def test_method_bindings_have_no_inferred_fallback(phase1_data, field):
    phase1_data["statistics"]["bindings"].pop(field)
    with pytest.raises(ValueError, match="SHA-256"):
        staged.validate_phase1_run(phase1_data)


@pytest.mark.parametrize("bad", ["case", "source", "time", "predecessor", "selected", "order", "source_index_type"])
def test_selection_identity_stops_before_io(phase1_data, bad):
    second = deepcopy(phase1_data["frames"][0]); second.update(case_id="case-two", frame_index=2, elapsed_h=2.)
    second["source"].update(work_index=2, elapsed_h=2., source_sha256="f" * 64, previous_source_sha256="c" * 64)
    if bad == "case": second["case_id"] = "case-one"
    elif bad == "source": second["source"]["source_sha256"] = "c" * 64
    elif bad == "time": second["elapsed_h"] = second["source"]["elapsed_h"] = .5
    elif bad == "predecessor": second["source"]["previous_source_sha256"] = "a" * 64
    elif bad == "selected": second["source"]["selected"] = False
    elif bad == "source_index_type": second["source"]["work_index"] = 2.0
    phase1_data["frames"].append(second); phase1_data["ordered_case_ids"] = ["case-one", "case-two"]
    if bad == "order": phase1_data["ordered_case_ids"].reverse()
    with pytest.raises(ValueError): staged.validate_phase1_run(phase1_data)


def test_seal_new_phase1_file_and_refuse_overwrite(phase1_data, tmp_path):
    path = tmp_path / "sealed.json"
    bound = staged.seal_phase1_configuration(path, phase1_data)
    before = path.read_bytes()
    assert bound == asset(path) and json.loads(before) == phase1_data
    with pytest.raises((FileExistsError, OSError)):
        staged.seal_phase1_configuration(path, phase1_data)
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".*.partial")), "the failed writer's own partial must remain"


@pytest.mark.parametrize("raw", [b'{"format":1,"format":2}', b'{"value":NaN}', b'{"value":Infinity}'])
def test_original_strict_reader_rejects_duplicate_and_nonfinite_metadata(raw, tmp_path):
    bound = _write(tmp_path / "malformed.json", raw=raw)
    with pytest.raises(ValueError):
        staged.verify_phase1_completion(bound, bound, expected_commit="1" * 40, reader=DataReader())


def test_metadata_only_reader_cannot_claim_byte_verification():
    class MetadataOnlyReader:
        def json(self, bound): raise AssertionError("an unverified reader was called")
    with pytest.raises(TypeError, match="original content-verifying DataReader"):
        staged.verify_phase1_completion(_bound("completion.json"), _bound("config.json"),
                                        expected_commit="1" * 40, reader=MetadataOnlyReader())


@pytest.mark.parametrize("pending", [False, True])
def test_phase2_seals_actual_bound_sources_without_native_or_mass_calls(phase1_data, tmp_path, monkeypatch, pending):
    import numpy as np
    from darsia.fluidflower.pipeline import series
    from darsia.fluidflower.phase import infer
    from darsia.fluidflower.mass import n1_pipeline
    def forbidden(*args, **kwargs): raise AssertionError("scientific producer called by DATA serializer")
    monkeypatch.setattr(series.SeriesInputs, "native", forbidden)
    monkeypatch.setattr(infer, "infer_labels", forbidden)
    monkeypatch.setattr(n1_pipeline.N1Pipeline, "evaluate", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    closure = _closure(tmp_path, phase1_data, pending=pending)
    original_bytes = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    target = tmp_path / "sealed_mass.json"
    anchor = dict(elapsed_h=2.25, affected_pore_volume_m3=.0001)
    result = staged.seal_phase2_configuration(target, closure["completion"], closure["configuration"],
        expected_phase1_commit="1" * 40, fitted_calibration=closure["calibration"], fitted_anchor=anchor, reader=DataReader())
    config = json.loads(target.read_bytes())
    assert config["format"] == "darsia-n1-run-v1" and config["calibration"] == closure["calibration"]
    assert config["anchor"] == anchor and "storage" not in config
    frame = config["frames"][0]; produced = closure["record"]
    assert frame["current"] == produced["native"]["full"]
    for role in ("labels", "classes", "area"): assert frame[role] == produced["phase"][role]
    assert frame["baseline"] == phase1_data["phase"]["reference"]["full"]
    assert config["phase1"]["frames"]["case-one"]["before_rules"] == produced["before_rules"]
    assert config["bindings"] == phase1_data["statistics"]["bindings"]
    assert config["phase1"]["configuration"] == closure["configuration"]
    assert result["execution_ready"] is False and result["job_created"] is False
    assert result["queue_ownership_or_retirement_verified"] is False
    assert result["native_or_phase_calls"] == result["mass_evaluation_calls"] == 0
    assert all(p.read_bytes() == raw for p, raw in original_bytes.items())


def test_phase2_binds_only_explicit_storage_data(phase1_data, tmp_path):
    closure = _closure(tmp_path, phase1_data)
    storage = dict(format="darsia-storage-output-request-v1", resolution_mm=1., gas_score_encoding="uint8-la-q255-v1", extent_m=[0., 1., 0., 1.])
    target = tmp_path / "mass_with_storage.json"
    staged.seal_phase2_configuration(target, closure["completion"], closure["configuration"], expected_phase1_commit="1" * 40,
        fitted_calibration=closure["calibration"], fitted_anchor=dict(elapsed_h=2.25, affected_pore_volume_m3=.0001),
        storage_data=storage, reader=DataReader())
    assert json.loads(target.read_bytes())["storage"] == storage


def test_tampered_stage_payload_stops_without_sealing(phase1_data, tmp_path):
    closure = _closure(tmp_path, phase1_data)
    Path(closure["record"]["phase"]["labels"]["path"]).write_bytes(b"changed")
    target = tmp_path / "must_not_exist.json"
    with pytest.raises(ValueError):
        staged.seal_phase2_configuration(target, closure["completion"], closure["configuration"], expected_phase1_commit="1" * 40,
            fitted_calibration=closure["calibration"], fitted_anchor=dict(elapsed_h=2.25, affected_pore_volume_m3=.0001), reader=DataReader())
    assert not target.exists()


@pytest.mark.parametrize("bad", ["unknown_role", "case", "precision", "outside_inventory", "before_encoding",
                                "before_source", "before_validity", "before_box", "before_shape", "before_unknown", "phase_area_key"])
def test_frame_closure_mismatch_is_not_repaired(phase1_data, tmp_path, bad):
    closure = _closure(tmp_path, phase1_data); record = deepcopy(closure["record"])
    if bad == "unknown_role": record["new_role"] = {}
    elif bad == "case": record["case_id"] = "case-other"
    elif bad == "precision": record["precision"] = "float64"
    elif bad == "outside_inventory":
        item = record["files"]["phase_labels.png"]; item["path"] = str(tmp_path / "phase_labels.png")
    elif bad == "phase_area_key": record["phase"]["area"]["array_key"] = "unrelated_array"
    elif bad.startswith("before_"):
        before = json.loads(Path(record["before_rules"]["receipt"]["path"]).read_bytes())
        evidence = before["before_rules"]
        if bad == "before_encoding": evidence["rescaled"] = True
        elif bad == "before_source": evidence["source_array"] = "final labels == 2"
        elif bad == "before_validity": evidence["validity_array"] = "post-rule domain"
        elif bad == "before_box": evidence["source_box_xywh"] = [0, 0, 3, 2]
        elif bad == "before_shape": evidence["target_shape_hw"] = [2, 3]
        else: evidence["undeclared_source"] = "not an original encoder field"
        bound = _rewrite_metadata(record["before_rules"]["receipt"], before)
        record["before_rules"]["receipt"] = bound; record["files"]["phase_before.json"] = bound
    completion = _rewrite_frame(closure, record)
    with pytest.raises(ValueError):
        staged.verify_phase1_completion(completion, closure["configuration"], expected_commit="1" * 40, reader=DataReader())


@pytest.mark.parametrize("anchor", [{}, {"elapsed_h": 2.25, "affected_pore_volume_m3": 0},
                                     {"elapsed_h": 2.25, "affected_pore_volume_m3": .1, "guessed": True}])
def test_fitted_anchor_is_explicit_and_not_inferred(phase1_data, tmp_path, anchor):
    closure = _closure(tmp_path, phase1_data)
    with pytest.raises(ValueError):
        staged.seal_phase2_configuration(tmp_path / "no.json", closure["completion"], closure["configuration"], expected_phase1_commit="1" * 40,
            fitted_calibration=closure["calibration"], fitted_anchor=anchor, reader=DataReader())


@pytest.mark.parametrize("field", ["quantization", "missing", "coarse_reduction"])
@pytest.mark.parametrize("replacement", ["different definition", None])
def test_before_receipt_retains_literal_original_encoding_contract(phase1_data, tmp_path, field, replacement):
    closure = _closure(tmp_path, phase1_data)
    record = deepcopy(closure["record"])
    before = json.loads(Path(record["before_rules"]["receipt"]["path"]).read_bytes())
    before["before_rules"][field] = replacement
    bound = _rewrite_metadata(record["before_rules"]["receipt"], before)
    record["before_rules"]["receipt"] = bound
    record["files"]["phase_before.json"] = bound
    completion = _rewrite_frame(closure, record)
    with pytest.raises(ValueError, match="original Q28 probability encoding"):
        staged.verify_phase1_completion(completion, closure["configuration"],
            expected_commit="1" * 40, reader=DataReader())


@pytest.mark.parametrize("bad", ["missing", "none", "shape", "bool_value"])
def test_clearwater_requires_original_explicit_matrix_before_sealing(phase1_data, tmp_path, bad):
    clear = phase1_data["statistics"]["shared"]["clearwater"]
    if bad == "missing": clear.pop("H_reference_to_grid")
    elif bad == "none": clear["H_reference_to_grid"] = None
    elif bad == "shape": clear["H_reference_to_grid"] = [[1., 0.], [0., 1.]]
    else: clear["H_reference_to_grid"][0][0] = True
    target = tmp_path / "must_not_seal.json"
    with pytest.raises(ValueError, match="clear-water"):
        staged.seal_phase1_configuration(target, phase1_data)
    assert not target.exists()


@pytest.mark.parametrize("bad", ["index_bool", "index_float", "elapsed_bool", "elapsed_int",
                                "source_bool", "source_index_bool", "source_elapsed_int",
                                "source_nested_bool", "configuration_size_bool"])
def test_frame_identity_preserves_exact_original_JSON_types(phase1_data, tmp_path, bad):
    closure = _closure(tmp_path, phase1_data)
    record = deepcopy(closure["record"])
    if bad == "index_bool": record["frame_index"] = True
    elif bad == "index_float": record["frame_index"] = 1.
    elif bad == "elapsed_bool": record["elapsed_h"] = True
    elif bad == "elapsed_int": record["elapsed_h"] = 1
    elif bad == "source_bool": record["source"]["studio_anchor"] = 0
    elif bad == "source_index_bool": record["source"]["work_index"] = True
    elif bad == "source_elapsed_int": record["source"]["elapsed_h"] = 1
    elif bad == "source_nested_bool": record["source"]["original_unmapped_metadata"]["keep_exactly"][0] = True
    else: record["configuration"]["size_bytes"] = float(record["configuration"]["size_bytes"])
    # The former Python equality guard considered every one of these equal.
    spec = phase1_data["frames"][0]
    assert record["frame_index"] == spec["frame_index"]
    assert record["elapsed_h"] == spec["elapsed_h"]
    assert record["source"] == spec["source"]
    assert record["configuration"] == closure["configuration"]
    completion = _rewrite_frame(closure, record)
    with pytest.raises(ValueError, match="identity differs"):
        staged.verify_phase1_completion(completion, closure["configuration"],
            expected_commit="1" * 40, reader=DataReader())


@pytest.mark.parametrize("where", ["before", "history"])
def test_all_saved_source_receipts_preserve_nested_JSON_types(phase1_data, tmp_path, where):
    closure = _closure(tmp_path, phase1_data)
    record = deepcopy(closure["record"])
    bound = record["before_rules"]["receipt"] if where == "before" else record["phase"]["history_input"]
    document = json.loads(Path(bound["path"]).read_bytes())
    document["source"]["original_unmapped_metadata"]["keep_exactly"][0] = True
    assert document["source"] == phase1_data["frames"][0]["source"]
    updated = _rewrite_metadata(bound, document)
    if where == "before":
        record["before_rules"]["receipt"] = updated
        record["files"]["phase_before.json"] = updated
    else:
        record["phase"]["history_input"] = updated
        record["files"]["history_input.json"] = updated
    completion = _rewrite_frame(closure, record)
    with pytest.raises(ValueError, match="BEFORE receipt/source/state differs|history is not its actual native acquisition"):
        staged.verify_phase1_completion(completion, closure["configuration"],
            expected_commit="1" * 40, reader=DataReader())


def test_historical_summary_anchor_may_be_omitted_without_materializing_default(phase1_data):
    phase1_data["statistics"]["summary"].pop("anchor_target_h")
    before = deepcopy(phase1_data)
    assert staged.validate_phase1_run(phase1_data) is phase1_data
    assert phase1_data == before
    assert "anchor_target_h" not in phase1_data["statistics"]["summary"]


@pytest.mark.parametrize("anchor", [1.75, 3])
def test_present_historical_summary_anchor_preserves_exact_value_and_type(phase1_data, anchor):
    phase1_data["statistics"]["summary"]["anchor_target_h"] = anchor
    before = deepcopy(phase1_data)
    assert staged.validate_phase1_run(phase1_data) is phase1_data
    actual = phase1_data["statistics"]["summary"]["anchor_target_h"]
    assert type(actual) is type(anchor)
    assert actual == anchor
    assert phase1_data == before


@pytest.mark.parametrize("anchor", [None, True, 0, -1, float("nan"), float("inf"), "2.25"])
def test_present_historical_summary_anchor_keeps_all_original_rejections(phase1_data, anchor):
    phase1_data["statistics"]["summary"]["anchor_target_h"] = anchor
    with pytest.raises(ValueError):
        staged.validate_phase1_run(phase1_data)
