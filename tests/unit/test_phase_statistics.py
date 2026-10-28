"""Private helper-only tests: tiny arrays/metadata, never a worker or source frame."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower.pipeline import phase_statistics as stage
from darsia.fluidflower.pipeline import staged_data as data
from darsia.fluidflower import storage_outputs
from .test_staged_data import phase1_data, _bound


def _capture():
    return dict(g3_pixel=np.array([[0., .25], [.5, 1.]], np.float32),
                flatfield_valid=np.array([[True, False], [True, True]]),
                source_box_xywh=[0, 0, 2, 2], probability_encoding=None)


def test_before_writer_copies_exact_existing_encoder_result(tmp_path):
    capture = _capture()
    expected, evidence = storage_outputs.encode_before(capture, (2, 2))
    spec = dict(source=dict(source_sha256="a" * 64))
    actual = stage._save_before(tmp_path, spec, capture, [2, 2])
    receipt = json.loads(Path(actual["receipt"]["path"]).read_bytes())
    with Image.open(actual["png"]["path"]) as image:
        assert image.mode == "LA" and np.array_equal(np.asarray(image), expected)
    assert receipt["before_rules"] == evidence and receipt["source"] == spec["source"]
    assert actual["state"] == "complete"


@pytest.mark.parametrize("case", ["missing", "outside", "valid_inf"])
def test_original_pending_BEFORE_is_saved_without_a_fabricated_PNG(tmp_path, case):
    capture = _capture()
    if case == "missing": capture = {}
    elif case == "outside": capture["g3_pixel"][0, 0] = 1.25
    else: capture["g3_pixel"][0, 0] = np.inf
    result = stage._save_before(tmp_path, dict(source={"source_sha256": "a" * 64}), capture, [2, 2])
    assert result["state"] == "pending" and "png" not in result
    assert not list(tmp_path.glob("*.png"))
    evidence = json.loads(Path(result["receipt"]["path"]).read_bytes())["before_rules"]
    assert evidence["reason"] in {"native_centre_not_computed", "pre_rule_score_outside_probability_contract"}


def test_missing_explicit_pre_light_photograph_remains_pending_without_reader(tmp_path):
    class MustNotRead:
        def path(self, *args): raise AssertionError("Pending photograph read an input")
    source = {"source_sha256": "a" * 64}
    result = stage._save_original_photo(tmp_path, {}, {"source": source}, None, MustNotRead())
    assert result["state"] == "pending" and "png" not in result
    receipt = json.loads(Path(result["receipt"]["path"]).read_bytes())
    assert receipt["source"] == source and receipt["reason"] == "explicit_saved_pre_light_photograph_not_bound"
    assert not list(tmp_path.glob("*.png"))


def test_statistics_serialization_keeps_exact_original_scalar_values(tmp_path):
    statistics = dict(original_histogram=[1, 3, 7], measured=1.125, original_float32=np.float32(.25),
                      anchor_target_h=2.25, baseline_gain=1.1, beta=0, U_rest=None)
    original = deepcopy(statistics)
    result = stage._save_statistics(tmp_path, dict(statistics=statistics, sandmap={"source": "original report"}))
    with gzip.open(result["path"], "rt", encoding="utf8") as stream:
        actual = json.load(stream)
    assert actual == original and type(actual["measured"]) is float
    assert statistics == original
    assert json.loads((tmp_path / "statistics_geometry_report.json").read_bytes()) == {"source": "original report"}


def test_existing_phase1_commit_is_never_rewritten_or_replayed(tmp_path):
    commit = tmp_path / "phase1_commit.json"
    commit.write_bytes(b"previous immutable commit")
    with pytest.raises(FileExistsError, match="no replay"):
        stage._commit_phase1(tmp_path, {}, task={}, configuration={}, native={}, phase={},
                             statistics={}, before={}, photograph={})
    assert commit.read_bytes() == b"previous immutable commit"


@pytest.mark.parametrize("case", ["legacy", "wrong_pipeline", "precision", "resume", "missing_binding"])
def test_phase1_requires_explicit_actual_owned_stage_context_before_source_IO(case, tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import run_io
    task = dict(pipeline=stage.PIPELINE, format="darsia-alias-task-v1", precision="float32",
                commit="1" * 40, phase1_configurations_sha256="2" * 64, threads=1)
    if case == "legacy": task.pop("format")
    elif case == "wrong_pipeline": task["pipeline"] = "native-phase-n1"
    elif case == "precision": task["precision"] = "float64"
    elif case == "resume": task["resume_from"] = {"attempt_directory": "must-not-be-opened"}
    def binding(_task):
        if case != "missing_binding": raise AssertionError("Invalid stage touched worker context")
        return None
    monkeypatch.setattr(run_io, "for_task", binding)
    with pytest.raises(ValueError): stage._task_configuration(task, tmp_path, "cpu", "float32")


@pytest.mark.parametrize("field", ["enclosed_hole_policy", "display_mode", "isolated_invalid_policy"])
def test_source_processor_never_infers_required_original_policy(phase1_data, field):
    phase1_data["native"]["source_certificate_fields"] = dict(renderer="renderer_certificate", photometry="v4r5_native_info")
    task = phase1_data["frames"][0]["native_task"]
    task["reference_to_frame_h"] = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    task.update(kind="source", processor=dict(source_row=_bound("source.json"), calibration_manifest=_bound("photometry.json"),
        photo_aggregate=_bound("photos.json"), enclosed_hole_policy="keep", display_mode="raw_log",
        isolated_invalid_policy="keep"), arguments=dict({key: _bound(key) for key in ("camera_jpeg", "camera_arw", "review_jpeg")},
            run="run-one", frame_id="frame_17", source_manifest_sha256="e" * 64,
            content_shape_hw=[2, 2], content_box_xywh=[0, 0, 2, 2]))
    task["processor"].pop(field)
    with pytest.raises(ValueError, match="missing"): data.validate_phase1_run(phase1_data)


def test_complete_source_processor_policies_are_retained_without_execution(phase1_data):
    phase1_data["native"]["source_certificate_fields"] = dict(renderer="renderer_certificate", photometry="v4r5_native_info")
    task = phase1_data["frames"][0]["native_task"]
    task["reference_to_frame_h"] = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    task.update(kind="source", processor=dict(source_row=_bound("source.json"), calibration_manifest=_bound("photometry.json"),
        photo_aggregate=_bound("photos.json"), enclosed_hole_policy="keep", display_mode="raw_log",
        isolated_invalid_policy="keep"), arguments=dict({key: _bound(key) for key in ("camera_jpeg", "camera_arw", "review_jpeg")},
            run="run-one", frame_id="frame_17", source_manifest_sha256="e" * 64,
            content_shape_hw=[2, 2], content_box_xywh=[0, 0, 2, 2]))
    before = deepcopy(phase1_data)
    assert data.validate_phase1_run(phase1_data) is phase1_data
    assert phase1_data == before


@pytest.mark.parametrize("case", ["unknown_case", "missing_bound", "unknown_role", "producer_sha"])
def test_optional_photo_requires_exact_explicit_closed_roles(phase1_data, case):
    request = {key: _bound(key) for key in ("native_geometry", "geometry_certificate", "current_geometry", "saved_native_review")}
    request["geometry_producer_sha256"] = "f" * 64
    phase1_data["photographs"] = {"case-one": request}
    if case == "unknown_case": phase1_data["photographs"]["other"] = request
    elif case == "missing_bound": request.pop("saved_native_review")
    elif case == "unknown_role": request["fallback"] = _bound("never-use")
    else: request["geometry_producer_sha256"] = "inferred"
    with pytest.raises(ValueError): data.validate_phase1_run(phase1_data)


def test_explicit_photo_bindings_are_preserved_as_DATA_only(phase1_data):
    request = {key: _bound(key) for key in ("native_geometry", "geometry_certificate", "current_geometry", "saved_native_review")}
    request["geometry_producer_sha256"] = "f" * 64
    phase1_data["photographs"] = {"case-one": request}
    before = deepcopy(phase1_data)
    data.validate_phase1_run(phase1_data)
    assert phase1_data == before
