"""Original-method DATA protocol tests, not source processing or model inference.

Native/frame consumers run only against explicit tiny unit protocol objects.
The native test deliberately stops before its first output write. Its synthetic
corrector checks the ORIGINAL SourceProcessor.correct Python signature and never
calls the numerical source producer. No fake Main/phase1 completion is reported.
"""
from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.pipeline import series, staged_data
from darsia.fluidflower.native.source import SourceProcessor
from .test_staged_data import phase1_data, _bound


@pytest.fixture
def source_phase1_data(phase1_data):
    """Every role is visible DATA; no autouse injection or inferred index/value."""
    value = deepcopy(phase1_data)
    value["native"]["source_certificate_fields"] = dict(renderer="renderer_certificate", photometry="v4r5_native_info")
    task = value["frames"][0]["native_task"]
    task.update(kind="source", reference_to_frame_h=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        processor=dict(source_row=_bound("source_row.json"), calibration_manifest=_bound("photometry.json"),
            photo_aggregate=_bound("photos.json"), enclosed_hole_policy="keep", display_mode="raw_log", isolated_invalid_policy="keep"),
        arguments=dict(run=value["run"], frame_id="frame_17", source_manifest_sha256="e" * 64,
            camera_jpeg=_bound("camera.jpg"), camera_arw=_bound("camera.arw"), review_jpeg=_bound("registered.jpg"),
            content_shape_hw=[2, 2], content_box_xywh=[0, 0, 2, 2]))
    # frame_17 is an explicit synthetic camera ID, deliberately different from
    # selected frame_index=1. No real source acquisition is invented or opened.
    return value


@pytest.mark.parametrize("missing", ["selected_revision", "placement_metadata", "inherited_spacing_m",
                                    "probability_spacing_m", "retained_frame_index", "reference_to_frame_h",
                                    "source_certificate_fields"])
def test_actual_consumer_roles_cannot_be_omitted(source_phase1_data, missing):
    frame = source_phase1_data["frames"][0]
    if missing == "selected_revision": frame["phase_roles"]["area"].pop("revision")
    elif missing == "reference_to_frame_h": frame["native_task"].pop(missing)
    elif missing == "source_certificate_fields": source_phase1_data["native"].pop(missing)
    elif missing == "retained_frame_index": frame["phase_roles"]["retained_source_cache"] = {"explicit": "unit bound role"}
    else: frame["phase_roles"].pop(missing)
    with pytest.raises(ValueError): staged_data.validate_phase1_run(source_phase1_data)


@pytest.mark.parametrize("missing", ["run", "frame_id", "source_manifest_sha256", "content_shape_hw", "content_box_xywh"])
def test_source_correct_signature_roles_cannot_be_omitted(source_phase1_data, missing):
    source_phase1_data["frames"][0]["native_task"]["arguments"].pop(missing)
    with pytest.raises(ValueError, match="missing"): staged_data.validate_phase1_run(source_phase1_data)


def test_complete_source_keywords_bind_original_correct_signature(source_phase1_data):
    staged_data.validate_phase1_run(source_phase1_data)
    spec = source_phase1_data["frames"][0]
    arguments = deepcopy(spec["native_task"]["arguments"])
    for key in ("camera_jpeg", "camera_arw", "review_jpeg"):
        arguments[key] = Path(arguments[key]["path"])
    bound = inspect.signature(SourceProcessor.correct).bind(object(), **arguments)
    assert bound.arguments["frame_id"] == "frame_17" and spec["frame_index"] == 1
    assert bound.arguments["source_manifest_sha256"] == "e" * 64


def test_original_native_consumer_reads_complete_source_ABI_before_output_boundary(source_phase1_data):
    staged_data.validate_phase1_run(source_phase1_data)
    spec = source_phase1_data["frames"][0]
    calls = []
    class CorrectorProtocol:
        def correct(self, **arguments):
            inspect.signature(SourceProcessor.correct).bind(self, **arguments)
            calls.append(deepcopy(arguments))
            return SimpleNamespace(log_linear_rgb=np.zeros((3, 2, 2), np.float32),
                certificate=dict(renderer_certificate={"unit": "renderer metadata"}, v4r5_native_info={"unit": "photometry metadata"}))
        def identity(self): return {"unit": "synthetic protocol object, no source producer called"}
    class ReaderProtocol:
        def json(self, _bound): return {}
        def path(self, bound): return Path(bound["path"])
    class ReachedOutputBoundary(Exception): pass
    class NoOutputStage:
        def mkdir(self, **_kwargs): raise ReachedOutputBoundary()
    obj = series.SeriesInputs.__new__(series.SeriesInputs)
    obj.config = source_phase1_data
    obj.reader = ReaderProtocol()
    obj.native_parameters = dict(extension={"reference_to_frame_h_key": "H_ac28_to_frame_native"})
    obj.processors = {json.dumps(spec["native_task"]["processor"], sort_keys=True): CorrectorProtocol()}
    with pytest.raises(ReachedOutputBoundary):
        series.SeriesInputs.native(obj, spec, NoOutputStage())
    assert len(calls) == 1 and calls[0]["frame_id"] == "frame_17"
    assert calls[0]["content_shape_hw"] == [2, 2]
    assert all(isinstance(calls[0][k], Path) for k in ("camera_jpeg", "camera_arw", "review_jpeg"))


def test_original_frame_consumer_retains_area_revision_and_unconditional_roles(phase1_data, monkeypatch):
    staged_data.validate_phase1_run(phase1_data)
    from darsia.fluidflower.pipeline import phase_inputs
    from darsia.fluidflower.geometry import barrier
    spec = phase1_data["frames"][0]
    roles = spec["phase_roles"]
    tiny = np.zeros((3, 2, 2), np.float32)
    valid = np.ones((2, 2), bool)
    masked = SimpleNamespace(log=tiny, valid=valid, provenance={"unit": "native protocol"})
    temporal = SimpleNamespace(provenance={"unit": "temporal protocol"})
    monkeypatch.setattr(phase_inputs, "original_photo_sand", lambda *a, **k: (valid, {"unit": "photo protocol"}))
    monkeypatch.setattr(phase_inputs, "original_native_input", lambda *a, **k: masked)
    monkeypatch.setattr(phase_inputs, "original_temporal_input", lambda *a, **k: temporal)
    monkeypatch.setattr(phase_inputs, "probability_encoding_contract", lambda *a, **k: deepcopy(roles["encoding_contract"]))
    monkeypatch.setattr(barrier, "prepare_static_components", lambda *a, **k: (np.zeros((2, 2), np.uint8), ["original"], {"unit": "static protocol"}))
    class ReaderProtocol:
        def json(self, bound):
            return dict(metadata={"spacing_m": roles["probability_spacing_m"]}, binding={"labelled": False})
        def archive(self, *args): return {"unit": "static inputs not decoded"}
    obj = series.SeriesInputs.__new__(series.SeriesInputs)
    obj.config = phase1_data; obj.reader = ReaderProtocol(); obj.aliases = {}
    obj.parameters = dict(inputs={}, retained={}, geometry=dict(native_shape_hw=[2, 2], source_box_xywh=[0, 0, 2, 2]))
    obj.native_parameters = {"extended_ring": {}}
    obj.phase = {"barrier_parameters": {}}
    obj.previous = dict(source={"source_sha256": spec["source"]["previous_source_sha256"]}, temporal_input=temporal)
    native = dict(log=tiny, valid=valid, full_log=tiny, full_valid=valid,
        context=SimpleNamespace(pixel_area_mm2=np.ones((2, 2)), rig_domain=valid), observed=valid, top=valid,
        record={"full": {"unit": "actual native protocol"}}, record_asset=_bound("native.json"))
    actual = series.SeriesInputs.frame(obj, spec, native)
    assert actual["physical_scale_revision"] == roles["area"]["revision"]
    assert actual["physical_scale_revision"] is not roles["area"]["revision"]
    assert actual["placement_metadata"] == roles["placement_metadata"]
    assert actual["inherited_spacing_m"] == roles["inherited_spacing_m"]
    assert actual["probability_spacing_m"] == roles["probability_spacing_m"]
    assert np.array_equal(actual["pixel_area_mm2"], np.ones((2, 2)))


@pytest.mark.parametrize("role", ["argument", "area", "certificate"])
def test_closed_ABI_additions_do_not_enable_unknown_role_catchall(source_phase1_data, role):
    spec = source_phase1_data["frames"][0]
    if role == "argument": spec["native_task"]["arguments"]["invented_fallback"] = "forbidden"
    elif role == "area": spec["phase_roles"]["area"]["invented_scale"] = .5
    else: source_phase1_data["native"]["source_certificate_fields"]["invented_role"] = "forbidden"
    with pytest.raises(ValueError, match="unknown"): staged_data.validate_phase1_run(source_phase1_data)
