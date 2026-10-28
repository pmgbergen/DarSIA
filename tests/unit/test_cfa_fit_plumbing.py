"""Explicit CFA fit policy reaches reference, baselines and scientific frames."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.native import cfa_clipping_diagnostics as cfa
from darsia.fluidflower.native import full_photometry as photo
from darsia.fluidflower.mass.n1_pipeline import asset
from .test_cfa_clipping_diagnostics import sensor_for
from .test_independent_full_photometry import photo_parameters


def test_no_clipped_photosites_fast_path_uses_existing_candidate_values_only(monkeypatch):
    sensor, parameters = sensor_for(np.full((360, 540), 16382, np.uint16))
    rgb = np.full((3, 360, 540), .9999, np.float32)
    rgb[1, 120:140, 130:150] = np.nan
    monkeypatch.setattr(cfa, "chart_clipping_diagnostics", lambda *a, **k: pytest.fail("dense source reconstruction"))
    valid, evidence = cfa.chart_fit_validity(sensor, parameters,
        [[20, 20], [340, 20], [340, 520], [20, 520]], camera_rgb=rgb)
    assert evidence["original_sensor_clipped_count"] == 0
    assert np.all(valid[..., 0]) and np.all(valid[..., 2]) and not np.all(valid[..., 1])
    assert np.count_nonzero(evidence["clipped_source_affected_pixels"]) == 0
    assert evidence["fit_validity_policy"] == cfa.FIT_VALIDITY_POLICY


def test_any_true_clipping_excludes_whole_candidate_channel_without_rgb_cutoff():
    mosaic = np.full((360, 540), 1008, np.uint16)
    mosaic[120, 131] = 16383  # A single original green photosite.
    sensor, parameters = sensor_for(mosaic)
    valid, evidence = cfa.chart_fit_validity(sensor, parameters,
        [[20, 20], [340, 20], [340, 520], [20, 520]], camera_rgb=np.zeros((3, 360, 540)))
    assert not np.all(valid[..., 1]) and np.all(valid[..., 0]) and np.all(valid[..., 2])
    assert evidence["original_sensor_clipped_count"] == 1
    assert np.max(evidence["clipped_source_affected_pixels"]) < 2500


def test_optional_colour_api_forwards_exact_masks_and_keeps_legacy_kwargs(monkeypatch):
    from darsia.fluidflower import image_correction as numeric
    calls = []
    def fit(image, reference, **kwargs):
        calls.append(kwargs)
        return dict(transform={}, diagnostics={})
    monkeypatch.setattr(numeric, "fit_uniform24_to_reference_swatches", fit)
    monkeypatch.setattr(numeric, "apply_uniform24_array_transform", lambda image, transform: image)
    image, ref, roi = np.full((3, 6, 8), .3), np.full((4, 6, 3), .3), [[0, 0], [5, 0], [5, 7], [0, 7]]
    valid = np.ones((4, 6, 3), bool)
    evidence = dict(source={"exact": "source"}, reference={"exact": "reference"})
    photo._colour_full(image, "uniform24", ref, roi)
    photo._colour_full(image, "uniform24", ref, roi, source_validity=valid,
                       reference_validity=valid, validity_evidence=evidence)
    assert set(calls[0]) == {"roi_rc", "settings"}
    assert calls[1]["source_validity"] is valid and calls[1]["reference_validity"] is valid
    assert calls[1]["validity_evidence"] is evidence


def test_actual_model_roundtrip_caches_reference_and_baseline_but_checks_each_new_frame(
        tmp_path, photo_parameters, monkeypatch):
    from darsia.fluidflower.native.raw_rig_source import RawRigSourceProcessor
    sensor_calls, colour_calls = [], []
    masks = {name: np.ones((4, 6, 3), bool) for name in ("reference", "target", "frame")}
    masks["reference"][0, 0, 0] = False
    masks["target"][3, 0, 1] = False
    masks["frame"][2, 5, 2] = False
    def evidence(sensor, registration, roi, *, camera_rgb):
        sensor_calls.append(sensor.fixture_run)
        valid = masks[sensor.fixture_run]
        return valid, dict(fit_validity_policy=cfa.FIT_VALIDITY_POLICY,
                           source=deepcopy(sensor.source), valid_patch_channels=valid.tolist())
    def colour(unit, method, reference, roi, **kwargs):
        colour_calls.append(kwargs)
        return np.asarray(unit, np.float32), dict(method=method, validity=kwargs["validity_evidence"])
    monkeypatch.setattr(cfa, "chart_fit_validity", evidence)
    monkeypatch.setattr(photo, "_colour_full", colour)
    def baseline(run):
        path = tmp_path / (run + ".synthetic.json")
        path.write_text(json.dumps({"fixture": run}))
        pin = asset(path)
        sensor = SimpleNamespace(source=pin, fixture_run=run,
                                 metadata=dict(white_level=100, black_level=0))
        return dict(run=run, sensor=sensor, rgb_full=np.full((3, 32, 48), 30., np.float32),
            rgb_native=np.full((32, 48, 3), .3, np.float32), white_range_dn=100.,
            native_to_camera_u_h=np.eye(3), chart_rgb=np.full((4, 6, 3), .3, np.float32),
            chart_roi_rc=[[2, 2], [29, 2], [29, 45], [2, 45]], shape_assets=[pin], image_metadata={},
            source_pins=dict(raw_acquisition=pin, camera_binding=dict(profile_id="camera")),
            parameters=dict(photometry=photo_parameters, registration={}))
    one, two = baseline("reference"), baseline("target")
    choice = dict(method="off", color_method="uniform24", reference_run="reference")
    result = photo.prepare_independent_photometry({"reference": one, "target": two}, choice, {},
        tmp_path / "prepared", choice_authority=one["source_pins"]["raw_acquisition"],
        cfa_validity_policy=cfa.FIT_VALIDITY_POLICY)
    assert sensor_calls == ["reference", "target"]
    assert len(colour_calls) == 3
    assert all(np.array_equal(call["reference_validity"], masks["reference"]) for call in colour_calls)
    assert np.array_equal(colour_calls[-1]["source_validity"], masks["target"])
    binding = json.loads(Path(result["runs"]["target"]["source_binding"]["path"]).read_bytes())
    owner = RawRigSourceProcessor.__new__(RawRigSourceProcessor)
    owner.binding = binding
    owner.parameters = dict(source=dict(asset_aliases={}, symbols=dict(reference_to_frame_h_key="h")),
        registration={}, photometry=photo_parameters, support=dict(coverage_tolerance=1e-6))
    prepared = photo.load_independent_model(owner)
    assert prepared["reference_validity"].dtype == bool
    assert np.array_equal(prepared["reference_validity"], masks["reference"])
    # The exact source and persisted array must agree with the sidecar proof;
    # an admissible sidecar object cannot substitute another mask or RAW source.
    from darsia.fluidflower.native import io
    for kind in ("reference", "baseline"):
        for field in ("source", "valid_patch_channels"):
            altered = deepcopy(prepared["record"])
            proof = altered[kind + "_validity_evidence"]
            if field == "source":
                proof["source"]["sha256"] = "f" * 64
            else:
                proof[field][0][0][0] = not proof[field][0][0][0]
            with monkeypatch.context() as local:
                local.setattr(io, "read_bound_json", lambda *a, **k: altered)
                with pytest.raises(ValueError, match="exact reference or baseline"):
                    photo.load_independent_model(owner)
    owner.processor = SimpleNamespace(independent=prepared)
    owner.baseline_sensor, owner.baseline_camera_u, owner.identity = two["sensor"], two["rgb_full"], {}
    owner._render = lambda *a, **k: (None, dict(native_to_camera_u_h=np.eye(3).tolist()))
    monkeypatch.setattr(photo, "_chart", lambda unit, roi=None: (np.ones((4, 6, 3)), two["chart_roi_rc"]))
    photo.correct_independent(owner, two["sensor"], np.eye(3), np.eye(3), np.eye(3))
    assert sensor_calls == ["reference", "target"]  # Reuse certified exact baseline evidence.
    new_frame = SimpleNamespace(fixture_run="frame", source=dict(path="frame.raw", sha256="b" * 64),
                               metadata=dict(white_level=100, black_level=0))
    monkeypatch.setattr(photo.raw_positioning, "camera_u", lambda *a, **k: two["rgb_full"])
    _, audit = photo.correct_independent(owner, new_frame, np.eye(3), np.eye(3), np.eye(3))
    assert sensor_calls == ["reference", "target", "frame"]
    assert np.array_equal(colour_calls[-1]["source_validity"], masks["frame"])
    assert np.array_equal(colour_calls[-1]["reference_validity"], masks["reference"])
    assert audit["photometry"]["color"]["validity"]["source"]["source"]["sha256"] == "b" * 64


def test_gui_preview_uses_same_explicit_policy_and_only_existing_two_sensor_objects(monkeypatch):
    from darsia.fluidflower.pipeline.current_image_preparation import colour_fit_preview_from_operations
    objects = {run: object() for run in ("reference", "target")}
    calls, fitted = [], []
    def evidence(sensor, registration, roi, **kwargs):
        calls.append(sensor)
        return np.ones((4, 6, 3), bool), dict(source={"sensor": "fixture"})
    def collect(context, reader, **kwargs):
        return dict(run=context.run, sensor=objects[context.run], parameters=dict(registration={}),
            rgb_full=np.full((3, 4, 6), .3), white_range_dn=1., chart_rgb=np.full((4, 6, 3), .3),
            chart_roi_rc=[[0, 0], [3, 0], [3, 5], [0, 5]], source_pins={})
    def colour(unit, method, reference, roi, **kwargs):
        fitted.append(kwargs)
        return unit, dict(diagnostics=kwargs["validity_evidence"])
    monkeypatch.setattr(cfa, "chart_fit_validity", evidence)
    monkeypatch.setattr(photo, "_colour_full", colour)
    result = colour_fit_preview_from_operations({"reference": "reference", "target": "target"}, "target",
        dict(method="off", color_method="uniform24", reference_run="reference"),
        context_loader=lambda operation, **kwargs: SimpleNamespace(run=operation), collector=collect,
        cfa_validity_policy=cfa.FIT_VALIDITY_POLICY)
    assert result["accepted"] is True and result["complete"] is False
    assert calls == [objects["reference"], objects["target"]]
    assert len(fitted) == 2 and all(call["source_validity"].dtype == bool for call in fitted)
