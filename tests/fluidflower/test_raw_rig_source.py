"""RAW source closure and unchanged frozen photometry on local fixtures."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower.native import photometry, raw_positioning, raw_rig_source as raw
from darsia.fluidflower.pipeline import job_inputs, phase_inputs


def descriptor(path):
    return dict(path=str(path), sha256=sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    shape, grid_shape = (16, 24), (4, 5)
    photo = json.loads((Path(__file__).parent / "fixtures/native_photometry_parameters.json").read_bytes())
    photo.update(native_shape_chw=[3, *shape], area_grid_shape_chw=[3, 2, 3])
    grid = photometry.ReferenceGrid(grid_shape, (-1, 0), (2, 3), parameters=photo)
    rng = np.random.default_rng(73)
    yy, xx = np.mgrid[:grid_shape[0], :grid_shape[1]]
    fit = photometry.Baseline("fixture", np.eye(3), np.eye(3), np.zeros(14),
        {1: .2}, {1: 20}, np.ones(grid_shape, np.int16), np.ones(grid_shape, bool),
        np.zeros(grid_shape, bool), np.zeros(3), xx.astype(np.float32), yy.astype(np.float32),
        np.ones(grid_shape, np.float32), rng.normal(6, .05, (3, *grid_shape)).astype(np.float32),
        np.zeros(grid_shape, np.float32), np.zeros((24, 2), np.float32),
        np.ones((3, 24), np.float32) * 6, None, .1)
    acquisition = tmp_path / "B.ARW"
    acquisition.write_bytes(b"bound-RAW-acquisition")
    raw_bound = descriptor(acquisition)
    models = {}
    for name, model in (("fit", fit), ("reference", replace(fit, run="donor_consensus"))):
        path = tmp_path / (name + ".npz")
        photometry.save_baseline(path, model, grid, dict(source_arw_sha256=raw_bound["sha256"],
            source_review_sha256="1" * 64, source_manifest_sha256="2" * 64,
            sand_mask_sha256="3" * 64), parameters=photo)
        models[name] = dict(data=descriptor(path), sidecar=descriptor(path.with_suffix(".json")))
    models["old_field"] = None
    shape_file = tmp_path / "existing_shape.npz"
    np.savez(shape_file, unchanged=np.ones(1))
    parameters = dict(source=dict(source_id_pattern="fixture", profiles=dict(fixture="camera"),
        native_shape_hw=list(shape), asset_aliases={}, homography_min_scale=1e-12,
        homography_match_atol=1e-7, symbols=dict(reference_to_frame_h_key="H_reference_to_frame")),
        registration={}, photometry=photo)
    parameters["source"]["raw_execution_identity"] = raw.execution_identity()
    binding = dict(contract=raw.CONTRACT, run="fixture", profile_id="camera", native_shape_hw=list(shape),
        baseline_raw=raw_bound, baseline_reference_to_native_h=np.eye(3).tolist(),
        shape_assets=[descriptor(shape_file)], image_metadata={}, models=models,
        execution_identity=raw.execution_identity())
    binding_path = tmp_path / "raw_binding.json"
    binding_path.write_text(json.dumps(binding))
    linear = rng.uniform(80, 1100, (3, *shape)).astype(np.float32)
    def sensor(path, *, expected_sha256, parameters, profile_id):
        assert Path(path).suffix == ".ARW" and descriptor(Path(path))["sha256"] == expected_sha256
        return raw_positioning.Sensor(np.zeros((16, 24, 3), np.uint16), descriptor(Path(path)),
            dict(white_level=16383, black_level=1008, raw_visible_wh=[24, 16]),
            (0,), (0,), (0,), profile_id)
    monkeypatch.setattr(raw_positioning, "read_sensor", sensor)
    monkeypatch.setattr(raw_positioning, "load_shape_chain", lambda assets: [object()])
    monkeypatch.setattr(raw_positioning, "camera_u", lambda *a, **k: linear.copy())
    monkeypatch.setattr(raw_positioning, "render_native", lambda *a, **k: (linear.copy(),
        dict(native_to_camera_u_h=np.eye(3).tolist(), camera_u_wh=[24, 16])))
    monkeypatch.setattr(Image, "open", lambda *a, **k: pytest.fail("Companion photograph opened"))
    row = dict(contract=raw.FRAME_CONTRACT, run="fixture", acquisition=raw_bound,
        reference_to_native_h=np.eye(3).tolist(), baseline_to_native_h=np.eye(3).tolist())
    return SimpleNamespace(root=tmp_path, parameters=parameters, binding=binding, path=binding_path,
        row=row, linear=linear, fit=fit, grid=grid)


def test_raw_B_uses_the_original_correction_bits_without_companions_or_parent_pixels(fixture):
    f = fixture
    processor = raw.RawRigSourceProcessor(f.path, parameters=f.parameters)
    actual, audit = processor.correct(f.row)
    p = processor.processor
    expected, info = photometry.correct_frame_native(f.linear, p.fit, p.reference, p.grid,
        np.eye(3), np.ones((16, 24), bool), exact_baseline_source=True,
        frame_chart_xy8=None, frame_chart_usable=None, old_field_ref8=None,
        parameters=f.parameters["photometry"])
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert audit["photometry"] == info and audit["exact_B"]
    assert audit["sources"]["camera_arw"]["sha256"] == f.row["acquisition"]["sha256"]
    assert audit["parent_cache"] is None and audit["jpeg_inputs"] is False
    assert audit["processor"] == processor.identity
    assert not list(f.root.glob("*.jpg"))
    with pytest.raises(ValueError, match="cannot be reused"):
        processor.correct(f.row, parent_reference=True)


@pytest.mark.parametrize("change", ["raw", "fit", "sidecar", "shape", "implementation"])
def test_source_refuses_changed_inputs_even_when_the_model_pair_is_internally_consistent(fixture, change):
    f = fixture
    if change == "implementation":
        f.binding["execution_identity"]["raw_source"] = "0" * 64
        f.path.write_text(json.dumps(f.binding))
    else:
        item = {"raw": f.binding["baseline_raw"], "fit": f.binding["models"]["fit"]["data"],
                "sidecar": f.binding["models"]["fit"]["sidecar"], "shape": f.binding["shape_assets"][0]}[change]
        with Path(item["path"]).open("ab") as stream:
            stream.write(b" ")
    with pytest.raises(ValueError, match="binding|identity"):
        raw.RawRigSourceProcessor(f.path, parameters=f.parameters)


def test_frame_geometry_cannot_change_the_frozen_reference_placement(fixture):
    f = fixture
    row = deepcopy(f.row)
    row["reference_to_native_h"][0][2] = 1
    with pytest.raises(ValueError, match="frozen reference"):
        raw.validate_frame(row, f.binding, f.parameters)
    row = deepcopy(f.row)
    row["camera_jpeg_path"] = "missing.jpg"
    with pytest.raises(ValueError, match="RAW-only fields"):
        raw.validate_frame(row, f.binding, f.parameters)


def test_model_sidecar_change_during_loading_is_rejected_before_rendering(fixture, monkeypatch):
    f = fixture
    original = photometry.load_baseline
    def changed(path, **kwargs):
        result = original(path, **kwargs)
        with Path(path).with_suffix(".json").open("ab") as stream:
            stream.write(b" ")
        return result
    monkeypatch.setattr(photometry, "load_baseline", changed)
    with pytest.raises(ValueError, match="binding"):
        raw.RawRigSourceProcessor(f.path, parameters=f.parameters)


def test_catalogue_contains_only_consumed_raw_and_unchanged_model_roles(fixture):
    f = fixture
    catalog = job_inputs.InputCatalog({"scratch": str(f.root)})
    task = dict(kind="raw_rig", processor=dict(raw_binding=descriptor(f.path)))
    catalog.native_task(task, f.row, f.parameters, "frame.native")
    relative = {row["reference"]["relative_path"] for row in catalog.entries.values()}
    assert relative == {"raw_binding.json", "B.ARW", "fit.npz", "fit.json",
                        "reference.npz", "reference.json", "existing_shape.npz"}
    assert not any(path.lower().endswith((".jpg", ".jpeg")) for path in relative)


def test_RAW_source_identity_matches_the_actual_admitted_Git_blobs():
    from darsia.fluidflower.pipeline.run_io import RAW_CODE_ROLES
    root = Path(__file__).resolve().parents[2]
    identity = raw.execution_identity()
    assert set(identity) == set(RAW_CODE_ROLES)
    for name, relative in RAW_CODE_ROLES.items():
        blob = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=root)
        assert identity[name] == sha256(blob).hexdigest()


def test_queued_RAW_binding_cannot_omit_or_change_current_code_roles(fixture):
    f = fixture
    parameters = deepcopy(f.parameters)
    parameters["source"].pop("raw_execution_identity")
    catalog = job_inputs.InputCatalog({"scratch": str(f.root)})
    with pytest.raises(ValueError, match="explicit native execution binding"):
        catalog.native_task(dict(kind="raw_rig", processor=dict(raw_binding=descriptor(f.path))),
                            f.row, parameters, "frame.native")


def test_new_RAW_baseline_history_rejects_relabelled_old_producer(fixture):
    f = fixture
    raw_sha = f.binding["baseline_raw"]["sha256"]
    _, audit = raw.RawRigSourceProcessor(f.path, parameters=f.parameters).correct(f.row)
    bound = dict(path="raw_audit.json", sha256="9" * 64)
    row = dict(contract=phase_inputs.RAW_ACQUISITION_CONTRACT, source_kind="raw",
        raw_acquisition=f.binding["baseline_raw"], raw_producer_audit=bound, source_sha256=raw_sha)
    record = dict(source_sha256=raw_sha, full=dict(source_sha256=raw_sha,
        binding=dict(producer_audit=bound)))
    parameters = dict(source_sha_field="source_sha256")
    phase_inputs.verify_raw_acquisition(record, row, audit, parameters=parameters)
    audit["jpeg_inputs"] = True
    with pytest.raises(ValueError, match="actual RAW producer"):
        phase_inputs.verify_raw_acquisition(record, row, audit, parameters=parameters)
