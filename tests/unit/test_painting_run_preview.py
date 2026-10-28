"""Unfinished maps can be placed for display without approval or writes."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import geometry_workspace_frame as wf
from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import layer_painting_manual as manual
from darsia.fluidflower import map_placement as placement
from darsia.fluidflower import painting_run_preview as preview
from darsia.fluidflower.series_geometry import RigFrame


SHAPE = (24, 32)
SIZE = (32, 24)
IDENTITY = np.eye(3)


def _descriptor(path, **extra):
    content = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content), **extra)


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return _descriptor(path)


def _files(folder):
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


def _models(tmp_path, *, target_matrix=None, target_shape=SHAPE, native=None):
    entries = {}
    photos = {}
    models = {}
    for number, run in enumerate(("source", "target")):
        shape = SHAPE if run == "source" else target_shape
        photo = tmp_path / (run + ".png")
        rgb = np.full((*shape, 3), 40 + number * 120, dtype=np.uint8)
        Image.fromarray(rgb).save(photo)
        photos[run] = photo
        model = dict(format=placement.GEOMETRY_FORMAT, run_id=run,
                     reference_image=_descriptor(photo), shape_hw=list(shape),
                     H_metric_mm_to_camera_u=(IDENTITY if run == "source" else
                         target_matrix if target_matrix is not None else IDENTITY).tolist())
        if native and run in native:
            transform, native_shape = native[run]
            model["baseline_native_grid"] = {
                "H_baseline_camera_u_to_rig_corrected_native": transform.tolist(),
                "shape_hw": list(native_shape),
            }
        models[run] = model
        model_path = tmp_path / (run + "-model.json")
        entries[run] = dict(reference_model=_write_json(model_path, model))
    index = tmp_path / "index.json"
    _write_json(index, dict(format=placement.GEOMETRY_FORMAT, runs=entries))
    session = lp.PaintingSession.default("source", photos["source"], SIZE)
    session.polygons = [lp.Polygon(2, [(3, 3), (3, 12), (12, 12), (12, 3)])]
    session.edge_fill_mm = 0
    return index, session, models, photos


def _load_photo(run, geometry):
    with Image.open(geometry.reference_image["path"]) as photo:
        return np.asarray(photo.convert("RGB"))


def _prepare(index, session, layers, **kwargs):
    return preview.prepare_preview(session, layers, geometry_index=index,
                                   target_run="target", config=kwargs.pop("config", {}),
                                   photo_loader=kwargs.pop("photo_loader", _load_photo),
                                   **kwargs)


def test_exact_native_and_display_grid_proof_is_additive_and_keeps_preview_products(tmp_path):
    index, session, _models_read, _photos = _models(tmp_path)
    rows, cols = np.indices(SHAPE)
    layers = (2 + ((rows // 4 + cols // 4) % 2)).astype(np.uint8)
    before = layers.copy(), deepcopy(session.to_dict()), _files(tmp_path)
    full = _prepare(index, session, layers, display_limit=1000)
    reduced = _prepare(index, session, layers, display_limit=16)
    assert full.native_shape_hw == SHAPE and full.display_shape_hw == SHAPE
    assert full.H_native_to_display == ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    assert np.array_equal(full.layers, layers)
    assert not full.uncovered.any()
    assert np.array_equal(full.photo, np.full((*SHAPE, 3), 160, np.uint8))
    assert reduced.native_shape_hw == SHAPE and reduced.display_shape_hw == (12, 16)
    assert reduced.H_native_to_display == ((.5, 0., -.25), (0., .5, -.25), (0., 0., 1.))
    assert np.array_equal(reduced.layers, layers[::2, ::2])
    assert np.array_equal(reduced.photo, np.full((12, 16, 3), 160, np.uint8))
    assert not reduced.uncovered.any()
    assert reduced.photo_path == full.photo_path and reduced.source_run == full.source_run
    assert reduced.measurements == full.measurements
    legacy = preview.RunPreview("target", full.photo, full.layers, full.uncovered,
                                full.photo_path, "source", "Drawing")
    assert legacy.native_shape_hw is None and legacy.display_shape_hw is None
    assert legacy.H_native_to_display is None
    assert np.array_equal(layers, before[0]) and session.to_dict() == before[1]
    assert _files(tmp_path) == before[2]


def test_early_preview_precedes_optional_reference_loading_and_preserves_exact_result(tmp_path):
    target_h = np.array([[1., 0., 3.], [0., 1., 2.], [0., 0., 1.]])
    index, session, _models_read, _photos = _models(tmp_path, target_matrix=target_h)
    layers = np.full(SHAPE, 2, np.uint8)
    baseline = _prepare(index, session, layers, display_limit=16)
    packets, stages, calls = [], [], []

    def load(run, geometry):
        calls.append(run)
        if run == "source":
            assert len(packets) == 1
            early = packets[0]
            assert early.photo_path == str(_photos["target"])
            assert np.array_equal(early.photo, baseline.photo)
            assert np.array_equal(early.layers, baseline.layers)
            assert np.array_equal(early.uncovered, baseline.uncovered)
            assert "layer_offsets" not in early.measurements
        return _load_photo(run, geometry)

    result = _prepare(index, session, layers, display_limit=16, measure_offsets=True,
                      photo_loader=load, on_progress=stages.append, on_preview=packets.append)
    assert calls == ["target", "source"]
    assert np.array_equal(result.photo, baseline.photo)
    assert np.array_equal(result.layers, baseline.layers)
    assert np.array_equal(result.uncovered, baseline.uncovered)
    assert "layer_offsets" in result.measurements and "layer_offsets" not in packets[0].measurements
    assert stages[0] == "Reading measured frame models" and stages[-1] == "Preview ready"
    assert any("Loading target" in stage for stage in stages)
    assert any("Measuring boundaries on source" in stage for stage in stages)


def test_optional_boundary_failure_leaves_a_bound_early_preview(tmp_path):
    index, session, _models_read, _photos = _models(tmp_path)
    packets = []

    def load(run, geometry):
        if run == "source":
            raise OSError("reference photograph unavailable")
        return _load_photo(run, geometry)

    with pytest.raises(OSError, match="reference photograph unavailable"):
        _prepare(index, session, np.full(SHAPE, 2, np.uint8), measure_offsets=True,
                 photo_loader=load, on_preview=packets.append)
    assert len(packets) == 1 and packets[0].run_id == "target"
    assert packets[0].photo_path == str(_photos["target"])
    assert "layer_offsets" not in packets[0].measurements


def test_projective_placement_matches_independent_pixel_landmarks_and_writes_nothing(tmp_path):
    target_h = np.array([[1, .05, 3], [.03, 1, 2], [.002, .001, 1]], dtype=float)
    index, session, _models_read, _photos = _models(tmp_path, target_matrix=target_h)
    # Every row and column has a distinct categorical value. Expected labels
    # below come from inverse projective coordinates, not another warp call.
    rows, cols = np.indices(SHAPE)
    layers = (1 + (rows // 4) * 8 + cols // 4).astype(np.uint8)
    before_files, before_session, before_layers = _files(tmp_path), deepcopy(session.to_dict()), layers.copy()
    result = _prepare(index, session, layers, source_name="Current drawing")
    inverse = np.linalg.inv(target_h)
    for row, col in [(6, 7), (12, 13), (18, 23), (20, 15)]:
        source = inverse @ np.array([col, row, 1.])
        source /= source[2]
        source_col, source_row = int(np.rint(source[0])), int(np.rint(source[1]))
        assert 0 <= source_row < SHAPE[0] and 0 <= source_col < SHAPE[1]
        assert result.layers[row, col] == layers[source_row, source_col]
        assert not result.uncovered[row, col]
    assert result.uncovered[0, 0]
    assert result.run_id == "target" and result.source_run == "source"
    assert result.source_name == "Current drawing"
    assert np.array_equal(result.photo, np.full((*SHAPE, 3), 160, np.uint8))
    assert np.array_equal(layers, before_layers)
    assert session.to_dict() == before_session and _files(tmp_path) == before_files


def test_integer_camera_shift_preserves_labels_and_marks_uncovered_edge(tmp_path):
    target_h = np.array([[1, 0, 3], [0, 1, 2], [0, 0, 1]], dtype=float)
    index, session, _models_read, _photos = _models(tmp_path, target_matrix=target_h)
    layers = np.arange(SHAPE[0] * SHAPE[1], dtype=np.uint16).reshape(SHAPE).astype(np.uint8)
    result = _prepare(index, session, layers)
    assert np.array_equal(result.layers[2:, 3:], layers[:-2, :-3])
    assert result.uncovered[:2].all() and result.uncovered[:, :3].all()
    assert not result.uncovered[2:, 3:].any()


def test_current_drawing_without_proposal_or_approval_remains_in_memory(tmp_path):
    index, session, _models_read, _photos = _models(tmp_path)
    session.strokes = [lp.Stroke(3, 1., [(17., 20.)])]
    before = _files(tmp_path), deepcopy(session.to_dict())
    returned, layers = preview.preview_layers(session, None, (), None, source="map")
    assert returned is session
    assert layers[6, 6] == 2 and layers[17, 20] == 3
    assert layers[20, 3] == lp.OUTSIDE
    result = _prepare(index, returned, layers)
    assert np.array_equal(result.layers, layers)
    assert _files(tmp_path) == before[0] and session.to_dict() == before[1]


def test_stale_proposal_refused_but_current_drawing_still_available(tmp_path):
    _index, session, _models_read, _photos = _models(tmp_path)
    proposal = lp.LayerProposal(np.ones(SHAPE, np.uint8), np.ones(SHAPE, np.float32),
                                SIZE, "stale")
    before = deepcopy(session.to_dict()), proposal.classes.copy()
    with pytest.raises(ValueError, match="out of date"):
        preview.preview_layers(session, proposal, (), None)
    returned, layers = preview.preview_layers(session, proposal, (), None, source="drawing")
    assert returned is session and layers[6, 6] == 2 and layers[20, 3] == lp.OUTSIDE
    assert session.to_dict() == before[0] and np.array_equal(proposal.classes, before[1])


def test_current_manual_map_does_not_preview_a_different_existing_proposal(tmp_path):
    _index, session, _models_read, _photos = _models(tmp_path)
    proposal = lp.LayerProposal(np.full(SHAPE, 3, np.uint8), np.ones(SHAPE, np.float32),
                                SIZE, "stale")
    expected = manual.manual_map(session, SIZE, frame=None).classes
    returned, layers = preview.preview_layers(session, proposal, (), None, source="manual")
    assert returned is session and np.array_equal(layers, expected)
    assert layers[6, 6] == 2 and layers[20, 3] == lp.OUTSIDE


def test_unapproved_saved_manual_map_with_gaps_is_previewed_exactly_without_writes(tmp_path):
    index, session, _models_read, _photos = _models(tmp_path)
    project = SimpleNamespace(root=tmp_path / "project")
    frame = RigFrame(corners={"top_left": (-.5, -.5), "top_right": (-.5, 31.5),
                              "bottom_right": (23.5, 31.5), "bottom_left": (23.5, -.5)},
                     width_m=.032, height_m=.024)
    saved_map = manual.manual_map(session, SIZE, frame=frame)
    assert saved_map.uncovered_pixels() > 0
    record = manual.save_manual_version(project, session, saved_map, name="Draft map")
    before = _files(tmp_path)
    # No approval is manufactured. The saved incomplete map is exactly what
    # the user needs to inspect before deciding whether it can be approved.
    assert not list(tmp_path.rglob("approvals.jsonl"))
    returned, layers = preview.preview_layers(session, None, (), frame, record=record)
    assert np.array_equal(layers, saved_map.classes)
    result = _prepare(index, returned, layers)
    assert np.array_equal(result.layers, saved_map.classes)
    assert (result.layers == lp.OUTSIDE).any()
    assert _files(tmp_path) == before


@pytest.mark.parametrize("which", ["source_shape", "source_path", "target_shape", "target_path"])
def test_unbound_photo_grid_or_path_is_refused_without_scaling_or_raw_fallback(tmp_path, which):
    index, session, models, _photos = _models(tmp_path)
    layers = np.ones(SHAPE, np.uint8)
    loader = _load_photo
    if which == "source_shape":
        layers = layers[:-1]
    elif which == "source_path":
        session.image_path = str(tmp_path / "another-photo.png")
    elif which == "target_shape":
        loader = lambda *_: np.zeros((23, 32, 3), np.uint8)
    else:
        models["target"]["reference_image"].pop("path")
        target_path = tmp_path / "target-model.json"
        bound = _write_json(target_path, models["target"])
        document = json.loads(index.read_text())
        document["runs"]["target"]["reference_model"] = bound
        _write_json(index, document)
        loader = lambda *_: np.zeros((*SHAPE, 3), np.uint8)
    before = _files(tmp_path)
    with pytest.raises(ValueError, match="grid|photograph|coordinate frame"):
        _prepare(index, session, layers, photo_loader=loader)
    assert _files(tmp_path) == before


def _corrected(tmp_path, *, shape=(20, 30)):
    runs = {}
    for number, run in enumerate(("source", "target")):
        fields = {}
        folder = tmp_path / (run + "-corrected")
        folder.mkdir()
        for name in wf.FIELDS:
            array = (np.full((*shape, 3), .25 + .5 * number, np.float32)
                     if name == "shape_corrected_baseline" else np.zeros(shape, np.int32))
            path = folder / (name + ".npz")
            np.savez(path, array=array)
            fields[name] = _descriptor(path, shape=list(array.shape), mtime_ns=path.stat().st_mtime_ns)
        runs[run] = dict(fields=fields, configured_depth_map=dict(path=str(folder / "original-depth.npz")))
    correction = tmp_path / "correction.json"
    correction.write_text("{}", encoding="utf-8")
    return dict(frame="rig-corrected", width_m=.03, height_m=.02,
                frame_binding=dict(format=wf.FRAME_BINDING_FORMAT, source_reference_run="source",
                    display_reference_run="source", reference_shape_corrections=[_descriptor(correction)], runs=runs))


def test_corrected_target_same_shape_and_transport_do_not_prove_saved_baseline_identity(tmp_path):
    shape = (20, 30)
    native_h = np.array([[1, 0, -2], [0, 1, -3], [0, 0, 1]], dtype=float)
    target_h = np.array([[1, 0, 3], [0, 1, 2], [0, 0, 1]], dtype=float)
    index, session, _models_read, _photos = _models(tmp_path, target_matrix=target_h,
                                                   native={"target": (native_h, shape)})
    config = _corrected(tmp_path, shape=shape)
    rows, cols = np.indices(SHAPE)
    layers = (rows * 8 + cols).astype(np.uint8)
    before = _files(tmp_path), deepcopy(config)
    # Both records name this same-sized grid, but neither establishes that
    # the renderer produced this particular saved baseline. A successful
    # preview would silently invent the missing coordinate-frame binding.
    with pytest.raises(ValueError, match="bind|renderer|saved|native|corrected"):
        _prepare(index, session, layers, config=config,
                 photo_loader=lambda *_: pytest.fail("unbound corrected target fell back to raw"))
    assert config == before[1] and _files(tmp_path) == before[0]


def test_corrected_source_also_refuses_shape_only_renderer_correspondence(tmp_path):
    shape = (20, 30)
    source_native = np.array([[1, 0, -4], [0, 1, -1], [0, 0, 1]], dtype=float)
    target_native = np.array([[1, 0, -2], [0, 1, -3], [0, 0, 1]], dtype=float)
    target_h = np.array([[1, 0, 3], [0, 1, 2], [0, 0, 1]], dtype=float)
    index, session, _models_read, _photos = _models(tmp_path, target_matrix=target_h,
        native={"source": (source_native, shape), "target": (target_native, shape)})
    config = _corrected(tmp_path, shape=shape)
    session.image_path = config["frame_binding"]["runs"]["source"]["fields"]["shape_corrected_baseline"]["path"]
    session.image_size = (30, 20)
    rows, cols = np.indices(shape)
    layers = (rows * 8 + cols).astype(np.uint8)
    before = _files(tmp_path), deepcopy(session.to_dict())
    with pytest.raises(ValueError, match="bind|renderer|saved|native|corrected"):
        _prepare(index, session, layers, config=config,
                 photo_loader=lambda *_: pytest.fail("unbound corrected source was treated as raw"))
    assert session.to_dict() == before[1] and _files(tmp_path) == before[0]


@pytest.mark.parametrize("mutation", ["missing_transport", "wrong_transport_shape", "missing_target_field"])
def test_corrected_preview_refuses_missing_or_different_target_binding_without_raw_fallback(tmp_path, mutation):
    shape = (20, 30)
    transport_shape = (19, 30) if mutation == "wrong_transport_shape" else shape
    native = None if mutation == "missing_transport" else {"target": (IDENTITY, transport_shape)}
    index, session, _models_read, _photos = _models(tmp_path, native=native)
    config = _corrected(tmp_path, shape=shape)
    if mutation == "missing_target_field":
        del config["frame_binding"]["runs"]["target"]["fields"]["shape_corrected_baseline"]
    before = _files(tmp_path)
    with pytest.raises(ValueError, match="native grid|corrected grid|explicit absolute|renderer|saved baseline|identity"):
        _prepare(index, session, np.ones(SHAPE, np.uint8), config=config,
                 photo_loader=lambda *_: pytest.fail("unsupported corrected preview fell back to raw"))
    assert _files(tmp_path) == before


def test_changed_bound_model_is_refused_before_any_photo_loader(tmp_path):
    index, session, _models_read, _photos = _models(tmp_path)
    model = tmp_path / "target-model.json"
    model.write_text(model.read_text().replace('"target"', '"other"'), encoding="utf-8")
    with pytest.raises(ValueError, match="reference model the index bound"):
        _prepare(index, session, np.ones(SHAPE, np.uint8),
                 photo_loader=lambda *_: pytest.fail("tampered model reached image reader"))


def test_native_grid_is_placed_before_display_reduction(tmp_path):
    index, session, _models_read, _photos = _models(tmp_path)
    layers = np.zeros(SHAPE, np.uint8)
    layers[:12, :16] = 2
    layers[12:, 16:] = 3
    result = _prepare(index, session, layers, display_limit=16)
    assert result.photo.shape == (12, 16, 3) and result.layers.shape == (12, 16)
    assert np.array_equal(result.layers, layers[::2, ::2])
    assert not result.uncovered.any()


def test_geometry_packet_binds_index_bytes_models_and_used_homographies(tmp_path):
    index, session, models, _photos = _models(tmp_path)
    result = _prepare(index, session, np.full(SHAPE, 2, np.uint8))
    binding = result.geometry_binding
    assert binding["geometry_index"] == _descriptor(index)
    for run in ("source", "target"):
        row = binding["models"][run]
        assert row["model"] == _descriptor(tmp_path / (run + "-model.json"))
        assert row["reference_image"] == models[run]["reference_image"]
        assert row["shape_hw"] == list(SHAPE)
        assert row["metric_to_grid"] == models[run]["H_metric_mm_to_camera_u"]
    assert result.native_shape_hw == SHAPE and result.display_shape_hw == SHAPE


@pytest.mark.parametrize("changed", ["index", "source", "target"])
def test_geometry_snapshot_refuses_changes_during_existing_reader(tmp_path, monkeypatch, changed):
    index, _session, _models_read, _photos = _models(tmp_path)
    original = placement.read_run_geometry
    def changed_after_read(path):
        models = original(path)
        if changed == "index":
            document = json.loads(index.read_text())
            document["display_note"] = "changed during the read"
            _write_json(index, document)
        else:
            model = tmp_path / (changed + "-model.json")
            document = json.loads(model.read_text())
            document["H_metric_mm_to_camera_u"][0][2] = 3.
            _write_json(model, document)
        return models
    monkeypatch.setattr(placement, "read_run_geometry", changed_after_read)
    with pytest.raises(ValueError, match="changed"):
        preview.read_preview_geometry(index)


def test_early_and_final_packets_have_independent_identical_geometry_proof(tmp_path, monkeypatch):
    index, session, _models_read, _photos = _models(tmp_path)
    packets = []
    monkeypatch.setattr(placement, "layer_offsets", lambda *args: {"boundaries": []})
    monkeypatch.setattr(placement, "add_changes_from_reference", lambda *args: None)
    result = _prepare(index, session, np.full(SHAPE, 2, np.uint8), measure_offsets=True,
                      on_preview=packets.append, display_limit=16)
    assert len(packets) == 1
    early = packets[0]
    assert early.geometry_binding == result.geometry_binding
    assert early.geometry_binding is not result.geometry_binding
    assert early.H_native_to_display == result.H_native_to_display
    assert early.native_shape_hw == result.native_shape_hw == SHAPE
    early.geometry_binding["models"]["source"]["metric_to_grid"][0][2] = 999.
    assert result.geometry_binding["models"]["source"]["metric_to_grid"][0][2] == 0.
